from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from lib.train.admin import multigpu


class BATModEEActor(BaseActor):
    """ Actor for training BAT models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        if self.cfg.TRAIN.STAGE == 2:
            loss, status = self.compute_losses2(out_dict, data)
        elif self.cfg.TRAIN.STAGE == 3:
            loss, status = self.compute_losses3(out_dict, data)
        else:
            loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
            # ce_keep_rate = 0.7

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            random_ee=self.cfg.TRAIN.RANDOM_EE,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        
        '''mod_choose loss
        '''
        # mod_choose_loss = 0.
        mod_choose = pred_dict['mod_choose']
        gt_choose = gt_dict['search_mod_choose']
        if self.cfg.MODEL.MODEE.LOSS_TYPE == 'cross_entropy':
            mod_choose_loss = self.objective['ce_loss'](mod_choose, gt_choose.squeeze(0).long())
        elif self.cfg.MODEL.MODEE.LOSS_TYPE == 'regress_mse':
            gt_choose_list = []
            for i in gt_choose.squeeze(0):
                if i == 0:
                    gt_choose_list.append([1,0,0])
                elif i == 1:
                    gt_choose_list.append([0,1,0])
                else:
                    gt_choose_list.append([0,0,1])
            gt_choose = torch.Tensor(gt_choose_list).to(device=mod_choose.device)
            mod_choose_loss = self.objective['mse'](mod_choose, gt_choose)
        
        # reconstruct loss, i.e. distill loss
        reconstruct_loss = torch.tensor(0.0).cuda()
        if self.cfg.TRAIN.STAGE == 'joint_1':
            target = torch.ones(1).cuda()
            n = 0
            if pred_dict['tir_reconstruct'] is not None:
                if self.cfg.MODEL.RECONSTRUCTOR.LOSS_TYPE == 'mse':
                    reconstruct_loss += self.objective['mse'](pred_dict['tir_reconstruct'][:,-320], pred_dict['supervise_tir'][:,-320].detach())
                elif self.cfg.MODEL.RECONSTRUCTOR.LOSS_TYPE == 'cosloss':
                    reconstruct_loss += self.objective['cosloss'](pred_dict['tir_reconstruct'][:,-320].reshape(-1,768), pred_dict['supervise_tir'][:,-320].reshape(-1,768).detach(), target)
                n += 1
            if pred_dict['rgb_reconstruct'] is not None: 
                if self.cfg.MODEL.RECONSTRUCTOR.LOSS_TYPE == 'mse':
                    reconstruct_loss += self.objective['mse'](pred_dict['rgb_reconstruct'][:,-320], pred_dict['supervise_rgb'][:,-320].detach())
                elif self.cfg.MODEL.RECONSTRUCTOR.LOSS_TYPE == 'cosloss':
                    reconstruct_loss += self.objective['cosloss'](pred_dict['rgb_reconstruct'][:,-320].reshape(-1,768), pred_dict['supervise_rgb'][:,-320].reshape(-1,768).detach(), target)
                n += 1
            reconstruct_loss /= n if n != 0 else torch.tensor(0.0).cuda()

        distill_loss = torch.tensor(0.0, device=l1_loss.device)
        if "distill" in self.cfg.MODEL.MODEE.TYPE:
            pred_distill = pred_dict['distill']
            distill_blk_x = pred_dict['distill_blk_x']
            distill_loss = self.objective['mse'](pred_distill[0], distill_blk_x[0].detach()) + self.objective['mse'](pred_distill[1], distill_blk_x[1].detach())
            distill_loss /= 2

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        loss += self.loss_weight['ce_loss'] * mod_choose_loss

        if self.cfg.TRAIN.STAGE == 'joint_1':
            loss += self.loss_weight['reconstruct'] * reconstruct_loss

        if self.cfg.MODEL.MODEE.TYPE == "linear_v3_distill":
            loss += self.loss_weight['distill_loss'] * distill_loss
        
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                    #   "Loss/giou": giou_loss.item(),
                    #   "Loss/l1": l1_loss.item(),
                    #   "Loss/location": location_loss.item(),
                      "Loss/mod_choose": mod_choose_loss.item(),
                      "Loss/distill": distill_loss.item(),
                      "Loss/reconstruct": reconstruct_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
        
    def compute_losses2(self, pred_dict, gt_dict, return_status=True):

        task_loss = torch.tensor(0.0).cuda()

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                        max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        
        if self.cfg.MODEL.RECONSTRUCTOR.TRAIN_HEAD:
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            # compute location loss
            if 'score_map' in pred_dict:
                location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            task_loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        
        # reconstruct loss
        reconstruct_loss = torch.tensor(0.0).cuda()
        target = torch.ones(1).cuda()
        n = 0
        if pred_dict['tir_reconstruct'] is not None:
            if self.cfg.MODEL.RECONSTRUCTOR.LOSS_TYPE == 'mse':
                reconstruct_loss += self.objective['mse'](pred_dict['tir_reconstruct'][:,-256], pred_dict['supervise_tir'][:,-256].detach())
            elif self.cfg.MODEL.RECONSTRUCTOR.LOSS_TYPE == 'cosloss':
                reconstruct_loss += self.objective['cosloss'](pred_dict['tir_reconstruct'][:,-256].reshape(-1,768), pred_dict['supervise_tir'][:,-256].reshape(-1,768).detach(), target)
            n += 1
        if pred_dict['rgb_reconstruct'] is not None: 
            if self.cfg.MODEL.RECONSTRUCTOR.LOSS_TYPE == 'mse':
                reconstruct_loss += self.objective['mse'](pred_dict['rgb_reconstruct'][:,-256], pred_dict['supervise_rgb'][:,-256].detach())
            elif self.cfg.MODEL.RECONSTRUCTOR.LOSS_TYPE == 'cosloss':
                reconstruct_loss += self.objective['cosloss'](pred_dict['rgb_reconstruct'][:,-256].reshape(-1,768), pred_dict['supervise_rgb'][:,-256].reshape(-1,768).detach(), target)
            n += 1
        reconstruct_loss /= n if n != 0 else torch.tensor(0.0).cuda()

        # weighted sum
        if self.cfg.MODEL.RECONSTRUCTOR.TRAIN_HEAD:
            loss = task_loss + self.loss_weight['reconstruct'] * reconstruct_loss
        else:
            loss = reconstruct_loss
        
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/task": task_loss.item(),
                      "Loss/reconstruct": reconstruct_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
        
    def compute_losses3(self, pred_dict, gt_dict, return_status=True):

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                        max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/task": loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss