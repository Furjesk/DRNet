import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath

from lib.models.layers.attn import Attention
from lib.models.layers.mimic_layer import IOUPredictor_Linear_Sig_Dim, DynamicLoRa, ModEEPredictor_v5_mamba, DynamicMlpMimicAttn, ModEEPredictor_v4_distill, IOUPredictor_Linear_Sig, DynamicMlp_DM, DynamicCrossMlp, DynamicViT, MIMICLayer, IOUPredictor, IOUPredictor_v2, IOUPredictor_v3, MIMICLayerV2, DynamicMlp, ModEEPredictor, ModEEPredictor_v2, ModEEPredictor_v3_distill, DynamicMMBlock
from lib.models.layers.mimic_layer import ModEEPredictor_token, IOUPredictor_Linear_Sig_Avg_Dim



def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t    
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]

    


    if box_mask_z is not None:
        #print("\n1\n1\n1")
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)



    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]
    
    keep_index = global_index.gather(dim=1, index=topk_idx)
    
    removed_index = global_index.gather(dim=1, index=non_topk_idx)
    

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)

    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens
    
    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    #print("finish ce func")

    return tokens_new, keep_index, removed_index                       # x, global_index_search, removed_index_search


def make_mimic_layer(layer_id, mimic_loc=[3,6,9], mimic_type=None, embed_dim=768, norm_layer=nn.LayerNorm, drop=0.,
                     attn_drop=0., num_heads=8, mlp_ratio=4., qkv_bias=False, drop_path=0., act_layer=nn.GELU, num_tokens=320):
    if layer_id in mimic_loc:
        if mimic_type == 'single_linear':
            return MIMICLayerV2(dim=embed_dim)
        elif mimic_type == 'single_mlp':
            return Mlp(in_features=embed_dim, act_layer=nn.GELU, drop=drop)
        elif mimic_type == 'single_adapter':
            return MIMICLayer()
        elif mimic_type == 'dynamic_mlp':
            layer_num = (11 - layer_id) // 2
            return DynamicMlp(in_features=embed_dim, layer_num=layer_num, norm_layer=norm_layer, drop=drop)
        elif mimic_type == 'single_linear_v2':
            layer_num = 1
            return DynamicMlp(in_features=embed_dim, layer_num=layer_num, norm_layer=norm_layer, drop=drop)
        elif mimic_type == 'dynamic_mlp_distill_medium':
            layer_num = (11 - layer_id) // 2
            return DynamicMlp_DM(in_features=embed_dim, layer_num=layer_num, norm_layer=norm_layer, drop=drop)
        elif mimic_type == 'dynamic_li_attn':
            layer_num = (11 - layer_id) // 2
            return DynamicMMBlock(in_features=embed_dim, layer_num=layer_num, drop=drop, attn_drop=attn_drop)
        elif mimic_type == 'single_li_attn':
            layer_num = 1
            return DynamicMMBlock(in_features=embed_dim, layer_num=layer_num, drop=drop, attn_drop=attn_drop)
        elif mimic_type == 'dynamic_vit':
            layer_num = 2 if (layer_id < 6) else 1
            return DynamicViT(embed_dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer, layer_num=layer_num)
        elif mimic_type == 'dynamic_cross_mlp':
            layer_num = (11 - layer_id) // 2
            return DynamicCrossMlp(num_tokens=num_tokens, in_features=embed_dim, layer_num=layer_num, norm_layer=norm_layer, drop=drop)
        elif mimic_type == 'dynamic_vit2':
            layer_num = (11 - layer_id) // 2
            return DynamicViT(embed_dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer, layer_num=layer_num)
        elif mimic_type == 'dynamic_lora':
            layer_num = (11 - layer_id) // 2
            return DynamicLoRa(layer_num=layer_num)
        elif mimic_type == 'dynamic_mimic_attn':
            layer_num = (11 - layer_id) // 2 # , norm_layer=norm_layer, drop=drop
            return DynamicMlpMimicAttn(num_tokens=320, layer_num=layer_num)
    else:
        return None
    
def make_iou_predictor(layer_id, mimic_loc=[3,6,9], embed_dim=768, predictor_type="two_conv"):
    if layer_id in mimic_loc and 'share' not in predictor_type:
        if predictor_type == 'two_conv':
            iou_predictor = IOUPredictor(embed_dim) # param: search_size=self.num_patches_search // 2
        elif predictor_type == 'two_linear':
            iou_predictor = IOUPredictor_v2(embed_dim)
        elif predictor_type == 'two_linear_sigmoid':
            iou_predictor = IOUPredictor_v3(embed_dim)
        elif predictor_type == 'discrete_two_conv':
            iou_predictor = nn.ModuleList([IOUPredictor(embed_dim),IOUPredictor(embed_dim)])
        elif predictor_type == 'linear_sig':
            iou_predictor = IOUPredictor_Linear_Sig(embed_dim)
        elif predictor_type == 'linear_sig_dim':
            iou_predictor = IOUPredictor_Linear_Sig_Dim(embed_dim)
        elif predictor_type == 'linear_sig_avg_dim':
            iou_predictor = IOUPredictor_Linear_Sig_Avg_Dim()
        return iou_predictor
    else:
        return None
    

class CEBlock_twobranch(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, backbone_type=None, layer_id=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search
        
        self.mimic_layer = None
        if 'mimic' in backbone_type and layer_id != 11:
            self.mimic_layer = MIMICLayer()

    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)
                     
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))

        if self.mimic_layer is not None:
            mimic_x = self.mimic_layer(x)
            mimic_xi = self.mimic_layer(xi)
            
            return x, global_index_template, global_index_search, removed_index_search, attn, mimic_x, \
                xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, mimic_xi
        else:
            return x, global_index_template, global_index_search, removed_index_search, attn, \
                xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn


class ModEEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, 
                 modee_type=None, use_softmax=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search
        ############
        self.modee_type = modee_type
        self.mod_ee_predictor = None
        if modee_type is not None:
            if modee_type == 'linear_v1':
                self.mod_ee_predictor = ModEEPredictor(dim, type_num=3, use_softmax=use_softmax)
            elif 'linear_v2' in modee_type:
                if modee_type == 'linear_v2_nol4':
                    self.mod_ee_predictor = ModEEPredictor_v2(dim, type_num=3, use_softmax=use_softmax, linear4=False)
                elif modee_type == 'linear_v2':
                    self.mod_ee_predictor = ModEEPredictor_v2(dim, type_num=3, use_softmax=use_softmax)
            elif modee_type == 'linear_v3_distill':
                self.mod_ee_predictor = ModEEPredictor_v3_distill(dim, type_num=3, use_softmax=use_softmax)
            elif modee_type == 'linear_v4_distill':
                self.mod_ee_predictor = ModEEPredictor_v4_distill(dim, type_num=3, use_softmax=use_softmax)
            elif modee_type == 'linear_v5_mamba':
                self.mod_ee_predictor = ModEEPredictor_v5_mamba(dim, type_num=3, use_softmax=use_softmax)
            elif modee_type == 'linear_sig_token':
                self.mod_ee_predictor = ModEEPredictor_token(dim, type_num=3, use_softmax=use_softmax)
            # elif modee_type == 'linear_sig_avgdim':
            #     self.mod_ee_predictor = ModEEPredictor_token(dim, type_num=3, use_softmax=use_softmax)


    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None,
                is_test=False, has_cls_token=False):
        mod_choose = None
        distill = None
        if self.mod_ee_predictor is not None:
            if has_cls_token:
                mod_choose = self.mod_ee_predictor(torch.stack((x[:,0], xi[:,0]),dim=1))
            else:
                if 'distill' in self.modee_type:
                    mod_choose, distill = self.mod_ee_predictor(x, xi)
                else:
                    mod_choose = self.mod_ee_predictor(x, xi)
            if is_test: # 测试可以这么写
                if mod_choose.max(dim=1)[1] == 0:
                    xi = None
                elif mod_choose.max(dim=1)[1] == 1:
                    x = None
        if x is not None and xi is not None:
            return *self.forward_two_modal(x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask, ce_template_mask, keep_ratio_search),\
                mod_choose, distill
        elif x is not None:
            x, global_index_template, global_index_search, removed_index_search, attn = self.forward_one_modal(x, global_index_template, global_index_search, mask, ce_template_mask, keep_ratio_search)
            return x, global_index_template, global_index_search, removed_index_search, attn, \
                xi, global_index_templatei, global_index_searchi, None, None, mod_choose, distill
        elif xi is not None:
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn = self.forward_one_modal(xi, global_index_templatei, global_index_searchi, mask, ce_template_mask, keep_ratio_search)
            return x, global_index_template, global_index_search, None, None, \
                xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, mod_choose, distill
        else:
            print("代码有误")
            return
    
    def forward_one_modal(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        lens_t = global_index_template.shape[1]
        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn
    
    def forward_two_modal(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)
                     
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))

        return x, global_index_template, global_index_search, removed_index_search, attn, \
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn
    
class ModLayerEEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, 
                 modee_type=None, use_softmax=False, mimic_loc=None, mimic_type=None, layer_id=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search
        ############
        self.mimic_layer = make_mimic_layer(layer_id, mimic_loc, mimic_type, embed_dim=dim, norm_layer=norm_layer, drop=drop)

        self.mod_ee_predictor = None
        if modee_type is not None:
            if modee_type == 'linear_v1':
                self.mod_ee_predictor = ModEEPredictor(dim, type_num=3, use_softmax=use_softmax)
            elif 'linear_v2' in modee_type:
                if modee_type == 'linear_v2_nol4':
                    self.mod_ee_predictor = ModEEPredictor_v2(dim, type_num=3, use_softmax=use_softmax, linear4=False)
                elif modee_type == 'linear_v2':
                    self.mod_ee_predictor = ModEEPredictor_v2(dim, type_num=3, use_softmax=use_softmax)


    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None,
                is_test=False):
        mod_choose = None
        if self.mod_ee_predictor is not None:
            mod_choose = self.mod_ee_predictor(x, xi)
            if is_test: # 测试可以这么写
                if mod_choose.max(dim=1)[1] == 0:
                    xi = None
                elif mod_choose.max(dim=1)[1] == 1:
                    x = None
        if x is not None and xi is not None:
            return *self.forward_two_modal(x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask, ce_template_mask, keep_ratio_search),\
                mod_choose
        elif x is not None:
            x, global_index_template, global_index_search, removed_index_search, attn = self.forward_one_modal(x, global_index_template, global_index_search, mask, ce_template_mask, keep_ratio_search)
            return x, global_index_template, global_index_search, removed_index_search, attn, \
                xi, global_index_templatei, global_index_searchi, None, None, mod_choose
        elif xi is not None:
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn = self.forward_one_modal(xi, global_index_templatei, global_index_searchi, mask, ce_template_mask, keep_ratio_search)
            return x, global_index_template, global_index_search, None, None, \
                xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, mod_choose
        else:
            print("代码有误")
            return
    
    def forward_one_modal(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        lens_t = global_index_template.shape[1]
        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn
    
    def forward_two_modal(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)
                     
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))

        return x, global_index_template, global_index_search, removed_index_search, attn, \
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn


class ModLayerCoEEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, 
                 modee_type=None, use_softmax=False, mimic_loc=None, mimic_type=None, layer_id=None,
                 has_mimic=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search
        ############
        self.mimic_layer = None
        if has_mimic:
            self.mimic_layer = make_mimic_layer(layer_id, mimic_loc, mimic_type, embed_dim=dim, norm_layer=norm_layer, drop=drop)

        self.mod_ee_predictor = None
        if modee_type is not None:
            if modee_type == 'linear_v1':
                self.mod_ee_predictor = ModEEPredictor(dim, type_num=3, use_softmax=use_softmax)
            elif 'linear_v2' in modee_type:
                if modee_type == 'linear_v2_nol4':
                    self.mod_ee_predictor = ModEEPredictor_v2(dim, type_num=3, use_softmax=use_softmax, linear4=False)
                elif modee_type == 'linear_v2':
                    self.mod_ee_predictor = ModEEPredictor_v2(dim, type_num=3, use_softmax=use_softmax)


    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None,
                is_test=False):
        mod_choose = None
        if self.mod_ee_predictor is not None:
            mod_choose = self.mod_ee_predictor(x[:,-320:], xi[:,-320:])
            if is_test: # 测试可以这么写
                if mod_choose.max(dim=1)[1] == 0:
                    xi = None
                elif mod_choose.max(dim=1)[1] == 1:
                    x = None
        if x is not None and xi is not None:
            return *self.forward_two_modal(x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask, ce_template_mask, keep_ratio_search),\
                mod_choose
        elif x is not None:
            x, global_index_template, global_index_search, removed_index_search, attn = self.forward_one_modal(x, global_index_template, global_index_search, mask, ce_template_mask, keep_ratio_search)
            return x, global_index_template, global_index_search, removed_index_search, attn, \
                xi, global_index_templatei, global_index_searchi, None, None, mod_choose
        elif xi is not None:
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn = self.forward_one_modal(xi, global_index_templatei, global_index_searchi, mask, ce_template_mask, keep_ratio_search)
            return x, global_index_template, global_index_search, None, None, \
                xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, mod_choose
        else:
            print("代码有误")
            return
    
    def forward_one_modal(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        lens_t = global_index_template.shape[1]
        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn
    
    def forward_two_modal(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)
                     
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))

        return x, global_index_template, global_index_search, removed_index_search, attn, \
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn

class FixEEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, 
                 mimic_loc=None, mimic_type=None, layer_id=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search
        ############
        self.mimic_layer = make_mimic_layer(layer_id, mimic_loc, mimic_type, embed_dim=dim, norm_layer=norm_layer, drop=drop, attn_drop=attn_drop)

    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None):
        
        if x is not None and xi is not None:
            return self.forward_two_modal(x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask, ce_template_mask, keep_ratio_search)
        elif x is not None:
            x, global_index_template, global_index_search, removed_index_search, attn = self.forward_one_modal(x, global_index_template, global_index_search, mask, ce_template_mask, keep_ratio_search)
            return x, global_index_template, global_index_search, removed_index_search, attn, \
                xi, global_index_templatei, global_index_searchi, None, None
        elif xi is not None:
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn = self.forward_one_modal(xi, global_index_templatei, global_index_searchi, mask, ce_template_mask, keep_ratio_search)
            return x, global_index_template, global_index_search, None, None, \
                xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn
        else:
            print("代码有误")
            return
    
    def forward_one_modal(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        lens_t = global_index_template.shape[1]
        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn
    
    def forward_two_modal(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)
                     
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))

        # mimic_x = mimic_xi = None
        # if self.mimic_layer is not None:
        #     mimic_x = self.mimic_layer(x)
        #     mimic_xi = self.mimic_layer(xi)
            
        return x, global_index_template, global_index_search, removed_index_search, attn, \
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn
    
class FixEEJointBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, 
                 mimic_loc=None, mimic_type=None, layer_id=None, predictor_type=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search
        ############
        self.mimic_type = mimic_type
        self.mimic_layer = make_mimic_layer(layer_id, mimic_loc, mimic_type, embed_dim=dim, norm_layer=norm_layer, drop=drop,
                                            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                            drop_path=drop_path, act_layer=act_layer)
        self.iou_predictor = make_iou_predictor(layer_id, mimic_loc, embed_dim=dim, predictor_type=predictor_type)

    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None,
                lens_x=256, is_test=False):
        if x is not None and xi is not None:
            return self.forward_two_modal(x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask, ce_template_mask, keep_ratio_search, lens_x, is_test)
        elif x is not None:
            x, global_index_template, global_index_search, removed_index_search, attn, iou_x, mimic_x = self.forward_one_modal(x, global_index_template, global_index_search, mask, ce_template_mask, keep_ratio_search, lens_x)
            return x, global_index_template, global_index_search, removed_index_search, attn, iou_x, mimic_x, \
                xi, global_index_templatei, global_index_searchi, None, None, None, None
        elif xi is not None:
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, iou_xi, mimic_xi = self.forward_one_modal(xi, global_index_templatei, global_index_searchi, mask, ce_template_mask, keep_ratio_search, lens_x)
            return x, global_index_template, global_index_search, None, None, None, None, \
                xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, iou_xi, mimic_xi
        else:
            print("代码有误")
            return
    
    def forward_one_modal(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None, lens_x=256):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            lens_t = global_index_template.shape[1]
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # mimic_x = None
        iou_x = None
        if self.iou_predictor is not None:
            iou_x = self.iou_predictor(x, lens_x=lens_x)
            # mimic_x = self.mimic_layer(x)
        return x, global_index_template, global_index_search, removed_index_search, attn, iou_x, None
    
    def forward_two_modal(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None, lens_x=256, is_test=False):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)
                             
        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            lens_t = global_index_template.shape[1]
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))

        mimic_x = mimic_xi = None
        iou_x = iou_xi = None
        if self.iou_predictor is not None:
            iou_x = self.iou_predictor(x, lens_x=lens_x)
            iou_xi = self.iou_predictor(xi, lens_x=lens_x)
            if not is_test: # 训练才执行
                if self.mimic_type == 'dynamic_mimic_attn':
                    mimic_x = self.mimic_layer(attn)
                    mimic_xi = self.mimic_layer(i_attn)
                else:
                    mimic_x = self.mimic_layer(x)
                    mimic_xi = self.mimic_layer(xi)
            
        return x, global_index_template, global_index_search, removed_index_search, attn, iou_x, mimic_x, \
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, iou_xi, mimic_xi
    
class FixEEJointTokenBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, 
                 mimic_loc=None, mimic_type=None, layer_id=None, predictor_type=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search
        ############
        self.mimic_layer = make_mimic_layer(layer_id, mimic_loc, mimic_type, embed_dim=dim, norm_layer=norm_layer, drop=drop,
                                            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                            drop_path=drop_path, act_layer=act_layer)
        self.iou_predictor = make_iou_predictor(layer_id, mimic_loc, embed_dim=dim, predictor_type=predictor_type)

    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None,
                lens_x=256, is_test=False):
        if x is not None and xi is not None:
            return self.forward_two_modal(x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask, ce_template_mask, keep_ratio_search, lens_x, is_test)
        elif x is not None:
            x, global_index_template, global_index_search, removed_index_search, attn, iou_x, mimic_x = self.forward_one_modal(x, global_index_template, global_index_search, mask, ce_template_mask, keep_ratio_search, lens_x)
            return x, global_index_template, global_index_search, removed_index_search, attn, iou_x, mimic_x, \
                xi, global_index_templatei, global_index_searchi, None, None, None, None
        elif xi is not None:
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, iou_xi, mimic_xi = self.forward_one_modal(xi, global_index_templatei, global_index_searchi, mask, ce_template_mask, keep_ratio_search, lens_x)
            return x, global_index_template, global_index_search, None, None, None, None, \
                xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, iou_xi, mimic_xi
        else:
            print("代码有误")
            return
    
    def forward_one_modal(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None, lens_x=256):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            lens_t = global_index_template.shape[1]
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # mimic_x = None
        iou_x = None
        if self.iou_predictor is not None:
            iou_x = self.iou_predictor(x[:,0])
            # mimic_x = self.mimic_layer(x)
        return x, global_index_template, global_index_search, removed_index_search, attn, iou_x, None
    
    def forward_two_modal(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None, lens_x=256, is_test=False):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)
                             
        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            lens_t = global_index_template.shape[1]
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))

        mimic_x = mimic_xi = None
        iou_x = iou_xi = None
        if self.iou_predictor is not None:
            iou_x = self.iou_predictor(x[:,0])
            iou_xi = self.iou_predictor(xi[:,0])
            if not is_test: # 训练才执行
                mimic_x = self.mimic_layer(x[:,1:])
                mimic_xi = self.mimic_layer(xi[:,1:])
            
        return x, global_index_template, global_index_search, removed_index_search, attn, iou_x, mimic_x, \
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, iou_xi, mimic_xi


class FixEEJointTokenBlock_nomimic(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, 
                 mimic_loc=None, mimic_type=None, layer_id=None, predictor_type=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search
        ############
        self.iou_predictor = make_iou_predictor(layer_id, mimic_loc, embed_dim=dim, predictor_type=predictor_type)

    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None,
                lens_x=256, is_test=False):
        if x is not None and xi is not None:
            return self.forward_two_modal(x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask, ce_template_mask, keep_ratio_search, lens_x, is_test)
        elif x is not None:
            x, global_index_template, global_index_search, removed_index_search, attn, iou_x = self.forward_one_modal(x, global_index_template, global_index_search, mask, ce_template_mask, keep_ratio_search, lens_x)
            return x, global_index_template, global_index_search, removed_index_search, attn, iou_x, \
                xi, global_index_templatei, global_index_searchi, None, None, None
        elif xi is not None:
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, iou_xi = self.forward_one_modal(xi, global_index_templatei, global_index_searchi, mask, ce_template_mask, keep_ratio_search, lens_x)
            return x, global_index_template, global_index_search, None, None, None, \
                xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, iou_xi
        else:
            print("代码有误")
            return
    
    def forward_one_modal(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None, lens_x=256):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            lens_t = global_index_template.shape[1]
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # mimic_x = None
        iou_x = None
        if self.iou_predictor is not None:
            iou_x = self.iou_predictor(x[:,0])
            # mimic_x = self.mimic_layer(x)
        return x, global_index_template, global_index_search, removed_index_search, attn, iou_x
    
    def forward_two_modal(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None, lens_x=256, is_test=False):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)
                             
        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            lens_t = global_index_template.shape[1]
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))

        # mimic_x = mimic_xi = None
        iou_x = iou_xi = None
        if self.iou_predictor is not None:
            iou_x = self.iou_predictor(x[:,0])
            iou_xi = self.iou_predictor(xi[:,0])
            # if not is_test: # 训练才执行
            #     mimic_x = self.mimic_layer(x[:,1:])
            #     mimic_xi = self.mimic_layer(xi[:,1:])
            
        return x, global_index_template, global_index_search, removed_index_search, attn, iou_x, \
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, iou_xi


class FixCoEEJointTokenBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, 
                 mimic_loc=None, mimic_type=None, layer_id=None, predictor_type=None, has_mimic=True):
        super().__init__()
        self.predictor_type = predictor_type
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search
        ############
        self.mimic_layer = None
        if has_mimic:
            self.mimic_layer = make_mimic_layer(layer_id, mimic_loc, mimic_type, embed_dim=dim, norm_layer=norm_layer, drop=drop,
                                            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                            drop_path=drop_path, act_layer=act_layer)
        self.iou_predictor = make_iou_predictor(layer_id, mimic_loc, embed_dim=dim, predictor_type=predictor_type)

    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None,
                lens_x=256, is_test=False):
        if x is not None and xi is not None:
            return self.forward_two_modal(x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask, ce_template_mask, keep_ratio_search, lens_x, is_test)
        elif x is not None:
            x, global_index_template, global_index_search, removed_index_search, attn, iou_x, mimic_x = self.forward_one_modal(x, global_index_template, global_index_search, mask, ce_template_mask, keep_ratio_search, lens_x)
            return x, global_index_template, global_index_search, removed_index_search, attn, iou_x, mimic_x, \
                xi, global_index_templatei, global_index_searchi, None, None, None, None
        elif xi is not None:
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, iou_xi, mimic_xi = self.forward_one_modal(xi, global_index_templatei, global_index_searchi, mask, ce_template_mask, keep_ratio_search, lens_x)
            return x, global_index_template, global_index_search, None, None, None, None, \
                xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, iou_xi, mimic_xi
        else:
            print("代码有误")
            return
    
    def forward_one_modal(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None, lens_x=256):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            lens_t = global_index_template.shape[1]
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # mimic_x = None
        iou_x = None
        if self.iou_predictor is not None:
            if self.predictor_type == 'linear_sig_avg_dim' or self.predictor_type == 'linear_sig_dim':
                iou_x = self.iou_predictor(x)
            else:
                iou_x = self.iou_predictor(x[:,0])
            # mimic_x = self.mimic_layer(x)
        return x, global_index_template, global_index_search, removed_index_search, attn, iou_x, None
    
    def forward_two_modal(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None, lens_x=256, is_test=False):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)
                             
        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            lens_t = global_index_template.shape[1]
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))

        mimic_x = mimic_xi = None
        iou_pred = None
        if self.iou_predictor is not None:
            if self.predictor_type == 'linear_sig_avg_dim' or self.predictor_type == 'linear_sig_dim':
                iou_pred = self.iou_predictor(x + xi)
            else:
                if 'cat' not in self.predictor_type:
                    iou_pred = self.iou_predictor(x[:,0] + xi[:,0])
                else:
                    iou_pred = self.iou_predictor(torch.stack((x[:,0],xi[:,0]), dim=1))
            if not is_test and self.mimic_layer is not None: # 训练才执行
                mimic_x = self.mimic_layer(x[:,-320:])
                mimic_xi = self.mimic_layer(xi[:,-320:])
            
        return x, global_index_template, global_index_search, removed_index_search, attn, iou_pred, mimic_x, \
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, mimic_xi
    

class CEXBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, 
                 exchange_type=None, ex_ratio_search=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search
        self.ex_ratio_search = ex_ratio_search
        self.exchange_type = exchange_type

    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None,
                ex_ratio_search=None):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)
                     
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        if self.ex_ratio_search > 0 and (ex_ratio_search is None or ex_ratio_search > 0):
            ex_ratio_search = self.ex_ratio_search if ex_ratio_search is None else ex_ratio_search
            # 实行token互换
            pass

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))

        return x, global_index_template, global_index_search, removed_index_search, attn, \
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn


class CEBlock_tradition(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, backbone_type=None, layer_id=None, mimic_type=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search
        
        self.mimic_layer = None
        if 'mimic' in backbone_type and layer_id != 11:
            if mimic_type == 'v1':
                self.mimic_layer = MIMICLayer()
            elif mimic_type == 'v2':
                self.mimic_layer = MIMICLayerV2(dim=dim)
        self.iou_predictor = None
        if 'iou' in backbone_type and layer_id != 11:
            self.iou_predictor = IOUPredictor()

    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)
                     
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))

        mimic_x = mimic_xi = None
        if self.mimic_layer is not None:
            mimic_x = self.mimic_layer(x) # 要不要加norm
            mimic_xi = self.mimic_layer(xi)
        iou_x = iou_xi = None
        if self.iou_predictor is not None:
            iou_x = self.iou_predictor(x)
            iou_xi = self.iou_predictor(xi)
            
        return x, global_index_template, global_index_search, removed_index_search, attn, mimic_x, iou_x,\
            xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, mimic_xi, iou_xi


class CoEEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None):
                
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)
                     
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))
        
        return x, global_index_template, global_index_search, removed_index_search, attn, xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn


# class CEABlock(nn.Module):

#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

#         self.keep_ratio_search = keep_ratio_search


#         self.adap_t = Bi_direct_adapter()        
#         self.adap2_t = Bi_direct_adapter()

#     def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None):
        
#         xori = x
        
#         x_attn, attn = self.attn(self.norm1(x), mask, True)   
#         x = x + self.drop_path(x_attn) + self.drop_path(self.adap_t(self.norm1(xi)))  #########-------------------------adapter

#         xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
#         xi = xi + self.drop_path(xi_attn) + self.drop_path(self.adap_t(self.norm1(xori)))  #########-------------------------adapter
                     
#         lens_t = global_index_template.shape[1]

#         removed_index_search = None
#         removed_index_searchi = None
#         if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
#             keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
#             x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
#             xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

#         xori = x

#         x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(xi)))   ###-------adapter

#         xi = xi + self.drop_path(self.mlp(self.norm2(xi))) + self.drop_path(self.adap2_t(self.norm2(xori)))   ###-------adapter
        
#         return x, global_index_template, global_index_search, removed_index_search, attn, xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #print("class Block ")
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        #print("class Block forward")
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class Block_two(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #print("class Block ")
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, xi, mask=None):
        #print("class Block forward")
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        xi = xi + self.drop_path(self.attn(self.norm1(xi), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))
        return x
