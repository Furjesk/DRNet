import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

# from lib.test.tracker.ostrack import OSTrack
from lib.test.tracker.bat import BATTrack
from lib.test.tracker.bat_fix_ee import BATFixEETrack
from lib.test.tracker.bat_coee import BATCoEETrack
import lib.test.parameter.bat as rgbt_adapter_params
import lib.test.parameter.bat_fix_ee as bat_fix_ee_params
import lib.test.parameter.bat_coee as bat_coee_params
import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F


def FeatureMapVisible(outputs,type):
    outputs = (outputs ** 2).sum(1)
    b, h, w = outputs.size()
    outputs = outputs.view(b, h * w)
    outputs = F.normalize(outputs, p=2, dim=1)
    outputs = outputs.view(b, h, w)

    font = cv2.FONT_HERSHEY_COMPLEX  # 设置字体
    # 图片对象、文本、像素、字体、字体大小、颜色、字体粗细
    for j in range(outputs.size(0)):
        am = outputs[j, ...].cpu().numpy()
        am = cv2.resize(am, (288, 288))
        am = 255 * (am - np.min(am)) / (
                np.max(am) - np.min(am) + 1e-12
        )
        am = np.uint8(np.floor(am))
        
        am=np.stack((am,am,am),axis=0)
        am=np.transpose(am,(1,2,0))  # 这里转换通道为RGB

        am=cv2.applyColorMap(am,cv2.COLORMAP_JET)
        return am

def genConfig(seq_path, set_type):

    if set_type == 'RGBT210':
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])

        RGB_gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')

    elif set_type == 'RGBT234':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif set_type == 'DroneT':
            ############################################  have to refine #############################################
            RGB_img_list = sorted([seq_path + '/rgb/' + p for p in os.listdir(seq_path + '/rgb') if
                                   os.path.splitext(p)[1] == '.jpg'])
            T_img_list = sorted([seq_path + '/ir/' + p for p in os.listdir(seq_path + '/ir') if
                                 os.path.splitext(p)[1] == '.jpg'])

            RGB_gt = np.loadtxt(seq_path + '/rgb.txt', delimiter=',')
            T_gt = np.loadtxt(seq_path + '/ir.txt', delimiter=',')

    elif set_type == 'GTOT':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] == '.png'])
        T_img_list = sorted([seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] == '.png'])

        RGB_gt = np.loadtxt(seq_path + '/groundTruth_v.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/groundTruth_i.txt', delimiter=' ')

        x_min = np.min(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(RGB_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(RGB_gt[:,[1,3]],axis=1)[:,None]
        RGB_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

        x_min = np.min(T_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(T_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(T_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(T_gt[:,[1,3]],axis=1)[:,None]
        T_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)
    
    elif set_type == 'LasHeR':
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')

    elif 'VTUAV' in set_type:
        RGB_img_list = sorted([seq_path + '/rgb/' + p for p in os.listdir(seq_path + '/rgb') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/ir/' + p for p in os.listdir(seq_path + '/ir') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/rgb.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/ir.txt', delimiter=' ')

    return RGB_img_list, T_img_list, RGB_gt, T_gt


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0, script_name='adapter', exit_threshold=0.65):
    #if 'VTUAV' in dataset_name:
    #    print(seq_name)
    #    seq_txt = seq_name.split('/')[1]
    #else:
    seq_txt = seq_name
    save_name = '{}_ep{}'.format(yaml_name, epoch)
    # save_name = '{}'.format(yaml_name)
    # save_path = f'./RGBT_workspace/results/{dataset_name}/' + save_name +  '/' + seq_txt + '.txt'
    # save_folder = f'./RGBT_workspace/results/{dataset_name}/' + save_name
    save_vis_folder = f'./RGBT_workspace/results/{dataset_name}/' + save_name +  '_vis/' + seq_txt
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    if not os.path.exists(save_vis_folder):
        os.makedirs(save_vis_folder)
    # if os.path.exists(save_path):
    #     print(f'-1 {seq_name}')
    #     return
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    if script_name == 'bat':
        params = rgbt_adapter_params.parameters(yaml_name, epoch)
        mmtrack = BATTrack(params)  # "GTOT" # dataset_name
        tracker = BAT_RGBT(tracker=mmtrack)
    elif script_name == 'bat_fix_ee':
        params = bat_fix_ee_params.parameters(yaml_name, epoch)
        mmtrack = BATFixEETrack(params)  # "GTOT" # dataset_name
        tracker = BAT_RGBT(tracker=mmtrack)
    elif script_name == 'bat_coee':
        params = bat_coee_params.parameters(args.yaml_name, args.epoch)
        mmtrack = BATCoEETrack(params)  # "GTOT" # dataset_name
        tracker = BAT_RGBT(tracker=mmtrack)

    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: '+seq_name +'——————————————')
    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    toc = 0
    for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
        tic = cv2.getTickCount()
        if frame_idx == 0:
            # initialization
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
            tracker.initialize(image, RGB_gt[0].tolist())  # xywh
        elif frame_idx > 0:
            # track
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
            box = RGB_gt[frame_idx]
            region, confidence, search_with_box, response, out_dict = tracker.track(image, vis=True, box=box, exit_threshold=exit_threshold)

            '''可视化特征图'''
            ### response 画热力图
            heatmap = FeatureMapVisible(response, None)

            # mimic_len = len(out_dict['mimic_token_list'])
            fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(10, 6))
            axs[0][0].imshow(heatmap[:,:,::-1])  
            axs[0][0].set_title('response map')  
            axs[0][0].axis('off')  

            axs[0][1].imshow(search_with_box[:,:3].squeeze().permute(1,2,0).cpu())  
            axs[0][1].set_title('RGB search')  
            axs[0][1].axis('off')

            axs[0][2].imshow(search_with_box[:,3:].squeeze().permute(1,2,0).cpu())  
            axs[0][2].set_title('TIR search')  
            axs[0][2].axis('off')  

            heatmap = FeatureMapVisible(out_dict['last_feature'][0][:,64:320].transpose(-1,-2).reshape(-1,768,16,16), None)
            axs[0][3].imshow(heatmap[:,:,::-1])  
            axs[0][3].set_title('RGB last {:d}'.format(out_dict['exit_layer']))  
            axs[0][3].axis('off')  
            heatmap = FeatureMapVisible(out_dict['last_feature'][1][:,64:320].transpose(-1,-2).reshape(-1,768,16,16), None)
            axs[0][4].imshow(heatmap[:,:,::-1])  
            axs[0][4].set_title('TIR last {:d}'.format(out_dict['exit_layer']))  
            axs[0][4].axis('off')  

            # 最后的测试可视化
            heatmap = FeatureMapVisible(out_dict['exit_feature'][:,64:320].transpose(-1,-2).reshape(-1,768,16,16), None)
            axs[1][0].imshow(heatmap[:,:,::-1])  
            axs[1][0].set_title('exit feature')  
            axs[1][0].axis('off')  
            
            # 一阶段mimic可视化
            # teacher_v = out_dict['last_feature'][0][:,64:].transpose(-1,-2)
            # teacher_i = out_dict['last_feature'][1][:,64:].transpose(-1,-2)
            # for i in range(mimic_len):
            #     # mimic token 与最后一层的相似度
            #     sim_v = torch.cosine_similarity(teacher_v, out_dict['mimic_token_list'][i][0][:,64:].transpose(-1,-2) ).mean()
            #     sim_i = torch.cosine_similarity(teacher_i, out_dict['mimic_token_list'][i][1][:,64:].transpose(-1,-2) ).mean()

            #     # 可视化 mimic token
            #     heatmap = FeatureMapVisible(out_dict['mimic_token_list'][i][0][:,64:].transpose(-1,-2).reshape(-1,768,16,16), None) # outputs['inter_token'][0]: search (1,320, 768)
            #     axs[i//mimic_len + 1][i%mimic_len].imshow(heatmap[:,:,::-1])  
            #     axs[i//mimic_len + 1][i%mimic_len].set_title('s_mv{:d}_{:.2f}'.format(i, sim_v))  
            #     axs[i//mimic_len + 1][i%mimic_len].axis('off')
            #     heatmap = FeatureMapVisible(out_dict['mimic_token_list'][i][1][:,64:].transpose(-1,-2).reshape(-1,768,16,16), None)
            #     axs[i//mimic_len + 2][i%mimic_len].imshow(heatmap[:,:,::-1])  
            #     axs[i//mimic_len + 2][i%mimic_len].set_title('s_mi{:d}_{:.2f}'.format(i, sim_i))  
            #     axs[i//mimic_len + 2][i%mimic_len].axis('off')  

            plt.savefig(os.path.join(save_vis_folder, str(frame_idx) + '.jpg'),bbox_inches='tight', dpi=200, pad_inches=0.0)
            plt.close()
            '''end'''

            # result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    if not debug:
        # np.savetxt(save_path, result)
        pass
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class BAT_RGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB, vis=False, box=None, exit_threshold=0.65):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB, vis=vis, box=box, exit_threshold=exit_threshold)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        if vis:
            search_with_box = outputs['search_with_box']
            response = outputs['response']
            return pred_bbox, pred_score, search_with_box, response, outputs
        else:
            return pred_bbox, pred_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='bat_coee', help='Name of tracking method(ostrack, adapter, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='rgbt_15_freeze_bk', help='Name of tracking method.')
    parser.add_argument('--dataset_name', type=str, default='LasHeR', help='Name of dataset (GTOT,RGBT234,LasHeR,VTUAVST,VTUAVLT).')
    parser.add_argument('--threads', default=1, type=int, help='Number of threads')   #################################-------------------------------##################
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=10, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='sequential', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    parser.add_argument('--exit_threshold', default=0.7, type=float, help='specific video name')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    # path initialization
    seq_list = None
    if dataset_name == 'GTOT':
        seq_home = '/home/lz/Videos/GTOT'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'RGBT210':
        seq_home = '/data1/Datasets/Tracking/RGBT210'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'RGBT234':
        seq_home = '/data1/Datasets/Tracking/RGBT234'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'DroneT':
        seq_home = '/root/nas-resource-linkdata/DroneT'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'LasHeR':
        seq_home = '/data1/Datasets/Tracking/LasHeR'
        seq_list = ['10runone', '11leftboy', '11runtwo', '1blackteacher', '1boycoming', '1stcol4thboy', '1strowleftboyturning', '1strowrightdrillmaster', '1strowrightgirl3540', '2girl', '2girlup', '2runseven', '3bike1', '3men', '3pinkleft', '3rdfatboy', '3rdgrouplastboy', '3thmoto', '4men', '4thboywithwhite', '7rightorangegirl', 'AQgirlwalkinrain', 'AQtruck2north', 'ab_bikeoccluded', 'ab_blkskirtgirl', 'ab_bolstershaking', 'ab_girlchoosesbike', 'ab_girlcrossroad', 'ab_pingpongball2', 'ab_rightlowerredcup_quezhen', 'ab_whiteboywithbluebag', 'advancedredcup', 'baggirl', 'ballshootatthebasket3times', 'basketball849', 'basketballathand', 'basketboy', 'bawgirl', 'belowdarkgirl', 'besom3', 'bike', 'bike2left', 'bike2trees', 'bikeboy', 'bikeboyintodark', 'bikeboyright', 'bikeboyturn', 'bikeboyturntimes', 'bikeboywithumbrella', 'bikefromlight', 'bikegoindark', 'bikeinrain', 'biketurnright', 'blackboy', 'blackboyoncall', 'blackcarturn', 'blackdown', 'blackgirl', 'blkboy`shead', 'blkboyback', 'blkboybetweenredandwhite', 'blkboydown', 'blkboyhead', 'blkboylefttheNo_21', 'blkboystand', 'blkboytakesumbrella', 'blkcaratfrontbluebus', 'blkgirlumbrella', 'blkhairgirltakingblkbag', 'blkmoto2north', 'blkstandboy', 'blktribikecome', 'blueboy', 'blueboy421', 'bluebuscoming', 'bluegirlbiketurn', 'bottlebetweenboy`sfeet', 'boy2basketballground', 'boy2buildings', 'boy2trees', 'boy2treesfindbike', 'boy`headwithouthat', 'boy`sheadingreycol', 'boyaftertree', 'boyaroundtrees', 'boyatdoorturnright', 'boydownplatform', 'boyfromdark', 'boyinlight', 'boyinplatform', 'boyinsnowfield3', 'boyleftblkrunning2crowd', 'boylefttheNo_9boy', 'boyoncall', 'boyplayphone', 'boyride2path', 'boyruninsnow', 'boyscomeleft', 'boyshead9684', 'boyss', 'boytakingbasketballfollowing', 'boytakingplate2left', 'boyunder2baskets', 'boywaitgirl', 'boywalkinginsnow2', 'broom', 'carbehindtrees', 'carcomeonlight', 'carcomingfromlight', 'carcominginlight', 'carlight2', 'carlightcome2', 'caronlight', 'carturn117', 'carwillturn', 'catbrown2', 'catbrownback2bush', 'couple', 'darkcarturn', 'darkgirl', 'darkouterwhiteboy', 'darktreesboy', 'drillmaster1117', 'drillmasterfollowingatright', 'farfatboy', 'firstexercisebook', 'foamatgirl`srighthand', 'foldedfolderatlefthand', 'girl2left3man1', 'girl`sblkbag', 'girlafterglassdoor', 'girldownstairfromlight', 'girlfromlight_quezhen', 'girlinrain', 'girllongskirt', 'girlof2leaders', 'girlrightthewautress', 'girlunderthestreetlamp', 'guardunderthecolumn', 'hugboy', 'hyalinepaperfrontface', 'large', 'lastleftgirl', 'leftblkTboy', 'leftbottle2hang', 'leftboy2jointhe4', 'leftboyoutofthetroop', 'leftchair', 'lefterbike', 'leftexcersicebookyellow', 'leftfarboycomingpicktheball', "leftgirl'swhitebag", 'lefthyalinepaper2rgb', 'lefthyalinepaperfrontpants', 'leftmirror', 'leftmirrorlikesky', 'leftmirrorside', 'leftopenexersicebook', 'leftpingpongball', 'leftrushingboy', 'leftunderbasket', 'leftuphand', 'littelbabycryingforahug', 'lowerfoamboard', 'mandownstair', 'manfromtoilet', 'mangetsoff', 'manoncall', 'mansimiliar', 'mantostartcar', 'midblkgirl', 'midboyNo_9', 'middrillmaster', 'midgreyboyrunningcoming', 'midof3girls', 'midredboy', 'midrunboywithwhite', 'minibus', 'minibusgoes2left', 'moto', 'motocomeonlight', 'motogoesaloongS', 'mototaking2boys306', 'mototurneast', 'motowithbluetop', 'pingpingpad3', 'pinkwithblktopcup', 'raincarturn', 'rainycarcome_ab', 'redboygoright', 'redcarcominginlight', 'redetricycle', 'redmidboy', 'redroadlatboy', 'redtricycle', 'right2ndflagformath', 'right5thflag', 'rightbike', 'rightbike-gai', 'rightblkboy4386', 'rightblkboystand', 'rightblkfatboyleftwhite', 'rightbluewhite', 'rightbottlecomes', 'rightboy504', 'rightcameraman', 'rightcar-chongT', 'rightcomingstrongboy', 'rightdarksingleman', 'rightgirltakingcup', 'rightwaiter1_quezhen', 'runningcameragirl', 'shinybikeboy2left', 'shinycarcoming', 'shinycarcoming2', 'silvercarturn', 'small-gai', 'standblkboy', 'swan_0109', 'truckgonorth', 'turning1strowleft2ndboy', 'umbreboyoncall', 'umbrella', 'umbrellabyboy', 'umbrellawillbefold', 'umbrellawillopen', 'waitresscoming', 'whitebikebelow', 'whiteboyrightcoccergoal', 'whitecarcomeinrain', 'whitecarturn683', 'whitecarturnleft', 'whitecarturnright', 'whitefardown', 'whitefargirl', 'whitegirlinlight', 'whitegirltakingchopsticks', 'whiteofboys', 'whiteridingbike', 'whiterunningboy', 'whiteskirtgirlcomingfromgoal', 'whitesuvturn', 'womanback2car', 'yellowgirl118', 'yellowskirt']
        seq_list.sort()
    elif dataset_name == 'VTUAVST':
        seq_home = '/root/nas-resource-linkdata/VTUAV/test/short-term'
        with open(join(join(seq_home, 'VTUAV-ST.txt')), 'r') as f:
            seq_list = f.read().splitlines()
    elif dataset_name == 'VTUAVLT':
        seq_home = '/root/nas-resource-linkdata/VTUAV/test/long-term'
        with open(join(seq_home, 'VTUAV-LT.txt'), 'r') as f:
            seq_list = f.read().splitlines()
    else:
        raise ValueError("Error dataset!")

    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, args.exit_threshold) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, args.exit_threshold) for s in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {time.time()-start} seconds!")
