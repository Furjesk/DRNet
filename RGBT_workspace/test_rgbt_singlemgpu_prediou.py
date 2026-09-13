import os
import cv2
import sys
from os.path import join, isdir, dirname
import numpy as np
import argparse
import ast
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

from lib.test.tracker.bat import BATTrack
from lib.test.tracker.bat_fix_ee import BATFixEETrack
from lib.test.tracker.bat_coee import BATCoEETrack
from lib.test.tracker.bat_fix_ee_cl import BATFixEECLTrack
from lib.test.tracker.bat_fix_ee_pg import BATFixEEPGTrack
from lib.test.tracker.bat_fix_ee_joint import BATFixEEJointTrack
from lib.test.tracker.bat_fix_ee_distill_medium import BATFixEEDMTrack
import lib.test.parameter.bat as rgbt_adapter_params
import lib.test.parameter.bat_fix_ee as bat_fix_ee_params
import lib.test.parameter.bat_coee as bat_coee_params
import lib.test.parameter.bat_fix_ee_cl as bat_fix_ee_cl_params
import lib.test.parameter.bat_fix_ee_pg as bat_fix_ee_pg_params
import lib.test.parameter.bat_fix_ee_joint as bat_fix_ee_joint_params
import lib.test.parameter.bat_fix_ee_distill_medium as bat_fix_ee_distill_medium_params
import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time


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


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0, script_name='adapter', exit_threshold=[0.65],
                ):

    seq_txt = seq_name
    save_name = '{}_{}_ep{}_dynamic_ee{:.2f}'.format(script_name, yaml_name, epoch, sum(exit_threshold)/len(exit_threshold))
    # save_name = '{}'.format(yaml_name)
    save_path = f'./RGBT_workspace/results/{dataset_name}/' + save_name +  '/' + seq_txt + '.txt'
    save_fps_path = f'./RGBT_workspace/results/{dataset_name}/' + save_name +  '/' + seq_txt + '_fps.txt'
    save_eelayer_path = f'./RGBT_workspace/results/{dataset_name}/' + save_name +  '/' + seq_txt + '_exit_layer.txt'
    save_iou_path = f'./RGBT_workspace/results/{dataset_name}/' + save_name +  '/' + seq_txt + '_iou.txt'
    save_folder = f'./RGBT_workspace/results/{dataset_name}/' + save_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    seq_path = seq_home + '/' + seq_name
    # print('——————————Process sequence: '+seq_name +'——————————————')
    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    toc = 0
    ee_layer = []
    iou_list = []
    for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
        image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
        tic = cv2.getTickCount()
        if frame_idx == 0:
            # initialization
            # image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
            tracker.initialize(image, RGB_gt[0].tolist())  # xywh
            tic_end = cv2.getTickCount()
        elif frame_idx > 0:
            # track
            # image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
            region, confidence, out_dict = tracker.track(image, exit_threshold=exit_threshold)  # xywh
            result[frame_idx] = np.array(region)
            tic_end = cv2.getTickCount()
            # if 'tir_exit_layer' in out_dict.keys() and out_dict['tir_exit_layer'] is not None:
            #     ee_layer.append([out_dict['rgb_exit_layer'], out_dict['tir_exit_layer']])
            # elif 'exit_layer' in out_dict.keys() and out_dict['tir_exit_layer'] is not None:
            #     ee_layer.append(out_dict['exit_layer'])
            if 'iou_list' in out_dict.keys() and out_dict['iou_list'] is not None:
                iou_list.append(np.array(out_dict['iou_list']).reshape(-1))
        toc += tic_end - tic
    toc /= cv2.getTickFrequency()
    if not debug:
        np.savetxt(save_path, result)
        # np.savetxt(save_fps_path, np.array([frame_idx / toc]))
        # np.savetxt(save_eelayer_path, np.array(ee_layer), fmt='%d')
        np.savetxt(save_iou_path, np.array(iou_list), fmt='%.3f')
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class BAT_RGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB, exit_threshold=0.65):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB, exit_threshold=exit_threshold)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score, outputs

def parse_list_arg(list_arg):
    try:
        return ast.literal_eval(list_arg)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid list argument: {list_arg}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='bat_fix_ee_joint', help='Name of tracking method(ostrack, adapter, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='rgbt_mimic_fix_ee_joint_train_20_mimic_only_task', help='Name of tracking method.')  # rgbt_mimic_fix_ee_stage2_mseloss_discrete2conv,rgbt_mimic_fix_ee_stage2_mseloss_twolinear
    parser.add_argument('--dataset_name', type=str, default='RGBT234', help='Name of dataset (GTOT,RGBT234,LasHeR,VTUAVST,VTUAVLT).')
    parser.add_argument('--threads', default=1, type=int, help='Number of threads')   #################################-------------------------------##################
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=10, type=int, help='epochs of ckpt')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    # parser.add_argument('--exit_threshold', default=0.7, type=float, help='specific video name')
    parser.add_argument('--exit_threshold', default="[0.8]", type=parse_list_arg, help='specific video name')
    # 可以再搞个模态分开的阈值，每次rgb的预测iou都比tir低0.05左右
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

    if args.script_name == 'bat':
        params = rgbt_adapter_params.parameters(args.yaml_name, args.epoch)
        mmtrack = BATTrack(params)  # "GTOT" # dataset_name
        tracker = BAT_RGBT(tracker=mmtrack)
    elif args.script_name == 'bat_fix_ee':
        params = bat_fix_ee_params.parameters(args.yaml_name, args.epoch)
        mmtrack = BATFixEETrack(params)  # "GTOT" # dataset_name
        tracker = BAT_RGBT(tracker=mmtrack)
    elif args.script_name == 'bat_coee':
        params = bat_coee_params.parameters(args.yaml_name, args.epoch)
        mmtrack = BATCoEETrack(params)  # "GTOT" # dataset_name
        tracker = BAT_RGBT(tracker=mmtrack)
    elif args.script_name == 'bat_fix_ee_cl':
        params = bat_fix_ee_cl_params.parameters(args.yaml_name, args.epoch)
        mmtrack = BATFixEECLTrack(params)  # "GTOT" # dataset_name
        tracker = BAT_RGBT(tracker=mmtrack)
    elif args.script_name == 'bat_fix_ee_pg':
        params = bat_fix_ee_pg_params.parameters(args.yaml_name, args.epoch)
        mmtrack = BATFixEEPGTrack(params)  # "GTOT" # dataset_name
        tracker = BAT_RGBT(tracker=mmtrack)
    elif args.script_name == 'bat_fix_ee_joint':
        params = bat_fix_ee_joint_params.parameters(args.yaml_name, args.epoch)
        mmtrack = BATFixEEJointTrack(params)  # "GTOT" # dataset_name
        tracker = BAT_RGBT(tracker=mmtrack)
    elif args.script_name == 'bat_fix_ee_distill_medium':
        params = bat_fix_ee_distill_medium_params.parameters(args.yaml_name, args.epoch)
        mmtrack = BATFixEEDMTrack(params)  # "GTOT" # dataset_name
        tracker = BAT_RGBT(tracker=mmtrack)

    start = time.time()

    seq_list = [args.video] if args.video != '' else seq_list
    sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, args.exit_threshold) for s in seq_list]
    for seqlist in sequence_list:
        run_sequence(*seqlist)

    print(f"Totally cost {time.time()-start} seconds!")
