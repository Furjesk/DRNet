import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
# from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.dataset import VisEvent, LasHeR, DepthTrack
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.fix_bn = getattr(cfg.TRAIN, "FIX_BN", False) # add for fixing base model bn layer


def names2datasets(name_list: list, settings, image_loader, modee_dir=''):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val", "COCO17", "VID", "TRACKINGNET",
                        "DepthTrack_train", "DepthTrack_val", "LasHeR_all", "LasHeR_train", "LasHeR_val", "LasHeR_test", "VisEvent"]
        if name == "DepthTrack_train":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='rgbcolormap', split='train'))
        if name == "DepthTrack_val":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='rgbcolormap', split='val'))
        if name == "LasHeR_all":
            datasets.append(LasHeR(settings.env.lasher_dir, dtype='rgbrgb', split='all', modee_dir=modee_dir))
        if name == "LasHeR_train":
            datasets.append(LasHeR(settings.env.lasher_dir, dtype='rgbrgb', split='train', modee_dir=modee_dir))
        if name == "LasHeR_val":
            datasets.append(LasHeR(settings.env.lasher_dir, dtype='rgbrgb', split='val', modee_dir=modee_dir))
        if name == "LasHeR_test":
            datasets.append(LasHeR(settings.env.lasher_dir, dtype='rgbrgb', split='test', modee_dir=modee_dir))
        if name == "VisEvent":
            datasets.append(VisEvent(settings.env.visevent_dir, dtype='rgbrgb', split='train'))
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    # Note: for multimodal data, ToGrayscale and Normalize need modify
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.BATProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.BATProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    modee_dir = getattr(cfg.TRAIN, "MODEE_PATH", "/data1/Code/luandong/wangjinhu/DATA/LasHeR_Mod_Use_Label/")
    print("sampler_mode", sampler_mode)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader, modee_dir),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders(visevent no val split)
    if cfg.DATA.VAL.DATASETS_NAME[0] is None:
        loader_val = None
    else:
        dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader, modee_dir),
                                            p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_val,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)
        val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
        loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                            epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_type = getattr(cfg.TRAIN.PROMPT, "TYPE", "")
    if 'bat' in train_type:
        print("Only training adapter parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "adap" in n and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if "adap" not in n:   #
                p.requires_grad = False
            else:
                print(n)
        
        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

        for n, p in net.named_parameters(): 
            if p.requires_grad:
                print(n, p.numel())
    
    elif train_type == 'train_mimic_iou':
        print("Only training mimic_iou and head parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if ("mimic" in n or 'iou_predictor' in n or 'box_head' in n) and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if "mimic" not in n and 'iou_predictor' not in n and 'box_head' not in n:
                p.requires_grad = False
            else:
                print(n)

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

        for n, p in net.named_parameters(): 
            if p.requires_grad:
                print(n, p.numel())
    
    elif train_type == 'only_predictor':
        print("Only training only_predictor parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "predictors" in n and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'predictors' not in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

    elif train_type == 'only_modee':
        print("Only training only_modee parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "mod_ee_predictor" in n and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'mod_ee_predictor' not in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)
    
    elif train_type == 'only_layeree':
        print("Only training only_layeree parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if ("iou_predictor" in n or "mimic_layer" in n) and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'iou_predictor' not in n and 'mimic_layer' not in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)
    
    elif train_type == 'exclude_predictor':
        print("Only training exclude_predictor parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "predictor" not in n and p.requires_grad ],
             "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
             }
        ]

        for n, p in net.named_parameters():
            
            if 'predictor' in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

    elif train_type == 'exclude_pred_recons':
        print("Only training exclude_pred_recons parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if ("predictor" not in n and "reconstructor" not in n) and p.requires_grad ],
             "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
             }
        ]

        for n, p in net.named_parameters():
            
            if 'predictor' in n or "reconstructor" in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

    elif train_type == 'only_policy':
        print("Only training only_policy parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "policy" in n and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'policy' not in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

    elif train_type == 'only_policy_and_head':
        print("Only training only_policy_and_head parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if ("policy" in n or "box_head" in n) and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'policy' not in n and "box_head" not in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)


    elif train_type == 'only_prompt':
        print("Only training only_prompt parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "prompt" in n and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'prompt' not in n:
                p.requires_grad = False
            else:
                print(n)

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

        for n, p in net.named_parameters(): 
            if p.requires_grad:
                print(n, p.numel())

    elif train_type == 'only_prompt_head':
        print("Only training only_prompt_head parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "prompt" in n and p.requires_grad ]},
            {
                "params": [p for n, p in net.named_parameters() if "head" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]

        for n, p in net.named_parameters():
            
            if 'prompt' not in n and 'head' not in n:
                p.requires_grad = False
            else:
                print(n)

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

        for n, p in net.named_parameters(): 
            if p.requires_grad:
                print(n, p.numel())
    
    elif train_type == 'freeze_backbone':
        print("Only training freeze_backbone parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if ("predictors" in n or "mimic_layers" in n or "mimic_head" in n) and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'predictors' not in n and 'mimic_layers' not in n and 'mimic_head' not in n:
                p.requires_grad = False
            else:
                print(n)

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

        for n, p in net.named_parameters(): 
            if p.requires_grad:
                print(n, p.numel())
    
    elif train_type == 'freeze_head_tar':
        print("Only training freeze_head_tar parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if ("box_head_tar" not in n) and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'box_head_tar' in n:
                p.requires_grad = False
            else:
                print(n)

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

        for n, p in net.named_parameters(): 
            if p.requires_grad:
                print(n, p.numel())
    
    elif train_type == 'only_mimic_head':
        print("Only training mimic and head parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if ("mimic_layers" in n or "mimic_head" in n) and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'mimic_layers' not in n and 'mimic_head' not in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

    elif train_type == 'only_mimic_and_head_delr':
        print("Only training mimic and head parameters. They are: ")

        param_dicts = [
            {
                "params": [p for n, p in net.named_parameters() if ("mimic_layer" in n or "box_head" in n) and p.requires_grad ],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER, # * 0.1
             }
        ]

        for n, p in net.named_parameters():
            
            if 'mimic_layer' not in n and 'box_head' not in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)
    
    elif train_type == 'only_head':
        print("Only training only_head parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "box_head" in n and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'box_head' not in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

    elif train_type == 'only_layeree_head':
        print("Only training only_layeree_head parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if ("box_head" in n or "iou_predictor" in n or "mimic_layer" in n) and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'box_head' not in n and "iou_predictor" not in n and "mimic_layer" not in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

    elif train_type == 'only_head_modee':
        print("Only training only_head_modee parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if ("box_head" in n or "mod_ee_predictor" in n) and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'box_head' not in n and 'mod_ee_predictor' not in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

    elif train_type == 'only_recons_head':
        print("Only training only_recons_head parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "box_head" in n and p.requires_grad ],
             "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,},
            {"params": [p for n, p in net.named_parameters() if "reconstructor" in n and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'box_head' not in n and 'reconstructor' not in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

    elif train_type == 'only_reconstructor':
        print("Only training only_reconstructor parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "reconstructor" in n and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'reconstructor' not in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

    
    elif train_type == 'only_mimic':
        print("Only training only_mimic parameters. They are: ")

        param_dicts = [
            # {"params": [p for n, p in net.named_parameters() if ("3.mimic_layer" in n or "6.mimic_layer" in n) and p.requires_grad ]}
            {"params": [p for n, p in net.named_parameters() if ("mimic_layer" in n) and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'mimic_layer' not in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

    elif train_type == 'exc_modee':
        print("Only training exc_modee parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if ("mod_ee_predictor" not in n) and p.requires_grad ]}
        ]

        for n, p in net.named_parameters():
            
            if 'mod_ee_predictor' in n:
                p.requires_grad = False
            else:
                print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)

    else:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n, p.numel())

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        # lr_scheduler = None
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
