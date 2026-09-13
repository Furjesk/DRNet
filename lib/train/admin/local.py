class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data2/jinhu.wang/Code/BAT_EE_bk_bk'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data2/jinhu.wang/Code/BAT_EE_bk/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data2/jinhu.wang/Code/BAT_EE_bk/pretrained_networks'
        self.got10k_val_dir = '/data1/Datasets/Tracking/got10k/val'
        self.lasot_lmdb_dir = '/data1/Datasets/Tracking/lasot_lmdb'
        self.got10k_lmdb_dir = '/data1/Datasets/Tracking/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/data1/Datasets/Tracking/trackingnet_lmdb'
        self.coco_lmdb_dir = '/data1/Datasets/Tracking/coco_lmdb'
        self.coco_dir = '/data1/Datasets/Tracking/coco'
        self.lasot_dir = '/data1/Datasets/Tracking/lasot'
        self.got10k_dir = '/data1/Datasets/Tracking/got10k/train'
        self.trackingnet_dir = '/data1/Datasets/Tracking/trackingnet'
        self.depthtrack_dir = '/data1/Datasets/Tracking/depthtrack/train'
        self.lasher_dir = '/data2/jinhu.wang/Data/LasHeR'
        self.visevent_dir = '/data1/Datasets/Tracking/visevent/train'
