"""Frozen Stanford2D3D S3D Fold 1 configuration for CMX-REL."""

import os.path as osp
import sys
import time

import numpy as np
from easydict import EasyDict as edict


C = edict()
config = C
cfg = C

C.experiment_name = "cmx_rel_fold1_seed12345"
C.code_root = "/home/zhuzhaoziao/RELPlus/CMX-REL"
C.root_dir = C.code_root
C.abs_dir = C.code_root
C.output_root = "/data/zhuzhaoziao/RELPlus/outputs/CMX_S3D_Fold1_reproduction"
C.run_dir = osp.join(C.output_root, C.experiment_name)
C.seed = 12345

C.dataset_name = "Stanford2D3DPano"
C.dataset_fold = 1
C.dataset_path = osp.join(C.output_root, "common", "Stanford2D3DPano")
C.rgb_root_folder = osp.join(C.dataset_path, "image")
C.rgb_format = "_rgb.png"
C.gt_root_folder = osp.join(C.dataset_path, "label")
C.gt_format = "_semantic.png"
C.gt_transform = True
C.x_name = "rel"
C.x_root_folder = osp.join(C.dataset_path, "rel")
C.x_format = "_rel.png"
C.x_is_single_channel = False
C.x_mode = "rel_original"
C.x_channel_semantics = ["EGVIA", "LOA", "ReD"]
C.train_source = osp.join(C.dataset_path, "fold1_train.txt")
C.eval_source = osp.join(C.dataset_path, "fold1_test.txt")
C.num_train_imgs = 1040
C.num_eval_imgs = 373
C.num_classes = 13
C.class_names = [
    "beam", "board", "bookcase", "ceiling", "chair", "clutter",
    "column", "door", "floor", "sofa", "table", "wall", "window",
]
C.is_test = False

C.background = 255
C.image_height = 1080
C.image_width = 1080
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

C.architecture = "Original CMX"
C.backbone = "mit_b2"
C.pretrained_model = "/data/zhuzhaoziao/cmx/raw/pretrained/segformer/mit_b2.pth"
C.decoder = "MLPDecoder"
C.decoder_embed_dim = 512
C.in_chans = 3
C.in_chans_x = 3
C.using_gate = False
C.using_smmf = False
C.using_dymm = False
C.random_reprojection = False
C.spherical_weighted_loss = False

C.optimizer = "AdamW"
C.criterion = "Focal"
C.lr = 6e-5
C.learning_rate = C.lr
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.global_batch_size = 8
C.per_gpu_batch_size = 1
C.gpu_count = 8
C.nepochs = 200
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.logical_train_length = C.batch_size * C.niters_per_epoch
C.num_workers = 16
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.random_mirror = True
C.random_crop = True
C.warm_up_epoch = 10
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.sync_batchnorm = True
C.amp = False

C.eval_start_epoch = 100
C.eval_step = 5
C.eval_stride_rate = 2 / 3
C.eval_stride_pixels = 720
C.eval_scale_array = [1]
C.eval_flip = False
C.eval_crop_size = [1080, 1080]
C.eval_reprojection = False
C.checkpoint_start_epoch = 100
C.checkpoint_step = 5

if C.root_dir not in sys.path:
    sys.path.insert(0, C.root_dir)
C.log_dir = C.run_dir
C.tb_dir = osp.join(C.run_dir, "tensorboard")
C.log_dir_link = C.run_dir
C.checkpoint_dir = osp.join(C.run_dir, "checkpoints")
C.prediction_dir = osp.join(C.run_dir, "predictions_best")
C.visualization_dir = osp.join(C.run_dir, "visualizations")
exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = osp.join(C.run_dir, "train_" + exp_time + ".log")
C.link_log_file = osp.join(C.run_dir, "train_last.log")
C.val_log_file = osp.join(C.run_dir, "eval_" + exp_time + ".log")
C.link_val_log_file = osp.join(C.run_dir, "eval_last.log")

C.data_setting = {
    "rgb_root": C.rgb_root_folder,
    "rgb_format": C.rgb_format,
    "gt_root": C.gt_root_folder,
    "gt_format": C.gt_format,
    "transform_gt": C.gt_transform,
    "x_root": C.x_root_folder,
    "x_format": C.x_format,
    "x_single_channel": C.x_is_single_channel,
    "x_mode": C.x_mode,
    "train_source": C.train_source,
    "eval_source": C.eval_source,
    "class_names": C.class_names,
}
