"""Original CMX MiT-B2 configuration for offline panoramic REL."""

import os.path as osp
import sys
import time
from pathlib import Path

import numpy as np
from easydict import EasyDict as edict


SOURCE_DATASET_ROOT = "/data/zhuzhaoziao/datasets/Stanford2D3D"
FORMAL_DATASET_ROOT = (
    "/data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration/formal_dataset"
)
OUTPUT_ROOT = "/data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration"
PRETRAINED_MODEL = "/data/zhuzhaoziao/cmx/raw/pretrained/segformer/mit_b2.pth"


def make_config(dataset_root=FORMAL_DATASET_ROOT, run_dir=None, smoke=False):
    c = edict()
    c.seed = 12345
    c.root_dir = str(Path(__file__).resolve().parents[2])
    c.abs_dir = osp.realpath(c.root_dir)

    c.dataset_name = "Stanford2D3DPano"
    c.dataset_source_root = SOURCE_DATASET_ROOT
    c.dataset_path = dataset_root
    c.dataset_fold = 1
    c.rgb_root_folder = osp.join(dataset_root, "RGB")
    c.rgb_format = ".png"
    c.gt_root_folder = osp.join(dataset_root, "Label")
    c.gt_format = ".png"
    c.gt_transform = True
    c.x_name = "rel"
    c.x_mode = "rel_original"
    c.x_root_folder = osp.join(dataset_root, "REL")
    c.x_format = ".png"
    c.x_is_single_channel = False
    c.in_chans = 3
    c.in_chans_x = 3
    c.train_source = osp.join(dataset_root, "train.txt")
    c.eval_source = osp.join(dataset_root, "test.txt")
    c.is_test = False
    c.num_train_imgs = 3 if smoke else 1040
    c.num_eval_imgs = 3 if smoke else 373
    c.num_classes = 13
    c.class_names = [
        "beam", "board", "bookcase", "ceiling", "chair", "clutter",
        "column", "door", "floor", "sofa", "table", "wall", "window",
    ]

    c.background = 255
    c.image_height = 256 if smoke else 1080
    c.image_width = 256 if smoke else 1080
    c.norm_mean = np.array([0.485, 0.456, 0.406])
    c.norm_std = np.array([0.229, 0.224, 0.225])
    c.rel_channel_semantics = ["EGVIA", "LOA", "ReD"]

    c.backbone = "mit_b2"
    c.pretrained_model = None if smoke else PRETRAINED_MODEL
    c.decoder = "MLPDecoder"
    c.decoder_embed_dim = 512
    c.optimizer = "AdamW"

    c.lr = 6e-5
    c.lr_power = 0.9
    c.momentum = 0.9
    c.weight_decay = 0.01
    c.batch_size = 1 if smoke else 8
    c.nepochs = 1 if smoke else 200
    c.niters_per_epoch = c.num_train_imgs // c.batch_size + 1
    c.num_workers = 0 if smoke else 16
    c.train_scale_array = [0.5] if smoke else [0.5, 0.75, 1, 1.25, 1.5, 1.75]
    c.warm_up_epoch = 10
    c.fix_bias = True
    c.bn_eps = 1e-3
    c.bn_momentum = 0.1

    c.using_gate = False
    c.using_smmf = False
    c.using_dymm = False
    c.using_relplus = False

    c.eval_iter = 25
    c.eval_stride_rate = 2 / 3
    c.eval_scale_array = [1]
    c.eval_flip = False
    c.eval_crop_size = [c.image_height, c.image_width]
    c.checkpoint_start_epoch = 100
    c.checkpoint_step = 5

    if c.root_dir not in sys.path:
        sys.path.insert(0, c.root_dir)
    if run_dir is None:
        run_dir = osp.join(OUTPUT_ROOT, "formal_run_not_started")
    c.log_dir = osp.abspath(run_dir)
    c.tb_dir = osp.join(c.log_dir, "tensorboard")
    c.log_dir_link = c.log_dir
    c.checkpoint_dir = osp.join(c.log_dir, "checkpoints")
    exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    c.log_file = osp.join(c.log_dir, "train_" + exp_time + ".log")
    c.link_log_file = osp.join(c.log_dir, "train_last.log")
    c.val_log_file = osp.join(c.log_dir, "val_" + exp_time + ".log")
    c.link_val_log_file = osp.join(c.log_dir, "val_last.log")
    c.data_setting = {
        "rgb_root": c.rgb_root_folder,
        "rgb_format": c.rgb_format,
        "gt_root": c.gt_root_folder,
        "gt_format": c.gt_format,
        "transform_gt": c.gt_transform,
        "x_root": c.x_root_folder,
        "x_format": c.x_format,
        "x_single_channel": c.x_is_single_channel,
        "x_mode": c.x_mode,
        "train_source": c.train_source,
        "eval_source": c.eval_source,
        "class_names": c.class_names,
    }
    return c


config = make_config()
cfg = config
config.config_path = __file__
