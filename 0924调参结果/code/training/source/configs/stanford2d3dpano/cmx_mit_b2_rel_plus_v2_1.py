"""Original CMX MiT-B2 wiring config for the 36-sample REL+ v2.1 pilot only."""

import os.path as osp
import sys
from pathlib import Path

import numpy as np
from easydict import EasyDict as edict


PROTOCOL_ID = "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"
INVALID_POLICY = "SOURCE_COMPAT_STORAGE_255"
DATASET_ROOT = "/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480"
PILOT_ROOT = (
    "/data/zhuzhaoziao/RELPlus/outputs/"
    "REL_plus_v2_1_implementation/pilot_cache"
)
OUTPUT_ROOT = (
    "/data/zhuzhaoziao/RELPlus/outputs/"
    "REL_plus_v2_1_implementation/cmx_wiring"
)
PRETRAINED_MODEL = "/data/zhuzhaoziao/cmx/raw/pretrained/segformer/mit_b2.pth"


def make_config():
    c = edict()
    c.seed = 12345
    c.root_dir = str(Path(__file__).resolve().parents[2])
    c.abs_dir = osp.realpath(c.root_dir)
    c.protocol_id = PROTOCOL_ID
    c.relplus_invalid_policy = INVALID_POLICY

    c.dataset_name = "Stanford2D3D_S2D_RELPlusV2_1_Pilot"
    c.dataset_path = DATASET_ROOT
    c.dataset_fold = 1
    c.rgb_root_folder = osp.join(DATASET_ROOT, "RGB")
    c.rgb_format = ".png"
    c.gt_root_folder = osp.join(DATASET_ROOT, "Label")
    c.gt_format = ".png"
    c.gt_transform = True
    c.x_name = "rel_plus_v2_1"
    c.x_mode = "rel_plus_v2_1"
    c.x_root_folder = osp.join(PILOT_ROOT, "RELPlus")
    c.x_format = ".png"
    c.x_valid_root_folder = osp.join(PILOT_ROOT, "ValidMask")
    c.x_valid_format = ".png"
    c.x_is_single_channel = False
    c.in_chans = 3
    c.in_chans_x = 3
    c.train_source = osp.join(PILOT_ROOT, "train.txt")
    c.eval_source = osp.join(PILOT_ROOT, "test.txt")
    c.is_test = False
    c.num_train_imgs = 36
    c.num_eval_imgs = 36
    c.num_classes = 13
    c.class_names = [
        "beam", "board", "bookcase", "ceiling", "chair", "clutter",
        "column", "door", "floor", "sofa", "table", "wall", "window",
    ]

    c.background = 255
    c.image_height = 480
    c.image_width = 480
    c.norm_mean = np.array([0.485, 0.456, 0.406])
    c.norm_std = np.array([0.229, 0.224, 0.225])
    c.rel_channel_semantics = ["EGVIA", "LOA", "ReD"]
    c.train_horizontal_flip = False

    c.backbone = "mit_b2"
    c.pretrained_model = PRETRAINED_MODEL
    c.decoder = "MLPDecoder"
    c.decoder_embed_dim = 512
    c.criterion = "CrossEntropy"
    c.optimizer = "AdamW"
    c.lr = 6e-5
    c.lr_power = 0.9
    c.momentum = 0.9
    c.weight_decay = 0.01
    c.batch_size = 1
    c.nepochs = 0
    c.niters_per_epoch = 36
    c.num_workers = 0
    c.train_scale_array = [0.75, 1.0, 1.25]
    c.warm_up_epoch = 0
    c.fix_bias = True
    c.bn_eps = 1e-3
    c.bn_momentum = 0.1

    c.using_gate = False
    c.using_smmf = False
    c.using_dymm = False
    c.using_sga = False
    c.using_relplus = False
    c.eval_flip = False
    c.eval_scale_array = [1]
    c.eval_crop_size = [480, 480]
    c.checkpoint_start_epoch = 10 ** 9
    c.checkpoint_step = 10 ** 9

    c.log_dir = osp.join(OUTPUT_ROOT, "not_a_training_run")
    c.tb_dir = osp.join(c.log_dir, "tensorboard")
    c.log_dir_link = c.log_dir
    c.checkpoint_dir = osp.join(c.log_dir, "checkpoints_prohibited")
    c.log_file = osp.join(c.log_dir, "train_prohibited.log")
    c.link_log_file = osp.join(c.log_dir, "train_last_prohibited.log")
    c.val_log_file = osp.join(c.log_dir, "val_prohibited.log")
    c.link_val_log_file = osp.join(c.log_dir, "val_last_prohibited.log")
    if c.root_dir not in sys.path:
        sys.path.insert(0, c.root_dir)
    return c


config = make_config()
cfg = config
config.config_path = __file__
