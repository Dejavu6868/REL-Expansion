"""Shared, explicit CMX-REL+ v2.1 S2D configuration fields."""

import os.path as osp
from pathlib import Path

import numpy as np
from easydict import EasyDict as edict

from dataloader.data_setting import build_data_setting


PROTOCOL_ID = "CMX_RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"
REPRESENTATION_PROTOCOL_ID = "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"
INVALID_POLICY = "SOURCE_COMPAT_STORAGE_255"
AUGMENTATION_PROFILE = "S2D_RELPLUS_COMPARISON_NO_FLIP"
CHANNEL_ORDER = ("EGVIA", "LOA", "ReD")
DATASET_ROOT = "/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480"
PILOT_ROOT = (
    "/data/zhuzhaoziao/RELPlus/outputs/"
    "REL_plus_v2_1_implementation/pilot_cache"
)
OUTPUT_ROOT = (
    "/data/zhuzhaoziao/RELPlus/outputs/"
    "CMX_RELPlus_v2_1_integration_fix"
)
PRETRAINED_MODEL = "/data/zhuzhaoziao/cmx/raw/pretrained/segformer/mit_b2.pth"


class ProtocolConfig(edict):
    """EasyDict-compatible config that preserves protocol tuples."""

    def __setattr__(self, name, value):
        if isinstance(value, tuple):
            object.__setattr__(self, name, value)
            dict.__setitem__(self, name, value)
            return
        super(ProtocolConfig, self).__setattr__(name, value)


def count_list(path, expected):
    with open(path, encoding="utf-8") as handle:
        count = sum(1 for line in handle if line.strip())
    if count != expected:
        raise RuntimeError(
            "split count mismatch for {}: expected {}, found {}".format(
                path, expected, count
            )
        )
    return count


def base_config(*, experiment_name, x_cache_root, output_name):
    c = ProtocolConfig()
    c.root_dir = str(Path(__file__).resolve().parents[2])
    c.abs_dir = osp.realpath(c.root_dir)
    c.experiment_name = experiment_name
    c.protocol_id = PROTOCOL_ID
    c.representation_protocol_id = REPRESENTATION_PROTOCOL_ID
    c.invalid_policy = INVALID_POLICY
    c.augmentation_profile = AUGMENTATION_PROFILE
    c.x_mode = "rel_plus_v2_1"
    c.channel_order = CHANNEL_ORDER

    c.dataset_name = "Stanford2D3D_S2D"
    c.dataset_path = DATASET_ROOT
    c.rgb_root_folder = osp.join(DATASET_ROOT, "RGB")
    c.rgb_format = ".png"
    c.gt_root_folder = osp.join(DATASET_ROOT, "Label")
    c.gt_format = ".png"
    c.gt_transform = True
    c.x_root_folder = osp.join(x_cache_root, "RELPlus")
    c.x_format = ".png"
    c.x_valid_root_folder = osp.join(x_cache_root, "ValidMask")
    c.x_valid_format = ".png"
    c.x_is_single_channel = False
    c.in_chans = 3
    c.in_chans_x = 3
    c.num_classes = 13
    c.class_names = [
        "beam", "board", "bookcase", "ceiling", "chair", "clutter",
        "column", "door", "floor", "sofa", "table", "wall", "window",
    ]
    c.background = 255

    c.image_height = 480
    c.image_width = 480
    c.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    c.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    c.train_horizontal_flip = False
    c.train_vertical_flip = False
    c.train_arbitrary_rotation = False
    c.train_perspective_warp = False
    c.train_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

    c.backbone = "mit_b2"
    c.pretrained_model = PRETRAINED_MODEL
    c.decoder = "MLPDecoder"
    c.decoder_embed_dim = 512
    c.using_gate = False
    c.using_smmf = False
    c.using_dymm = False
    c.using_sga = False

    c.seed = 12345
    c.seed_policy = "base_seed + epoch + local_rank * 1000"
    c.worker_seed_policy = "author_default_no_explicit_worker_init"
    c.cudnn_benchmark = False
    c.cudnn_deterministic = False
    c.criterion = "Focal"
    c.focal_gamma = 2
    c.loss_reduction = "none_then_mean"
    c.optimizer = "AdamW"
    c.lr = 6e-5
    c.lr_power = 0.9
    c.momentum = 0.9
    c.weight_decay = 0.01
    c.scheduler = "WarmUpPolyLR"
    c.warm_up_epoch = 10
    c.batch_size = 8
    c.num_workers = 16
    c.nepochs = 200
    c.fix_bias = True
    c.bn_eps = 1e-3
    c.bn_momentum = 0.1

    c.eval_batch_size = 1
    c.eval_scale_array = [1]
    c.eval_flip = False
    c.eval_crop_size = [480, 480]
    c.eval_stride_rate = 1.0
    c.eval_align_corners = False
    c.eval_start_epoch = 100
    c.eval_step = 5
    c.checkpoint_start_epoch = 100
    c.checkpoint_step = 5

    c.output_dir = osp.join(OUTPUT_ROOT, output_name)
    c.log_dir = osp.join(c.output_dir, "logs")
    c.tb_dir = osp.join(c.output_dir, "tensorboard")
    c.log_dir_link = c.log_dir
    c.checkpoint_dir = osp.join(c.output_dir, "checkpoints")
    c.log_file = osp.join(c.log_dir, "train.log")
    c.link_log_file = osp.join(c.log_dir, "train_last.log")
    c.val_log_file = osp.join(c.log_dir, "val.log")
    c.link_val_log_file = osp.join(c.log_dir, "val_last.log")
    return c


def finish_config(c):
    c.data_setting = build_data_setting(c, split="train")
    return c
