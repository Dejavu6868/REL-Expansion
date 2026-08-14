import os
import os.path as osp
import sys
import time

import numpy as np
from easydict import EasyDict as edict


DATASET_ROOT = "/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480"
OUTPUT_ROOT = "/data/zhuzhaoziao/cmx/outputs"
PRETRAINED_MODEL = "/data/zhuzhaoziao/cmx/raw/pretrained/segformer/mit_b2.pth"


def make_config(modality):
    if modality not in {"hha", "rawdepth"}:
        raise ValueError("modality must be 'hha' or 'rawdepth'")

    run_dir = os.environ.get("CMX_RUN_DIR")
    if not run_dir:
        raise RuntimeError("CMX_RUN_DIR must point to this run's output directory")

    c = edict()
    c.seed = 12345
    c.effective_distributed_seeds = [0, 1, 2, 3]
    c.root_dir = osp.abspath(os.getcwd())
    c.abs_dir = osp.realpath(".")

    c.dataset_name = "Stanford2D3D"
    c.dataset_path = DATASET_ROOT
    c.rgb_root_folder = osp.join(c.dataset_path, "RGB")
    c.rgb_format = ".png"
    c.gt_root_folder = osp.join(c.dataset_path, "Label")
    c.gt_format = ".png"
    c.gt_transform = True
    c.x_root_folder = osp.join(c.dataset_path, "HHA" if modality == "hha" else "RawDepth")
    c.x_format = ".png"
    c.x_is_single_channel = modality == "rawdepth"
    c.train_source = osp.join(c.dataset_path, "train.txt")
    c.eval_source = osp.join(c.dataset_path, "test.txt")
    c.is_test = False
    c.num_train_imgs = 52903
    c.num_eval_imgs = 17593
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

    c.backbone = "mit_b2"
    c.pretrained_model = PRETRAINED_MODEL
    c.decoder = "MLPDecoder"
    c.decoder_embed_dim = 512
    c.optimizer = "AdamW"

    c.lr = 6e-5
    c.lr_power = 0.9
    c.momentum = 0.9
    c.weight_decay = 0.01
    c.batch_size = 12
    c.nepochs = 32
    c.niters_per_epoch = c.num_train_imgs // c.batch_size + 1
    c.num_workers = 16
    c.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
    c.train_horizontal_flip = False
    c.warm_up_epoch = 10

    c.fix_bias = True
    c.bn_eps = 1e-3
    c.bn_momentum = 0.1

    c.eval_iter = 25
    c.eval_stride_rate = 2 / 3
    c.eval_scale_array = [1]
    c.eval_flip = False
    c.eval_crop_size = [480, 480]

    c.checkpoint_start_epoch = 4
    c.checkpoint_step = 4

    if c.root_dir not in sys.path:
        sys.path.insert(0, c.root_dir)

    c.modality = modality
    c.output_root = OUTPUT_ROOT
    c.log_dir = osp.abspath(run_dir)
    c.tb_dir = osp.join(c.log_dir, "tensorboard")
    c.log_dir_link = c.log_dir
    c.checkpoint_dir = osp.join(c.log_dir, "checkpoints")
    exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    c.log_file = osp.join(c.log_dir, "train_" + exp_time + ".log")
    c.link_log_file = osp.join(c.log_dir, "train_last.log")
    c.val_log_file = osp.join(c.log_dir, "val_" + exp_time + ".log")
    c.link_val_log_file = osp.join(c.log_dir, "val_last.log")

    c.pretrained_sha256 = "ced22617efb7bae3c34ad0a80f20a9b8afb4d27368cb0835a23456baa9d0e092"
    c.hha_generation = "Depth2HHA at 1080x1080 with pose K, then bilinear resize to 480x480"
    c.raw_depth_encoding = "uint16 sensor depth plus one, high byte, nearest resize to 480x480"
    return c
