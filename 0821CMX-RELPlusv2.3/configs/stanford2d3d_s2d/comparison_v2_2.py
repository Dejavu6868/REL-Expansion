"""Builders for fail-closed RGBD/HHA/REL+ V2.2 comparison configs."""

import copy
import os.path as osp

from dataloader.data_setting import build_data_setting

from .common import DATASET_ROOT
from .cmx_mit_b2_rel_plus_v2_2_formal import config as relplus_formal


COMPARISON_PROTOCOL_ID = "CMX_THREE_ARM_V2_2_NO_FLIP"


def build_comparison_config(arm):
    config = copy.deepcopy(relplus_formal)
    config.comparison_protocol_id = COMPARISON_PROTOCOL_ID
    config.comparison_arm = arm
    config.training_authorized = False
    config.full_cache_authorized = False
    config.data_ready = False
    if arm == "rgbd":
        config.experiment_name = "CMX_RGBD_v2_2_S2D_formal"
        config.x_mode = "standard"
        config.x_root_folder = osp.join(DATASET_ROOT, "RawDepth")
        config.x_is_single_channel = True
        config.representation_protocol_id = "CMX_RGBD_RAWDEPTH_480"
        config.cache_audit_report = None
    elif arm == "hha":
        config.experiment_name = "CMX_HHA_v2_2_S2D_formal"
        config.x_mode = "standard"
        config.x_root_folder = osp.join(DATASET_ROOT, "HHA")
        config.x_is_single_channel = False
        config.representation_protocol_id = "CMX_HHA_480"
        config.cache_audit_report = None
    elif arm == "rel_plus_v2_1":
        config.experiment_name = "CMX_RELPlus_v2_2_S2D_formal"
    else:
        raise ValueError("unknown comparison arm: {}".format(arm))
    config.data_setting = build_data_setting(config, split="train")
    return config


def frozen_control_fields(config):
    names = (
        "dataset_name",
        "dataset_split",
        "train_source",
        "eval_source",
        "num_train_imgs",
        "num_eval_imgs",
        "logical_samples_per_epoch",
        "sampler",
        "backbone",
        "decoder",
        "using_gate",
        "using_smmf",
        "using_dymm",
        "using_sga",
        "augmentation_profile",
        "train_horizontal_flip",
        "seed",
        "criterion",
        "focal_gamma",
        "loss_reduction",
        "optimizer",
        "scheduler",
        "primary_endpoint",
        "secondary_endpoint",
    )
    return {name: getattr(config, name) for name in names}
