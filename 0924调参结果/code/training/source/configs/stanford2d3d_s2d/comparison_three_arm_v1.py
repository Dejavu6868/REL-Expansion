"""Frozen RGBD/HHA configs based on the executed CMX-REL+ V2.3 runtime."""

import copy
import os.path as osp

from dataloader.data_setting import build_data_setting

from .common import DATASET_ROOT
from .cmx_mit_b2_rel_plus_v2_3_formal import config as relplus_formal


INTEGRATION_PROTOCOL_ID = "CMX_S2D_THREE_ARM_V1"
COMPARISON_PROTOCOL_ID = "S2D_THREE_ARM_SEED12345_NO_FLIP_V1"
OUTPUT_ROOT = "/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_ThreeArm_v1"
INPUT_AUDIT_REPORT = (
    "/data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v2_3/"
    "reports/three_arm_data_contract.json"
)
TRAINING_CODE_EQUIVALENCE_REPORT = osp.join(
    OUTPUT_ROOT,
    "reports/training_code_equivalence/TRAINING_CODE_EQUIVALENCE_AUDIT.json",
)
MODEL_INITIALIZATION_REPORT = osp.join(
    OUTPUT_ROOT,
    "reports/initialization/THREE_ARM_MODEL_INITIALIZATION_AUDIT.json",
)
SAMPLER_SEQUENCE_REPORT = osp.join(
    OUTPUT_ROOT,
    "reports/sampler/OLD_VS_NEW_SAMPLER_SEQUENCE_AUDIT.json",
)
DATALOADER_TRACE_REPORT = osp.join(
    OUTPUT_ROOT,
    "reports/dataloader/THREE_ARM_DATALOADER_CROSS_EPOCH_TRACE.json",
)
EVALUATOR_EQUIVALENCE_REPORT = osp.join(
    OUTPUT_ROOT,
    "reports/evaluator/RELPLUS_EPOCH200_EVALUATOR_EQUIVALENCE.json",
)
EVALUATOR_RANK_CONSISTENCY_REPORT = osp.join(
    OUTPUT_ROOT,
    "reports/evaluator/RELPLUS_EPOCH200_1GPU_VS_8GPU_CONFUSION_AUDIT.json",
)


def build_three_arm_config(arm):
    config = copy.deepcopy(relplus_formal)
    if arm not in ("rgbd", "hha"):
        raise ValueError("unsupported training arm: {}".format(arm))

    config.integration_protocol_id = INTEGRATION_PROTOCOL_ID
    config.protocol_id = INTEGRATION_PROTOCOL_ID
    config.comparison_protocol_id = COMPARISON_PROTOCOL_ID
    config.comparison_arm = arm
    config.arm_name = "CMX-RGBD" if arm == "rgbd" else "CMX-HHA"
    config.input_audit_report = INPUT_AUDIT_REPORT
    config.training_code_equivalence_report = TRAINING_CODE_EQUIVALENCE_REPORT
    config.model_initialization_report = MODEL_INITIALIZATION_REPORT
    config.sampler_sequence_report = SAMPLER_SEQUENCE_REPORT
    config.dataloader_trace_report = DATALOADER_TRACE_REPORT
    config.evaluator_equivalence_report = EVALUATOR_EQUIVALENCE_REPORT
    config.evaluator_rank_consistency_report = EVALUATOR_RANK_CONSISTENCY_REPORT
    config.training_authorized = False
    config.full_cache_authorized = False
    config.source_compatible_invalid_accepted = False
    config.data_ready = False

    config.pop("x_valid_root_folder", None)
    config.pop("x_valid_format", None)
    config.channel_order = None
    if arm == "rgbd":
        config.experiment_name = "CMX_RGBD_S2D_three_arm_v1_seed12345"
        config.representation_protocol_id = "RGBD_UINT8_REPEAT3_SOURCECOMPAT"
        config.x_mode = "rawdepth_uint8_repeat3"
        config.x_root_folder = osp.join(DATASET_ROOT, "RawDepth")
        config.x_is_single_channel = True
        config.x_loader_adapter = "IMREAD_UNCHANGED_UINT8_REPEAT3"
        config.model_visible_x_channels = (
            "rawdepth_high8",
            "rawdepth_high8",
            "rawdepth_high8",
        )
    else:
        config.experiment_name = "CMX_HHA_S2D_three_arm_v1_seed12345"
        config.representation_protocol_id = "HHA_FROZEN_CACHE_EXECUTABLE_ORDER"
        config.x_mode = "hha_frozen_cache"
        config.x_root_folder = osp.join(DATASET_ROOT, "HHA")
        config.x_is_single_channel = False
        config.x_loader_adapter = "IMREAD_UNCHANGED_PRESERVE_ARRAY_ORDER"
        config.model_visible_x_channels = "frozen executable order; physical names unresolved"

    run_dir = osp.join(
        OUTPUT_ROOT,
        "cmx_{}_seed12345".format(arm),
    )
    config.output_dir = run_dir
    config.log_dir = osp.join(run_dir, "logs")
    config.tb_dir = osp.join(run_dir, "tensorboard")
    config.log_dir_link = osp.join(
        OUTPUT_ROOT, "latest_cmx_{}_seed12345_logs".format(arm)
    )
    config.checkpoint_dir = osp.join(run_dir, "checkpoints")
    config.log_file = osp.join(config.log_dir, "train.log")
    config.link_log_file = osp.join(config.log_dir, "train_last.log")
    config.val_log_file = osp.join(config.log_dir, "val.log")
    config.link_val_log_file = osp.join(config.log_dir, "val_last.log")
    config.ddp_smoke_report = osp.join(
        OUTPUT_ROOT,
        "ddp_smoke",
        arm,
        "ddp_optimizer_smoke_summary.json",
    )
    config.recovery_checkpoint_step = 5
    config.recovery_checkpoint_before_epoch = 100
    config.data_setting = build_data_setting(config, split="train")
    return config
