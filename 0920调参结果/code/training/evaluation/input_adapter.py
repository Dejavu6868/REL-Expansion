#!/usr/bin/env python3
"""Input-only adapter for the frozen V2.3 full evaluator on RGBD/HHA arms."""

import sys
from pathlib import Path

import numpy as np
import torch


SOURCE_ROOT = Path("/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-FG1-20260920/source")
sys.path.insert(0, str(SOURCE_ROOT))

from engine import relplus_evaluator as evaluator_core
from tools import eval_rel_plus_v2_3_full as frozen_evaluator


SUPPORTED_MODES = {
    "rgbd": "rawdepth_uint8_repeat3",
    "hha": "hha_frozen_cache",
}


def unbatch_three_arm(batch):
    sample = {"fn": batch["fn"][0]}
    for key in ("data", "modal_x", "label"):
        value = batch[key][0]
        sample[key] = (
            value.detach().cpu().numpy() if torch.is_tensor(value) else value
        )
    return sample


def prepare_three_arm_eval_sample(sample, config):
    arm = getattr(config, "comparison_arm", None)
    expected_mode = SUPPORTED_MODES.get(arm)
    if expected_mode is None:
        raise ValueError("three-arm evaluator requires comparison_arm=rgbd or hha")
    if getattr(config, "x_mode", None) != expected_mode:
        raise ValueError(
            "three-arm evaluator x_mode mismatch: expected {}, found {}".format(
                expected_mode, getattr(config, "x_mode", None)
            )
        )
    if getattr(config, "eval_flip", None) is not False:
        raise ValueError("three-arm evaluator requires eval_flip=False")
    if list(getattr(config, "eval_scale_array", [])) != [1]:
        raise ValueError("three-arm S2D evaluator requires eval_scale_array=[1]")

    rgb = evaluator_core._as_uint8_hwc3(sample["data"], "RGB")
    modal_x = evaluator_core._as_uint8_hwc3(
        sample["modal_x"], "three-arm modality"
    )
    label = np.asarray(sample["label"])
    if label.shape != rgb.shape[:2] or modal_x.shape[:2] != rgb.shape[:2]:
        raise ValueError("RGB, modality and label shapes must match")
    crop_size = tuple(int(value) for value in config.eval_crop_size)
    if crop_size != rgb.shape[:2]:
        raise ValueError(
            "full-image S2D evaluation requires sample shape {} but found {}".format(
                crop_size, rgb.shape[:2]
            )
        )

    return evaluator_core.PreparedEvalSample(
        rgb=evaluator_core._normalized_bchw(
            rgb, config.norm_mean, config.norm_std
        ),
        modal_x=evaluator_core._normalized_bchw(
            modal_x, config.norm_mean, config.norm_std
        ),
        label=np.ascontiguousarray(label),
        valid_mask=None,
        sample_id=str(sample["fn"]),
    )


def main():
    evaluator_core.prepare_eval_sample = prepare_three_arm_eval_sample
    frozen_evaluator._unbatch = unbatch_three_arm
    return frozen_evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
