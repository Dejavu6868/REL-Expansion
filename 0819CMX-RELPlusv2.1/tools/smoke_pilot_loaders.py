#!/usr/bin/env python3
"""Run real pilot train/validation DataLoaders without model construction."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _summary(batch):
    return {
        "sample_id": batch["fn"][0],
        "rgb_shape": list(batch["data"].shape),
        "rgb_dtype": str(batch["data"].dtype),
        "modal_x_shape": list(batch["modal_x"].shape),
        "modal_x_dtype": str(batch["modal_x"].dtype),
        "label_shape": list(batch["label"].shape),
        "label_dtype": str(batch["label"].dtype),
        "valid_mask_shape": list(batch["modal_x_valid_mask"].shape),
        "valid_mask_dtype": str(batch["modal_x_valid_mask"].dtype),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selected = importlib.import_module(
        "configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_1_pilot"
    )
    sys.modules["config"] = selected
    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.data_setting import build_data_setting
    from dataloader.dataloader import TrainPre, ValPre
    from utils.training_protocol import set_author_seed

    config = selected.config
    set_author_seed(config.seed, epoch=0, local_rank=0, distributed=False)
    train_setting = build_data_setting(config, split="train")
    train_dataset = RGBXDataset(
        train_setting,
        "train",
        TrainPre(config.norm_mean, config.norm_std, cfg=config),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    train_batch = next(iter(train_loader))

    val_setting = build_data_setting(config, split="val")
    val_dataset = RGBXDataset(
        val_setting,
        "val",
        ValPre(x_mode=val_setting["x_mode"]),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    val_batch = next(iter(val_loader))

    train = _summary(train_batch)
    validation = _summary(val_batch)
    status = (
        "PASS"
        if train["rgb_shape"] == [1, 3, 480, 480]
        and train["modal_x_shape"] == [1, 3, 480, 480]
        and train["rgb_dtype"] == "torch.float32"
        and train["modal_x_dtype"] == "torch.float32"
        and train["valid_mask_dtype"] == "torch.bool"
        and validation["modal_x_shape"] == [1, 480, 480, 3]
        and validation["valid_mask_shape"] == [1, 480, 480]
        else "FAIL"
    )
    report = {
        "status": status,
        "representation_protocol_id": config.representation_protocol_id,
        "augmentation_profile": config.augmentation_profile,
        "x_mode": config.x_mode,
        "channel_order": list(config.channel_order),
        "train_horizontal_flip": config.train_horizontal_flip,
        "train": train,
        "validation": validation,
        "valid_mask_passed_to_model": False,
        "model_constructed": False,
        "training_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
