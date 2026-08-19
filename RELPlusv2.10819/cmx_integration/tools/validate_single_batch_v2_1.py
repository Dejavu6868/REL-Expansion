#!/usr/bin/env python3
"""Run one real CMX forward/loss/backward with no optimizer or checkpoint."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
RELPLUS_ROOT = REPO_ROOT.parent
sys.path[:0] = [str(REPO_ROOT), str(RELPLUS_ROOT)]

from configs.stanford2d3dpano.cmx_mit_b2_rel_plus_v2_1 import make_config
from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import TrainPre
from models.builder import EncoderDecoder


def _gradient_summary(model, predicate):
    tensors = []
    names = []
    for name, parameter in model.named_parameters():
        if predicate(name) and parameter.grad is not None:
            tensors.append(parameter.grad.detach())
            names.append(name)
    finite = bool(tensors) and all(bool(torch.isfinite(value).all()) for value in tensors)
    nonzero = bool(tensors) and any(bool(torch.count_nonzero(value)) for value in tensors)
    norm = float(
        sum(value.float().norm().item() for value in tensors)
    )
    return {
        "tensor_count": len(tensors),
        "finite": finite,
        "nonzero": nonzero,
        "sum_tensor_norm": norm,
        "first_parameter": names[0] if names else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = make_config()
    cfg.train_scale_array = [1.0]
    cfg.num_workers = 0
    cfg.batch_size = 1
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real MiT-B2 wiring check")
    device = torch.device(args.device)

    setting = {
        "rgb_root": cfg.rgb_root_folder,
        "rgb_format": cfg.rgb_format,
        "gt_root": cfg.gt_root_folder,
        "gt_format": cfg.gt_format,
        "transform_gt": cfg.gt_transform,
        "x_root": cfg.x_root_folder,
        "x_format": cfg.x_format,
        "x_single_channel": cfg.x_is_single_channel,
        "x_mode": cfg.x_mode,
        "x_valid_root": cfg.x_valid_root_folder,
        "x_valid_format": cfg.x_valid_format,
        "train_source": cfg.train_source,
        "eval_source": cfg.eval_source,
        "class_names": cfg.class_names,
    }
    preprocess = TrainPre(
        cfg.norm_mean, cfg.norm_std, cfg=cfg, rng=np.random.default_rng(cfg.seed)
    )
    dataset = RGBXDataset(setting, "train", preprocess)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )
    batch = next(iter(loader))
    rgb = batch["data"].to(device, non_blocking=True)
    modal_x = batch["modal_x"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    valid_label = (label == 255) | ((label >= 0) & (label < cfg.num_classes))
    if not bool(valid_label.all()):
        raise ValueError("label contains values outside classes and ignore index")
    if rgb.dtype != torch.float32 or modal_x.dtype != torch.float32:
        raise TypeError("model inputs must be explicit float32")
    if rgb.shape != modal_x.shape or rgb.shape[1] != 3:
        raise ValueError("RGB and modal_x must be matching BCHW three-channel tensors")

    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=255)
    model = EncoderDecoder(cfg=cfg, criterion=criterion, norm_layer=nn.BatchNorm2d)
    model.to(device)
    model.train()
    model.zero_grad()
    loss = model(rgb, modal_x, label)
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise FloatingPointError("single-batch loss is not finite scalar")
    loss.backward()

    gradients = {
        "rgb_encoder": _gradient_summary(
            model,
            lambda name: name.startswith("backbone.")
            and "extra_" not in name
            and ".FRMs." not in name
            and ".FFMs." not in name,
        ),
        "x_encoder": _gradient_summary(
            model, lambda name: name.startswith("backbone.extra_")
        ),
        "fusion": _gradient_summary(
            model, lambda name: ".FRMs." in name or ".FFMs." in name
        ),
        "decoder": _gradient_summary(
            model, lambda name: name.startswith("decode_head.")
        ),
    }
    if not all(item["finite"] and item["nonzero"] for item in gradients.values()):
        raise RuntimeError("one or more CMX parameter groups lacked finite nonzero gradients")

    report = {
        "status": "PASS",
        "protocol_id": cfg.protocol_id,
        "architecture": {
            "model": "Original CMX",
            "backbone": cfg.backbone,
            "decoder": cfg.decoder,
            "Gate": cfg.using_gate,
            "DyMM": cfg.using_dymm,
            "SMMF": cfg.using_smmf,
            "SGA": cfg.using_sga,
        },
        "sample_id": batch["fn"][0],
        "rgb_shape": list(rgb.shape),
        "modal_x_shape": list(modal_x.shape),
        "label_shape": list(label.shape),
        "rgb_dtype": str(rgb.dtype),
        "modal_x_dtype": str(modal_x.dtype),
        "loss": float(loss.detach().cpu()),
        "ignore_index": criterion.ignore_index,
        "ignore_pixel_count": int(torch.count_nonzero(label == 255).item()),
        "gradients": gradients,
        "diagnostic_valid_mask_shape": list(
            batch["modal_x_valid_mask"].shape
        ),
        "diagnostic_valid_mask_passed_to_model": False,
        "backward_executed": True,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "scheduler_step_executed": False,
        "checkpoint_written": False,
        "epoch_loop_started": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    del loss, model, rgb, modal_x, label
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
