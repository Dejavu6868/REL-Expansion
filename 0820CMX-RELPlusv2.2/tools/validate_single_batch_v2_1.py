#!/usr/bin/env python3
"""Run one real CMX forward/loss/backward with no optimizer or checkpoint."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
RELPLUS_ROOT = REPO_ROOT.parent
sys.path[:0] = [str(REPO_ROOT), str(RELPLUS_ROOT)]


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

    selected = importlib.import_module(
        "configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_1_pilot"
    )
    sys.modules["config"] = selected
    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.data_setting import build_data_setting
    from dataloader.dataloader import TrainPre
    from models.builder import EncoderDecoder
    from utils.training_protocol import (
        build_author_criterion,
        configure_author_cudnn,
        set_author_seed,
    )

    cfg = selected.config
    cfg.train_scale_array = [1.0]
    cfg.num_workers = 0
    cfg.batch_size = 1
    seed = set_author_seed(cfg.seed, epoch=0, local_rank=0, distributed=False)
    cudnn = configure_author_cudnn()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real MiT-B2 wiring check")
    device = torch.device(args.device)

    setting = build_data_setting(cfg, split="train")
    preprocess = TrainPre(
        cfg.norm_mean, cfg.norm_std, cfg=cfg
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

    criterion = build_author_criterion(cfg)
    model = EncoderDecoder(cfg=cfg, criterion=criterion, norm_layer=nn.BatchNorm2d)
    model.to(device)
    model.train()
    model.zero_grad()
    logits = model(rgb, modal_x)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("single-batch logits are not finite")
    loss_map = criterion(logits, label.long())
    loss = loss_map.mean()
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
        "loss_map_shape": list(loss_map.shape),
        "logits_finite": True,
        "criterion": "FocalLoss2d(gamma=2,reduction=none)->mean",
        "ignore_index": criterion.loss.ignore_index,
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
        "seed": seed,
        "cudnn": cudnn,
        "pretrained_backbone_requested": cfg.pretrained_model,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    del loss, loss_map, logits, model, rgb, modal_x, label
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
