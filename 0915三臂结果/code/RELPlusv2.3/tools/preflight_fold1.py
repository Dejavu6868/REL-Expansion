#!/usr/bin/env python3
"""Run the real 1080 CMX S3D forward/backward/update preflight."""

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np


def tensor_stats(tensor):
    values = tensor.detach().float()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std().item()),
        "finite": bool(values.isfinite().all().item()),
    }


def parameter_group(name):
    if name.startswith("backbone.extra_"):
        return "x_encoder"
    if name.startswith("backbone.FRMs.") or name.startswith("backbone.FFMs."):
        return "fusion"
    if name.startswith("backbone.patch_embed") or name.startswith("backbone.block") or name.startswith("backbone.norm"):
        return "rgb_encoder"
    if name.startswith("decode_head."):
        return "decoder"
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-module", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    selected = importlib.import_module(args.config_module)
    sys.modules["config"] = selected

    import torch
    import torch.nn as nn
    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.dataloader import TrainPre
    from models.builder import EncoderDecoder as SegModel
    from utils.init_func import group_weight
    from utils.loss_opr import FocalLoss2d

    config = selected.config
    seed = int(config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 1080 preflight")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU for each preflight")

    dataset = RGBXDataset(
        config.data_setting,
        "train",
        TrainPre(config.norm_mean, config.norm_std),
    )
    sample = dataset[0]
    rgb = sample["data"].unsqueeze(0).cuda(non_blocking=False)
    modal_x = sample["modal_x"].unsqueeze(0).cuda(non_blocking=False)
    label = sample["label"].unsqueeze(0).cuda(non_blocking=False)

    criterion = FocalLoss2d(reduction="mean", ignore_index=config.background)
    model = SegModel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)

    counts = {"rgb_encoder": 0, "x_encoder": 0, "fusion": 0, "decoder": 0, "other": 0}
    for name, parameter in model.named_parameters():
        counts[parameter_group(name)] += parameter.numel()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    raw = torch.load(config.pretrained_model, map_location="cpu")
    if "model" in raw:
        raw = raw["model"]
    backbone_state = model.backbone.state_dict()
    mapped = []
    unused_raw = []
    for key, value in raw.items():
        targets = []
        if "patch_embed" in key:
            targets = [key, key.replace("patch_embed", "extra_patch_embed")]
        elif "block" in key:
            targets = [key, key.replace("block", "extra_block")]
        elif "norm" in key:
            targets = [key, key.replace("norm", "extra_norm")]
        else:
            unused_raw.append(key)
        for target in targets:
            loaded = target in backbone_state and tuple(backbone_state[target].shape) == tuple(value.shape)
            equal = loaded and torch.equal(backbone_state[target].cpu(), value.cpu())
            mapped.append({"source": key, "target": target, "loaded": bool(loaded), "equal": bool(equal)})
    missing_mapped = [item["target"] for item in mapped if not item["loaded"] or not item["equal"]]
    rgb_loaded = any(item["equal"] and not item["target"].startswith("extra_") for item in mapped)
    x_loaded = any(item["equal"] and item["target"].startswith("extra_") for item in mapped)

    params_list = group_weight([], model, nn.BatchNorm2d, config.lr)
    optimizer = torch.optim.AdamW(
        params_list,
        lr=config.lr,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    model.cuda().train()
    optimizer.zero_grad()
    logits = model(rgb, modal_x)
    loss = criterion(logits, label.long())
    loss.backward()

    selected_parameters = {}
    for group_name in ("rgb_encoder", "x_encoder", "fusion", "decoder"):
        for name, parameter in model.named_parameters():
            if parameter_group(name) != group_name or id(parameter) not in optimizer_ids:
                continue
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                continue
            if parameter.grad.detach().abs().max().item() == 0:
                continue
            selected_parameters[group_name] = {
                "name": name,
                "before": parameter.detach().cpu().clone(),
                "parameter": parameter,
                "gradient_max_abs": float(parameter.grad.detach().abs().max().item()),
            }
            break

    optimizer.step()
    updates = {}
    for group_name, item in selected_parameters.items():
        after = item["parameter"].detach().cpu()
        updates[group_name] = {
            "parameter": item["name"],
            "in_optimizer": True,
            "gradient_max_abs": item["gradient_max_abs"],
            "parameter_change_max_abs": float((after - item["before"]).abs().max().item()),
        }

    label_values = sorted(int(value) for value in torch.unique(label).cpu().tolist())
    valid_pixels = int((label != config.background).sum().item())
    ignore_pixels = int((label == config.background).sum().item())
    required_groups = {"rgb_encoder", "x_encoder", "fusion", "decoder"}
    errors = []
    if list(rgb.shape) != [1, 3, 1080, 1080]:
        errors.append("RGB shape mismatch")
    if list(modal_x.shape) != [1, 3, 1080, 1080]:
        errors.append("X shape mismatch")
    if list(label.shape) != [1, 1080, 1080]:
        errors.append("label shape mismatch")
    if list(logits.shape) != [1, 13, 1080, 1080]:
        errors.append("logits shape mismatch")
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        errors.append("non-finite logits or loss")
    if not set(label_values).issubset(set(range(13)) | {255}):
        errors.append("label IDs outside 0..12 and 255")
    if missing_mapped or not rgb_loaded or not x_loaded:
        errors.append("MiT-B2 initialization did not load both encoder branches")
    for group_name in required_groups:
        if group_name not in updates or updates[group_name]["parameter_change_max_abs"] <= 0:
            errors.append("{} did not update".format(group_name))
    if total_parameters != 66567573:
        errors.append("parameter count is {}, expected 66567573".format(total_parameters))

    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": 0,
                "iteration": 0,
            },
            str(args.checkpoint),
        )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "experiment": config.experiment_name,
        "sample": sample["fn"],
        "architecture": config.architecture,
        "backbone": config.backbone,
        "gate": config.using_gate,
        "smmf": config.using_smmf,
        "dymm": config.using_dymm,
        "dataset": config.dataset_name,
        "fold": config.dataset_fold,
        "seed": seed,
        "amp": False,
        "rgb": tensor_stats(rgb),
        "x_channels": [tensor_stats(modal_x[:, channel]) for channel in range(3)],
        "label_shape": list(label.shape),
        "label_values": label_values,
        "valid_pixels": valid_pixels,
        "ignore_pixels": ignore_pixels,
        "logits": tensor_stats(logits),
        "loss": float(loss.detach().item()),
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
            **counts,
        },
        "pretrained": {
            "path": config.pretrained_model,
            "mapped_target_count": len(mapped),
            "successfully_loaded_target_count": len(mapped) - len(missing_mapped),
            "missing_mapped_targets": missing_mapped,
            "unused_source_keys": unused_raw,
            "rgb_branch_loaded": rgb_loaded,
            "x_branch_loaded": x_loaded,
        },
        "updates": updates,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
