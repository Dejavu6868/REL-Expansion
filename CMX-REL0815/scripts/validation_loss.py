#!/usr/bin/env python3
"""Compute diagnostic full-split CE for one fixed checkpoint."""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checked_checkpoint(path, expected_epoch):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError("checkpoint is absent or empty: {}".format(path))
    checksum = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint root must be a dictionary: {}".format(path))
    checkpoint_epoch = checkpoint.get("epoch")
    if type(checkpoint_epoch) is not int or checkpoint_epoch != expected_epoch:
        raise RuntimeError(
            "checkpoint epoch mismatch for {}: expected {}, found {!r}".format(
                path, expected_epoch, checkpoint_epoch
            )
        )
    if "model" not in checkpoint:
        raise KeyError("checkpoint has no model state: {}".format(path))
    return checkpoint, checksum


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def canonical_normalize(batch, mean, std):
    """Match CMX evaluator normalization: NumPy float64, then float32 tensor."""
    array = batch.detach().cpu().numpy()
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("expected NHWC three-channel input, got {}".format(array.shape))
    if array.dtype != np.uint8:
        raise TypeError("expected raw uint8 input, got {}".format(array.dtype))
    mean64 = np.asarray(mean, dtype=np.float64).reshape(1, 1, 1, 3)
    std64 = np.asarray(std, dtype=np.float64).reshape(1, 1, 1, 3)
    normalized = (array.astype(np.float64) / 255.0 - mean64) / std64
    normalized = np.ascontiguousarray(
        normalized.transpose(0, 3, 1, 2), dtype=np.float32
    )
    return torch.from_numpy(normalized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and args.output is None:
        parser.error("--output is required unless --preflight-only is used")

    checkpoint_path = args.run_dir / "checkpoints" / "epoch-{}.pth".format(args.epoch)
    checkpoint, checkpoint_sha256 = load_checked_checkpoint(checkpoint_path, args.epoch)
    checkpoint_report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "expected_epoch": args.epoch,
    }
    if args.preflight_only:
        if args.output is not None:
            atomic_write_json(args.output, checkpoint_report)
        print(json.dumps(checkpoint_report, indent=2, sort_keys=True))
        return

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    selected_config = importlib.import_module("configs.cmx_relplus_2d")
    config = selected_config.config
    sys.modules["config"] = selected_config

    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.dataloader import ValPre
    from models.builder import EncoderDecoder

    setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
        "class_names": config.class_names,
    }
    dataset = RGBXDataset(setting, "val", ValPre())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    model = EncoderDecoder(config, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device).eval()

    total_loss = 0.0
    valid_pixels = 0
    start = time.time()
    with torch.no_grad():
        for index, batch in enumerate(loader, 1):
            rgb = canonical_normalize(batch["data"], config.norm_mean, config.norm_std)
            relplus = canonical_normalize(
                batch["modal_x"], config.norm_mean, config.norm_std
            )
            rgb = rgb.to(device, non_blocking=True)
            relplus = relplus.to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True).long()
            logits = model(rgb, relplus)
            loss_sum = F.cross_entropy(
                logits, label, ignore_index=config.background, reduction="sum"
            )
            count = int((label != config.background).sum().item())
            if not torch.isfinite(loss_sum).item():
                raise RuntimeError("non-finite validation loss at batch {}".format(index))
            total_loss += float(loss_sum.item())
            valid_pixels += count
            if index % 200 == 0 or index == len(loader):
                if valid_pixels <= 0:
                    raise RuntimeError("no valid validation pixels through batch {}".format(index))
                print(
                    "epoch={} batch={}/{} mean_ce={:.8f}".format(
                        args.epoch, index, len(loader), total_loss / valid_pixels
                    ),
                    flush=True,
                )
    if valid_pixels <= 0:
        raise RuntimeError("validation split contains no valid pixels")
    mean_cross_entropy = total_loss / valid_pixels
    if not math.isfinite(total_loss) or not math.isfinite(mean_cross_entropy):
        raise RuntimeError("validation aggregate is non-finite")
    report = dict(checkpoint_report)
    report.update(
        {
            "epoch": args.epoch,
            "mean_cross_entropy": mean_cross_entropy,
            "cross_entropy_sum": total_loss,
            "valid_pixels": valid_pixels,
            "sample_count": len(dataset),
            "elapsed_seconds": time.time() - start,
            "normalization": "NumPy float64 (uint8/255 - mean) / std, then contiguous float32 NCHW",
            "protocol": "diagnostic direct 480x480 forward on fixed Area-5 split; pixel-weighted CE over non-ignore pixels; not used for checkpoint selection",
        }
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
