#!/usr/bin/env python3
"""Evaluate one CMX S3D Fold 1 checkpoint with deterministic rank partitioning."""

import argparse
import csv
import importlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def nullable(values):
    return [None if not np.isfinite(value) else float(value) for value in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-module", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--per-class", required=True, type=Path)
    parser.add_argument("--confusion", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    selected = importlib.import_module(args.config_module)
    sys.modules["config"] = selected

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.data_setting import build_data_setting
    from dataloader.dataloader import ValPre
    from engine.evaluator import Evaluator
    from models.builder import EncoderDecoder as SegModel
    from utils.metric import compute_score, hist_info

    config = selected.config
    if getattr(config, "x_mode", None) == "rel_plus_v2_1":
        raise RuntimeError(
            "Legacy eval_fold1.py rejects REL+ v2.1; "
            "use tools/eval_rel_plus_v2_1.py"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = args.local_rank
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
    else:
        rank = 0
        torch.cuda.set_device(0)
        local_rank = 0

    data_setting = build_data_setting(config, split="val")
    dataset = RGBXDataset(
        data_setting, "val", ValPre(x_mode=data_setting["x_mode"])
    )
    if args.limit:
        dataset._file_names = dataset._file_names[: args.limit]

    network = SegModel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    saved = torch.load(str(args.checkpoint), map_location="cpu")
    state = saved["model"] if "model" in saved else saved
    incompatible = network.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint mismatch: missing={} unexpected={}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )
    network.cuda(local_rank).eval()
    helper = Evaluator(
        dataset,
        config.num_classes,
        config.norm_mean,
        config.norm_std,
        network,
        config.eval_scale_array,
        config.eval_flip,
        [local_rank],
    )
    helper.val_func = network

    hist = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)
    correct = 0
    labeled = 0
    ignore = 0
    processed = 0
    started = time.time()
    with torch.no_grad():
        for index in range(rank, len(dataset), world_size):
            sample = dataset[index]
            prediction = helper.sliding_eval_rgbX(
                sample["data"],
                sample["modal_x"],
                config.eval_crop_size,
                config.eval_stride_rate,
                local_rank,
            )
            hist_one, labeled_one, correct_one = hist_info(
                config.num_classes, prediction, sample["label"]
            )
            hist += hist_one.astype(np.int64)
            labeled += int(labeled_one)
            correct += int(correct_one)
            ignore += int(sample["label"].size - labeled_one)
            processed += 1
            if args.predictions:
                output = args.predictions / (sample["fn"] + "_pred.png")
                output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output), prediction.astype(np.uint8)):
                    raise RuntimeError("failed to save prediction {}".format(output))

    hist_tensor = torch.as_tensor(hist, dtype=torch.int64, device="cuda:{}".format(local_rank))
    count_tensor = torch.tensor(
        [correct, labeled, ignore, processed],
        dtype=torch.int64,
        device="cuda:{}".format(local_rank),
    )
    if distributed:
        dist.all_reduce(hist_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        hist = hist_tensor.cpu().numpy()
        correct, labeled, ignore, processed = [int(value) for value in count_tensor.cpu().tolist()]
        iou, mean_iou, _, frequency_iou, mean_class_accuracy, pixel_accuracy = compute_score(
            hist, correct, labeled
        )
        class_accuracy = np.diag(hist) / hist.sum(axis=1)
        epoch = saved.get("epoch") if isinstance(saved, dict) else None
        payload = {
            "status": "PASS",
            "experiment": config.experiment_name,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(epoch) if epoch is not None else None,
            "sample_count": processed,
            "expected_sample_count": len(dataset),
            "world_size": world_size,
            "crop": config.eval_crop_size,
            "stride_rate": config.eval_stride_rate,
            "stride_pixels": config.eval_stride_pixels,
            "scales": config.eval_scale_array,
            "flip": config.eval_flip,
            "mIoU": float(mean_iou),
            "mIoU_percent": float(mean_iou * 100),
            "pixel_accuracy": float(pixel_accuracy),
            "pixel_accuracy_percent": float(pixel_accuracy * 100),
            "mean_class_accuracy": float(mean_class_accuracy),
            "mean_class_accuracy_percent": float(mean_class_accuracy * 100),
            "frequency_weighted_iou": float(frequency_iou),
            "iou": nullable(iou),
            "class_accuracy": nullable(class_accuracy),
            "class_names": config.class_names,
            "valid_pixels": labeled,
            "ignore_pixels": ignore,
            "correct_pixels": correct,
            "elapsed_seconds": time.time() - started,
        }
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.per_class.parent.mkdir(parents=True, exist_ok=True)
        args.confusion.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with args.per_class.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class_id", "class_name", "iou", "class_accuracy"])
            for class_id, class_name in enumerate(config.class_names):
                writer.writerow([class_id, class_name, iou[class_id], class_accuracy[class_id]])
        np.savetxt(str(args.confusion), hist, fmt="%d", delimiter=",")
        print(json.dumps(payload, ensure_ascii=False))

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
