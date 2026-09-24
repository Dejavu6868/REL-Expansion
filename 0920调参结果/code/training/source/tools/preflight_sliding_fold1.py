#!/usr/bin/env python3
"""Run one full 2048x4096 S3D panorama through the current CMX slider."""

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def coverage_map(height, width, crop_height, crop_width, stride_height, stride_width):
    coverage = np.zeros((height, width), dtype=np.uint8)
    rows = int(math.ceil((height - crop_height) / stride_height)) + 1
    cols = int(math.ceil((width - crop_width) / stride_width)) + 1
    windows = []
    for row in range(rows):
        for col in range(cols):
            start_x = col * stride_width
            start_y = row * stride_height
            end_x = min(start_x + crop_width, width)
            end_y = min(start_y + crop_height, height)
            start_x = end_x - crop_width
            start_y = end_y - crop_height
            coverage[start_y:end_y, start_x:end_x] += 1
            windows.append([start_y, end_y, start_x, end_x])
    return coverage, windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-module", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    selected = importlib.import_module(args.config_module)
    sys.modules["config"] = selected

    import torch
    import torch.nn as nn
    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.dataloader import ValPre
    from engine.evaluator import Evaluator
    from models.builder import EncoderDecoder as SegModel
    from utils.metric import hist_info

    config = selected.config
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU for each sliding preflight")

    dataset = RGBXDataset(config.data_setting, "val", ValPre())
    sample = dataset[0]
    image = sample["data"]
    modal_x = sample["modal_x"]
    label = sample["label"]

    network = SegModel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    incompatible = network.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint mismatch: missing={} unexpected={}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )
    network.cuda().eval()

    helper = Evaluator(
        dataset,
        config.num_classes,
        config.norm_mean,
        config.norm_std,
        network,
        config.eval_scale_array,
        config.eval_flip,
        [0],
    )
    helper.val_func = network

    started = time.time()
    with torch.no_grad():
        prediction = helper.sliding_eval_rgbX(
            image,
            modal_x,
            config.eval_crop_size,
            config.eval_stride_rate,
            0,
        )
    elapsed = time.time() - started

    crop_height, crop_width = config.eval_crop_size
    stride_height = int(math.ceil(crop_height * config.eval_stride_rate))
    stride_width = int(math.ceil(crop_width * config.eval_stride_rate))
    coverage, windows = coverage_map(
        image.shape[0], image.shape[1], crop_height, crop_width, stride_height, stride_width
    )
    hist, labeled, correct = hist_info(config.num_classes, prediction, label)
    errors = []
    if list(image.shape) != [2048, 4096, 3]:
        errors.append("RGB panorama shape mismatch")
    if list(modal_x.shape) != [2048, 4096, 3]:
        errors.append("X panorama shape mismatch")
    if list(prediction.shape) != [2048, 4096]:
        errors.append("prediction shape mismatch")
    if prediction.shape != label.shape:
        errors.append("prediction and label shapes differ")
    if int(coverage.min()) < 1:
        errors.append("sliding windows leave uncovered pixels")
    if int(hist.sum()) != int(labeled):
        errors.append("confusion matrix count mismatch")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "experiment": config.experiment_name,
        "sample": sample["fn"],
        "image_shape": list(image.shape),
        "x_shape": list(modal_x.shape),
        "label_shape": list(label.shape),
        "prediction_shape": list(prediction.shape),
        "crop": [crop_height, crop_width],
        "stride_rate": config.eval_stride_rate,
        "stride_pixels": [stride_height, stride_width],
        "scale": config.eval_scale_array,
        "flip": config.eval_flip,
        "window_count": len(windows),
        "windows": windows,
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "coverage_zero_pixels": int((coverage == 0).sum()),
        "confusion_shape": list(hist.shape),
        "confusion_sum": int(hist.sum()),
        "labeled_pixels": int(labeled),
        "correct_pixels": int(correct),
        "prediction_min": int(prediction.min()),
        "prediction_max": int(prediction.max()),
        "elapsed_seconds": elapsed,
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
