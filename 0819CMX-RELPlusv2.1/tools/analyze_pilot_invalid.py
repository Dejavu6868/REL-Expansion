#!/usr/bin/env python3
"""Diagnose invalid=255 interpolation jointly with labels; never alter inputs."""

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=36)
    args = parser.parse_args()
    selected = importlib.import_module(
        "configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_1_pilot"
    )
    config = selected.config
    from dataloader.profiles import author_epoch_seed, sample_comparison_transform
    from rel_plus.integration.cmx_preprocess import analyze_invalid_interpolation

    with open(config.eval_source, encoding="utf-8") as handle:
        sample_ids = [line.strip() for line in handle if line.strip()][: args.limit]
    rng = random.Random(author_epoch_seed(config.seed, 0, 0))
    rows = []
    for sample_id in sample_ids:
        rel_path = Path(config.x_root_folder) / (sample_id + config.x_format)
        mask_path = Path(config.x_valid_root_folder) / (
            sample_id + config.x_valid_format
        )
        label_path = Path(config.gt_root_folder) / (
            sample_id + config.gt_format
        )
        rel_plus = cv2.imread(str(rel_path), cv2.IMREAD_UNCHANGED)
        valid = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if rel_plus is None or valid is None or label is None:
            raise FileNotFoundError("pilot diagnostic input missing: {}".format(sample_id))
        label = label.astype(np.uint8) - np.uint8(1)
        transform = sample_comparison_transform(
            rel_plus.shape[:2],
            config.train_scale_array,
            (config.image_height, config.image_width),
            rng,
        )
        diagnostic = analyze_invalid_interpolation(
            rel_plus,
            valid.astype(bool),
            transform,
            label=label,
            ignore_index=config.background,
            num_classes=config.num_classes,
        )
        rows.append({"sample_id": sample_id, "diagnostic": diagnostic})

    mean_keys = (
        "source_invalid_ratio",
        "transformed_nearest_invalid_ratio",
        "bilinear_invalid_affected_ratio",
        "affected_mean_channel_deviation",
        "affected_max_channel_deviation",
    )
    summary = {
        key: float(np.mean([row["diagnostic"][key] for row in rows]))
        for key in mean_keys
    }
    summary["affected_pixel_count"] = int(
        sum(row["diagnostic"]["affected_pixel_count"] for row in rows)
    )
    summary["affected_label_class_counts"] = {
        str(class_id): int(
            sum(
                row["diagnostic"]["affected_label_class_counts"][str(class_id)]
                for row in rows
            )
        )
        for class_id in range(config.num_classes)
    }
    ignore_count = int(
        sum(row["diagnostic"]["affected_label_ignore_count"] for row in rows)
    )
    semantic_count = int(
        sum(
            row["diagnostic"]["affected_label_valid_semantic_count"]
            for row in rows
        )
    )
    denominator = max(1, summary["affected_pixel_count"])
    summary["affected_label_ignore_count"] = ignore_count
    summary["affected_label_valid_semantic_count"] = semantic_count
    summary["affected_label_ignore_ratio"] = float(ignore_count / denominator)
    summary["affected_label_valid_semantic_ratio"] = float(
        semantic_count / denominator
    )
    report = {
        "status": "PASS" if rows else "FAIL",
        "policy": config.invalid_policy,
        "sample_count": len(rows),
        "formal_input_changed": False,
        "summary": summary,
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
