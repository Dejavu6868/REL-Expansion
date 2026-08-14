#!/usr/bin/env python3
"""Render declared Stage2B gravity/performance-stratum prediction montages."""

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np


SEEDS = (12345, 23456, 34567)
CLASS_NAMES = (
    "beam",
    "board",
    "bookcase",
    "ceiling",
    "chair",
    "clutter",
    "column",
    "door",
    "floor",
    "sofa",
    "table",
    "wall",
    "window",
)
PALETTE = np.asarray(
    [
        (255, 179, 0),
        (128, 62, 117),
        (255, 104, 0),
        (166, 189, 215),
        (193, 0, 32),
        (206, 162, 98),
        (129, 112, 102),
        (0, 125, 52),
        (246, 118, 142),
        (0, 83, 138),
        (255, 122, 92),
        (83, 55, 122),
        (255, 142, 0),
    ],
    dtype=np.uint8,
)
PALETTE_BGR = PALETTE[:, ::-1]


def _run_dir(stage2a_root, stage2b_root, seed, arm):
    if seed == 12345:
        return stage2a_root / arm
    return stage2b_root / "seed_{}".format(seed) / arm


def _read_image(path, flags):
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError("missing image: {}".format(path))
    return image


def _colorize(labels):
    labels = np.asarray(labels)
    output = np.zeros(labels.shape + (3,), dtype=np.uint8)
    valid = (labels >= 0) & (labels < len(PALETTE))
    output[valid] = PALETTE_BGR[labels[valid].astype(np.int64)]
    return output


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representatives", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--stage2a-root", required=True, type=Path)
    parser.add_argument("--stage2b-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    with args.representatives.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        raise ValueError("expected six declared representative samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for index, row in enumerate(rows, start=1):
        name = row["name"]
        rgb_path = args.dataset_root / "RGB" / (name + ".png")
        label_path = args.dataset_root / "Label" / (name + ".png")
        rgb = _read_image(rgb_path, cv2.IMREAD_COLOR)
        stored_label = _read_image(label_path, cv2.IMREAD_UNCHANGED)
        gt = stored_label.astype(np.int16) - 1
        gt[stored_label == 0] = 255

        panels = [("RGB", rgb), ("Ground truth", _colorize(gt))]
        for seed in SEEDS:
            for arm, label in (("relplus_local", "Local"), ("relplus_pose", "Pose")):
                prediction_path = (
                    _run_dir(args.stage2a_root, args.stage2b_root, seed, arm)
                    / "visualizations/predictions"
                    / (name + ".png")
                )
                prediction = _read_image(prediction_path, cv2.IMREAD_UNCHANGED)
                if prediction.ndim != 2:
                    raise ValueError("prediction must be a label-index image")
                panels.append(("seed {} {}".format(seed, label), _colorize(prediction)))

        panel_height, panel_width, title_height = 480, 480, 42
        header_height, legend_height = 100, 100
        canvas = np.full(
            (
                header_height + 2 * (panel_height + title_height) + legend_height,
                4 * panel_width,
                3,
            ),
            255,
            dtype=np.uint8,
        )
        cv2.putText(
            canvas,
            name,
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        subtitle = (
            "gravity={} ({:.3f} deg) | {} | mean Pose-Local present-mIoU={:+.3f} pp"
        ).format(
            row["gravity_group"],
            float(row["gravity_error_deg"]),
            row["performance_stratum"],
            float(row["mean_pose_minus_local_present_class_miou_pp"]),
        )
        cv2.putText(
            canvas,
            subtitle,
            (18, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        for panel_index, (title, image) in enumerate(panels):
            row_index, column_index = divmod(panel_index, 4)
            y0 = header_height + row_index * (panel_height + title_height)
            x0 = column_index * panel_width
            canvas[y0 + title_height : y0 + title_height + panel_height, x0 : x0 + panel_width] = image
            cv2.putText(
                canvas,
                title,
                (x0 + 12, y0 + 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        legend_y = header_height + 2 * (panel_height + title_height) + 24
        cell_width = canvas.shape[1] // 7
        for class_index, class_name in enumerate(CLASS_NAMES):
            legend_row, legend_column = divmod(class_index, 7)
            x0 = legend_column * cell_width + 12
            y0 = legend_y + legend_row * 38
            cv2.rectangle(
                canvas,
                (x0, y0),
                (x0 + 22, y0 + 22),
                tuple(int(value) for value in PALETTE_BGR[class_index]),
                thickness=-1,
            )
            cv2.putText(
                canvas,
                class_name,
                (x0 + 30, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        output = args.output_dir / "{:02d}_{}_{}_{}.png".format(
            index,
            row["gravity_group"],
            row["performance_stratum"],
            _safe_name(name.split("/", 1)[1]),
        )
        if not cv2.imwrite(str(output), canvas):
            raise ValueError("failed to write visualization: {}".format(output))
        manifest.append(
            {
                "name": name,
                "gravity_group": row["gravity_group"],
                "performance_stratum": row["performance_stratum"],
                "figure": str(output),
            }
        )
        print("visualization={}".format(output), flush=True)

    with (args.output_dir / "visualization_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)


if __name__ == "__main__":
    main()
