#!/usr/bin/env python3
"""Build fixed cross-arm S3D Fold 1 qualitative comparisons."""

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


OUTPUT_ROOT = Path("/data/zhuzhaoziao/RELPlus/outputs/CMX_S3D_Fold1_reproduction")
DATA_ROOT = OUTPUT_ROOT / "common" / "Stanford2D3DPano"
RUNS = {
    "rgbd": OUTPUT_ROOT / "cmx_rgbd_fold1_seed12345",
    "hha": OUTPUT_ROOT / "cmx_hha_fold1_seed12345",
    "rel": OUTPUT_ROOT / "cmx_rel_fold1_seed12345",
}
PALETTE = np.array(
    [
        [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
        [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
        [192, 0, 128],
    ], dtype=np.uint8
)


def paths(sample_id):
    return {
        "rgb": DATA_ROOT / "image" / (sample_id + "_rgb.png"),
        "depth": DATA_ROOT / "depth" / (sample_id + "_depth.png"),
        "hha": DATA_ROOT / "hha" / (sample_id + "_hha.png"),
        "rel": DATA_ROOT / "rel" / (sample_id + "_rel.png"),
        "label": DATA_ROOT / "label" / (sample_id + "_semantic.png"),
        "rgbd_pred": RUNS["rgbd"] / "predictions_best" / (sample_id + "_pred.png"),
        "hha_pred": RUNS["hha"] / "predictions_best" / (sample_id + "_pred.png"),
        "rel_pred": RUNS["rel"] / "predictions_best" / (sample_id + "_pred.png"),
    }


def color_classes(values, valid=None):
    output = PALETTE[np.clip(values, 0, 12)]
    if valid is not None:
        output[~valid] = 0
    return output


def label_and_valid(raw_label):
    valid = raw_label > 0
    label = np.zeros_like(raw_label)
    label[valid] = raw_label[valid] - 1
    return label, valid


def score_sample(sample_id):
    item = paths(sample_id)
    raw_label = cv2.imread(str(item["label"]), cv2.IMREAD_GRAYSCALE)
    label, valid = label_and_valid(raw_label)
    predictions = {
        name: cv2.imread(str(item[name + "_pred"]), cv2.IMREAD_GRAYSCALE)
        for name in ("rgbd", "hha", "rel")
    }
    correct = {name: (prediction == label) & valid for name, prediction in predictions.items()}
    denominator = int(valid.sum())
    return {
        "sample_id": sample_id,
        "all_correct": float((correct["rgbd"] & correct["hha"] & correct["rel"]).sum() / denominator),
        "all_wrong": float((~correct["rgbd"] & ~correct["hha"] & ~correct["rel"] & valid).sum() / denominator),
        "rel_over_hha": float((correct["rel"] & ~correct["hha"]).sum() / denominator),
        "hha_over_rel": float((correct["hha"] & ~correct["rel"]).sum() / denominator),
        "rgbd_only": float((correct["rgbd"] & ~correct["hha"] & ~correct["rel"]).sum() / denominator),
    }


def add_title(image, title):
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((288, 512, 3), dtype=np.uint8)
    canvas[32:] = image
    cv2.putText(canvas, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def make_panels(sample_id, output_dir):
    item = paths(sample_id)
    rgb = cv2.imread(str(item["rgb"]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(item["depth"]), cv2.IMREAD_UNCHANGED)
    depth = cv2.cvtColor((depth >> 8).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    hha = cv2.imread(str(item["hha"]), cv2.IMREAD_COLOR)
    rel = cv2.imread(str(item["rel"]), cv2.IMREAD_UNCHANGED)
    raw_label = cv2.imread(str(item["label"]), cv2.IMREAD_GRAYSCALE)
    label, valid = label_and_valid(raw_label)
    predictions = {
        name: cv2.imread(str(item[name + "_pred"]), cv2.IMREAD_GRAYSCALE)
        for name in ("rgbd", "hha", "rel")
    }

    panels = [
        add_title(rgb, "RGB"), add_title(depth, "Raw depth"), add_title(hha, "HHA"),
        add_title(rel, "REL [EGVIA, LOA, ReD]"), add_title(color_classes(label, valid), "Ground truth"),
        add_title(color_classes(predictions["rgbd"]), "CMX-RGBD"),
        add_title(color_classes(predictions["hha"]), "CMX-HHA"),
        add_title(color_classes(predictions["rel"]), "CMX-REL"),
    ]
    comparison = np.hstack(panels)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / (sample_id.replace("/", "__") + "_comparison.png")), comparison)

    correct = {name: (prediction == label) & valid for name, prediction in predictions.items()}
    masks = [
        (correct["hha"] & ~correct["rgbd"], "HHA correct / RGBD wrong"),
        (correct["rel"] & ~correct["rgbd"], "REL correct / RGBD wrong"),
        (correct["rel"] & ~correct["hha"], "REL correct / HHA wrong"),
        (correct["hha"] & ~correct["rel"], "HHA correct / REL wrong"),
    ]
    mask_panels = []
    for mask, title in masks:
        image = np.zeros((*mask.shape, 3), dtype=np.uint8)
        image[mask] = (0, 255, 0)
        mask_panels.append(add_title(image, title))
    cv2.imwrite(
        str(output_dir / (sample_id.replace("/", "__") + "_difference_masks.png")),
        np.hstack(mask_panels),
    )


def main():
    sample_ids = (DATA_ROOT / "fold1_test.txt").read_text(encoding="utf-8").splitlines()
    scores = [score_sample(sample_id) for sample_id in sample_ids]
    categories = ("all_correct", "all_wrong", "rel_over_hha", "hha_over_rel", "rgbd_only")
    selected = []
    selected_set = set()
    selected_category = {}
    for category in categories:
        for row in sorted(scores, key=lambda item: item[category], reverse=True):
            if row["sample_id"] in selected_set:
                continue
            selected.append(row["sample_id"])
            selected_set.add(row["sample_id"])
            selected_category[row["sample_id"]] = category
            if sum(selected_category.get(item) == category for item in selected) == 4:
                break
    for sample_id in sample_ids:
        if len(selected) >= 20:
            break
        if sample_id not in selected_set:
            selected.append(sample_id)
            selected_set.add(sample_id)
            selected_category[sample_id] = "fixed_fill"

    output_dir = OUTPUT_ROOT / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "selection_category", *categories])
        by_id = {row["sample_id"]: row for row in scores}
        for sample_id in selected:
            writer.writerow([sample_id, selected_category[sample_id], *(by_id[sample_id][key] for key in categories)])
    for sample_id in selected:
        make_panels(sample_id, output_dir)

    for run_dir in RUNS.values():
        destination = run_dir / "visualizations"
        destination.mkdir(parents=True, exist_ok=True)
        for source in output_dir.iterdir():
            if source.is_file():
                shutil.copy2(source, destination / source.name)
    print(json.dumps({"status": "PASS", "selected_count": len(selected), "output": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
