#!/usr/bin/env python3
"""Audit the complete S3D Fold 1 RGB/label/depth/HHA/REL data chain."""

import argparse
import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


CLASSES = [
    "beam", "board", "bookcase", "ceiling", "chair", "clutter",
    "column", "door", "floor", "sofa", "table", "wall", "window",
]
EXPECTED_SHAPE = (2048, 4096)


def paths_for(root, sample_id):
    return {
        "rgb": root / "image" / (sample_id + "_rgb.png"),
        "label": root / "label" / (sample_id + "_semantic.png"),
        "depth": root / "depth" / (sample_id + "_depth.png"),
        "hha": root / "hha" / (sample_id + "_hha.png"),
        "rel": root / "rel" / (sample_id + "_rel.png"),
    }


def inspect_sample(task):
    root_text, sample_id = task
    root = Path(root_text)
    paths = paths_for(root, sample_id)
    record = {"sample_id": sample_id, "errors": []}
    label_hist = np.zeros(14, dtype=np.int64)
    depth_invalid = 0

    for name, path in paths.items():
        record[name + "_path"] = str(path)
        record[name + "_exists"] = path.is_file()
        if not path.is_file():
            record["errors"].append("missing " + name)
            continue
        flag = cv2.IMREAD_UNCHANGED if name in ("depth", "rel") else cv2.IMREAD_COLOR
        if name == "label":
            flag = cv2.IMREAD_GRAYSCALE
        array = cv2.imread(str(path), flag)
        if array is None:
            record["errors"].append("unreadable " + name)
            continue
        record[name + "_shape"] = "x".join(str(value) for value in array.shape)
        record[name + "_dtype"] = str(array.dtype)
        if array.shape[:2] != EXPECTED_SHAPE:
            record["errors"].append("shape " + name)
        if name == "depth":
            if array.dtype != np.uint16 or array.ndim != 2:
                record["errors"].append("depth protocol")
            else:
                depth_invalid = int((array == 65535).sum())
                record["depth_min"] = int(array.min())
                record["depth_max"] = int(array.max())
        elif name == "label":
            values = np.unique(array)
            if not set(int(value) for value in values).issubset(set(range(14))):
                record["errors"].append("label IDs")
            label_hist = np.bincount(array.reshape(-1), minlength=14)[:14].astype(np.int64)
        elif name in ("rgb", "hha", "rel") and (array.ndim != 3 or array.shape[2] != 3):
            record["errors"].append("channels " + name)

    record["depth_invalid_65535"] = depth_invalid
    record["status"] = "PASS" if not record["errors"] else "FAIL"
    record["errors"] = "; ".join(record["errors"])
    record["label_hist"] = label_hist.tolist()
    return record


def channel_summary(arrays):
    result = []
    for channel in range(3):
        flat = np.concatenate([array[:, :, channel].reshape(-1) for array in arrays])
        result.append(
            {
                "min": int(flat.min()),
                "max": int(flat.max()),
                "mean": float(flat.mean()),
                "std": float(flat.std()),
                "constant_images": int(sum(np.ptp(array[:, :, channel]) == 0 for array in arrays)),
                "non_finite": 0,
            }
        )
    return result


def color_label(label):
    palette = np.array(
        [
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128],
        ], dtype=np.uint8
    )
    return palette[label]


def thumbnail(array):
    return cv2.resize(array, (512, 256), interpolation=cv2.INTER_NEAREST)


def write_visual_checks(root, sample_ids, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    hha_arrays = []
    rel_arrays = []
    records = []
    for index, sample_id in enumerate(sample_ids):
        paths = paths_for(root, sample_id)
        rgb_bgr = cv2.imread(str(paths["rgb"]), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
        label = cv2.imread(str(paths["label"]), cv2.IMREAD_GRAYSCALE)
        hha_bgr = cv2.imread(str(paths["hha"]), cv2.IMREAD_COLOR)
        rel = cv2.imread(str(paths["rel"]), cv2.IMREAD_UNCHANGED)
        hha_network = cv2.cvtColor(hha_bgr, cv2.COLOR_BGR2RGB)
        hha_arrays.append(hha_network)
        rel_arrays.append(rel)
        depth8 = (depth >> 8).astype(np.uint8)
        depth_panel = cv2.cvtColor(depth8, cv2.COLOR_GRAY2BGR)
        label_panel = color_label(label)

        hha_panels = [rgb_bgr, depth_panel]
        for channel in range(3):
            hha_panels.append(cv2.cvtColor(hha_network[:, :, channel], cv2.COLOR_GRAY2BGR))
        hha_panels.extend([hha_bgr, label_panel])
        rel_panels = [rgb_bgr, depth_panel]
        for channel in range(3):
            rel_panels.append(cv2.cvtColor(rel[:, :, channel], cv2.COLOR_GRAY2BGR))
        rel_panels.extend([rel, label_panel])
        cv2.imwrite(str(output_dir / "hha_{:02d}.png".format(index)), np.hstack([thumbnail(panel) for panel in hha_panels]))
        cv2.imwrite(str(output_dir / "rel_{:02d}.png".format(index)), np.hstack([thumbnail(panel) for panel in rel_panels]))
        records.append({"index": index, "sample_id": sample_id})
    return records, channel_summary(hha_arrays), channel_summary(rel_arrays)


def write_manifest(path, records):
    fields = [
        "sample_id", "status", "errors",
        "rgb_path", "label_path", "depth_path", "hha_path", "rel_path",
        "rgb_shape", "label_shape", "depth_shape", "hha_shape", "rel_shape",
        "rgb_dtype", "label_dtype", "depth_dtype", "hha_dtype", "rel_dtype",
        "depth_min", "depth_max", "depth_invalid_65535",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    train = (args.dataset_root / "fold1_train.txt").read_text(encoding="utf-8").splitlines()
    test = (args.dataset_root / "fold1_test.txt").read_text(encoding="utf-8").splitlines()
    all_ids = train + test
    duplicates_train = sorted({sample for sample in train if train.count(sample) > 1})
    duplicates_test = sorted({sample for sample in test if test.count(sample) > 1})
    overlap = sorted(set(train) & set(test))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(inspect_sample, [(str(args.dataset_root), sample) for sample in all_ids]))
    by_id = {record["sample_id"]: record for record in records}
    train_records = [by_id[sample] for sample in train]
    test_records = [by_id[sample] for sample in test]
    train_hist = np.sum([record["label_hist"] for record in train_records], axis=0, dtype=np.int64)
    test_hist = np.sum([record["label_hist"] for record in test_records], axis=0, dtype=np.int64)
    failures = [record for record in records if record["status"] != "PASS"]
    modality_counts = {
        name: sum(bool(record.get(name + "_exists")) for record in records)
        for name in ("rgb", "label", "depth", "hha", "rel")
    }
    depth_invalid = sum(int(record.get("depth_invalid_65535", 0)) for record in records)
    depth_min = min(int(record["depth_min"]) for record in records if "depth_min" in record)
    depth_max = max(int(record["depth_max"]) for record in records if "depth_max" in record)

    rng = random.Random(12345)
    sampled_ids = sorted(rng.sample(all_ids, 20))
    visual_records, hha_stats, rel_stats = write_visual_checks(
        args.dataset_root, sampled_ids, args.audit_dir / "modality_checks"
    )

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.audit_dir / "fold1_train_manifest.csv", train_records)
    write_manifest(args.audit_dir / "fold1_test_manifest.csv", test_records)
    status = "PASS" if not failures and not overlap and not duplicates_train and not duplicates_test else "FAIL"
    payload = {
        "status": status,
        "dataset_root": str(args.dataset_root),
        "total": len(set(all_ids)),
        "train": len(train),
        "test": len(test),
        "train_test_overlap": overlap,
        "train_duplicates": duplicates_train,
        "test_duplicates": duplicates_test,
        "all_modalities_readable_and_aligned": len(records) - len(failures),
        "modality_file_counts": modality_counts,
        "failures": failures,
        "class_names": CLASSES,
        "train_class_pixels": train_hist[1:].tolist(),
        "train_ignore_pixels": int(train_hist[0]),
        "test_class_pixels": test_hist[1:].tolist(),
        "test_ignore_pixels": int(test_hist[0]),
        "raw_label_protocol": "0=ignore, 1..13=classes",
        "training_label_protocol": "255=ignore, 0..12=classes after uint8 subtraction",
        "raw_depth": {
            "dtype": "uint16",
            "min": depth_min,
            "max": depth_max,
            "invalid_value": 65535,
            "invalid_pixel_count": depth_invalid,
            "network_encoding": "explicit uint16 high byte, copied to three channels",
            "per_image_minmax": False,
        },
        "sampled_visual_checks": visual_records,
        "hha_network_channel_stats": hha_stats,
        "rel_network_channel_stats": rel_stats,
    }
    (args.audit_dir / "s3d_data_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# S3D data audit", "", "Status: **{}**".format(status), "",
        "- Source chain: Stanford2D3D `no_xyz` archives only",
        "- ERP panoramas: {}".format(payload["total"]),
        "- Fold 1 train/test: {}/{}".format(len(train), len(test)),
        "- Complete readable RGB/Label/Depth/HHA/REL intersection: {}".format(payload["all_modalities_readable_and_aligned"]),
        "- Per-modality file counts: {}".format(modality_counts),
        "- Train/test overlap: {}".format(len(overlap)),
        "- Shape: 2048×4096",
        "- Raw labels: 0 ignore, 1–13 classes; loader converts to 255 ignore, 0–12 classes",
        "", "## Pixel counts", "",
        "| ID | Class | Train | Test |", "|---:|---|---:|---:|",
    ]
    for index, class_name in enumerate(CLASSES):
        lines.append("| {} | {} | {} | {} |".format(index, class_name, train_hist[index + 1], test_hist[index + 1]))
    lines.append("| 255 | ignore | {} | {} |".format(train_hist[0], test_hist[0]))
    lines.extend(["", "Twenty fixed seed-12345 modality panels are stored in `modality_checks/`."])
    (args.audit_dir / "S3D_DATA_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(1 if status != "PASS" else 0)


if __name__ == "__main__":
    main()
