#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


EXPECTED_TRAIN = 52903
EXPECTED_TEST = 17593
EXPECTED_FOLDERS = {
    "RGB": (".png", (480, 480, 3), np.uint8),
    "HHA": (".png", (480, 480, 3), np.uint8),
    "RawDepth": (".png", (480, 480), np.uint8),
    "Label": (".png", (480, 480), np.uint8),
}


def read_list(path):
    entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(entries) != len(set(entries)):
        duplicates = [item for item, count in Counter(entries).items() if count > 1]
        raise ValueError("duplicate entries in {}: {}".format(path, duplicates[:20]))
    return entries


def disk_items(root, folder, suffix):
    return {
        str(path.relative_to(root / folder).with_suffix(""))
        for path in (root / folder).glob("area_*/*" + suffix)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480",
    )
    parser.add_argument(
        "--report", default="/data/zhuzhaoziao/cmx/outputs/preflight/dataset_check.json",
    )
    parser.add_argument("--max-items", type=int, help="Only decode this many items per split")
    parser.add_argument(
        "--folders", nargs="+", choices=tuple(EXPECTED_FOLDERS),
        default=list(EXPECTED_FOLDERS),
    )
    args = parser.parse_args()

    root = Path(args.root)
    train = read_list(root / "train.txt")
    test = read_list(root / "test.txt")
    if len(train) != EXPECTED_TRAIN or len(test) != EXPECTED_TEST:
        raise ValueError("unexpected split counts: train={} test={}".format(len(train), len(test)))
    overlap = set(train) & set(test)
    if overlap:
        raise ValueError("train/test overlap: {}".format(sorted(overlap)[:20]))
    if any(not item.startswith(("area_1/", "area_2/", "area_3/", "area_4/", "area_6/")) for item in train):
        raise ValueError("train.txt contains an item outside Areas 1,2,3,4,6")
    if any(not item.startswith(("area_5a/", "area_5b/")) for item in test):
        raise ValueError("test.txt contains an item outside Area 5")

    listed = set(train) | set(test)
    folder_counts = {}
    for folder in args.folders:
        suffix, _, _ = EXPECTED_FOLDERS[folder]
        files = disk_items(root, folder, suffix)
        missing = listed - files
        extra = files - listed
        if missing or extra:
            raise ValueError(
                "{} mismatch: missing={} extra={}".format(
                    folder, sorted(missing)[:20], sorted(extra)[:20],
                )
            )
        folder_counts[folder] = len(files)

    selected_train = train[:args.max_items] if args.max_items else train
    selected_test = test[:args.max_items] if args.max_items else test
    selected = selected_train + selected_test
    label_pixels = np.zeros(14, dtype=np.int64)
    missing_or_invalid = []
    for item in tqdm(selected, desc="decode and validate"):
        for folder in args.folders:
            suffix, expected_shape, expected_dtype = EXPECTED_FOLDERS[folder]
            path = root / folder / (item + suffix)
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.shape != expected_shape or image.dtype != expected_dtype:
                missing_or_invalid.append({
                    "item": item,
                    "folder": folder,
                    "shape": None if image is None else image.shape,
                    "dtype": None if image is None else str(image.dtype),
                })
                continue
            if folder == "Label":
                values, counts = np.unique(image, return_counts=True)
                if np.any(values > 13):
                    missing_or_invalid.append({"item": item, "invalid_label_ids": values.tolist()})
                else:
                    label_pixels[values] += counts
    if missing_or_invalid:
        raise ValueError("invalid files: {}".format(missing_or_invalid[:20]))

    report = {
        "dataset_root": str(root.resolve()),
        "train_count": len(train),
        "test_count": len(test),
        "train_areas": ["area_1", "area_2", "area_3", "area_4", "area_6"],
        "test_areas": ["area_5a", "area_5b"],
        "checked_folders": args.folders,
        "folder_counts": folder_counts,
        "decoded_items": len(selected),
        "partial_decode": args.max_items is not None,
        "stored_label_pixel_counts": {str(i): int(count) for i, count in enumerate(label_pixels)},
        "stored_label_ids": "0 is ignore; 1..13 are semantic classes",
        "loader_label_ids": "stored labels are decremented; 0 becomes ignore_index 255, 1..13 become 0..12",
        "status": "PASS",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
