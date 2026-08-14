import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


SAMPLE_PATTERN = re.compile(
    r"^(?P<area>[^/]+)/camera_(?P<camera>[0-9a-f]+)_(?P<room>.+)_frame_(?P<frame>[0-9]+)$"
)


def _read_split(path, split):
    return [(line.strip(), split) for line in path.read_text().splitlines() if line.strip()]


def _parse(sample_id):
    match = SAMPLE_PATTERN.match(sample_id)
    if not match:
        raise ValueError("unexpected sample ID: {}".format(sample_id))
    fields = match.groupdict()
    fields["frame"] = int(fields["frame"])
    return fields


def _spread_indices(length, count):
    if length <= count:
        return list(range(length))
    return sorted({int(round(index * (length - 1) / (count - 1))) for index in range(count)})


def _candidate_records(dataset_root):
    entries = _read_split(dataset_root / "train.txt", "train")
    entries += _read_split(dataset_root / "test.txt", "test")
    grouped = defaultdict(lambda: defaultdict(list))
    split_by_id = {}
    for sample_id, split in entries:
        parsed = _parse(sample_id)
        grouped[parsed["area"]][(parsed["camera"], parsed["room"])].append(
            (parsed["frame"], sample_id)
        )
        split_by_id[sample_id] = split

    records = []
    for area in sorted(grouped):
        room_cameras = sorted(grouped[area])
        for group_index in _spread_indices(len(room_cameras), 4):
            camera, room = room_cameras[group_index]
            frames = sorted(grouped[area][(camera, room)])
            _, sample_id = frames[len(frames) // 2]
            depth_path = dataset_root / "Depth16" / (sample_id + ".png")
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
                raise ValueError("invalid uint16 depth: {}".format(depth_path))
            valid = (depth > 0) & (depth != 65535)
            parsed = _parse(sample_id)
            records.append(
                {
                    "sample_id": sample_id,
                    "split": split_by_id[sample_id],
                    "area": area,
                    "room": room,
                    "camera_uuid": camera,
                    "frame": parsed["frame"],
                    "depth_height": depth.shape[0],
                    "depth_width": depth.shape[1],
                    "valid_pixels": int(valid.sum()),
                    "total_pixels": int(valid.size),
                    "valid_ratio": float(valid.mean()),
                }
            )
    return records


def freeze_samples(dataset_root, output_path):
    candidates = _candidate_records(dataset_root)
    by_area = defaultdict(list)
    for record in candidates:
        by_area[record["area"]].append(record)

    selected = []
    for area in sorted(by_area):
        ordered = sorted(by_area[area], key=lambda item: (item["valid_ratio"], item["sample_id"]))
        low = dict(ordered[0])
        high = dict(ordered[-1])
        low["selection_reason"] = "area_low_valid_ratio_candidate"
        high["selection_reason"] = "area_high_valid_ratio_candidate"
        selected.extend([low, high])

    selected_ids = {record["sample_id"] for record in selected}
    remaining = [record for record in candidates if record["sample_id"] not in selected_ids]
    median = float(np.median([record["valid_ratio"] for record in candidates]))
    for record in sorted(
        remaining, key=lambda item: (abs(item["valid_ratio"] - median), item["sample_id"])
    )[:2]:
        record = dict(record)
        record["selection_reason"] = "global_median_valid_ratio_candidate"
        selected.append(record)

    if not 12 <= len(selected) <= 20:
        raise AssertionError("review sample count must be between 12 and 20")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "area",
        "room",
        "camera_uuid",
        "frame",
        "depth_height",
        "depth_width",
        "valid_pixels",
        "total_pixels",
        "valid_ratio",
        "selection_reason",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return selected, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected, candidates = freeze_samples(args.dataset_root, args.output)
    print("FROZEN_SAMPLE_COUNT={}".format(len(selected)))
    print("CANDIDATE_COUNT={}".format(len(candidates)))
    print("AREAS={}".format(",".join(sorted({item["area"] for item in selected}))))
    print(
        "VALID_RATIO_RANGE={:.6f},{:.6f}".format(
            min(item["valid_ratio"] for item in selected),
            max(item["valid_ratio"] for item in selected),
        )
    )


if __name__ == "__main__":
    main()
