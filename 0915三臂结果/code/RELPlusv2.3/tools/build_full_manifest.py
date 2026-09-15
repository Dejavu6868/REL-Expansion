#!/usr/bin/env python3
"""Build the exact 70,496-row Stanford2D3D S2D manifest without generating cache."""

import argparse
import csv
import json
import re
from pathlib import Path


PROTOCOL_ID = "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"
SAMPLE_PATTERN = re.compile(r"^camera_([^_]+)_(.+)_frame_([0-9]+)$")


def _read_split(path, split):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        sample_id = line.strip()
        if sample_id:
            rows.append((sample_id, split))
    return rows


def build_rows(dataset_root):
    root = Path(dataset_root).resolve()
    entries = _read_split(root / "train.txt", "train")
    entries.extend(_read_split(root / "test.txt", "test"))
    sample_ids = [sample_id for sample_id, _ in entries]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("train/test manifests contain duplicate sample IDs")
    rows = []
    for sample_id, split in entries:
        area, stem = sample_id.split("/", 1)
        match = SAMPLE_PATTERN.match(stem)
        if match is None:
            raise ValueError("unexpected Stanford sample ID: {}".format(sample_id))
        camera, room, frame = match.groups()
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "area": area,
                "area_group": "area_5" if area in ("area_5a", "area_5b") else area,
                "room": room,
                "camera": camera,
                "frame": int(frame),
                "rgb_path": str(root / "RGB" / (sample_id + ".png")),
                "label_path": str(root / "Label" / (sample_id + ".png")),
                "depth_path": str(root / "Depth16" / (sample_id + ".png")),
                "camera_metadata_path": str(root / "Pose" / (sample_id + ".json")),
                "dataset_profile": "stanford2d3d_s2d",
                "protocol_id": PROTOCOL_ID,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = build_rows(args.dataset_root)
    if len(rows) != 70496:
        raise ValueError("formal manifest must contain 70496 rows, got {}".format(len(rows)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "sample_count": len(rows),
        "train_count": sum(row["split"] == "train" for row in rows),
        "test_count": sum(row["split"] == "test" for row in rows),
        "area_counts": {
            area: sum(row["area"] == area for row in rows)
            for area in sorted({row["area"] for row in rows})
        },
        "output": str(output.resolve()),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
