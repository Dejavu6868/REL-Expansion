#!/usr/bin/env python3
"""Prepare a three-sample CMX layout from real S3D panoramas and REL files."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


EXPECTED_CLASSES = [
    "<UNK>", "beam", "board", "bookcase", "ceiling", "chair", "clutter",
    "column", "door", "floor", "sofa", "table", "wall", "window",
]


def paired_paths(depth_path):
    text = str(depth_path)
    rgb = text.replace("/pano/depth/", "/pano/rgb/").replace(
        "_depth.png", "_rgb.png"
    )
    semantic = text.replace("/pano/depth/", "/pano/semantic/").replace(
        "_depth.png", "_semantic.png"
    )
    return Path(rgb), Path(semantic)


def label_lookup(path):
    labels = json.loads(path.read_text(encoding="utf-8"))
    classes = []
    class_ids = []
    for label in labels:
        class_name = label.split("_", 1)[0]
        if class_name not in classes:
            classes.append(class_name)
        class_ids.append(classes.index(class_name))
    if classes != EXPECTED_CLASSES:
        raise ValueError("Unexpected Stanford2D3D semantic class order")
    return np.asarray(class_ids, dtype=np.uint8)


def convert_label(source, destination, lookup):
    semantic = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if semantic is None:
        raise FileNotFoundError(source)
    indices = (
        semantic[:, :, 0].astype(np.uint32)
        + semantic[:, :, 1].astype(np.uint32) * 256
        + semantic[:, :, 2].astype(np.uint32) * 65536
    )
    label = np.zeros(indices.shape, dtype=np.uint8)
    valid = indices < len(lookup)
    label[valid] = lookup[indices[valid]]
    if not cv2.imwrite(str(destination), label):
        raise RuntimeError("Failed to write {}".format(destination))
    return label


def link(source, destination):
    destination.symlink_to(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-manifest", required=True, type=Path)
    parser.add_argument("--semantic-labels", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    with args.generation_manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["status"] in ("OK", "SKIPPED")
        ][: args.limit]
    if len(rows) != args.limit:
        raise ValueError("Generation manifest has fewer usable rows than requested")

    rgb_root = args.output_root / "RGB"
    rel_root = args.output_root / "REL"
    label_root = args.output_root / "Label"
    for path in (rgb_root, rel_root, label_root):
        path.mkdir(parents=True, exist_ok=False)

    lookup = label_lookup(args.semantic_labels)
    manifest_rows = []
    for index, row in enumerate(rows, start=1):
        sample_id = "s3d_{:02d}".format(index)
        depth_path = Path(row["depth_path"])
        rel_path = Path(row["rel_path"])
        rgb_path, semantic_path = paired_paths(depth_path)
        link(rgb_path, rgb_root / (sample_id + ".png"))
        link(rel_path, rel_root / (sample_id + ".png"))
        label_path = label_root / (sample_id + ".png")
        label = convert_label(semantic_path, label_path, lookup)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "depth_path": str(depth_path),
                "rgb_path": str(rgb_path),
                "semantic_path": str(semantic_path),
                "rel_path": str(rel_path),
                "label_path": str(label_path),
                "label_values": ",".join(str(int(v)) for v in np.unique(label)),
            }
        )

    names = "\n".join(row["sample_id"] for row in manifest_rows) + "\n"
    (args.output_root / "train.txt").write_text(names, encoding="utf-8")
    (args.output_root / "test.txt").write_text(names, encoding="utf-8")
    with (args.output_root / "sample_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print("Prepared {} real S3D smoke samples".format(len(manifest_rows)))


if __name__ == "__main__":
    main()
