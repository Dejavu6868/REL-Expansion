#!/usr/bin/env python3
"""Check whether the frozen S2D HHA cache equals Depth2HHA on the raw data.

The ERP HHA in ``hha.py`` follows Depth2HHA-python, the generator the CMX
README names. It describes the same modality as the HHA checkpoint only if
the cached training HHA is Depth2HHA output. This script recomputes HHA for a
spread of S2D samples from native Depth16 + pose K and compares it with the
cached 480x480 files. Two resolution orders and two channel orders are
tried, because the cache's generation recipe is not recorded.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_projection as cp  # noqa: E402
import hha  # noqa: E402


def decode(raw):
    valid = (raw != 0) & (raw != 65535)
    return np.where(valid, raw.astype(np.float64) / 512.0, 0.0), valid


def compare(candidate, cached):
    report = {}
    for order_name, order in (("as_stored", [0, 1, 2]), ("reversed", [2, 1, 0])):
        diff = np.abs(candidate[:, :, order].astype(np.int16) - cached.astype(np.int16))
        report[order_name] = [
            {
                "exact_fraction": float((diff[:, :, c] == 0).mean()),
                "median_abs": float(np.median(diff[:, :, c])),
                "p95_abs": float(np.percentile(diff[:, :, c], 95)),
            }
            for c in range(3)
        ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="full_manifest.csv")
    parser.add_argument("--hha-root", required=True, type=Path)
    parser.add_argument("--hha-format", default=".png")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=cp.DEFAULT_SOURCE_ROOT)
    args = parser.parse_args()

    cp.import_frozen_source(args.source_root)
    from rel_plus.camera import load_stanford_s2d_camera_geometry, resize_camera_geometry
    from rel_plus.depth import resize_raw_depth_nearest

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    picks = np.linspace(0, len(rows) - 1, num=min(args.samples, len(rows))).astype(int)

    results = []
    for index in picks:
        row = rows[index]
        cached = cv2.imread(
            str(args.hha_root / (row["sample_id"] + args.hha_format)), cv2.IMREAD_UNCHANGED
        )
        raw = cv2.imread(row["depth_path"], cv2.IMREAD_UNCHANGED)
        if cached is None or raw is None:
            raise FileNotFoundError(row["sample_id"])
        camera = load_stanford_s2d_camera_geometry(row["camera_metadata_path"], raw.shape)
        shape = cached.shape[:2]

        depth, valid = decode(raw)
        native, _ = hha.pinhole_hha(depth, valid, camera.K_json)
        native = cv2.resize(native, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

        small_raw = resize_raw_depth_nearest(raw, shape)
        small_camera = resize_camera_geometry(camera, shape)
        depth, valid = decode(small_raw)
        resized, _ = hha.pinhole_hha(depth, valid, small_camera.K_json)

        results.append(
            {
                "sample_id": row["sample_id"],
                "native_then_resize": compare(native, cached),
                "resize_then_hha": compare(resized, cached),
            }
        )
        print(row["sample_id"], flush=True)

    best = None
    for variant in ("native_then_resize", "resize_then_hha"):
        for order in ("as_stored", "reversed"):
            medians = [
                max(channel["median_abs"] for channel in item[variant][order])
                for item in results
            ]
            score = float(np.median(medians))
            if best is None or score < best["median_of_worst_channel_median"]:
                best = {
                    "variant": variant,
                    "channel_order": order,
                    "median_of_worst_channel_median": score,
                }
    status = "MATCH" if best["median_of_worst_channel_median"] <= 1.0 else "NO_MATCH"
    summary = {"status": status, "best": best, "samples": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("{}: {}".format(status, best))
    return 0 if status == "MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
