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

MATCH_RULE = "all_sampled_bytes_equal"
VARIANTS = ("native_then_resize", "resize_then_hha")
CHANNEL_ORDERS = {"as_stored": [0, 1, 2], "reversed": [2, 1, 0]}


def decode(raw):
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError("raw depth must be a 2D uint16 array")
    valid = (raw != 0) & (raw != 65535)
    return np.where(valid, raw.astype(np.float64) / 512.0, 0.0), valid


def compare(candidate, cached):
    if (
        candidate.dtype != np.uint8 or cached.dtype != np.uint8
        or candidate.shape != cached.shape or candidate.ndim != 3
        or candidate.shape[2] != 3 or candidate.size == 0
    ):
        raise ValueError("HHA comparison needs matching nonempty uint8 HxWx3 arrays")
    report = {}
    for order_name, order in CHANNEL_ORDERS.items():
        diff = np.abs(candidate[:, :, order].astype(np.int16) - cached.astype(np.int16))
        report[order_name] = [
            {
                "exact_fraction": float((diff[:, :, c] == 0).mean()),
                "median_abs": float(np.median(diff[:, :, c])),
                "p95_abs": float(np.percentile(diff[:, :, c], 95)),
                "max_abs": int(diff[:, :, c].max()),
            }
            for c in range(3)
        ]
    return report


def summarize_results(results):
    """Unlock a recipe only when every byte of every sampled cache matches."""
    if not results:
        raise ValueError("no HHA samples to compare")
    candidates = []
    for variant in VARIANTS:
        for order in CHANNEL_ORDERS:
            channels = [c for item in results for c in item[variant][order]]
            if len(channels) != 3 * len(results):
                raise ValueError("each HHA sample must report all three channels")
            candidates.append({
                "variant": variant,
                "channel_order": order,
                "min_exact_fraction": min(c["exact_fraction"] for c in channels),
                "max_abs": max(c["max_abs"] for c in channels),
                "median_of_worst_channel_median": float(np.median([
                    max(c["median_abs"] for c in item[variant][order])
                    for item in results
                ])),
            })
    best = min(candidates, key=lambda c: (c["max_abs"], -c["min_exact_fraction"]))
    exact = best["max_abs"] == 0 and best["min_exact_fraction"] == 1.0
    return {
        "status": "MATCH" if exact else "NO_MATCH",
        "match_rule": MATCH_RULE,
        "best": best,
        "samples": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="full_manifest.csv")
    parser.add_argument("--hha-root", required=True, type=Path)
    parser.add_argument("--hha-format", default=".png")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=cp.DEFAULT_SOURCE_ROOT)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")

    cp.import_frozen_source(args.source_root)
    from rel_plus.camera import load_stanford_s2d_camera_geometry, resize_camera_geometry
    from rel_plus.depth import resize_raw_depth_nearest

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("no HHA samples in manifest")
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
        if cached.dtype != np.uint8 or cached.ndim != 3 or cached.shape[2] != 3:
            raise ValueError("HHA cache must be uint8 HxWx3: " + row["sample_id"])
        depth, valid = decode(raw)
        camera = load_stanford_s2d_camera_geometry(row["camera_metadata_path"], raw.shape)
        shape = cached.shape[:2]

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

    summary = summarize_results(results)
    summary.update(manifest=str(args.manifest.resolve()), hha_root=str(args.hha_root.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("{}: {}".format(summary["status"], summary["best"]))
    return 0 if summary["status"] == "MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
