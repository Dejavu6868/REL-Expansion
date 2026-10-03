#!/usr/bin/env python3
"""Check whether the frozen S2D HHA cache equals Depth2HHA on the raw data.

The ERP HHA in ``hha.py`` follows Depth2HHA-python, the generator the CMX
README names. It describes the same modality as the HHA checkpoint only if
the cached training HHA is Depth2HHA output. This script recomputes HHA for a
spread of S2D samples from native Depth16 + pose K and compares it with the
cached 480x480 files. Two resolution orders, several resize kernels and two
channel orders are tried, because the cache's generation recipe is not
recorded.
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
NEAR_MATCH_RULE = (
    "per sample: every channel p95 <= 1; at most one channel with max diff > 2, "
    "and that channel exact on >= 90% of pixels"
)
NATIVE_RESIZES = {
    "native_then_resize": cv2.INTER_NEAREST,
    "native_then_resize_nearest_center": None,
    "native_then_resize_linear": cv2.INTER_LINEAR,
    "native_then_resize_area": cv2.INTER_AREA,
    "native_then_resize_cubic": cv2.INTER_CUBIC,
}
VARIANTS = tuple(NATIVE_RESIZES) + ("resize_then_hha",)
CHANNEL_ORDERS = {"as_stored": [0, 1, 2], "reversed": [2, 1, 0]}


def decode(raw):
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError("raw depth must be a 2D uint16 array")
    valid = (raw != 0) & (raw != 65535)
    return np.where(valid, raw.astype(np.float64) / 512.0, 0.0), valid


def resize_native_hha(image, shape, variant):
    """Resize native-resolution HHA to ``shape`` with the kernel ``variant`` names."""
    height, width = shape
    interpolation = NATIVE_RESIZES[variant]
    if interpolation is not None:
        return cv2.resize(image, (width, height), interpolation=interpolation)
    # Pixel-centre nearest neighbour (PIL / skimage convention), unlike OpenCV's.
    rows = np.minimum(((np.arange(height) + 0.5) * image.shape[0] / height).astype(int), image.shape[0] - 1)
    columns = np.minimum(((np.arange(width) + 0.5) * image.shape[1] / width).astype(int), image.shape[1] - 1)
    return image[rows][:, columns]


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


def near_match(channels_per_sample):
    """Rounding-level agreement: what remains after the right kernel is found."""
    for channels in channels_per_sample:
        if any(c["p95_abs"] > 1 for c in channels):
            return False
        loose = [c for c in channels if c["max_abs"] > 2]
        if len(loose) > 1 or any(c["exact_fraction"] < 0.9 for c in loose):
            return False
    return True


def summarize_results(results):
    """MATCH only when every sampled byte matches; NEAR_MATCH only by rounding."""
    if not results:
        raise ValueError("no HHA samples to compare")
    candidates = []
    for variant in VARIANTS:
        for order in CHANNEL_ORDERS:
            if any(variant not in item for item in results):
                continue
            channels = [c for item in results for c in item[variant][order]]
            if len(channels) != 3 * len(results):
                raise ValueError("each HHA sample must report all three channels")
            candidates.append({
                "variant": variant,
                "channel_order": order,
                "near": near_match([item[variant][order] for item in results]),
                "min_exact_fraction": min(c["exact_fraction"] for c in channels),
                "max_abs": max(c["max_abs"] for c in channels),
                "median_of_worst_channel_median": float(np.median([
                    max(c["median_abs"] for c in item[variant][order])
                    for item in results
                ])),
            })
    best = min(
        candidates, key=lambda c: (not c["near"], -c["min_exact_fraction"], c["max_abs"])
    )
    exact = best["max_abs"] == 0 and best["min_exact_fraction"] == 1.0
    near = best.pop("near")
    for candidate in candidates:
        candidate.pop("near", None)
    return {
        "status": "MATCH" if exact else "NEAR_MATCH" if near else "NO_MATCH",
        "match_rule": MATCH_RULE,
        "near_match_rule": NEAR_MATCH_RULE,
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

        small_raw = resize_raw_depth_nearest(raw, shape)
        small_camera = resize_camera_geometry(camera, shape)
        depth, valid = decode(small_raw)
        resized, _ = hha.pinhole_hha(depth, valid, small_camera.K_json)

        result = {"sample_id": row["sample_id"]}
        for variant in NATIVE_RESIZES:
            result[variant] = compare(resize_native_hha(native, shape, variant), cached)
        result["resize_then_hha"] = compare(resized, cached)
        results.append(result)
        print(row["sample_id"], flush=True)

    summary = summarize_results(results)
    summary.update(manifest=str(args.manifest.resolve()), hha_root=str(args.hha_root.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("{}: {}".format(summary["status"], summary["best"]))
    return 0 if summary["status"] in ("MATCH", "NEAR_MATCH") else 1


if __name__ == "__main__":
    raise SystemExit(main())
