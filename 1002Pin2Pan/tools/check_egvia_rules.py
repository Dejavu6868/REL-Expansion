#!/usr/bin/env python3
"""Compare EGVIA height rules per class on cached training data, before any retraining.

EGVIA puts the surface angle A (0 facing up, 127.5 vertical, 255 facing down)
and the height H into one byte. The rules compared, all from the same A and H:

* ``v2_1``: REL's public code and REL+ v2.1: height blended only where the
  surface is more than alpha = 45 degrees from horizontal, a jump at alpha.
* ``paper_eq4``: REL paper Eq. 4: height blended only into near-horizontal
  surfaces, the same jump on the other side.
* ``v3_0``: REL+ 3.0: 0.5 A + 0.5 H everywhere. No jump, but up-facing
  surfaces span 0..127 and walls 64..191, so a high table top and a low wall
  can share a value.
* ``banded``: 85 o(A) + H / 3, where o(A) steps smoothly from 0 (up-facing) to
  1 (vertical) to 2 (down-facing) over +-15 degrees around 45 and 135. No jump,
  and flat up-facing, vertical and down-facing surfaces get the disjoint bands
  0..85, 85..170 and 170..255, with per-image height inside each band.
* ``banded_metric``: as ``banded``, but H is metric height relative to the
  camera, clipped to ``--metric-range-cm``, instead of the per-image
  P1..P99 normalisation. The same table top then gets the same value in every
  frame, and in pinhole frames and panoramas alike.

Every rule is checked byte for byte against the frozen encoders' v2_1 output
on every image. For each rule it reports, per pair of classes, the overlap of
their EGVIA histograms (sum of the minimum of the two normalised histograms:
1 identical, 0 disjoint), and per class the share of pixels within 5 degrees of
the 45 and 135 degree switch (where v2_1 and paper_eq4 jump). It also reports
per-class percentiles of camera-relative height, to set the metric range from
the training split. EGVIA overlap is a proxy: RGB, LOA and ReD can still
separate classes that EGVIA confuses.
"""

import argparse
import csv
import json
import multiprocessing
import time
from pathlib import Path

import cv2
import numpy as np

import cross_projection as cp
import eval_pano_transfer as ev


TRAIN_AREAS = ("area_1", "area_2", "area_3", "area_4", "area_6")
RULES = ("v2_1", "paper_eq4", "v3_0", "banded", "banded_metric")
ALPHA_DEG = 45.0
LAMBDA = 0.5
BAND_HALF_WIDTH_DEG = 15.0
SWITCH_ZONE_DEG = 5.0
HEIGHT_BINS_CM = np.arange(-400, 401, 5)
KEY_PAIRS = (
    ("floor", "table"), ("table", "wall"), ("chair", "table"), ("sofa", "table"),
    ("wall", "door"), ("wall", "window"), ("wall", "board"), ("wall", "bookcase"),
    ("ceiling", "wall"), ("ceiling", "beam"), ("floor", "wall"),
)
_JOB = {}  # set before the worker pool forks


def smooth_step(x):
    t = np.clip((x + 1.0) / 2.0, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def orientation_level(angle):
    """0 up-facing, 1 vertical, 2 down-facing, smooth over +-15 degrees around 45 and 135."""
    degrees = angle * 180.0 / 255.0
    return (
        smooth_step((degrees - ALPHA_DEG) / BAND_HALF_WIDTH_DEG)
        + smooth_step((degrees - (180.0 - ALPHA_DEG)) / BAND_HALF_WIDTH_DEG)
    )


def metric_height(height_cm, metric_range_cm):
    low, high = metric_range_cm
    return np.clip((height_cm - low) * 255.0 / (high - low), 0.0, 255.0)


def egvia_rules(angle, height_normalized, height_cm, metric_range_cm):
    """Float EGVIA per rule from the unblended angle (0..255) and the normalised height (0..255)."""
    threshold = ALPHA_DEG * 255.0 / 180.0
    horizontal = (angle <= threshold) | (angle >= 255.0 - threshold)
    blended = LAMBDA * angle + (1.0 - LAMBDA) * height_normalized
    level = orientation_level(angle)
    return {
        "v2_1": np.where(horizontal, angle, blended),
        "paper_eq4": np.where(horizontal, blended, angle),
        "v3_0": blended,
        "banded": 85.0 * level + height_normalized / 3.0,
        "banded_metric": 85.0 * level + metric_height(height_cm, metric_range_cm) / 3.0,
    }


def to_byte(egvia):
    """The frozen encoders' cast: NaN to 255, clip, truncate."""
    return np.clip(np.nan_to_num(egvia, nan=255.0), 0, 255).astype(np.uint8)


def image_statistics(angle, height_normalized, height_cm, valid, label, frozen_egvia):
    """Per-class EGVIA histograms per rule, switch-zone counts and height histograms of one image."""
    rules = egvia_rules(angle, height_normalized, height_cm, _JOB["metric_range_cm"])
    if not np.array_equal(to_byte(rules["v2_1"])[valid], frozen_egvia[valid]):
        raise AssertionError("v2_1 rule does not reproduce the frozen EGVIA bytes")
    keep = valid & (label < _JOB["classes"])
    classes = label[keep].astype(np.int64)
    n = _JOB["classes"]
    histograms = {
        name: np.bincount(classes * 256 + to_byte(egvia)[keep], minlength=n * 256).reshape(n, 256)
        for name, egvia in rules.items()
    }
    degrees = angle[keep] * 180.0 / 255.0
    near_switch = (np.abs(degrees - ALPHA_DEG) <= SWITCH_ZONE_DEG) | (
        np.abs(degrees - (180.0 - ALPHA_DEG)) <= SWITCH_ZONE_DEG
    )
    height_bin = np.clip(np.digitize(height_cm[keep], HEIGHT_BINS_CM) - 1, 0, len(HEIGHT_BINS_CM) - 2)
    height = np.bincount(classes * (len(HEIGHT_BINS_CM) - 1) + height_bin, minlength=n * (len(HEIGHT_BINS_CM) - 1))
    return {
        "egvia": histograms,
        "pixels": np.bincount(classes, minlength=n),
        "near_switch": np.bincount(classes[near_switch], minlength=n),
        "height_cm": height.reshape(n, -1),
    }


def pinhole_frame(row):
    from rel_plus.generator import generate_rel_plus_v2
    from rel_plus.profiles import STANFORD_S2D_PROFILE
    from rel_plus.stanford_s2d import load_canonical_frame

    raw, camera, _ = load_canonical_frame(
        row["depth_path"], row["camera_metadata_path"], dataset_profile=STANFORD_S2D_PROFILE
    )
    _, debug = generate_rel_plus_v2(raw, camera, return_debug=True)
    path = Path(_JOB["gt_root"]) / (row["sample_id"] + _JOB["gt_format"])
    label = _JOB["gt_transform"](_JOB["open_image"](str(path), cv2.IMREAD_GRAYSCALE, dtype=np.uint8))
    if label.shape != raw.shape:
        raise ValueError("label {} does not match the 480 depth".format(path))
    return image_statistics(
        debug["egvia_angle_before_blend"], debug["height_normalized"], debug["height_raw_cm"],
        debug["encoding_valid_mask"], label, debug["egvia_encoded"],
    )


def erp_angle_and_height(depth):
    """The unblended angle and the height inside ERP ``getREL``, with its statements."""
    pc_rot, normals, _ = _JOB["processDepthImage_ERP"](depth * 100, depth == 0)
    height_cm = pc_rot[:, :, 2]
    height = height_cm
    low, high = np.percentile(height, 1), np.percentile(height, 99)
    if high > low:
        height = (height - low) * 255.0 / (high - low)
    height = np.clip(height, 0, 255).astype(np.float32)
    normal_z = np.clip(-normals[:, :, 2], -1.0, 1.0)
    angle = np.clip((np.arccos(normal_z, dtype=np.float32) / np.pi) * 255.0, 0, 255).astype(np.float32)
    return angle, height, height_cm


def panorama(sample):
    height, width = _JOB["size"]
    raw = cv2.imread(str(sample["depth"]), cv2.IMREAD_UNCHANGED)
    depth = cp.decode_erp_depth(cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST))
    angle, height_normalized, height_cm = erp_angle_and_height(depth)
    frozen_egvia = _JOB["getREL"](depth)[..., 0]
    semantic = cv2.imread(str(sample["semantic"]), cv2.IMREAD_COLOR)
    semantic = cv2.resize(semantic, (width, height), interpolation=cv2.INTER_NEAREST)
    label = ev.decode_semantic(semantic, _JOB["lookup"])
    return image_statistics(angle, height_normalized, height_cm, depth != 0, label, frozen_egvia)


def add(total, statistics):
    for key, value in statistics.items():
        if isinstance(value, dict):
            add(total.setdefault(key, {}), value)
        else:
            total[key] = total.get(key, 0) + value


def overlap_matrix(histograms):
    """Sum of the minimum of two classes' normalised EGVIA histograms, for every pair."""
    counts = histograms.sum(axis=1, keepdims=True)
    normalised = np.divide(histograms, counts, out=np.zeros(histograms.shape), where=counts > 0)
    return np.minimum(normalised[:, None, :], normalised[None, :, :]).sum(axis=2)


def histogram_percentiles(histogram, edges):
    count = histogram.sum()
    if count == 0:
        return None
    cumulative = np.cumsum(histogram)
    return {
        name: float(edges[min(np.searchsorted(cumulative, q / 100.0 * count), len(edges) - 2)])
        for name, q in (("p5", 5), ("median", 50), ("p95", 95))
    }


def summarize(total, names):
    present = total["pixels"] > 0
    overlaps = {rule: overlap_matrix(total["egvia"][rule]) for rule in RULES}
    upper = np.triu(np.ones((len(names), len(names)), dtype=bool), 1) & present[:, None] & present[None, :]
    return {
        "pixels": dict(zip(names, total["pixels"].tolist())),
        "share_within_{:g}_deg_of_switch".format(SWITCH_ZONE_DEG): {
            name: float(near / pixels) if pixels else None
            for name, near, pixels in zip(names, total["near_switch"], total["pixels"])
        },
        "all_pixels_within_switch_zone": float(total["near_switch"].sum() / total["pixels"].sum()),
        "egvia_overlap_key_pairs": {
            "{}-{}".format(a, b): {rule: float(overlaps[rule][names.index(a), names.index(b)]) for rule in RULES}
            for a, b in KEY_PAIRS
        },
        "egvia_overlap_mean_over_class_pairs": {rule: float(overlaps[rule][upper].mean()) for rule in RULES},
        "camera_relative_height_cm": {
            name: histogram_percentiles(histogram, HEIGHT_BINS_CM)
            for name, histogram in zip(names, total["height_cm"])
        },
    }, overlaps


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, type=Path, help="full_manifest.csv")
    parser.add_argument("--config", required=True, type=Path, help="relplus.json (480 label root)")
    parser.add_argument("--stanford-root", required=True, type=Path)
    parser.add_argument("--semantic-labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=ev.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--frames", type=int, default=3000, help="random pinhole training frames")
    parser.add_argument("--panoramas", type=int, default=None, help="default: every training-area panorama")
    parser.add_argument(
        "--metric-range-cm", type=float, nargs=2, default=(-200.0, 200.0),
        help="camera-relative height mapped to 0..255 by banded_metric",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))

    frozen = cp.import_frozen_source(args.source_root)
    from dataloader.RGBXDataset import RGBXDataset

    config = json.loads(args.config.read_text(encoding="utf-8"))
    names = list(config["class_names"])
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    rng = np.random.default_rng(args.seed)
    frames = [rows[i] for i in sorted(rng.choice(len(rows), min(args.frames, len(rows)), replace=False))]
    samples = ev.list_panoramas(args.stanford_root, list(TRAIN_AREAS))
    if args.panoramas is not None:
        samples = [samples[i] for i in sorted(rng.choice(len(samples), min(args.panoramas, len(samples)), replace=False))]
    _JOB.update(
        classes=len(names),
        metric_range_cm=tuple(args.metric_range_cm),
        gt_root=config["gt_root_folder"],
        gt_format=config["gt_format"],
        gt_transform=RGBXDataset._gt_transform,
        open_image=RGBXDataset._open_image,
        getREL=getattr(frozen["getREL"], "func", frozen["getREL"]),
        processDepthImage_ERP=frozen["processDepthImage_ERP"],
        lookup=ev.label_lookup(args.semantic_labels),
        size=(args.height, args.width),
    )

    started = time.time()
    totals = {"pinhole": {}, "panorama": {}}
    with multiprocessing.get_context("fork").Pool(args.workers, cv2.setNumThreads, (1,)) as pool:
        for index, statistics in enumerate(pool.imap_unordered(pinhole_frame, frames), start=1):
            add(totals["pinhole"], statistics)
            if index % 500 == 0 or index == len(frames):
                print("{}/{} pinhole frames".format(index, len(frames)), flush=True)
        for index, statistics in enumerate(pool.imap_unordered(panorama, samples), start=1):
            add(totals["panorama"], statistics)
            if index % 100 == 0 or index == len(samples):
                print("{}/{} panoramas".format(index, len(samples)), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "rules": list(RULES),
        "alpha_deg": ALPHA_DEG,
        "band_half_width_deg": BAND_HALF_WIDTH_DEG,
        "metric_range_cm": list(args.metric_range_cm),
        "pinhole_frames": len(frames),
        "panoramas": len(samples),
        "panorama_areas": list(TRAIN_AREAS),
        "panorama_size": [args.height, args.width],
        "seed": args.seed,
        "seconds": time.time() - started,
        "subsets": {},
    }
    with (args.output / "egvia_overlap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subset", "rule", "class"] + names)
        for subset, total in totals.items():
            report["subsets"][subset], overlaps = summarize(total, names)
            for rule in RULES:
                for name, row in zip(names, overlaps[rule]):
                    writer.writerow([subset, rule, name] + ["{:.4f}".format(v) for v in row])
    with (args.output / "egvia_histograms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subset", "rule", "class"] + ["egvia_{}".format(b) for b in range(256)])
        for subset, total in totals.items():
            for rule in RULES:
                for name, row in zip(names, total["egvia"][rule]):
                    writer.writerow([subset, rule, name] + row.tolist())
    (args.output / "egvia_rules.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for subset, summary in report["subsets"].items():
        print(subset, "switch zone", summary["all_pixels_within_switch_zone"])
        print(subset, "mean overlap", json.dumps(summary["egvia_overlap_mean_over_class_pairs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
