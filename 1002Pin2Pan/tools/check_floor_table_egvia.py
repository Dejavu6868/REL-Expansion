#!/usr/bin/env python3
"""How well REL+ 3.0 EGVIA tells table tops from floor, before any retraining.

Frozen v2.1 EGVIA carries no height on horizontal surfaces, so floor and table
tops share one value. 3.0 (``relplus_variant.py``) blends per-image normalised
height into them at half weight. This measures the cue that gives, in both domains:

* Pinhole: a random sample of the training split of the frozen manifest,
  encoded with the frozen generator exactly as the cache tool does, with the
  480 labels the training config reads.
* Panorama: panoramas of the training areas (never area 5), encoded with the
  ERP ``getREL`` at the evaluation size, with their labels.

Floor and table-top pixels are floor- or table-labelled pixels whose surface
faces up: unblended EGVIA angle at most 63 (45 degrees, the v2.1 rule). On
those pixels v2.1 EGVIA is that angle. For each variant the report gives the
EGVIA medians of both surfaces and ``table_above_floor``, the chance that a
random table-top pixel has higher EGVIA than a random floor pixel (0.5: no
separation, 1.0: complete); for pinhole frames with table tops, the per-frame
median gap, split by whether floor is in view (3.0 height is normalised per
image, so a table top that is the lowest surface in view can sit near 0).
"""

import argparse
import csv
import functools
import json
import multiprocessing
import time
from pathlib import Path

import cv2
import numpy as np

import cross_projection as cp
import eval_pano_transfer as ev
import relplus_variant as rv


TRAIN_AREAS = ("area_1", "area_2", "area_3", "area_4", "area_6")
RAW_ANGLE_ALPHA = 180.0  # every pixel counts as horizontal, so EGVIA is the unblended angle
UP_FACING_MAX = 63  # 45 degrees: the v2.1 horizontal rule, upward side
MIN_PIXELS = 200  # a surface counts as in view of a pinhole frame from this many pixels
VARIANTS = ("v2_1", "v3_0")
_JOB = {}  # set before the worker pool forks


def surface_histograms(raw_angle, v3_0, label, floor_id, table_id):
    """256-bin EGVIA histograms of up-facing floor and table pixels, per variant."""
    up = raw_angle <= UP_FACING_MAX
    result = {}
    for surface, class_id in (("floor", floor_id), ("table", table_id)):
        mask = up & (label == class_id)
        for variant, egvia in (("v2_1", raw_angle), ("v3_0", v3_0)):
            result[variant, surface] = np.bincount(egvia[mask], minlength=256)
    return result


def histogram_median(histogram):
    count = histogram.sum()
    return None if count == 0 else int(np.searchsorted(np.cumsum(histogram), (count - 1) / 2, side="right"))


def table_above_floor(floor, table):
    """Chance that a random table-top pixel has higher EGVIA than a random floor pixel (ties count half).

    0.5: EGVIA cannot tell them apart; 1.0: every table top sits above every floor pixel.
    """
    below = np.concatenate([[0], np.cumsum(floor)[:-1]])
    return float((table * (below + 0.5 * floor)).sum() / (table.sum() * floor.sum()))


def summarize(histograms):
    summary = {}
    for variant in VARIANTS:
        floor, table = histograms[variant, "floor"], histograms[variant, "table"]
        summary[variant] = {
            "floor_pixels": int(floor.sum()),
            "table_top_pixels": int(table.sum()),
            "floor_median": histogram_median(floor),
            "table_top_median": histogram_median(table),
            "table_above_floor": table_above_floor(floor, table) if floor.sum() and table.sum() else None,
        }
    return summary


def pinhole_frame(row):
    from rel_plus.profiles import STANFORD_S2D_PROFILE
    from rel_plus.stanford_s2d import load_canonical_frame
    import rel_plus.generator as generator

    raw, camera, _ = load_canonical_frame(
        row["depth_path"], row["camera_metadata_path"], dataset_profile=STANFORD_S2D_PROFILE
    )
    egvia = {}
    for name, alpha in (("raw", RAW_ANGLE_ALPHA), ("v3_0", rv.VARIANT_ALPHA["v3_0"])):
        generator.REL_PLUS_V2_ALPHA = alpha
        egvia[name] = generator.generate_rel_plus_v2_1(raw, camera)[..., 0]
    path = Path(_JOB["gt_root"]) / (row["sample_id"] + _JOB["gt_format"])
    label = _JOB["gt_transform"](_JOB["open_image"](str(path), cv2.IMREAD_GRAYSCALE, dtype=np.uint8))
    if label.shape != raw.shape:
        raise ValueError("label {} does not match the 480 depth".format(path))
    return surface_histograms(egvia["raw"], egvia["v3_0"], label, _JOB["floor_id"], _JOB["table_id"])


def panorama(sample):
    height, width = _JOB["size"]
    raw = cv2.imread(str(sample["depth"]), cv2.IMREAD_UNCHANGED)
    depth = cp.decode_erp_depth(cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST))
    get_rel = _JOB["getREL"]
    raw_angle = get_rel(depth, alpha=RAW_ANGLE_ALPHA)[..., 0]
    v3_0 = get_rel(depth, alpha=rv.VARIANT_ALPHA["v3_0"])[..., 0]
    semantic = cv2.imread(str(sample["semantic"]), cv2.IMREAD_COLOR)
    semantic = cv2.resize(semantic, (width, height), interpolation=cv2.INTER_NEAREST)
    label = ev.decode_semantic(semantic, _JOB["lookup"])
    bottom = slice(int(np.ceil(height * 150 / 180)), height)  # elevation -60..-90, as the band scores
    ids = (_JOB["floor_id"], _JOB["table_id"])
    return (
        surface_histograms(raw_angle, v3_0, label, *ids),
        surface_histograms(raw_angle[bottom], v3_0[bottom], label[bottom], *ids),
    )


def add(total, histograms):
    for key, value in histograms.items():
        total[key] = total.get(key, 0) + value


def percentiles(values):
    if not values:
        return None
    return {name: float(np.percentile(values, q)) for name, q in (("p5", 5), ("p25", 25), ("median", 50), ("p75", 75), ("p95", 95))}


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
        gt_root=config["gt_root_folder"],
        gt_format=config["gt_format"],
        gt_transform=RGBXDataset._gt_transform,
        open_image=RGBXDataset._open_image,
        floor_id=names.index("floor"),
        table_id=names.index("table"),
        getREL=getattr(frozen["getREL"], "func", frozen["getREL"]),
        lookup=ev.label_lookup(args.semantic_labels),
        size=(args.height, args.width),
    )

    started = time.time()
    totals = {"pinhole": {}, "pinhole_floor_in_view": {}, "pinhole_no_floor_in_view": {},
              "panorama": {}, "panorama_bottom_band": {}}
    gaps, no_floor_table_medians = [], []
    with multiprocessing.get_context("fork").Pool(args.workers, cv2.setNumThreads, (1,)) as pool:
        for index, histograms in enumerate(pool.imap_unordered(pinhole_frame, frames), start=1):
            add(totals["pinhole"], histograms)
            floor_px = histograms["v3_0", "floor"].sum()
            if histograms["v3_0", "table"].sum() >= MIN_PIXELS:
                table_median = histogram_median(histograms["v3_0", "table"])
                if floor_px >= MIN_PIXELS:
                    add(totals["pinhole_floor_in_view"], histograms)
                    gaps.append(table_median - histogram_median(histograms["v3_0", "floor"]))
                else:
                    add(totals["pinhole_no_floor_in_view"], histograms)
                    no_floor_table_medians.append(table_median)
            if index % 500 == 0 or index == len(frames):
                print("{}/{} pinhole frames".format(index, len(frames)), flush=True)
        for index, (whole, bottom) in enumerate(pool.imap_unordered(panorama, samples), start=1):
            add(totals["panorama"], whole)
            add(totals["panorama_bottom_band"], bottom)
            if index % 100 == 0 or index == len(samples):
                print("{}/{} panoramas".format(index, len(samples)), flush=True)

    report = {
        "question": "does REL+ 3.0 EGVIA separate up-facing table tops from floor?",
        "up_facing_rule": "unblended EGVIA angle <= {} (45 degrees)".format(UP_FACING_MAX),
        "pinhole_frames": len(frames),
        "pinhole_table_frames_with_floor_in_view": len(gaps),
        "pinhole_table_frames_without_floor_in_view": len(no_floor_table_medians),
        "min_pixels_in_view": MIN_PIXELS,
        "panoramas": len(samples),
        "panorama_areas": list(TRAIN_AREAS),
        "panorama_size": [args.height, args.width],
        "seed": args.seed,
        "seconds": time.time() - started,
        "subsets": {name: summarize(histograms) for name, histograms in totals.items() if histograms},
        "pinhole_v3_0_table_minus_floor_median_gap": percentiles(gaps),
        "pinhole_v3_0_table_top_median_without_floor": percentiles(no_floor_table_medians),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "floor_table_egvia.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "floor_table_egvia_histograms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subset", "variant", "surface"] + ["egvia_{}".format(b) for b in range(256)])
        for subset, histograms in totals.items():
            for (variant, surface), histogram in sorted(histograms.items()):
                writer.writerow([subset, variant, surface] + histogram.tolist())
    for subset, summary in report["subsets"].items():
        print(subset, json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
