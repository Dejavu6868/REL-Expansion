#!/usr/bin/env python3
"""How much of the panorama score sits on black RGB, CPU only.

Trans4PASS (``_color2id`` in stanford2d3d_pan.py) sets every label whose native RGB is
(0, 0, 0) to ignore; ``eval_pano_transfer.py`` scores those pixels. Two steps:

``count``  reads the area-5 panoramas and counts labelled pixels on black RGB per elevation
           band (the bands of ``eval_pano_transfer.py``) and per class, at each label size.
           Black is decided on the native panorama and nearest-resized like the labels.

``bounds`` takes saved confusions (``elevation_band_confusion.npy`` or ``confusion_matrix.csv``)
           and gives the lowest and highest strict 8-class mIoU the run could have once those
           pixels are left out. The predictions on them are not saved, so this is an interval
           that holds the exact value, not a rescore (``eval_pano_transfer.py --ignore-black-rgb``
           gives the exact value). The label size is found by matching the confusion's
           ground-truth counts to ``count``'s, which also checks the two used the same labels.

usage:
  python3 black_pixel_check.py count --stanford-root ROOT --semantic-labels JSON --output DIR
  python3 black_pixel_check.py bounds --counts DIR/black_pixel_counts.npz --output OUT.csv \\
      name=path/elevation_band_confusion.npy [name=path/confusion_matrix.csv ...]
"""

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pano_transfer as ev  # noqa: E402

NUM_CLASSES = len(ev.EXPECTED_CLASSES) - 1  # 13, <UNK> is ignore
CLASS_NAMES = ev.EXPECTED_CLASSES[1:]
KEEP = [CLASS_NAMES.index(name) for name in ev.EIGHT_CLASS_NAMES]
EDGES = ev.ELEVATION_BAND_EDGES_DEG
BANDS = ["{}..{}".format(top, bottom) for top, bottom in zip(EDGES[:-1], EDGES[1:])]


def band_of_rows(height):
    """Band index of every row, by the rule of ``ev.elevation_band_confusions`` (top edge included)."""
    elevation = 90.0 - np.arange(height) * 180.0 / height
    band = np.full(height, -1)
    for index, (top, bottom) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        band[(elevation <= top) & (elevation > bottom)] = index
    assert (band >= 0).all()
    return band


def count_one(label, black, band):
    """Per band: labelled pixels per class, the black ones per class, and black pixels of any label."""
    labelled = np.zeros((len(BANDS), NUM_CLASSES), dtype=np.int64)
    black_labelled = np.zeros_like(labelled)
    black_any = np.zeros(len(BANDS), dtype=np.int64)
    for index in range(len(BANDS)):
        rows_label = label[band == index]
        rows_black = black[band == index]
        valid = rows_label < NUM_CLASSES
        labelled[index] = np.bincount(rows_label[valid], minlength=NUM_CLASSES)
        black_labelled[index] = np.bincount(rows_label[valid & rows_black], minlength=NUM_CLASSES)
        black_any[index] = rows_black.sum()
    return labelled, black_labelled, black_any


def class_table(labelled, black):
    return {
        name: {"labelled": int(labelled[i]), "black": int(black[i]),
               "black_percent": 100.0 * black[i] / labelled[i] if labelled[i] else None}
        for i, name in enumerate(CLASS_NAMES)
    }


def run_count(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))
    sizes = [tuple(size) for size in args.label_size]
    lookup = ev.label_lookup(args.semantic_labels)
    samples = ev.list_panoramas(args.stanford_root, args.areas)[: args.limit]
    totals = {
        size: [np.zeros((len(BANDS), NUM_CLASSES), np.int64), np.zeros((len(BANDS), NUM_CLASSES), np.int64),
               np.zeros(len(BANDS), np.int64)]
        for size in sizes
    }
    per_panorama = []
    for index, sample in enumerate(samples, start=1):
        semantic = cv2.imread(str(sample["semantic"]), cv2.IMREAD_COLOR)
        if semantic is None:
            raise FileNotFoundError(sample["semantic"])
        row = {"sample_id": sample["sample_id"]}
        for size in sizes:
            resized = cv2.resize(semantic, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
            label = ev.decode_semantic(resized, lookup)
            black = ev.black_rgb_mask(sample["rgb"], size)
            counts = count_one(label, black, band_of_rows(size[0]))
            for total, part in zip(totals[size], counts):
                total += part
            if size == sizes[0]:
                row.update(labelled=int(counts[0].sum()), black_labelled=int(counts[1].sum()),
                           black_any=int(counts[2].sum()), pixels=size[0] * size[1])
        per_panorama.append(row)
        if index % 20 == 0 or index == len(samples):
            print("{}/{} panoramas".format(index, len(samples)), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {}
    summary = {
        "status": "SMOKE" if args.limit is not None else "COMPLETED",
        "areas": args.areas, "sample_count": len(samples),
        "black_rule": "native RGB == (0, 0, 0), nearest-resized to the label size (Trans4PASS _color2id)",
        "bands_deg": BANDS, "sizes": {},
    }
    for size in sizes:
        key = "{}x{}".format(*size)
        labelled, black, black_any = totals[size]
        arrays.update({key + "_labelled": labelled, key + "_black_labelled": black, key + "_black_any": black_any})
        band_pixels = np.bincount(band_of_rows(size[0]), minlength=len(BANDS)) * size[1] * len(samples)
        summary["sizes"][key] = {
            "pixels": int(band_pixels.sum()),
            "labelled": int(labelled.sum()),
            "black_any": int(black_any.sum()),
            "black_labelled": int(black.sum()),
            "black_labelled_percent_of_labelled": 100.0 * black.sum() / labelled.sum(),
            "black_labelled_8class": int(black[:, KEEP].sum()),
            "black_labelled_8class_percent": 100.0 * black[:, KEEP].sum() / labelled[:, KEEP].sum(),
            "per_class_13": class_table(labelled.sum(0), black.sum(0)),
            "per_band": [
                {"band_deg": name, "pixels": int(band_pixels[i]), "black_any": int(black_any[i]),
                 "black_any_percent_of_band": 100.0 * black_any[i] / band_pixels[i],
                 "labelled": int(labelled[i].sum()), "black_labelled": int(black[i].sum()),
                 "black_labelled_8class": int(black[i, KEEP].sum()),
                 "share_of_all_black_labelled_percent": 100.0 * black[i].sum() / black.sum() if black.sum() else None,
                 "per_class_13": class_table(labelled[i], black[i])}
                for i, name in enumerate(BANDS)
            ],
        }
    np.savez(args.output / "black_pixel_counts.npz", **arrays)
    (args.output / "black_pixel_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output / "per_panorama.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_panorama[0]))
        writer.writeheader()
        writer.writerows(per_panorama)
    for key, size_summary in summary["sizes"].items():
        print("{}: {} of {} labelled pixels on black RGB ({:.4f}%); 8-class {:.4f}%".format(
            key, size_summary["black_labelled"], size_summary["labelled"],
            size_summary["black_labelled_percent_of_labelled"], size_summary["black_labelled_8class_percent"]))
    return 0


def strict8_bounds(confusion, black):
    """Strict 8-class IoU per class (percent) now, and its lowest and highest value without ``black``.

    ``confusion`` is 13 x 13 (ground truth rows); ``black`` is the 13 per-class counts of
    labelled pixels on black RGB. Dropped-class rows never count (as rescore8_bands.py), so only
    black pixels of the 8 classes matter. With a of them correct for class k and f of them
    predicted k with other ground truth, IoU_k becomes (tp - a) / (union - b - f): lowest at the
    largest a and f = 0, highest at the smallest a and the largest f.

    The per-class lows happen together (every black pixel predicted right), so ``mIoU_low`` is
    reachable. The per-class highs are not: a black pixel is a false positive of one class at most,
    so ``mIoU_high`` caps the sum of f by the black pixels predicted wrong. The mean IoU is convex
    in f, so its maximum is at a vertex of that budget polytope; there are few, and all are tried.
    """
    rows = confusion[KEEP].astype(np.float64)
    tp = rows[np.arange(len(KEEP)), KEEP]
    gt = rows.sum(1)
    pred = rows[:, KEEP].sum(0)
    union = gt + pred - tp
    b = np.asarray(black, dtype=np.float64)[KEEP]
    if (b > gt).any():
        raise ValueError("more black pixels than labelled pixels in a class")
    a_max = np.minimum(b, tp)
    a_min = np.maximum(0.0, b - (gt - tp))
    f_max = np.minimum(pred - tp, b.sum() - b)

    def iou(numerator, denominator):
        return np.divide(numerator, denominator, out=np.full(len(KEEP), np.nan), where=denominator > 0) * 100

    present = gt - b > 0
    low = iou(tp - a_max, union - b)
    budget = b.sum() - a_min.sum()
    best = -np.inf
    for at_cap in itertools.product((False, True), repeat=len(KEEP)):
        at_cap = np.array(at_cap)
        left = budget - f_max[at_cap].sum()
        if left < 0:
            continue
        for partial in [None] + list(np.flatnonzero(~at_cap)):  # at most one class off its bounds
            f = np.where(at_cap, f_max, 0.0)
            if partial is not None:
                f[partial] = min(f_max[partial], left)
            best = max(best, np.nanmean(iou(tp - a_min, union - b - f)[present]) if present.any() else np.nan)
    return {
        "now": iou(tp, union), "low": low, "high": iou(tp - a_min, union - b - f_max),
        "present_now": gt > 0, "present_after": present,
        "mIoU_low": np.nanmean(low[present]) if present.any() else np.nan, "mIoU_high": best,
    }


def load_confusion(path):
    path = Path(path)
    confusion = np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.loadtxt(path, delimiter=",")
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape not in ((len(BANDS), NUM_CLASSES, NUM_CLASSES), (NUM_CLASSES, NUM_CLASSES)):
        raise ValueError("{}: unexpected confusion shape {}".format(path, confusion.shape))
    return confusion


def match_size(confusion, counts):
    """Label size whose ground-truth counts equal the confusion's (per band when it has bands)."""
    for key in sorted(name[: -len("_black_any")] for name in counts.files if name.endswith("_black_any")):
        labelled = counts[key + "_labelled"]
        if confusion.ndim == 3 and np.array_equal(confusion.sum(2), labelled):
            return key
        if confusion.ndim == 2 and np.array_equal(confusion.sum(1), labelled.sum(0)):
            return key
    raise ValueError("no counted label size has this confusion's ground-truth counts; "
                     "count at the run's --height/--width on the same areas")


def run_bounds(args):
    counts = np.load(args.counts, allow_pickle=False)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "label_size", "band", "gt_pixels_8class", "black_gt_pixels_8class",
                         "mIoU_now", "mIoU_low", "mIoU_high", "width"]
                        + ["{}_{}".format(name, end) for name in ev.EIGHT_CLASS_NAMES for end in ("low", "high")])
        for arg in args.runs:
            name, path = arg.split("=", 1)
            confusion = load_confusion(path)
            key = match_size(confusion, counts)
            black = counts[key + "_black_labelled"]
            if confusion.ndim == 3:
                parts = [("all", confusion.sum(0), black.sum(0))] + list(zip(BANDS, confusion, black))
            else:
                parts = [("all", confusion, black.sum(0))]
            for band, part, part_black in parts:
                s = strict8_bounds(part, part_black)
                # overall: mean over the 8 classes; a band: mean over the classes present in it
                now = np.nanmean(s["now"][s["present_now"]]) if s["present_now"].any() else np.nan
                low, high = s["mIoU_low"], s["mIoU_high"]  # nan when every labelled pixel of the band is black
                writer.writerow(
                    [name, key, band, int(part[KEEP].sum()), int(part_black[KEEP].sum()),
                     "{:.2f}".format(now), "{:.2f}".format(low), "{:.2f}".format(high), "{:.2f}".format(high - low)]
                    + ["{:.2f}".format(value) for pair in zip(s["low"], s["high"]) for value in pair]
                )
                if band == "all":
                    print("{} ({}): strict 8-class mIoU {:.2f} now, {:.2f}..{:.2f} without black RGB".format(
                        name, key, now, low, high))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    count = commands.add_parser("count")
    count.add_argument("--stanford-root", required=True, type=Path)
    count.add_argument("--semantic-labels", required=True, type=Path)
    count.add_argument("--output", required=True, type=Path)
    count.add_argument("--areas", nargs="+", default=["area_5a", "area_5b"])
    count.add_argument("--label-size", nargs=2, type=int, action="append", metavar=("HEIGHT", "WIDTH"),
                       help="label sizes to count at (default: 2048 4096 and 1024 2048)")
    count.add_argument("--limit", type=int, default=None, help="smoke runs only")
    bounds = commands.add_parser("bounds")
    bounds.add_argument("--counts", required=True, type=Path)
    bounds.add_argument("--output", required=True, type=Path)
    bounds.add_argument("runs", nargs="+", metavar="name=confusion")
    args = parser.parse_args(argv)
    if args.command == "count":
        args.label_size = args.label_size or [[2048, 4096], [1024, 2048]]
        if args.limit is not None and args.limit <= 0:
            parser.error("--limit must be positive; omit it for full counting")
        return run_count(args)
    return run_bounds(args)


if __name__ == "__main__":
    raise SystemExit(main())
