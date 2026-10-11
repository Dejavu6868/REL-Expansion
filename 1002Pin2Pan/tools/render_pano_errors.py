#!/usr/bin/env python3
"""Render panorama prediction errors, with the REL+ EGVIA channel beside them.

Reads the class-id maps that ``eval_pano_transfer.py --save-predictions`` wrote,
ranks every panorama by two error counts against the ground truth, and draws
the worst ones:

* ``floor_as_table``: ground-truth floor predicted as table (the v2.1 nadir failure).
* ``wall_errors``: ground-truth wall predicted as another class, plus other
  classes predicted as wall.

Each figure stacks, top to bottom: RGB, EGVIA under v2.1 (alpha 45, height only
on surfaces steeper than 45 degrees), EGVIA under 3.0 (height on every
surface), ground truth, the prediction, and an error map (red = wrong,
magenta = floor read as table, yellow = wall error), with an optional second
prediction folder below for comparison. EGVIA comes from the frozen ERP
``getREL`` on the same nearest-resized depth the eval uses; 255 (no depth) is
drawn black. ``counts.csv`` holds both counts and their per-class split for
every panorama.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import eval_pano_transfer as ev
import relplus_variant as rv
from cross_projection import decode_erp_depth


CLASS_NAMES = ev.EXPECTED_CLASSES[1:]  # class ids 0..12 in the frozen config order
FLOOR, TABLE, WALL = (CLASS_NAMES.index(name) for name in ("floor", "table", "wall"))
# BGR, one fixed colour per class id; 255 (ignore) is black.
PALETTE = np.array([
    [60, 20, 220], [180, 130, 70], [48, 167, 255], [230, 230, 230], [0, 200, 120],
    [128, 128, 128], [200, 80, 200], [0, 100, 160], [140, 180, 210], [80, 60, 140],
    [30, 90, 230], [190, 210, 160], [250, 200, 60],
], dtype=np.uint8)
ERROR_COLOURS = {"wrong": (40, 40, 220), "floor_as_table": (220, 0, 220), "wall": (0, 220, 255)}


def read_class_names(eval_dir):
    with (Path(eval_dir) / "per_class_iou.csv").open(encoding="utf-8") as handle:
        return [row["class_name"] for row in csv.DictReader(handle)]


def error_counts(prediction, label):
    """Floor-as-table and wall error counts, with the classes walls are confused with."""
    valid = label != 255
    wrong = valid & (prediction != label)
    wall_missed = wrong & (label == WALL)
    wall_false = wrong & (prediction == WALL)
    counts = {
        "valid_pixels": int(valid.sum()),
        "wrong_pixels": int(wrong.sum()),
        "floor_as_table": int((valid & (label == FLOOR) & (prediction == TABLE)).sum()),
        "wall_errors": int(wall_missed.sum() + wall_false.sum()),
    }
    for class_id, name in enumerate(CLASS_NAMES):
        if class_id != WALL:
            counts["wall_as_" + name] = int((wall_missed & (prediction == class_id)).sum())
            counts[name + "_as_wall"] = int((wall_false & (label == class_id)).sum())
    return counts


def colourise(class_map):
    out = np.zeros(class_map.shape + (3,), dtype=np.uint8)
    known = class_map < len(PALETTE)
    out[known] = PALETTE[class_map[known]]
    return out


def error_map(bgr, prediction, label):
    out = (bgr.astype(np.float32) * 0.35).astype(np.uint8)
    valid = label != 255
    wrong = valid & (prediction != label)
    out[wrong] = ERROR_COLOURS["wrong"]
    out[wrong & ((label == WALL) | (prediction == WALL))] = ERROR_COLOURS["wall"]
    out[valid & (label == FLOOR) & (prediction == TABLE)] = ERROR_COLOURS["floor_as_table"]
    return out


def egvia_image(rel):
    egvia = rel[:, :, 0]
    out = cv2.applyColorMap(egvia, cv2.COLORMAP_VIRIDIS)
    out[egvia == 255] = 0
    return out


def titled(image, title, width):
    image = cv2.resize(image, (width, width // 2), interpolation=cv2.INTER_NEAREST)
    bar = np.full((28, width, 3), 255, dtype=np.uint8)
    cv2.putText(bar, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def legend(width):
    rows = []
    per_row = 7
    names = list(CLASS_NAMES) + ["ignore"]
    colours = [tuple(int(v) for v in c) for c in PALETTE] + [(0, 0, 0)]
    for start in range(0, len(names), per_row):
        bar = np.full((26, width, 3), 255, dtype=np.uint8)
        for slot, (name, colour) in enumerate(zip(names[start:start + per_row], colours[start:start + per_row])):
            x = 8 + slot * (width // per_row)
            cv2.rectangle(bar, (x, 6), (x + 14, 20), colour, -1)
            cv2.putText(bar, name, (x + 20, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        rows.append(bar)
    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stanford-root", required=True, type=Path)
    parser.add_argument("--semantic-labels", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path,
                        help="eval_pano_transfer.py --output folder run with --save-predictions")
    parser.add_argument("--label", default="REL+ v2.1", help="name of --predictions in the figure")
    parser.add_argument("--compare-predictions", type=Path, default=None,
                        help="optional second eval folder (e.g. REL+ 3.0 or HHA), drawn below")
    parser.add_argument("--compare-label", default="comparison")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", type=int, default=8, help="panoramas drawn per ranking")
    parser.add_argument("--source-root", type=Path, default=ev.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--areas", nargs="+", default=["area_5a", "area_5b"])
    parser.add_argument("--panel-width", type=int, default=1024)
    args = parser.parse_args()
    if args.top <= 0:
        parser.error("--top must be positive")

    for folder in [args.predictions] + ([args.compare_predictions] if args.compare_predictions else []):
        names = read_class_names(folder)
        if names != CLASS_NAMES:
            raise ValueError("{} class order {} differs from {}".format(folder, names, CLASS_NAMES))
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))
    args.output.mkdir(parents=True, exist_ok=True)

    lookup = ev.label_lookup(args.semantic_labels)
    samples = [s for s in ev.list_panoramas(args.stanford_root, args.areas)
               if (args.predictions / "predictions" / (s["sample_id"] + ".png")).is_file()]
    if not samples:
        raise FileNotFoundError("no saved predictions under {}/predictions".format(args.predictions))

    def read_prediction(folder, sample, shape):
        path = folder / "predictions" / (sample["sample_id"] + ".png")
        prediction = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if prediction is None or prediction.shape != shape:
            raise ValueError("{} is missing or not {}".format(path, shape))
        return prediction

    def read_label(sample, shape):
        semantic = cv2.imread(str(sample["semantic"]), cv2.IMREAD_COLOR)
        semantic = cv2.resize(semantic, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return ev.decode_semantic(semantic, lookup)

    rows = []
    for sample in samples:
        prediction = cv2.imread(str(args.predictions / "predictions" / (sample["sample_id"] + ".png")),
                                cv2.IMREAD_UNCHANGED)
        counts = error_counts(prediction, read_label(sample, prediction.shape))
        rows.append(dict(sample_id=sample["sample_id"], **counts))
    with (args.output / "counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    frozen = ev.import_frozen(args.source_root)
    by_id = {s["sample_id"]: s for s in samples}
    width = args.panel_width
    for ranking in ("floor_as_table", "wall_errors"):
        ranked = sorted(rows, key=lambda row: row[ranking], reverse=True)[: args.top]
        for rank, row in enumerate(ranked, start=1):
            sample = by_id[row["sample_id"]]
            prediction = cv2.imread(str(args.predictions / "predictions" / (sample["sample_id"] + ".png")),
                                    cv2.IMREAD_UNCHANGED)
            height_px, width_px = prediction.shape
            label = read_label(sample, prediction.shape)
            bgr = cv2.resize(cv2.imread(str(sample["rgb"]), cv2.IMREAD_COLOR), (width_px, height_px),
                             interpolation=cv2.INTER_LINEAR)
            raw = cv2.imread(str(sample["depth"]), cv2.IMREAD_UNCHANGED)
            depth = decode_erp_depth(cv2.resize(raw, (width_px, height_px), interpolation=cv2.INTER_NEAREST))
            panels = [
                titled(bgr, sample["sample_id"] + "  RGB", width),
                titled(egvia_image(frozen["getREL"](depth, alpha=rv.VARIANT_ALPHA["v2_1"])),
                       "EGVIA v2.1 (height only on surfaces steeper than 45 deg)", width),
                titled(egvia_image(frozen["getREL"](depth, alpha=rv.VARIANT_ALPHA["v3_0"])),
                       "EGVIA 3.0 (height on every surface)", width),
                titled(colourise(label), "ground truth", width),
                titled(colourise(prediction), args.label + " prediction", width),
                titled(error_map(bgr, prediction, label),
                       "{} errors: magenta floor-as-table {:,} px, yellow wall {:,} px, red other".format(
                           args.label, row["floor_as_table"], row["wall_errors"]), width),
            ]
            if args.compare_predictions:
                other = read_prediction(args.compare_predictions, sample, prediction.shape)
                other_counts = error_counts(other, label)
                panels.append(titled(colourise(other), args.compare_label + " prediction", width))
                panels.append(titled(error_map(bgr, other, label),
                                     "{} errors: floor as table {:,} px, wall {:,} px".format(
                                         args.compare_label, other_counts["floor_as_table"],
                                         other_counts["wall_errors"]), width))
            name = "{}_{:02d}_{}.png".format(ranking, rank, sample["sample_id"].replace("/", "_"))
            cv2.imwrite(str(args.output / name), np.vstack(panels + [legend(width)]))
            print("wrote", name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
