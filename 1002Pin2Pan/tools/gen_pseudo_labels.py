#!/usr/bin/env python3
"""Pseudo-label the cached target panoramas with a checkpoint (Trans4PASS ``gen_pseudo_label.py``).

Each panorama gets one whole-panorama pass with circular padding, as in
``eval_pano_transfer.py``. A pixel keeps its predicted class when its softmax
confidence reaches that class's threshold and becomes 255 otherwise. The
threshold is Trans4PASS's: the median confidence of all pixels predicted as the
class over the whole target set, capped at 0.9. Trans4PASS averages six input
scales; this uses the evaluation size only (1.75x of a 1024x2048 panorama does
not fit a 24 GB card).

Writes ``PseudoLabel/<area>/<stem>.png`` and ``pseudo_labels.json`` under
``--output``. The cache's ground truth is read only for the report: pseudo-label
coverage and accuracy per class, and the checkpoint's scores on these training
panoramas, overall and per elevation band. Do not tune anything on it.
"""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

import build_target_cache as tc
import eval_pano_transfer as ev
import pin2pan_adapt as pa


def read(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image


def write(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError("could not write " + str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--target-cache", required=True, type=Path, help="build_target_cache.py output")
    parser.add_argument("--config", required=True, type=Path, help="resolved arm config JSON")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-epoch", type=int, default=200)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=ev.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--wrap-pad", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="smoke runs only")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    import torch
    import torch.nn as nn

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))
    frozen = ev.import_frozen(args.source_root)
    config = ev.load_config(args.config)
    manifest = tc.load_manifest(args.target_cache, config)
    samples = manifest["samples"][: args.limit]
    num_classes = config.num_classes

    device = torch.device(args.device)
    network = frozen["EncoderDecoder"](cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    epoch = frozen["load_checkpoint_once"](network, args.checkpoint, expected_epoch=args.expected_epoch)
    network.to(device)
    network.eval()

    started = time.time()
    work = args.output / "_work"
    levels = pa.CONFIDENCE_LEVELS + 1
    histograms = np.zeros((num_classes, levels), dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    band_confusion = np.zeros((len(ev.ELEVATION_BAND_EDGES_DEG) - 1,) + confusion.shape, dtype=np.int64)
    for index, sample_id in enumerate(samples, start=1):
        rgb = read(args.target_cache / "RGB" / (sample_id + ".png"))
        modal_x = read(args.target_cache / "X" / (sample_id + ".png"))
        label = read(args.target_cache / "Label" / (sample_id + ".png"))
        logits = ev.predict_with_wrap(
            network,
            frozen["normalized_bchw"](rgb, config.norm_mean, config.norm_std),
            frozen["normalized_bchw"](modal_x, config.norm_mean, config.norm_std),
            args.wrap_pad,
            device,
        )
        confidence, prediction = torch.softmax(logits, dim=1)[0].max(dim=0)
        prediction = prediction.cpu().numpy().astype(np.uint8)
        confidence = pa.quantize_confidence(confidence.cpu().numpy())
        histograms += np.bincount(
            prediction.ravel().astype(np.int64) * levels + confidence.ravel(),
            minlength=num_classes * levels,
        ).reshape(num_classes, levels)
        write(work / (sample_id + "_prediction.png"), prediction)
        write(work / (sample_id + "_confidence.png"), confidence)
        confusion += frozen["hist_info"](num_classes, prediction, label)[0].astype(np.int64)
        band_confusion += ev.elevation_band_confusions(num_classes, prediction, label, frozen["hist_info"])
        if index % 50 == 0 or index == len(samples):
            print("predicted {}/{} panoramas".format(index, len(samples)), flush=True)

    thresholds = pa.class_thresholds(histograms)
    pseudo_confusion = np.zeros_like(confusion)
    kept = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0
    for sample_id in samples:
        prediction = read(work / (sample_id + "_prediction.png"))
        confidence = read(work / (sample_id + "_confidence.png"))
        pseudo = pa.apply_thresholds(prediction, confidence, thresholds)
        write(args.output / "PseudoLabel" / (sample_id + ".png"), pseudo)
        label = read(args.target_cache / "Label" / (sample_id + ".png"))
        keep = pseudo != pa.IGNORE_INDEX
        kept += np.bincount(pseudo[keep], minlength=num_classes)
        total_pixels += pseudo.size
        pseudo_confusion += frozen["hist_info"](num_classes, pseudo[keep], label[keep])[0].astype(np.int64)
    shutil.rmtree(work)

    names = list(config.class_names)
    metrics = frozen["metrics_from_confusion"](confusion)
    labelled_kept = pseudo_confusion.sum(axis=0)
    per_class = [
        {
            "class_name": name,
            "threshold": float(thresholds[class_id]),
            "predicted_pixels": int(histograms[class_id].sum()),
            "kept_pixels": int(kept[class_id]),
            "kept_precision_percent": (
                100.0 * pseudo_confusion[class_id, class_id] / labelled_kept[class_id]
                if labelled_kept[class_id] else None
            ),
        }
        for class_id, name in enumerate(names)
    ]
    report = {
        "status": "SMOKE" if args.limit is not None else "COMPLETED",
        "rule": "Trans4PASS gen_pseudo_label.py: per-class median confidence capped at 0.9; one scale",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": epoch,
        "config": str(args.config),
        "target_cache": str(args.target_cache),
        "x_mode": config.x_mode,
        "relplus_variant": manifest["relplus_variant"],
        "areas": manifest["areas"],
        "sample_count": len(samples),
        "wrap_pad": args.wrap_pad,
        "seconds": time.time() - started,
        "pseudo_labels": {
            "coverage_percent": 100.0 * kept.sum() / total_pixels,
            "accuracy_percent": 100.0 * np.trace(pseudo_confusion) / max(pseudo_confusion.sum(), 1),
            "per_class": per_class,
        },
        "ground_truth_note": "Label/ of the target cache, used for this report only",
        "checkpoint_on_target_train": {
            "metrics": metrics,
            "elevation_bands": ev.write_elevation_bands(
                args.output, band_confusion, frozen["metrics_from_confusion"], names
            ),
            "bottom_band_floor_as_table_pixels": int(
                band_confusion[-1, names.index("floor"), names.index("table")]
            ),
        },
    }
    (args.output / "pseudo_labels.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "samples.txt").write_text("".join(s + "\n" for s in samples), encoding="utf-8")
    np.savetxt(args.output / "confusion_matrix.csv", confusion, fmt="%d", delimiter=",")
    np.savetxt(args.output / "pseudo_label_confusion.csv", pseudo_confusion, fmt="%d", delimiter=",")
    with (args.output / "pseudo_label_classes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    print(
        "pseudo-labels cover {:.1f}% of pixels at {:.1f}% accuracy; checkpoint mIoU {:.2f}% on {} target "
        "training panoramas".format(
            report["pseudo_labels"]["coverage_percent"],
            report["pseudo_labels"]["accuracy_percent"],
            metrics["mIoU_percent"],
            len(samples),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
