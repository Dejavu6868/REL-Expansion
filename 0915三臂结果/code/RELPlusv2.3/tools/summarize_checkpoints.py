#!/usr/bin/env python3
"""Summarize the 21 required Fold 1 checkpoint evaluations for one arm."""

import argparse
import csv
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--target", required=True, type=float)
    args = parser.parse_args()

    metric_files = sorted(
        args.run_dir.glob("metrics/metrics_epoch*.json"),
        key=lambda path: int(path.stem.replace("metrics_epoch", "")),
    )
    if len(metric_files) != 21:
        raise RuntimeError("expected 21 checkpoint metrics, found {}".format(len(metric_files)))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in metric_files]
    expected_epochs = list(range(100, 201, 5))
    observed_epochs = [int(row["checkpoint_epoch"]) for row in rows]
    if observed_epochs != expected_epochs:
        raise RuntimeError("checkpoint epochs differ: {}".format(observed_epochs))

    best = max(rows, key=lambda row: row["mIoU_percent"])
    epoch200 = next(row for row in rows if int(row["checkpoint_epoch"]) == 200)
    best["paper_target_mIoU_percent"] = args.target
    best["difference_from_target_pp"] = best["mIoU_percent"] - args.target
    epoch200["paper_target_mIoU_percent"] = args.target
    epoch200["difference_from_target_pp"] = epoch200["mIoU_percent"] - args.target

    with (args.run_dir / "metrics_all_checkpoints.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["epoch", "mIoU_percent", "pixel_accuracy_percent", "mean_class_accuracy_percent", "valid_pixels", "ignore_pixels", "sample_count"]
        )
        for row in rows:
            writer.writerow(
                [
                    row["checkpoint_epoch"], row["mIoU_percent"],
                    row["pixel_accuracy_percent"], row["mean_class_accuracy_percent"],
                    row["valid_pixels"], row["ignore_pixels"], row["sample_count"],
                ]
            )
    (args.run_dir / "metrics_best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.run_dir / "metrics_epoch200.json").write_text(
        json.dumps(epoch200, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    best_epoch = int(best["checkpoint_epoch"])
    shutil.copy2(
        args.run_dir / "metrics" / "per_class_epoch{}.csv".format(best_epoch),
        args.run_dir / "per_class_iou_best.csv",
    )
    shutil.copy2(
        args.run_dir / "metrics" / "per_class_epoch200.csv",
        args.run_dir / "per_class_iou_epoch200.csv",
    )
    shutil.copy2(
        args.run_dir / "metrics" / "confusion_epoch{}.csv".format(best_epoch),
        args.run_dir / "confusion_matrix_best.csv",
    )
    shutil.copy2(
        args.run_dir / "metrics" / "confusion_epoch200.csv",
        args.run_dir / "confusion_matrix_epoch200.csv",
    )
    print("best_epoch={}".format(best_epoch))
    print("best_mIoU_percent={}".format(best["mIoU_percent"]))
    print("epoch200_mIoU_percent={}".format(epoch200["mIoU_percent"]))


if __name__ == "__main__":
    main()
