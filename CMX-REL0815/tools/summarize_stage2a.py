#!/usr/bin/env python3
import csv
import json
import math
import sys
from pathlib import Path


ARMS = ("rawdepth", "hha", "relplus_local", "relplus_pose")
LABELS = {
    "rawdepth": "RawDepth",
    "hha": "HHA",
    "relplus_local": "REL+-Local",
    "relplus_pose": "REL+-Pose",
}
CLASSES = (
    "beam", "board", "bookcase", "ceiling", "chair", "clutter",
    "column", "door", "floor", "sofa", "table", "wall", "window",
)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(root_arg):
    root = Path(root_arg)
    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    metrics = {}
    histories = {}
    summary_rows = []
    for arm in ARMS:
        metrics[arm] = json.loads((root / arm / "metrics" / "final_metrics.json").read_text())
        histories[arm] = read_csv(root / arm / "metrics.csv")
        history = histories[arm]
        summary_rows.append({
            "arm": arm,
            "label": LABELS[arm],
            "seed": 12345,
            "selected_epoch": 32,
            "miou": metrics[arm]["miou"],
            "mean_pixel_accuracy": metrics[arm]["mean_pixel_accuracy"],
            "pixel_accuracy": metrics[arm]["pixel_accuracy"],
            "final_train_loss": float(history[-1]["train_loss"]),
            "mean_iteration_time_seconds": sum(float(r["mean_iteration_time_seconds"]) for r in history) / len(history),
            "total_epoch_time_seconds": sum(float(r["epoch_time_seconds"]) for r in history),
            "peak_gpu_memory_bytes": max(int(r["peak_gpu_memory_bytes"]) for r in history),
        })
    fields = list(summary_rows[0])
    write_csv(out / "four_arm_metrics.csv", fields, summary_rows)

    per_class_rows = []
    for name in CLASSES:
        values = {arm: metrics[arm]["per_class_iou"][name] for arm in ARMS}
        per_class_rows.append({
            "class": name,
            "rawdepth": values["rawdepth"],
            "hha": values["hha"],
            "relplus_local": values["relplus_local"],
            "relplus_pose": values["relplus_pose"],
            "hha_minus_rawdepth": values["hha"] - values["rawdepth"],
            "local_minus_rawdepth": values["relplus_local"] - values["rawdepth"],
            "local_minus_hha": values["relplus_local"] - values["hha"],
            "pose_minus_local": values["relplus_pose"] - values["relplus_local"],
        })
    write_csv(out / "per_class_iou.csv", list(per_class_rows[0]), per_class_rows)

    comparisons = {
        "hha_minus_rawdepth": metrics["hha"]["miou"] - metrics["rawdepth"]["miou"],
        "local_minus_rawdepth": metrics["relplus_local"]["miou"] - metrics["rawdepth"]["miou"],
        "local_minus_hha": metrics["relplus_local"]["miou"] - metrics["hha"]["miou"],
        "pose_minus_local": metrics["relplus_pose"]["miou"] - metrics["relplus_local"]["miou"],
    }
    result = {
        "status": "COMPLETE_STAGE2A_FOUR_ARM_SINGLE_SEED",
        "seed": 12345,
        "selected_epoch": 32,
        "primary_endpoint": "mIoU",
        "metrics": {arm: {
            "miou": metrics[arm]["miou"],
            "mean_pixel_accuracy": metrics[arm]["mean_pixel_accuracy"],
            "pixel_accuracy": metrics[arm]["pixel_accuracy"],
        } for arm in ARMS},
        "planned_miou_differences": comparisons,
        "highest_miou_arm": max(ARMS, key=lambda arm: metrics[arm]["miou"]),
        "inference_boundary": "single seed; descriptive effects only; no significance or reproducibility claim",
        "next_experiment_started": False,
    }
    (out / "FINAL_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Stage2A four-arm single-seed comparison", "",
        "| Arm | mIoU | mAcc | pAcc | Final train loss | Mean iter (s) | Peak GPU (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append("| {label} | {miou:.4%} | {mean_pixel_accuracy:.4%} | {pixel_accuracy:.4%} | {final_train_loss:.6f} | {mean_iteration_time_seconds:.4f} | {gib:.3f} |".format(
            gib=row["peak_gpu_memory_bytes"] / 1024 ** 3, **row))
    lines += ["", "Planned mIoU differences (percentage points):", ""]
    for key, value in comparisons.items():
        lines.append("- `{}`: {:+.4f} pp".format(key, value * 100.0))
    lines += ["", "> Single seed only: these are descriptive differences, not significance or reproducibility evidence.", ""]
    (out / "final_comparison_table.md").write_text("\n".join(lines))

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = small = ImageFont.load_default()
    left, top, right, bottom = 110, 90, 1340, 780
    draw.text((left, 30), "Stage2A training loss (single seed 12345)", fill="black", font=font)
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    all_losses = [float(row["train_loss"]) for arm in ARMS for row in histories[arm]]
    lo, hi = math.log10(min(all_losses)), math.log10(max(all_losses))
    def xy(epoch, loss):
        x = left + (epoch - 1) / 31.0 * (right - left)
        y = bottom - (math.log10(loss) - lo) / (hi - lo) * (bottom - top)
        return x, y
    for epoch in (1, 4, 8, 12, 16, 20, 24, 28, 32):
        x, _ = xy(epoch, min(all_losses))
        draw.line((x, top, x, bottom), fill=(225, 225, 225), width=1)
        draw.text((x - 8, bottom + 12), str(epoch), fill="black", font=small)
    for exponent in range(math.floor(lo), math.ceil(hi) + 1):
        value = 10 ** exponent
        _, y = xy(1, value)
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((20, y - 10), "10^{}".format(exponent), fill="black", font=small)
    colors = {"rawdepth": (31, 119, 180), "hha": (255, 127, 14), "relplus_local": (44, 160, 44), "relplus_pose": (214, 39, 40)}
    for index, arm in enumerate(ARMS):
        points = [xy(int(row["epoch"]), float(row["train_loss"])) for row in histories[arm]]
        draw.line(points, fill=colors[arm], width=4)
        lx = left + index * 285
        draw.line((lx, 830, lx + 45, 830), fill=colors[arm], width=5)
        draw.text((lx + 55, 816), LABELS[arm], fill="black", font=small)
    draw.text((650, 830), "Epoch", fill="black", font=small)
    image.save(out / "training_curves.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_stage2a.py STAGE2A_ROOT")
    main(sys.argv[1])
