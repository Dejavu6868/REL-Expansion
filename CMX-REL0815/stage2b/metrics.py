import csv
import json
import os
from pathlib import Path


def _safe_ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator) / float(denominator)


def per_image_from_hist(name, hist, class_names):
    matrix = [[int(value) for value in row] for row in hist]
    size = len(class_names)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("confusion matrix and class names do not match")
    gt = [sum(matrix[index]) for index in range(size)]
    predicted = [sum(matrix[row][index] for row in range(size)) for index in range(size)]
    correct = [matrix[index][index] for index in range(size)]
    precision = [_safe_ratio(correct[index], predicted[index]) for index in range(size)]
    recall = [_safe_ratio(correct[index], gt[index]) for index in range(size)]
    iou = [
        _safe_ratio(correct[index], gt[index] + predicted[index] - correct[index])
        for index in range(size)
    ]
    present = [iou[index] for index in range(size) if gt[index] > 0 and iou[index] is not None]
    labeled = sum(gt)
    return {
        "name": name,
        "pixel_accuracy": _safe_ratio(sum(correct), labeled),
        "present_class_miou": sum(present) / len(present) if present else None,
        "labeled_pixels": labeled,
        "gt_pixels": gt,
        "predicted_pixels": predicted,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_iou": iou,
    }


def write_per_image_csv(path, results, class_names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "name", "pixel_accuracy", "present_class_miou", "labeled_pixels",
        "gt_pixels", "predicted_pixels", "per_class_precision",
        "per_class_recall", "per_class_iou",
    )
    temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for raw in sorted(results, key=lambda row: row["name"]):
            row = per_image_from_hist(raw["name"], raw["hist"], class_names)
            writer.writerow({
                key: json.dumps(row[key], separators=(",", ":"))
                if isinstance(row[key], list) else row[key]
                for key in fields
            })
    os.replace(str(temporary), str(path))
