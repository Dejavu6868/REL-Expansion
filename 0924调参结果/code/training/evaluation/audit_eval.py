"""Independent, read-only integrity audit for fixed epoch-200 evaluation."""

import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


SAMPLE_COUNT = 17593
CLASS_COUNT = 13
WORLD_SIZE = 8
VALID_PIXEL_COUNT = 3973620198


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _csv(path, fieldnames):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == fieldnames, "CSV columns differ: {}".format(path))
        rows = list(reader)
    _require(all(set(row) == set(fieldnames) and all(v is not None for v in row.values())
                 for row in rows), "malformed CSV rows: {}".format(path))
    return rows


def _integer(value, label):
    if isinstance(value, str):
        _require(re.fullmatch(r"[0-9]+", value) is not None,
                 "{} must be a nonnegative integer".format(label))
        value = int(value)
    _require(type(value) is int and 0 <= value <= np.iinfo(np.int64).max,
             "{} must be a nonnegative int64 integer".format(label))
    return value


def _matrix(value, label, csv_strings=False):
    _require(isinstance(value, list) and len(value) == CLASS_COUNT,
             "{} must have 13 rows".format(label))
    rows = []
    for row in value:
        _require(isinstance(row, list) and len(row) == CLASS_COUNT,
                 "{} must have 13 columns".format(label))
        if not csv_strings:
            _require(all(type(item) is int for item in row),
                     "{} JSON values must be integers".format(label))
        rows.append([_integer(item, label) for item in row])
    # Bound the Python-integer sum before converting or adding int64 matrices.
    _require(sum(sum(row) for row in rows) <= VALID_PIXEL_COUNT,
             "{} has too many pixels".format(label))
    return np.asarray(rows, dtype=np.int64)


def _close(observed, expected, label, atol=1e-12):
    try:
        actual = np.asarray(observed, dtype=np.float64)
        target = np.asarray(expected, dtype=np.float64)
    except (ValueError, TypeError):
        raise ValueError("{} is not numeric".format(label))
    _require(actual.shape == target.shape, "{} shape mismatch".format(label))
    _require(bool(np.allclose(actual, target, atol=atol, rtol=0, equal_nan=True)),
             "{} does not match independent recomputation".format(label))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_arm(output_dir, config, expected_checkpoint):
    """Return a PASS report or raise ValueError; never modify any input file."""
    output_dir = Path(output_dir)
    evaluation = output_dir / "evaluation"
    _require((output_dir / "exitcode").read_text().strip() == "0",
             "evaluation exit code is not zero")
    names = list(config["class_names"])
    _require(len(names) == CLASS_COUNT and len(set(names)) == CLASS_COUNT,
             "expected class names must contain 13 unique classes")
    test_source = Path(config["eval_source"])
    test_ids = [line.strip() for line in test_source.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    _require(len(test_ids) == len(set(test_ids)) == SAMPLE_COUNT,
             "test list must have 17593 unique IDs")

    metrics = _json(evaluation / "metrics.json")
    required = {
        "status": "PASS", "checkpoint_epoch": 200,
        "evaluation_sample_count": SAMPLE_COUNT, "class_count": CLASS_COUNT,
        "ignore_index": 255, "world_size": WORLD_SIZE,
        "eval_scale_array": [1], "eval_flip": False,
        "eval_crop_size": [480, 480], "eval_align_corners": False,
        "representation_protocol_id": config["representation_protocol_id"],
        "metric_unit": "fraction_0_to_1", "scientific_metric_reported": True,
    }
    for key, value in required.items():
        _require(key in metrics and metrics[key] == value,
                 "metrics {} differs from expected {!r}".format(key, value))
    for key in ("checkpoint_epoch", "evaluation_sample_count", "class_count",
                "ignore_index", "world_size", "valid_pixel_count"):
        _integer(metrics[key], "metrics " + key)
    if "integration_protocol_id" in config:
        _require(metrics.get("integration_protocol_id") == config["integration_protocol_id"],
                 "integration protocol mismatch")

    reports = [_json(evaluation / "rank_{:02d}_evaluation.json".format(rank))
               for rank in range(WORLD_SIZE)]
    matrices, owned, manifest_expected, rank_counts = [], [], [], []
    for rank, report in enumerate(reports):
        for key, expected in (("rank", rank), ("local_rank", rank), ("world_size", WORLD_SIZE)):
            _require(type(report.get(key)) is int and report[key] == expected,
                     "rank {} {} mismatch".format(rank, key))
        expected_ids = test_ids[rank::WORLD_SIZE]
        _require(report.get("owned_sample_ids") == expected_ids,
                 "rank {} ordered shard mismatch".format(rank))
        _require(type(report.get("sample_count")) is int and
                 report["sample_count"] == len(expected_ids),
                 "rank {} sample count mismatch".format(rank))
        owned.extend(report["owned_sample_ids"])
        rank_counts.append(report["sample_count"])
        manifest_expected.extend({"sample_id": sample_id, "rank": str(rank)}
                                 for sample_id in expected_ids)
        matrices.append(_matrix(report["confusion_matrix"], "rank {} matrix".format(rank)))
    _require(len(owned) == len(set(owned)) == SAMPLE_COUNT and set(owned) == set(test_ids),
             "rank sample union is incomplete or duplicated")
    manifest = _csv(evaluation / "evaluation_manifest.csv", ["sample_id", "rank"])
    _require(manifest == manifest_expected, "manifest differs from exact ordered rank shards")

    matrix_rows = _csv(evaluation / "confusion_matrix.csv",
                       ["true_class"] + ["pred_{}".format(i) for i in range(CLASS_COUNT)])
    _require([_integer(row["true_class"], "true_class") for row in matrix_rows] ==
             list(range(CLASS_COUNT)), "confusion CSV class order mismatch")
    matrix = _matrix([[row["pred_{}".format(i)] for i in range(CLASS_COUNT)]
                      for row in matrix_rows], "confusion CSV", csv_strings=True)
    summed = np.sum(np.stack(matrices), axis=0, dtype=np.int64)
    _require(np.array_equal(matrix, summed), "rank matrices do not sum to confusion CSV")
    _require(int(matrix.sum()) == metrics["valid_pixel_count"] == VALID_PIXEL_COUNT,
             "valid pixel count mismatch")

    diagonal = np.diag(matrix).astype(np.float64)
    gt = matrix.sum(axis=1, dtype=np.int64)
    pred = matrix.sum(axis=0, dtype=np.int64)
    union = gt.astype(np.float64) + pred.astype(np.float64) - diagonal
    iou = np.divide(diagonal, union, out=np.full(CLASS_COUNT, np.nan), where=union > 0)
    class_accuracy = np.divide(diagonal, gt, out=np.full(CLASS_COUNT, np.nan), where=gt > 0)
    computed = {"mIoU": float(np.nanmean(iou)),
                "pixel_accuracy": float(diagonal.sum() / VALID_PIXEL_COUNT),
                "mean_accuracy": float(np.nanmean(class_accuracy))}
    for key, value in computed.items():
        _close(metrics[key], value, key)
        _close(metrics[key + "_percent"], value * 100, key + "_percent", atol=1e-10)
    _close(metrics["per_class_iou"], iou, "per_class_iou")
    _close(metrics["per_class_iou_percent"], iou * 100, "per_class_iou_percent", atol=1e-10)
    per_class = _csv(evaluation / "per_class_iou.csv",
                     ["class_id", "class_name", "IoU_fraction", "IoU_percent"])
    _require([_integer(row["class_id"], "class_id") for row in per_class] ==
             list(range(CLASS_COUNT)), "per-class CSV IDs mismatch")
    _require([row["class_name"] for row in per_class] == names,
             "per-class CSV names/order mismatch")
    _close([row["IoU_fraction"] for row in per_class], iou, "CSV per-class fractions")
    _close([row["IoU_percent"] for row in per_class], iou * 100,
           "CSV per-class percentages", atol=1e-10)

    checkpoint = Path(expected_checkpoint["path"])
    _require(metrics["checkpoint"] == str(checkpoint), "checkpoint path mismatch")
    expected_size = _integer(expected_checkpoint["size"], "checkpoint size")
    expected_mtime = _integer(expected_checkpoint["mtime_ns"], "checkpoint mtime_ns")
    before = checkpoint.stat()
    _require(before.st_size == expected_size and before.st_mtime_ns == expected_mtime,
             "checkpoint size/mtime changed")
    checkpoint_hash = _sha256(checkpoint)
    after = checkpoint.stat()
    _require(checkpoint_hash == expected_checkpoint["sha256"], "checkpoint SHA-256 changed")
    _require(after.st_size == expected_size and after.st_mtime_ns == expected_mtime,
             "checkpoint changed while hashing")
    return {
        "status": "PASS", "metrics": metrics, "gt_histogram": gt.tolist(),
        "rank_counts": rank_counts, "checkpoint_sha256": checkpoint_hash,
        "test_source_sha256": _sha256(test_source),
        "checks": {
            "exitcode_zero": True, "fixed_epoch_200": True,
            "ordered_rank_shards_match": True, "unique_test_samples": SAMPLE_COUNT,
            "manifest_matches_rank_shards": True, "rank_confusions_match_csv": True,
            "nonnegative_integer_confusions": True, "metrics_independently_recomputed": True,
            "valid_pixel_count": VALID_PIXEL_COUNT, "class_order_matches": True,
            "checkpoint_path_size_mtime_sha256_unchanged": True,
        },
    }
