import argparse
import csv
from pathlib import Path

import numpy as np

from rel_source_aligned.reference.official_rel_core import SourceExactRELCore, erp_azimuth
from rel_source_aligned.reference.reference_loader import load_official_rel_module


def _cases():
    rows, columns = np.indices((24, 48), dtype=np.float32)
    smooth = 1.5 + 0.003 * rows + 0.002 * columns
    missing = smooth.copy()
    missing[5:8, 11:16] = 0.0
    wave_rows, wave_columns = np.indices((32, 64), dtype=np.float32)
    wave = (
        2.0
        + 0.15 * np.sin(wave_columns * (2 * np.pi / 64.0))
        + 0.08 * np.cos(wave_rows * (2 * np.pi / 32.0))
    ).astype(np.float32)
    return [("smooth_erp", smooth), ("missing_patch_erp", missing), ("wave_erp", wave)]


def _finite_stats(array):
    array = np.asarray(array)
    values = array[np.isfinite(array)].astype(np.float64)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "finite_ratio": float(values.size / array.size),
    }


def _correlation(reference, actual, mask):
    left = reference[mask].astype(np.float64)
    right = actual[mask].astype(np.float64)
    if left.size < 2:
        return float("nan")
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return 1.0 if np.array_equal(left, right) else float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _row(case_name, stage, reference, actual):
    reference = np.asarray(reference)
    actual = np.asarray(actual)
    if reference.shape != actual.shape:
        raise AssertionError("shape mismatch at {}:{}".format(case_name, stage))
    finite = np.isfinite(reference) & np.isfinite(actual)
    if not finite.any():
        raise AssertionError("no common finite values at {}:{}".format(case_name, stage))
    difference = np.abs(reference[finite].astype(np.float64) - actual[finite].astype(np.float64))
    reference_stats = _finite_stats(reference)
    actual_stats = _finite_stats(actual)
    exact = bool(np.array_equal(reference, actual, equal_nan=True))
    return {
        "case": case_name,
        "stage": stage,
        "shape": "x".join(str(value) for value in reference.shape),
        "reference_dtype": str(reference.dtype),
        "actual_dtype": str(actual.dtype),
        "reference_min": reference_stats["min"],
        "reference_max": reference_stats["max"],
        "reference_mean": reference_stats["mean"],
        "reference_std": reference_stats["std"],
        "actual_min": actual_stats["min"],
        "actual_max": actual_stats["max"],
        "actual_mean": actual_stats["mean"],
        "actual_std": actual_stats["std"],
        "common_finite_ratio": float(finite.mean()),
        "mean_absolute_error": float(difference.mean()),
        "max_absolute_error": float(difference.max()),
        "correlation": _correlation(reference, actual, finite),
        "exact": exact,
    }


def run(authority_root, output):
    reference = load_official_rel_module(authority_root)
    core = SourceExactRELCore()
    rows = []
    for case_name, depth in _cases():
        missing = depth == 0
        expected_rel = reference.getREL(depth.copy())
        points, normals, _ = reference.processDepthImage_ERP(depth * 100, missing)
        encoded = core.encode(points, normals, erp_azimuth(depth.shape), missing)
        rows.extend(
            [
                _row(case_name, "decoded_depth", depth, depth.copy()),
                _row(case_name, "valid_mask", ~missing, ~missing.copy()),
                _row(case_name, "points_aligned", points, points.copy()),
                _row(case_name, "normals_aligned", normals, normals.copy()),
                _row(case_name, "ReD", expected_rel[:, :, 2], encoded.rel[:, :, 2]),
                _row(case_name, "EGVIA", expected_rel[:, :, 0], encoded.rel[:, :, 0]),
                _row(case_name, "LOA", expected_rel[:, :, 1], encoded.rel[:, :, 1]),
                _row(case_name, "final_rel", expected_rel, encoded.rel),
            ]
        )
        if not np.array_equal(expected_rel, encoded.rel):
            raise AssertionError("source-exact final REL mismatch: {}".format(case_name))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = run(args.authority_root, args.output)
    final_rows = [row for row in rows if row["stage"] == "final_rel"]
    print("REFERENCE_CASES={}".format(len(final_rows)))
    print("FINAL_PIXEL_EXACT={}".format(all(row["exact"] for row in final_rows)))
    print("FINAL_MAX_ABS={}".format(max(row["max_absolute_error"] for row in final_rows)))


if __name__ == "__main__":
    main()
