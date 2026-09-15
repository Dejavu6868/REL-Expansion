#!/usr/bin/env python3
"""Run controlled encoder-level v1-to-v2 byte-difference accounting."""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rel_plus.encoding import encode_rel_channels, perspective_tangent_field
from rel_plus.validation.v1_v2_diff import summarize_difference


def load_v1_package(v1_root):
    package_path = Path(v1_root) / "rel_plus"
    spec = importlib.util.spec_from_file_location(
        "rel_plus_v1",
        str(package_path / "__init__.py"),
        submodule_search_locations=[str(package_path)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return importlib.import_module("rel_plus_v1.encoding")


def encode(encoder, points, normals, valid):
    tangent, _ = perspective_tangent_field(points)
    return encoder.encode_rel_channels(
        points, normals, valid, tangent=tangent, alpha=45.0, lam=0.5
    )[0]


def run_cases(v1_encoder):
    rows, columns = np.indices((7, 9), dtype=np.float64)
    dense_points = np.stack(
        [0.2 + columns, 0.3 + rows, 1.0 + 0.1 * columns + 0.2 * rows], axis=-1
    )
    dense_normals = np.stack(
        [0.2 + 0.01 * columns, -0.3 + 0.02 * rows, np.ones_like(rows)], axis=-1
    )
    dense_normals /= np.linalg.norm(dense_normals, axis=-1, keepdims=True)
    valid = np.ones(rows.shape, dtype=bool)
    cases = []

    def add(name, points, normals, v1_valid, v2_valid, classification):
        before = encode(v1_encoder, points, normals, v1_valid)
        after = encode(sys.modules["rel_plus.encoding"], points * 100.0, normals, v2_valid)
        cases.append(summarize_difference(name, before, after, classification))

    add("dense_valid", dense_points, dense_normals, valid, valid, "UNCHANGED_DENSE_VALID")
    invalid = valid.copy()
    invalid[0, 0] = False
    add("depth_invalid", dense_points, dense_normals, invalid, invalid, "UNCHANGED_DENSE_VALID")
    nan_normals = dense_normals.copy()
    nan_normals[2, 3] = np.nan
    v1_normal_valid = valid.copy()
    v1_normal_valid[2, 3] = False
    add(
        "nan_normal", dense_points, nan_normals, v1_normal_valid, valid,
        "INTENTIONAL_NORMAL_MASK_FIX"
    )
    zero_normals = dense_normals.copy()
    zero_normals[3, 4] = 0.0
    add("zero_normal", dense_points, zero_normals, valid, valid, "UNCHANGED_DENSE_VALID")

    degenerate_points = np.array([[[0.01, 0.02, 0.03]]], dtype=np.float64)
    degenerate_normal = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float64)
    add(
        "single_valid_pixel", degenerate_points, degenerate_normal,
        np.ones((1, 1), bool), np.ones((1, 1), bool),
        "INTENTIONAL_SOURCE_UNIT_FIX"
    )
    constant_height = dense_points.copy()
    constant_height[..., 2] = 0.02
    add(
        "constant_height", constant_height, dense_normals, valid, valid,
        "INTENTIONAL_SOURCE_UNIT_FIX"
    )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    v1_encoder = load_v1_package(args.v1_root)
    cases = run_cases(v1_encoder)
    with (output_root / "v1_v2_diff.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    summary = {
        "status": "PASS" if sum(row["unexpected_difference_count"] for row in cases) == 0 else "FAIL",
        "case_count": len(cases),
        "changed_pixel_count": sum(row["changed_pixel_count"] for row in cases),
        "normal_mask_changed_pixels": sum(
            row["intentional_changed_pixel_count"]
            for row in cases if row["classification"] == "INTENTIONAL_NORMAL_MASK_FIX"
        ),
        "unit_branch_changed_pixels": sum(
            row["intentional_changed_pixel_count"]
            for row in cases if row["classification"] == "INTENTIONAL_SOURCE_UNIT_FIX"
        ),
        "unexpected_difference_count": sum(row["unexpected_difference_count"] for row in cases),
        "cases": cases,
    }
    (output_root / "v1_v2_diff.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
