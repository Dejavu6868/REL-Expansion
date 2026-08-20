#!/usr/bin/env python3
"""Resume-capable full-manifest geometry and normal preflight for REL+ v2.1."""

import argparse
import csv
import json
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rel_plus.camera import (
    gravity_in_camera,
    load_stanford_s2d_camera_geometry,
    resize_camera_geometry,
)
from rel_plus.constants import REL_PLUS_V2_1_PROTOCOL_ID, REL_PLUS_V2_NORMAL_RADIUS
from rel_plus.depth import decode_stanford_s2d_depth, resize_raw_depth_nearest
from rel_plus.geometry import GravityAlignmentSingularity, align_points_and_normals_to_gravity
from rel_plus.profiles import DatasetCameraProfile, STANFORD_S2D_PROFILE
from rel_plus.source_helpers import estimate_source_perspective_normals


STRUCTURAL_FAILURES = {
    "DATASET_PROFILE_MISMATCH",
    "PROTOCOL_ID_MISMATCH",
    "FILE_PAIR_MISSING",
    "DEPTH_DTYPE_OR_SHAPE_FAILURE",
    "DEPTH_NATIVE_SHAPE_MISMATCH",
    "CANONICAL_LABEL_FAILURE",
    "K_OR_POSE_FAILURE",
    "GRAVITY_SINGULARITY",
    "ALL_DEPTH_INVALID",
    "ALL_NORMAL_NONFINITE",
}

_WORKER_PROFILE = None
_WORKER_SEMANTIC_IDS = None


def _worker_init(profile, semantic_ids):
    global _WORKER_PROFILE, _WORKER_SEMANTIC_IDS
    _WORKER_PROFILE = profile
    _WORKER_SEMANTIC_IDS = semantic_ids
    cv2.setNumThreads(1)


def _worker_scan(row):
    return scan_row(row, _WORKER_PROFILE, _WORKER_SEMANTIC_IDS)


def _semantic_ids(path):
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_name = {name: int(index) for index, name in payload["stored_ids"].items()}
    return {
        name: by_name[name]
        for name in ("floor", "ceiling", "wall")
        if name in by_name
    }


def _base_result(row):
    result = dict(row)
    result.update(
        {
            "depth_dtype": "UNAVAILABLE",
            "native_depth_shape": "UNAVAILABLE",
            "canonical_depth_shape": "UNAVAILABLE",
            "principal_x": "",
            "principal_y": "",
            "skew_xy": "",
            "skew_yx": "",
            "camera_center": "[]",
            "gravity_camera": "[]",
            "gravity_alignment_angle_deg": "",
            "depth_invalid_ratio": "",
            "normal_nonfinite_ratio": "",
            "zero_normal_ratio": "",
            "low_support_ratio": "",
            "normal_quality_ratio": "",
            "floor_ratio": "",
            "ceiling_ratio": "",
            "wall_ratio": "",
            "status": "FAIL",
            "reasons": "",
        }
    )
    return result


def scan_row(row, dataset_profile=STANFORD_S2D_PROFILE, semantic_ids=None):
    result = _base_result(row)
    reasons = []
    semantic_ids = semantic_ids or {}
    if row.get("dataset_profile") not in (None, "", dataset_profile.name):
        reasons.append("DATASET_PROFILE_MISMATCH")
    if row.get("protocol_id") not in (None, "", REL_PLUS_V2_1_PROTOCOL_ID):
        reasons.append("PROTOCOL_ID_MISMATCH")
    required_paths = (
        "rgb_path",
        "label_path",
        "depth_path",
        "camera_metadata_path",
    )
    for field in required_paths:
        if not row.get(field) or not Path(row[field]).is_file():
            reasons.append("FILE_PAIR_MISSING:{}".format(field))

    depth_path = Path(row.get("depth_path", ""))
    raw_native = (
        cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_path.is_file()
        else None
    )
    if raw_native is None or raw_native.ndim != 2 or raw_native.dtype != np.uint16:
        reasons.append("DEPTH_DTYPE_OR_SHAPE_FAILURE")
    else:
        result["depth_dtype"] = str(raw_native.dtype)
        result["native_depth_shape"] = "{}x{}".format(*raw_native.shape)
        if tuple(raw_native.shape) != dataset_profile.native_image_shape:
            reasons.append("DEPTH_NATIVE_SHAPE_MISMATCH")

    label_path = Path(row.get("label_path", ""))
    label = (
        cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label_path.is_file()
        else None
    )
    if (
        label is None
        or label.ndim != 2
        or tuple(label.shape) != dataset_profile.canonical_image_shape
    ):
        reasons.append("CANONICAL_LABEL_FAILURE")
        label = None

    camera = None
    if (
        Path(row.get("camera_metadata_path", "")).is_file()
        and raw_native is not None
        and tuple(raw_native.shape) == dataset_profile.native_image_shape
    ):
        try:
            camera = load_stanford_s2d_camera_geometry(
                row["camera_metadata_path"], dataset_profile.native_image_shape
            )
            dataset_profile.assert_camera_reference(camera)
            centre = -camera.R_world_to_camera.T @ camera.t_world_to_camera
            gravity = gravity_in_camera(camera.R_world_to_camera)
            probe = gravity.reshape(1, 1, 3)
            align_points_and_normals_to_gravity(
                probe, probe, gravity, sample_id=row.get("sample_id", "<unknown>")
            )
            result["principal_x"] = float(camera.K_json[0, 2])
            result["principal_y"] = float(camera.K_json[1, 2])
            result["skew_xy"] = float(camera.K_json[0, 1])
            result["skew_yx"] = float(camera.K_json[1, 0])
            result["camera_center"] = json.dumps(centre.tolist())
            result["gravity_camera"] = json.dumps(gravity.tolist())
            result["gravity_alignment_angle_deg"] = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            gravity
                            @ np.array([0.0, 0.0, -1.0], dtype=np.float64),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
        except GravityAlignmentSingularity:
            reasons.append("GRAVITY_SINGULARITY")
        except (OSError, ValueError, json.JSONDecodeError):
            reasons.append("K_OR_POSE_FAILURE")
    elif Path(row.get("camera_metadata_path", "")).is_file():
        reasons.append("K_OR_POSE_FAILURE")

    if (
        raw_native is not None
        and tuple(raw_native.shape) == dataset_profile.native_image_shape
        and camera is not None
    ):
        raw_canonical = resize_raw_depth_nearest(
            raw_native, dataset_profile.canonical_image_shape
        )
        camera_canonical = resize_camera_geometry(
            camera, dataset_profile.canonical_image_shape
        )
        camera_canonical.assert_matches_image_shape(raw_canonical.shape)
        result["canonical_depth_shape"] = "{}x{}".format(*raw_canonical.shape)
        depth_m, depth_valid = decode_stanford_s2d_depth(raw_canonical)
        result["depth_invalid_ratio"] = float(1.0 - np.mean(depth_valid))
        if not np.any(depth_valid):
            reasons.append("ALL_DEPTH_INVALID")
        with np.errstate(divide="ignore", invalid="ignore"):
            normals, diagnostics = estimate_source_perspective_normals(
                depth_m,
                depth_valid,
                camera_canonical.K_json,
                radius=REL_PLUS_V2_NORMAL_RADIUS,
            )
        ratios = diagnostics.ratios(depth_valid)
        result["normal_nonfinite_ratio"] = ratios["normal_invalid_ratio"]
        result["zero_normal_ratio"] = ratios["zero_normal_ratio"]
        result["low_support_ratio"] = ratios["low_support_ratio"]
        result["normal_quality_ratio"] = ratios["normal_quality_ratio"]
        if np.any(depth_valid) and not np.any(
            depth_valid & np.all(np.isfinite(normals), axis=2)
        ):
            reasons.append("ALL_NORMAL_NONFINITE")

        if label is not None and semantic_ids:
            denominator = float(label.size)
            for name in ("floor", "ceiling", "wall"):
                result[name + "_ratio"] = float(
                    np.count_nonzero(label == semantic_ids[name]) / denominator
                )

    result["status"] = "PASS" if not reasons else "FAIL"
    result["reasons"] = ";".join(sorted(set(reasons)))
    return result


def scan_manifest(path, dataset_profile=STANFORD_S2D_PROFILE, semantic_ids=None):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("manifest is empty")
    return [
        scan_row(row, dataset_profile=dataset_profile, semantic_ids=semantic_ids)
        for row in rows
    ]


def _distribution(rows, field):
    values = [
        float(row[field])
        for row in rows
        if row.get(field) not in (None, "", "UNAVAILABLE")
    ]
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _ranked(rows, field, reverse=True, limit=20):
    available = [
        row for row in rows if row.get(field) not in (None, "", "UNAVAILABLE")
    ]
    available.sort(
        key=lambda row: (float(row[field]), row["sample_id"]), reverse=reverse
    )
    return [
        {"sample_id": row["sample_id"], "value": float(row[field])}
        for row in available[:limit]
    ]


def summarize(rows):
    failures = [row for row in rows if row["status"] == "FAIL"]
    reasons = {}
    for row in failures:
        for reason in row["reasons"].split(";"):
            category = reason.split(":", 1)[0]
            reasons[category] = reasons.get(category, 0) + 1
    status = "PASS" if not failures else "FAIL"
    return {
        "status": status,
        "protocol_id": REL_PLUS_V2_1_PROTOCOL_ID,
        "dataset_profile": STANFORD_S2D_PROFILE.name,
        "sample_count": len(rows),
        "pass_count": len(rows) - len(failures),
        "fail_count": len(failures),
        "failure_reason_counts": dict(sorted(reasons.items())),
        "area_counts": {
            area: {
                "sample_count": sum(row.get("area") == area for row in rows),
                "fail_count": sum(
                    row.get("area") == area and row["status"] == "FAIL"
                    for row in rows
                ),
            }
            for area in sorted({row.get("area", "") for row in rows})
        },
        "distributions": {
            field: _distribution(rows, field)
            for field in (
                "depth_invalid_ratio",
                "normal_nonfinite_ratio",
                "zero_normal_ratio",
                "low_support_ratio",
                "normal_quality_ratio",
                "gravity_alignment_angle_deg",
            )
        },
        "top_outliers": {
            "depth_invalid_ratio": _ranked(rows, "depth_invalid_ratio"),
            "normal_nonfinite_ratio": _ranked(rows, "normal_nonfinite_ratio"),
            "zero_normal_ratio": _ranked(rows, "zero_normal_ratio"),
            "low_support_ratio": _ranked(rows, "low_support_ratio"),
            "normal_quality_ratio_low": _ranked(
                rows, "normal_quality_ratio", reverse=False
            ),
        },
        "pilot_cache_status": (
            "NOT_READY_FOR_PILOT_CACHE"
            if failures
            else "READY_FOR_PILOT_CACHE_WITH_REVIEW"
        ),
    }


def _read_csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_preflight(
    manifest_path,
    output_path,
    *,
    dataset_profile=STANFORD_S2D_PROFILE,
    semantic_ids=None,
    workers=1,
    resume=False
):
    manifest_rows = _read_csv(manifest_path)
    if not manifest_rows:
        raise ValueError("manifest is empty")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.csv")
    existing = _read_csv(partial) if resume else []
    by_id = {row["sample_id"]: row for row in existing}
    pending = [row for row in manifest_rows if row["sample_id"] not in by_id]
    mode = "a" if existing else "w"
    handle = None
    writer = None
    try:
        if pending:
            if workers > 1:
                pool = mp.Pool(
                    processes=workers,
                    initializer=_worker_init,
                    initargs=(dataset_profile, semantic_ids or {}),
                )
                iterator = pool.imap_unordered(_worker_scan, pending, chunksize=8)
            else:
                pool = None
                iterator = (
                    scan_row(row, dataset_profile, semantic_ids or {}) for row in pending
                )
            for index, result in enumerate(iterator, 1):
                if writer is None:
                    handle = partial.open(mode, encoding="utf-8", newline="")
                    writer = csv.DictWriter(handle, fieldnames=list(result))
                    if not existing:
                        writer.writeheader()
                writer.writerow(result)
                by_id[result["sample_id"]] = result
                if index % 100 == 0:
                    handle.flush()
                if index % 1000 == 0:
                    print(
                        json.dumps(
                            {
                                "status": "RUNNING",
                                "completed_this_run": index,
                                "completed_total": len(by_id),
                                "manifest_total": len(manifest_rows),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            if pool is not None:
                pool.close()
                pool.join()
    finally:
        if handle is not None:
            handle.close()

    missing = [row["sample_id"] for row in manifest_rows if row["sample_id"] not in by_id]
    if missing:
        raise RuntimeError("preflight incomplete; {} samples missing".format(len(missing)))
    ordered = [by_id[row["sample_id"]] for row in manifest_rows]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    summary = summarize(ordered)
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-mapping")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run_preflight(
        args.manifest,
        args.output,
        semantic_ids=_semantic_ids(args.class_mapping),
        workers=max(1, args.workers),
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
