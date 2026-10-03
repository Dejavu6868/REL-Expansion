#!/usr/bin/env python3
"""Cross-projection consistency check: perspective REL+ v2.1 vs ERP REL.

For each panorama, render gnomonic (pinhole) crops from the ERP depth with a
known rotation, run the frozen perspective REL+ generator on each crop, run
the original ERP REL on the full panorama, sample the ERP result at the same
3D rays, and compare.

Two groups of quantities are reported separately:

* raw geometry (height, horizontal radius, EGVIA angle before the height
  blend, LOA): these must agree if both code paths describe the same 3D
  scene; a disagreement is a geometry/convention bug;
* encoded bytes (EGVIA, LOA, ReD): these also include the per-image ReD
  min/max and height 1st/99th-percentile normalisation, which differs between
  a crop and a full 360-degree panorama by construction.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT / "0927调参结果" / "selected_recipe" / "code" / "training" / "source"
)

DEFAULT_YAWS = (0.0, 90.0, 180.0, 270.0)
DEFAULT_PITCHES = (-20.0, 0.0, 20.0)
DEFAULT_THRESHOLDS = {
    "height_cm_median": 2.0,
    "radius_cm_median": 2.0,
    "egvia_angle_deg_median": 2.0,
    "loa_deg_median": 2.0,
}
HHA_THRESHOLDS = {
    "angle_deg_median": 2.0,
    "height_cm_median": 2.0,
}


def import_frozen_source(source_root):
    """Put the frozen training snapshot first on sys.path and import it."""
    source_root = Path(source_root).resolve()
    sys.dont_write_bytecode = True  # keep the frozen archive's file set unchanged
    if not (source_root / "rel_plus" / "generator.py").is_file():
        raise FileNotFoundError(
            "{} is not a frozen REL+ source snapshot".format(source_root)
        )
    sys.path.insert(0, str(source_root))
    from rel_plus.camera import CameraGeometry
    from rel_plus.generator import generate_rel_plus_v2_1
    from third_party.rel_original.rel import getREL
    from third_party.rel_original.rgbd_util import (
        computeNormalsSquareSupport_ERP,
        getPointCloud_ERP,
        processDepthImage_ERP,
    )

    return {
        "CameraGeometry": CameraGeometry,
        "generate_rel_plus_v2_1": generate_rel_plus_v2_1,
        "getREL": getREL,
        "processDepthImage_ERP": processDepthImage_ERP,
        "getPointCloud_ERP": getPointCloud_ERP,
        "computeNormalsSquareSupport_ERP": computeNormalsSquareSupport_ERP,
    }


def decode_erp_depth(raw_depth):
    """Decode uint16 ERP depth exactly as ``rel.getImage`` does (metres)."""
    raw = np.asarray(raw_depth)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError("ERP depth must be a 2D uint16 array")
    shifted = raw + np.uint16(1)  # 65535 wraps to 0, the source missing value
    return shifted.astype(np.float32) / 512.0


def erp_directions(height, width):
    """Unit ray per ERP pixel in the ``getPointCloud_ERP`` convention."""
    u = np.arange(width)[np.newaxis, :]
    v = np.arange(height)[:, np.newaxis]
    phi = (u / width) * 2 * np.pi - np.pi
    theta = (0.5 - v / height) * np.pi
    return np.stack(
        np.broadcast_arrays(
            np.cos(theta) * np.sin(phi),
            np.cos(theta) * np.cos(phi),
            np.sin(theta),
        ),
        axis=2,
    )


def crop_camera_to_world(yaw_deg, pitch_deg):
    """Rotation with columns [right, down, forward] in the ERP (z-up) frame."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    forward = np.array(
        [np.cos(pitch) * np.sin(yaw), np.cos(pitch) * np.cos(yaw), np.sin(pitch)]
    )
    right = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)


def crop_intrinsics(size, fov_deg):
    """Square pinhole K in the Stanford JSON half-pixel convention."""
    focal = (size / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return np.array(
        [[focal, 0.0, size / 2.0], [0.0, focal, size / 2.0], [0.0, 0.0, 1.0]]
    )


def crop_rays(size, k_json, camera_to_world):
    """Camera-frame rays (z=1) and their unit world directions per pixel."""
    rows, columns = np.indices((size, size), dtype=np.float64)
    rays_camera = np.stack(
        [
            (columns + 0.5 - k_json[0, 2]) / k_json[0, 0],
            (rows + 0.5 - k_json[1, 2]) / k_json[1, 1],
            np.ones((size, size)),
        ],
        axis=2,
    )
    rays_world = np.einsum("ij,klj->kli", camera_to_world, rays_camera)
    rays_world /= np.linalg.norm(rays_world, axis=2, keepdims=True)
    return rays_camera, rays_world


def erp_indices_for_directions(directions, height, width):
    """Nearest ERP pixel (row, column) for each unit world direction."""
    phi = np.arctan2(directions[..., 0], directions[..., 1])
    theta = np.arcsin(np.clip(directions[..., 2], -1.0, 1.0))
    columns = np.rint((phi + np.pi) / (2 * np.pi) * width).astype(np.int64) % width
    rows = np.clip(
        np.rint((0.5 - theta / np.pi) * height).astype(np.int64), 0, height - 1
    )
    return rows, columns


def render_crop_raw_depth(depth_erp_m, rows, columns, rays_camera):
    """Turn sampled ERP range into a Stanford-encoded uint16 z-depth crop."""
    range_m = depth_erp_m[rows, columns].astype(np.float64)
    valid = range_m > 0
    z_m = range_m / np.linalg.norm(rays_camera, axis=2)
    raw = np.zeros(z_m.shape, dtype=np.uint16)
    raw[valid] = np.clip(np.rint(z_m[valid] * 512.0), 1, 65534).astype(np.uint16)
    return raw


def erp_reference(depth_erp_m, frozen):
    """Raw ERP-REL geometry plus the encoded ``getREL`` bytes."""
    missing = depth_erp_m == 0
    points, normals, gravity_rotation = frozen["processDepthImage_ERP"](
        depth_erp_m * 100, missing
    )
    height, width = depth_erp_m.shape
    phi = (np.arange(width) / width) * 2 * np.pi - np.pi
    hcos = normals[:, :, 0] * np.cos(phi)[np.newaxis] - normals[
        :, :, 1
    ] * np.sin(phi)[np.newaxis]
    hcos = np.clip(np.nan_to_num(hcos, nan=0.0), -1.0, 1.0)
    normal_z = np.clip(-normals[:, :, 2], -1.0, 1.0)
    return {
        "valid": ~missing,
        "height_cm": points[:, :, 2],
        "radius_cm": np.hypot(points[:, :, 0], points[:, :, 1]),
        "egvia_angle_deg": np.degrees(np.arccos(normal_z)),
        "loa_deg": np.degrees(np.arccos(hcos)),
        "normals": normals,
        "encoded": frozen["getREL"](depth_erp_m),
        "estimated_gravity_rotation_deg": [float(v) for v in gravity_rotation],
    }


def stable_mask(range_m, valid, relative_jump=0.10, radius=3):
    """Valid pixels at least ``radius`` pixels from a depth discontinuity.

    A discontinuity is an adjacent-pixel range jump above ``relative_jump``;
    ``radius`` covers the normal estimator's 5x5 support plus one pixel.
    """
    filled = np.where(valid, range_m, 0).astype(np.float64)
    scale = np.maximum(filled, 1e-6)
    jump = np.zeros(filled.shape, dtype=bool)
    vertical = np.abs(np.diff(filled, axis=0)) > relative_jump * scale[:-1]
    horizontal = np.abs(np.diff(filled, axis=1)) > relative_jump * scale[:, :-1]
    jump[:-1] |= vertical
    jump[1:] |= vertical
    jump[:, :-1] |= horizontal
    jump[:, 1:] |= horizontal
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    near_jump = cv2.dilate(jump.astype(np.uint8), kernel).astype(bool)
    eroded_valid = cv2.erode(valid.astype(np.uint8), kernel).astype(bool)
    return eroded_valid & ~near_jump


def summarize(values):
    values = np.abs(np.asarray(values, dtype=np.float64))
    if values.size == 0:
        return {"count": 0, "median": None, "p95": None}
    if not np.isfinite(values).all():
        return {"count": int(values.size), "median": None, "p95": None}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def render_crop(depth_erp_m, yaw, pitch, size, fov):
    """Crop geometry and its Stanford-encoded z-depth, sampled from the ERP."""
    height, width = depth_erp_m.shape
    k_json = crop_intrinsics(size, fov)
    camera_to_world = crop_camera_to_world(yaw, pitch)
    rays_camera, rays_world = crop_rays(size, k_json, camera_to_world)
    rows, columns = erp_indices_for_directions(rays_world, height, width)
    raw_crop = render_crop_raw_depth(depth_erp_m, rows, columns, rays_camera)
    return k_json, camera_to_world, rows, columns, raw_crop


def compare_crop(depth_erp_m, reference, frozen, yaw, pitch, size, fov):
    """Generate one crop through the perspective path and compare it."""
    k_json, camera_to_world, rows, columns, raw_crop = render_crop(
        depth_erp_m, yaw, pitch, size, fov
    )
    camera = frozen["CameraGeometry"].from_json_k(
        k_json,
        (size, size),
        camera_to_world.T,
        sample_id="yaw{:g}_pitch{:g}".format(yaw, pitch),
    )
    rel_plus, debug = frozen["generate_rel_plus_v2_1"](
        raw_crop, camera, return_debug=True
    )

    crop_valid = np.asarray(debug["encoding_valid_mask"], dtype=bool)
    sampled_range = depth_erp_m[rows, columns]
    mask = crop_valid & reference["valid"][rows, columns]
    mask &= stable_mask(sampled_range, mask)
    mask &= np.isfinite(debug["normals_aligned"]).all(axis=2)
    mask &= np.isfinite(reference["normals"][rows, columns]).all(axis=2)

    perspective_egvia_deg = np.degrees(
        np.arccos(np.clip(-debug["normals_aligned"][:, :, 2], -1.0, 1.0))
    )
    perspective_loa_deg = np.degrees(
        np.arccos(np.clip(np.nan_to_num(debug["hcos"], nan=0.0), -1.0, 1.0))
    )
    sample = lambda name: reference[name][rows, columns]
    encoded_erp = reference["encoded"][rows, columns].astype(np.int16)
    encoded_crop = rel_plus.astype(np.int16)

    report = {
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "compared_pixels": int(mask.sum()),
        "crop_valid_pixels": int(crop_valid.sum()),
        "raw": {
            "height_cm": summarize(
                debug["height_raw_cm"][mask] - sample("height_cm")[mask]
            ),
            "radius_cm": summarize(
                debug["red_raw_cm"][mask] - sample("radius_cm")[mask]
            ),
            "egvia_angle_deg": summarize(
                perspective_egvia_deg[mask] - sample("egvia_angle_deg")[mask]
            ),
            "loa_deg": summarize(
                perspective_loa_deg[mask] - sample("loa_deg")[mask]
            ),
        },
        "encoded_bytes": {
            channel: summarize(
                encoded_crop[:, :, index][mask] - encoded_erp[:, :, index][mask]
            )
            for index, channel in enumerate(("EGVIA", "LOA", "ReD"))
        },
        "normalisation": {
            "crop_radius_cm_min_max": [
                float(debug["red_raw_cm"].min()),
                float(debug["red_raw_cm"].max()),
            ],
            "crop_height_cm_p1_p99": [
                float(np.percentile(debug["height_raw_cm"], 1)),
                float(np.percentile(debug["height_raw_cm"], 99)),
            ],
        },
    }
    return report


def compare_crop_hha(depth_erp_m, reference, yaw, pitch, size, fov):
    """Depth2HHA on one crop against the ERP HHA sampled at the same rays."""
    import hha

    k_json, camera_to_world, rows, columns, raw_crop = render_crop(
        depth_erp_m, yaw, pitch, size, fov
    )
    crop_valid = (raw_crop != 0) & (raw_crop != 65535)
    encoded, debug = hha.pinhole_hha(raw_crop / 512.0, crop_valid, k_json)
    sampled_range = depth_erp_m[rows, columns]
    mask = crop_valid & reference["valid"][rows, columns]
    mask &= stable_mask(sampled_range, mask)
    mask &= np.isfinite(debug["normals"]).all(axis=2)
    mask &= np.isfinite(reference["normals"][rows, columns]).all(axis=2)
    sample = lambda name: reference[name][rows, columns]
    gravity_world = camera_to_world @ debug["gravity_camera"]
    encoded_erp = reference["encoded"][rows, columns].astype(np.int16)
    return {
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "compared_pixels": int(mask.sum()),
        "crop_valid_pixels": int(crop_valid.sum()),
        "crop_gravity_error_deg": float(
            np.degrees(np.arccos(np.clip(-gravity_world[2], -1.0, 1.0)))
        ),
        "raw": {
            "angle_deg": summarize(debug["angle_deg"][mask] - sample("angle_deg")[mask]),
            "height_cm": summarize(debug["height_cm"][mask] - sample("height_cm")[mask]),
        },
        "disparity_depth_ratio_crop_over_erp": summarize(
            debug["disparity_depth_cm"][mask] / sample("disparity_depth_cm")[mask]
        ),
        "encoded_bytes": {
            channel: summarize(
                encoded.astype(np.int16)[:, :, index][mask]
                - encoded_erp[:, :, index][mask]
            )
            for index, channel in enumerate(("angle", "height", "disparity"))
        },
    }


def check_panorama_hha(depth_erp_m, frozen, *, yaws, pitches, size, fov):
    import hha

    encoded, debug = hha.erp_hha(depth_erp_m, frozen)
    reference = dict(debug, encoded=encoded, valid=depth_erp_m > 0)
    crops = [
        compare_crop_hha(depth_erp_m, reference, yaw, pitch, size, fov)
        for yaw in yaws
        for pitch in pitches
    ]
    failures = []
    for crop in crops:
        for key, limit in HHA_THRESHOLDS.items():
            quantity = key[: -len("_median")]
            median = crop["raw"][quantity]["median"]
            if median is None or not np.isfinite(median) or median > limit:
                failures.append(
                    "HHA yaw={yaw_deg:g} pitch={pitch_deg:g}: ".format(**crop)
                    + "{} median {} > {}".format(quantity, median, limit)
                )
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "erp_gravity_error_deg": float(
            np.degrees(np.arccos(np.clip(-debug["gravity_erp"][2], -1.0, 1.0)))
        ),
        "crops": crops,
    }


def check_panorama(raw_depth, frozen, *, yaws, pitches, size, fov, thresholds, with_hha=False):
    depth_erp_m = decode_erp_depth(raw_depth)
    reference = erp_reference(depth_erp_m, frozen)
    crops = [
        compare_crop(depth_erp_m, reference, frozen, yaw, pitch, size, fov)
        for yaw in yaws
        for pitch in pitches
    ]
    failures = []
    for crop in crops:
        for key, limit in thresholds.items():
            quantity = key[: -len("_median")]
            median = crop["raw"][quantity]["median"]
            if median is None or not np.isfinite(median) or median > limit:
                failures.append(
                    "yaw={yaw_deg:g} pitch={pitch_deg:g}: ".format(**crop)
                    + "{} median {} > {}".format(quantity, median, limit)
                )
    points = reference["height_cm"][reference["valid"]]
    radius = reference["radius_cm"][reference["valid"]]
    hha_report = None
    if with_hha:
        hha_report = check_panorama_hha(
            depth_erp_m, frozen, yaws=yaws, pitches=pitches, size=size, fov=fov
        )
        failures += hha_report["failures"]
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "hha": hha_report,
        "erp_estimated_gravity_rotation_deg": reference[
            "estimated_gravity_rotation_deg"
        ],
        "erp_normalisation": {
            "radius_cm_min_max": [
                float(reference["radius_cm"].min()),
                float(reference["radius_cm"].max()),
            ],
            "height_cm_p1_p99": [
                float(np.percentile(reference["height_cm"], 1)),
                float(np.percentile(reference["height_cm"], 99)),
            ],
            "valid_height_cm_range": [float(points.min()), float(points.max())],
            "valid_radius_cm_range": [float(radius.min()), float(radius.max())],
        },
        "crops": crops,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "depth", nargs="+", type=Path, help="Stanford2D3D pano/depth/*_depth.png"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--crop-size", type=int, default=480)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--yaws", type=float, nargs="+", default=DEFAULT_YAWS)
    parser.add_argument("--pitches", type=float, nargs="+", default=DEFAULT_PITCHES)
    parser.add_argument("--hha", action="store_true", help="also check ERP HHA")
    args = parser.parse_args()

    frozen = import_frozen_source(args.source_root)
    results = {}
    for path in args.depth:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)
        results[str(path)] = check_panorama(
            raw,
            frozen,
            yaws=args.yaws,
            pitches=args.pitches,
            size=args.crop_size,
            fov=args.fov,
            thresholds=DEFAULT_THRESHOLDS,
            with_hha=args.hha,
        )
        print("{}: {}".format(path.name, results[str(path)]["status"]))
    summary = {
        "status": "PASS"
        if all(item["status"] == "PASS" for item in results.values())
        else "FAIL",
        "thresholds": DEFAULT_THRESHOLDS,
        "hha_thresholds": HHA_THRESHOLDS if args.hha else None,
        "crop_size": args.crop_size,
        "fov_deg": args.fov,
        "panoramas": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("overall: {}".format(summary["status"]))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
