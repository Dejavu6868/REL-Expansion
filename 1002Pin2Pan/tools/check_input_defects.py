#!/usr/bin/env python3
"""Phase 0 checks: size the REL+ input defects on cached training data, before any retraining.

CPU only. Nothing here writes into a cache, a checkpoint or the frozen source.
Four checks, each matched to an issue in notes/relplus-code-review-2026-10-15.md:

* ReD (issue 1): on pinhole training frames, how often invalid pixels (stored
  at the camera, so horizontal distance 0) force the per-image ReD minimum to
  0, and how far that moves the ReD byte of valid pixels against a valid-only
  min/max. Also the percentiles of horizontal distance in cm, to set a fixed
  metric range. On pinhole crops rendered from panoramas
  (``cross_projection.check_panorama``), how far the crop's ReD byte is from
  the panorama's at the same ray: the current per-image ReD, and a metric ReD
  (distance / R, R = training p99) derived from the raw distance difference.
* Blending (issue 2): ``analyze_invalid_interpolation`` from the frozen
  loader, on the REL+ cache under test and the HHA cache, with one training
  transform per frame shared by both arms. It reports the share of labelled
  pixels whose input the bilinear resize mixes with invalid pixels.
* Flip (issue 4): encodes each frame and its mirror image (mirrored depth,
  mirrored intrinsics and gravity) and checks that the mirror's encoding is the
  flipped encoding with LOA remapped (180 - LOA and 179 - LOA are both scored),
  for v2.1 and REL+ 3.1 bytes.
* Panorama inputs (issues 5 and 6): raw-0 and 65535 counts in panorama depth
  (the ERP decoder turns raw 0 into a valid 2 mm point), and the tilt of the
  ``getGDir`` gravity against the ERP vertical (the S2D panoramas are stored
  level, as ``cross_projection.py`` also assumes), with what that tilt does to
  REL+ 3.1 height (cm) and EGVIA bytes.

No panorama labels are read. Panorama checks cover every area unless
``--areas`` says otherwise; crops use the training areas only.
"""

import argparse
import csv
import json
import multiprocessing
import random
import time
from pathlib import Path

import cv2
import numpy as np

import cross_projection as cp
import eval_pano_transfer as ev
import relplus_variant as rv


TRAIN_AREAS = ("area_1", "area_2", "area_3", "area_4", "area_6")
ALL_AREAS = ("area_1", "area_2", "area_3", "area_4", "area_5a", "area_5b", "area_6")
DISTANCE_BIN_CM = 10.0
DISTANCE_BINS = 300  # 0..3000 cm, last bin open
CROP_YAWS = (0.0, 90.0, 180.0, 270.0)
CROP_PITCHES = (-20.0, 0.0, 20.0)
# Pre-stated in the REL+ fix plan (2026-10-15).
PHASE1_CURRENT_LEVELS = 10.0
PHASE1_METRIC_LEVELS = 3.0
PHASE2_AFFECTED_SHARE = 0.01
_JOB = {}  # set before the worker pool forks


def gt_transform(stored):
    """Frozen ``RGBXDataset._gt_transform``: 0 and 255 to ignore, 1..13 to 0..12."""
    mapped = stored.astype(np.int16) - 1
    mapped[(stored == 0) | (stored == 255)] = 255
    return mapped.astype(np.uint8)


def percentiles(values, qs=(50, 95)):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"p{}".format(q): None for q in qs}
    return {"p{}".format(q): float(np.percentile(values, q)) for q in qs}


# ---------------------------------------------------------------- ReD (issue 1)


def red_statistics(red_raw_cm, red_encoded, valid):
    """ReD byte shift of valid pixels when the min/max ignores invalid pixels."""
    stats = {"invalid_share": float(1.0 - valid.mean()), "red_min_all_cm": float(red_raw_cm.min())}
    histogram = np.zeros(DISTANCE_BINS, dtype=np.int64)
    if not valid.any():
        stats.update(red_min_valid_cm=None, red_max_valid_cm=None, shift_median=None, shift_p95=None)
        return stats, histogram
    distance = red_raw_cm[valid]
    low, high = distance.min(), distance.max()
    scaled = (red_raw_cm - low) * 255.0 / (high - low) if high > low else red_raw_cm.copy()
    valid_only = np.clip(scaled, 0, 255).astype(np.uint8)
    shift = np.abs(valid_only[valid].astype(np.int16) - red_encoded[valid].astype(np.int16))
    bins = np.minimum((distance / DISTANCE_BIN_CM).astype(np.int64), DISTANCE_BINS - 1)
    histogram += np.bincount(bins, minlength=DISTANCE_BINS)
    stats.update(
        red_min_valid_cm=float(low),
        red_max_valid_cm=float(high),
        shift_median=float(np.median(shift)),
        shift_p95=float(np.percentile(shift, 95)),
    )
    return stats, histogram


def histogram_quantile(histogram, q):
    cumulative = np.cumsum(histogram)
    if cumulative[-1] == 0:
        return None
    return float((np.searchsorted(cumulative, q / 100.0 * cumulative[-1]) + 1) * DISTANCE_BIN_CM)


# ---------------------------------------------------------- blending (issue 2)


def blend_statistics(x, valid, label, transform):
    """Frozen ``analyze_invalid_interpolation`` plus the labelled-pixel denominator."""
    from rel_plus.integration.cmx_preprocess import analyze_invalid_interpolation

    n = _JOB["classes"]
    diagnostic = analyze_invalid_interpolation(
        x, valid, transform, label=label, ignore_index=255, num_classes=n
    )
    scaled = cv2.resize(label, (transform.scaled_width, transform.scaled_height), interpolation=cv2.INTER_NEAREST)
    crop = scaled[
        transform.crop_top: transform.crop_top + transform.output_height,
        transform.crop_left: transform.crop_left + transform.output_width,
    ]
    affected = diagnostic["affected_pixel_count"]
    return {
        "output_pixels": transform.output_height * transform.output_width,
        "labelled_pixels": int(np.count_nonzero(crop < n)),
        "affected_pixels": affected,
        "affected_labelled_pixels": diagnostic["affected_label_valid_semantic_count"],
        "deviation_sum": diagnostic["affected_mean_channel_deviation"] * affected,
        "deviation_max": diagnostic["affected_max_channel_deviation"],
        "affected_by_class": np.array(
            [diagnostic["affected_label_class_counts"][str(c)] for c in range(n)], dtype=np.int64
        ),
    }


# --------------------------------------------------------------- flip (issue 4)


def encode_frame(raw, k_json, gravity_camera):
    """``generate_rel_plus_v2`` for any intrinsics and gravity; v2.1 and 3.1 bytes."""
    from rel_plus.camera import backproject_z_depth
    from rel_plus.constants import REL_PLUS_V2_ALPHA, REL_PLUS_V2_LAMBDA, REL_PLUS_V2_NORMAL_RADIUS
    from rel_plus.depth import decode_stanford_s2d_depth
    from rel_plus.encoding import encode_rel_channels, perspective_tangent_field
    from rel_plus.geometry import align_points_and_normals_to_gravity
    from rel_plus.source_helpers import estimate_source_perspective_normals

    depth_m, valid = decode_stanford_s2d_depth(raw)
    points = backproject_z_depth(depth_m, valid, k_json)
    normals, _ = estimate_source_perspective_normals(depth_m, valid, k_json, radius=REL_PLUS_V2_NORMAL_RADIUS)
    points, normals, _ = align_points_and_normals_to_gravity(points, normals, gravity_camera)
    points_cm = points * 100.0
    tangent, _ = perspective_tangent_field(points_cm)
    kwargs = dict(tangent=tangent, alpha=REL_PLUS_V2_ALPHA, lam=REL_PLUS_V2_LAMBDA)
    v2_1, _ = encode_rel_channels(points_cm, normals, valid, **kwargs)
    v3_1, _ = rv.encode_banded_metric(points_cm, normals, valid, **kwargs)
    return {"v2_1": v2_1, "v3_1": v3_1}, valid


def mirror_camera(k_json, gravity_camera, width):
    """Intrinsics and gravity of the left-right mirrored image (camera x negated)."""
    k_mirror = np.array(k_json, dtype=np.float64)
    k_mirror[0, 1] = -k_mirror[0, 1]
    k_mirror[0, 2] = width - k_mirror[0, 2]  # JSON pixel centres sit at u + 0.5
    return k_mirror, np.asarray(gravity_camera, dtype=np.float64) * np.array([-1.0, 1.0, 1.0])


def flip_statistics(raw, k_json, gravity_camera, frozen_bytes):
    """Mirror-encoded bytes against the flipped bytes, per channel, over valid pixels."""
    encoded, valid = encode_frame(raw, k_json, gravity_camera)
    if not np.array_equal(encoded["v2_1"], frozen_bytes):
        raise AssertionError("encode_frame does not reproduce generate_rel_plus_v2")
    k_mirror, gravity_mirror = mirror_camera(k_json, gravity_camera, raw.shape[1])
    mirrored, valid_mirror = encode_frame(np.ascontiguousarray(raw[:, ::-1]), k_mirror, gravity_mirror)
    stats = {"valid_pixels": int(valid.sum()), "valid_mask_mismatch": int((valid[:, ::-1] != valid_mirror).sum())}
    keep = valid[:, ::-1] & valid_mirror
    for variant in ("v2_1", "v3_1"):
        flipped = encoded[variant][:, ::-1].astype(np.int16)[keep]
        mirror = mirrored[variant].astype(np.int16)[keep]
        for index, channel in ((0, "EGVIA"), (2, "ReD")):
            diff = np.abs(mirror[:, index] - flipped[:, index])
            stats["{}_{}_exact".format(variant, channel)] = int((diff == 0).sum())
            stats["{}_{}_over_1".format(variant, channel)] = int((diff > 1).sum())
        for offset in (180, 179):
            diff = np.abs(mirror[:, 1] - (offset - flipped[:, 1]))
            stats["{}_LOA_{}_exact".format(variant, offset)] = int((diff == 0).sum())
            stats["{}_LOA_{}_over_1".format(variant, offset)] = int((diff > 1).sum())
    stats["compared_pixels"] = int(keep.sum())
    return stats


# ------------------------------------------------------- pinhole worker


def pinhole_frame(job):
    index, row = job
    from rel_plus.camera import gravity_in_camera
    from rel_plus.generator import generate_rel_plus_v2
    from rel_plus.profiles import STANFORD_S2D_PROFILE
    from rel_plus.stanford_s2d import load_canonical_frame

    raw, camera, _ = load_canonical_frame(
        row["depth_path"], row["camera_metadata_path"], dataset_profile=STANFORD_S2D_PROFILE
    )
    rel_plus, debug = generate_rel_plus_v2(raw, camera, return_debug=True)
    valid = np.asarray(debug["encoding_valid_mask"], dtype=bool)
    red, histogram = red_statistics(debug["red_raw_cm"], debug["red_encoded"], valid)
    result = {"row": dict(sample_id=row["sample_id"], **red), "distance_histogram": histogram}

    if index < _JOB["flip_frames"]:
        result["flip"] = flip_statistics(
            raw, camera.K_json, gravity_in_camera(camera.R_world_to_camera), rel_plus
        )

    if _JOB["arms"]:
        from dataloader.profiles import sample_comparison_transform

        stored = cv2.imread(str(Path(_JOB["gt_root"]) / (row["sample_id"] + _JOB["gt_format"])), cv2.IMREAD_GRAYSCALE)
        if stored is None or stored.shape != raw.shape:
            raise ValueError("label of {} is missing or not 480x480".format(row["sample_id"]))
        label = gt_transform(stored)
        transform = sample_comparison_transform(
            raw.shape, _JOB["scales"], _JOB["output_shape"], random.Random(_JOB["seed"] + index)
        )
        result["blend"] = {}
        for arm, spec in _JOB["arms"].items():
            x = cv2.imread(str(Path(spec["x_root"]) / (row["sample_id"] + ".png")), cv2.IMREAD_UNCHANGED)
            if x is None or x.shape != raw.shape + (3,) or x.dtype != np.uint8:
                raise ValueError("{} input of {} is missing or not 480x480x3 uint8".format(arm, row["sample_id"]))
            if spec.get("valid_root"):
                stored_valid = cv2.imread(str(Path(spec["valid_root"]) / (row["sample_id"] + ".png")), cv2.IMREAD_UNCHANGED)
                if stored_valid is None or stored_valid.shape != raw.shape:
                    raise ValueError("valid mask of {} is missing".format(row["sample_id"]))
                arm_valid = stored_valid.astype(bool)
            else:
                arm_valid = valid  # HHA has no stored mask: the 480 depth's valid pixels
            stats = blend_statistics(x, arm_valid, label, transform)
            stats["valid_mask_mismatch"] = int((arm_valid != valid).sum())
            result["blend"][arm] = stats
    return result


# ------------------------------------------------------ panorama worker


def rotation_between(points, rotated, valid):
    """Least-squares M with rotated = points @ M (``rotatePC`` is linear)."""
    keep = valid & (np.hypot(points[..., 0], points[..., 1]) > 50.0)
    m, _, _, _ = np.linalg.lstsq(points[keep].astype(np.float64), rotated[keep].astype(np.float64), rcond=None)
    return m


def egvia_v3_1(normals, height_cm, valid):
    normal_z = np.clip(-normals[:, :, 2], -1.0, 1.0)
    angle = np.clip((np.arccos(normal_z, dtype=np.float32) / np.pi) * 255.0, 0, 255).astype(np.float32)
    return rv.egvia_byte(rv.banded_metric_egvia(angle, height_cm), valid)


def panorama(job):
    sample, with_crops = job
    height, width = _JOB["size"]
    raw_native = cv2.imread(str(sample["depth"]), cv2.IMREAD_UNCHANGED)
    if raw_native is None or raw_native.dtype != np.uint16 or raw_native.ndim != 2:
        raise ValueError("{} is not a uint16 depth map".format(sample["depth"]))
    raw = cv2.resize(raw_native, (width, height), interpolation=cv2.INTER_NEAREST)
    depth = cp.decode_erp_depth(raw)
    valid = depth != 0
    points = _JOB["getPointCloud_ERP"](depth * 100)
    rotated, normals, rot = _JOB["processDepthImage_ERP"](depth * 100, ~valid)
    m = rotation_between(points, rotated, valid)
    up = m[2] / np.linalg.norm(m[2])  # the ERP vertical after the gravity rotation
    shift_cm = np.abs(rotated[..., 2] - points[..., 2])[valid]
    egvia_estimated = egvia_v3_1(normals, rotated[..., 2], valid)
    egvia_level = egvia_v3_1(np.einsum("hwj,ij->hwi", normals, np.linalg.inv(m).T), points[..., 2], valid)
    egvia_diff = np.abs(egvia_estimated.astype(np.int16) - egvia_level.astype(np.int16))[valid]
    distance = np.hypot(rotated[..., 0], rotated[..., 1])
    row = {
        "sample_id": sample["sample_id"],
        "native_raw0_share": float(np.mean(raw_native == 0)),
        "native_raw65535_share": float(np.mean(raw_native == 65535)),
        "raw0_share": float(np.mean(raw == 0)),
        "missing_share": float(1.0 - valid.mean()),
        "estimated_rotation_deg": "{:g} {:g}".format(*rot),
        "tilt_deg": float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))),
        "height_shift_cm_p50": float(np.median(shift_cm)),
        "height_shift_cm_p95": float(np.percentile(shift_cm, 95)),
        "egvia_v3_1_diff_p50": float(np.median(egvia_diff)),
        "egvia_v3_1_diff_p95": float(np.percentile(egvia_diff, 95)),
        "egvia_v3_1_diff_over_2_share": float(np.mean(egvia_diff > 2)),
        "red_min_all_cm": float(distance.min()),
        "red_min_valid_cm": float(distance[valid].min()),
        "red_max_cm": float(distance.max()),
    }
    result = {"row": row}
    if with_crops:
        report = cp.check_panorama(
            raw, _JOB["frozen"], yaws=CROP_YAWS, pitches=CROP_PITCHES, size=_JOB["crop_size"],
            fov=_JOB["crop_fov"], thresholds=cp.DEFAULT_THRESHOLDS,
        )
        result["crops"] = [
            {
                "sample_id": sample["sample_id"],
                "yaw_deg": crop["yaw_deg"],
                "pitch_deg": crop["pitch_deg"],
                "compared_pixels": crop["compared_pixels"],
                "red_byte_diff_p50": crop["encoded_bytes"]["ReD"]["median"],
                "red_byte_diff_p95": crop["encoded_bytes"]["ReD"]["p95"],
                "distance_cm_diff_p50": crop["raw"]["radius_cm"]["median"],
                "distance_cm_diff_p95": crop["raw"]["radius_cm"]["p95"],
            }
            for crop in report["crops"]
        ]
        result["crop_geometry_status"] = report["status"]
    return result


# ------------------------------------------------------------- summaries


def summarize_pinhole(results, names):
    rows = [r["row"] for r in results]
    histogram = sum(r["distance_histogram"] for r in results)
    holed = [row for row in rows if row["invalid_share"] > 0]
    medians = [row["shift_median"] for row in rows if row["shift_median"] is not None]
    summary = {
        "frames": len(rows),
        "red": {
            "frames_with_invalid_share": len(holed) / max(1, len(rows)),
            "invalid_share_of_holed_frames": percentiles([row["invalid_share"] for row in holed], (50, 95)),
            "valid_red_min_cm_of_holed_frames": percentiles([row["red_min_valid_cm"] for row in holed if row["red_min_valid_cm"] is not None], (5, 50, 95)),
            "byte_shift_frame_median": percentiles(medians, (50, 75, 95)),
            "frames_with_median_shift_at_least_5": float(np.mean(np.array(medians) >= 5)) if medians else None,
            "frames_with_median_shift_at_least_10": float(np.mean(np.array(medians) >= 10)) if medians else None,
            "horizontal_distance_cm": {
                "p{}".format(q): histogram_quantile(histogram, q) for q in (50, 90, 95, 99, 99.9)
            },
        },
    }
    flips = [r["flip"] for r in results if "flip" in r]
    if flips:
        total = {key: sum(f[key] for f in flips) for key in flips[0]}
        compared = max(1, total["compared_pixels"])
        summary["flip"] = {
            "frames": len(flips),
            "compared_pixels": total["compared_pixels"],
            "valid_mask_mismatch_pixels": total["valid_mask_mismatch"],
            "share": {
                key: total[key] / compared
                for key in sorted(total) if key.endswith("_exact") or key.endswith("_over_1")
            },
        }
    blends = [r["blend"] for r in results if "blend" in r]
    if blends:
        summary["blend"] = {}
        for arm in blends[0]:
            stats = [b[arm] for b in blends]
            labelled = sum(s["labelled_pixels"] for s in stats)
            affected = sum(s["affected_pixels"] for s in stats)
            by_class = sum(s["affected_by_class"] for s in stats)
            summary["blend"][arm] = {
                "affected_share_of_labelled_pixels": sum(s["affected_labelled_pixels"] for s in stats) / max(1, labelled),
                "affected_share_of_all_pixels": affected / max(1, sum(s["output_pixels"] for s in stats)),
                "affected_mean_channel_deviation": sum(s["deviation_sum"] for s in stats) / max(1, affected),
                "affected_max_channel_deviation": max(s["deviation_max"] for s in stats),
                "affected_labelled_pixels_by_class": dict(zip(names, by_class.tolist())),
                "valid_mask_mismatch_pixels": sum(s["valid_mask_mismatch"] for s in stats),
            }
    return summary, rows


def summarize_panoramas(results, metric_range_cm):
    rows = [r["row"] for r in results]
    by_area = {}
    for row in rows:
        by_area.setdefault(row["sample_id"].split("/")[0], []).append(row)
    summary = {
        "panoramas": len(rows),
        "by_area": {
            area: {
                "panoramas": len(group),
                "with_native_raw0_share": float(np.mean([row["native_raw0_share"] > 0 for row in group])),
                "native_raw0_pixel_share": percentiles([row["native_raw0_share"] for row in group], (50, 95, 100)),
                "missing_pixel_share": percentiles([row["missing_share"] for row in group], (50, 95)),
                "tilt_deg": percentiles([row["tilt_deg"] for row in group], (50, 95, 100)),
                "height_shift_cm_p95": percentiles([row["height_shift_cm_p95"] for row in group], (50, 95)),
                "egvia_v3_1_diff_p95": percentiles([row["egvia_v3_1_diff_p95"] for row in group], (50, 95)),
                "egvia_v3_1_diff_over_2_share": percentiles([row["egvia_v3_1_diff_over_2_share"] for row in group], (50, 95)),
            }
            for area, group in sorted(by_area.items())
        },
    }
    crops = [crop for r in results for crop in r.get("crops", [])]
    if crops:
        current = [c["red_byte_diff_p50"] for c in crops if c["red_byte_diff_p50"] is not None]
        metric = [
            c["distance_cm_diff_p50"] * 255.0 / metric_range_cm
            for c in crops if c["distance_cm_diff_p50"] is not None
        ]
        summary["crops"] = {
            "crops": len(crops),
            "geometry_fail_panoramas": sum(r.get("crop_geometry_status") == "FAIL" for r in results),
            "current_red_byte_diff_crop_median": percentiles(current, (25, 50, 75)),
            "metric_red_byte_diff_crop_median": percentiles(metric, (25, 50, 75)),
            "metric_range_cm": metric_range_cm,
            "crops_with_current_median_at_least_10": float(np.mean(np.array(current) >= PHASE1_CURRENT_LEVELS)) if current else None,
        }
    return summary, rows, crops


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plan_gates(report):
    """The fix plan's pre-stated thresholds, read off this report."""
    gates = {}
    crops = report.get("panorama", {}).get("crops")
    if crops and crops["current_red_byte_diff_crop_median"]["p50"] is not None:
        gates["phase1_metric_red"] = {
            "rule": "current crop-vs-panorama ReD median >= {:g} levels on most crops and metric ReD <= {:g}".format(
                PHASE1_CURRENT_LEVELS, PHASE1_METRIC_LEVELS
            ),
            "met": bool(
                crops["crops_with_current_median_at_least_10"] > 0.5
                and crops["metric_red_byte_diff_crop_median"]["p50"] <= PHASE1_METRIC_LEVELS
            ),
        }
    blend = report.get("pinhole", {}).get("blend")
    if blend:
        gates["phase2_resize_fix"] = {
            "rule": "more than {:g}% of labelled training pixels affected in either arm".format(100 * PHASE2_AFFECTED_SHARE),
            "met": any(arm["affected_share_of_labelled_pixels"] > PHASE2_AFFECTED_SHARE for arm in blend.values()),
        }
    return gates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path, help="full_manifest.csv")
    parser.add_argument("--config", required=True, type=Path, help="the REL+ 3.1 run's config json (labels, scales, cache paths)")
    parser.add_argument("--stanford-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hha-root", type=Path, help="HHA cache folder (hha.json x_root_folder); omit to skip HHA blending")
    parser.add_argument("--skip-blend", action="store_true", help="no cache reads (no torch needed)")
    parser.add_argument("--source-root", type=Path, default=ev.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--frames", type=int, default=3000, help="random pinhole training frames")
    parser.add_argument("--flip-frames", type=int, default=200, help="the first N of those frames also get the flip check")
    parser.add_argument("--areas", nargs="+", default=list(ALL_AREAS), help="panorama areas for the input counts")
    parser.add_argument("--crop-panoramas", type=int, default=60, help="random training-area panoramas for the crop ReD check")
    parser.add_argument("--crop-fov", type=float, default=60.0, help="training frames: median HFOV 62 deg")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))

    frozen = cp.import_frozen_source(args.source_root)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    names = list(config["class_names"])
    arms = {}
    if not args.skip_blend:
        cache_root = Path(config["x_root_folder"]).parent
        marker = rv.read_marker(cache_root)
        arms["relplus_" + ("v2_1" if marker is None else marker["variant"])] = {
            "x_root": config["x_root_folder"], "valid_root": config["x_valid_root_folder"],
        }
        if args.hha_root is not None:
            arms["hha"] = {"x_root": str(args.hha_root)}

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    rng = np.random.default_rng(args.seed)
    frames = [rows[i] for i in sorted(rng.choice(len(rows), min(args.frames, len(rows)), replace=False))]
    samples = ev.list_panoramas(args.stanford_root, list(args.areas))
    train_ids = [i for i, s in enumerate(samples) if s["sample_id"].split("/")[0] in TRAIN_AREAS]
    crop_ids = set(rng.choice(train_ids, min(args.crop_panoramas, len(train_ids)), replace=False).tolist()) if train_ids else set()

    _JOB.update(
        classes=len(names),
        flip_frames=args.flip_frames,
        arms=arms,
        gt_root=config["gt_root_folder"],
        gt_format=config["gt_format"],
        scales=list(config["train_scale_array"]),
        output_shape=(int(config["image_height"]), int(config["image_width"])),
        seed=args.seed,
        size=(args.height, args.width),
        crop_fov=args.crop_fov,
        crop_size=480,
        frozen=frozen,
        getPointCloud_ERP=frozen["getPointCloud_ERP"],
        processDepthImage_ERP=frozen["processDepthImage_ERP"],
    )

    started = time.time()
    pinhole_results, panorama_results = [], []
    with multiprocessing.get_context("fork").Pool(args.workers, cv2.setNumThreads, (1,)) as pool:
        for count, result in enumerate(pool.imap_unordered(pinhole_frame, list(enumerate(frames))), start=1):
            pinhole_results.append(result)
            if count % 500 == 0 or count == len(frames):
                print("{}/{} pinhole frames".format(count, len(frames)), flush=True)
        jobs = [(sample, i in crop_ids) for i, sample in enumerate(samples)]
        for count, result in enumerate(pool.imap_unordered(panorama, jobs), start=1):
            panorama_results.append(result)
            if count % 100 == 0 or count == len(jobs):
                print("{}/{} panoramas".format(count, len(jobs)), flush=True)

    pinhole_summary, frame_rows = summarize_pinhole(pinhole_results, names)
    metric_range = pinhole_summary["red"]["horizontal_distance_cm"]["p99"]
    panorama_summary, pano_rows, crop_rows = summarize_panoramas(panorama_results, metric_range)
    report = {
        "frames": len(frames),
        "flip_frames": min(args.flip_frames, len(frames)),
        "arms": arms,
        "panorama_areas": list(args.areas),
        "panorama_size": [args.height, args.width],
        "crop_fov_deg": args.crop_fov,
        "seed": args.seed,
        "seconds": time.time() - started,
        "pinhole": pinhole_summary,
        "panorama": panorama_summary,
    }
    report["plan_gates"] = plan_gates(report)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "input_defects.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output / "pinhole_frames.csv", sorted(frame_rows, key=lambda r: r["sample_id"]))
    write_csv(args.output / "panoramas.csv", sorted(pano_rows, key=lambda r: r["sample_id"]))
    write_csv(args.output / "crops.csv", crop_rows)
    print(json.dumps(report["plan_gates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
