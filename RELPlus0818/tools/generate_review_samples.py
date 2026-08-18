#!/usr/bin/env python3
"""Select ten pose-diverse S2D frames and generate REL+ review montages."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rel_plus.camera import gravity_in_camera, load_stanford_s2d_camera_geometry
from rel_plus.generator import generate_rel_plus
from rel_plus.source_helpers import align_points_and_normals_to_gravity
from rel_plus.stanford_s2d import load_canonical_frame
from rel_plus.storage import save_rel_plus_png
from visualize_rel_plus import save_contact_sheet, save_review_bundle


AREAS = ("area_2", "area_3", "area_4", "area_5a", "area_5b", "area_6")
POSE_NAME = re.compile(
    r"^camera_([0-9a-f]{32})_(.+)_frame_([0-9]+)[.]json$"
)


def matrix_zyx_degrees(rotation):
    """Report one explicit ZYX decomposition of the W2C rotation."""
    sy_value = np.hypot(rotation[0, 0], rotation[1, 0])
    if sy_value > 1e-8:
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        pitch = np.arctan2(-rotation[2, 0], sy_value)
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    else:
        yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy_value)
        roll = 0.0
    return tuple(float(np.degrees(value)) for value in (yaw, pitch, roll))


def enumerate_candidates(production_root, native_root):
    candidates = []
    for area in AREAS:
        for pose_path in sorted((production_root / "Pose" / area).glob("*.json")):
            match = POSE_NAME.fullmatch(pose_path.name)
            if match is None:
                continue
            camera_id, room, frame_number = match.groups()
            stem = pose_path.stem
            depth_path = production_root / "Depth16" / area / (stem + ".png")
            rgb_path = production_root / "RGB" / area / (stem + ".png")
            xyz_path = (
                native_root
                / area
                / "data/global_xyz"
                / (stem + "_domain_global_xyz.exr")
            )
            if not (depth_path.is_file() and rgb_path.is_file() and xyz_path.is_file()):
                continue
            try:
                camera = load_stanford_s2d_camera_geometry(pose_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            yaw, pitch, roll = matrix_zyx_degrees(camera.R_world_to_camera)
            gravity = gravity_in_camera(camera.R_world_to_camera)
            tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1.0, 1.0))))
            candidates.append(
                {
                    "sample_id": area + "/" + stem,
                    "area": area,
                    "room": room,
                    "camera": camera_id,
                    "frame": int(frame_number),
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "camera_metadata_path": str(pose_path),
                    "global_xyz_path": str(xyz_path),
                    "w2c_yaw_deg": yaw,
                    "w2c_pitch_deg": pitch,
                    "w2c_roll_deg": roll,
                    "gravity_tilt_deg": tilt,
                    "rotation": camera.R_world_to_camera,
                }
            )
    return candidates


def select_pose_diverse_ten(candidates):
    if not candidates:
        raise RuntimeError("no complete production/native S2D candidates found")
    selected = []
    used_cameras = set()

    def add_first(rows, reason):
        for source in rows:
            if source["camera"] not in used_cameras:
                row = dict(source)
                row["selection_reason"] = reason
                selected.append(row)
                used_cameras.add(source["camera"])
                return
        raise RuntimeError("unable to satisfy selection objective: {}".format(reason))

    for area in AREAS:
        area_rows = [row for row in candidates if row["area"] == area]
        add_first(
            sorted(
                area_rows,
                key=lambda row: (-row["gravity_tilt_deg"], row["sample_id"]),
            ),
            "area coverage; largest available gravity tilt",
        )

    objectives = (
        ("most positive W2C pitch", "w2c_pitch_deg", True),
        ("most negative W2C pitch", "w2c_pitch_deg", False),
        ("most positive W2C roll", "w2c_roll_deg", True),
        ("most negative W2C roll", "w2c_roll_deg", False),
    )
    for reason, field, descending in objectives:
        add_first(
            sorted(
                candidates,
                key=lambda row: (
                    -row[field] if descending else row[field],
                    row["sample_id"],
                ),
            ),
            reason,
        )
    if len(selected) != 10:
        raise RuntimeError("selection produced {} samples instead of 10".format(len(selected)))
    return selected


def write_manifest(path, rows):
    fields = [
        "selection_index",
        "selection_reason",
        "sample_id",
        "area",
        "room",
        "camera",
        "frame",
        "rgb_path",
        "depth_path",
        "camera_metadata_path",
        "global_xyz_path",
        "image_resolution",
        "k_source",
        "pose_source",
        "w2c_yaw_deg",
        "w2c_pitch_deg",
        "w2c_roll_deg",
        "gravity_tilt_deg",
        "invalid_depth_ratio",
        "rel_valid_ratio",
        "review_directory",
        "rel_plus_path",
        "montage_path",
        "numeric_review_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_selected_samples(selected, output_root):
    review_root = output_root / "review_samples"
    review_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    montage_paths = []
    labels = []

    for index, row in enumerate(selected):
        raw_native = cv2.imread(row["depth_path"], cv2.IMREAD_UNCHANGED)
        if raw_native is None or raw_native.dtype != np.uint16 or raw_native.ndim != 2:
            raise RuntimeError("invalid Depth16: {}".format(row["depth_path"]))
        raw_depth, camera, source_shape = load_canonical_frame(
            row["depth_path"], row["camera_metadata_path"], (480, 480)
        )
        rgb = cv2.imread(row["rgb_path"], cv2.IMREAD_COLOR)
        if rgb is None or rgb.shape[:2] != (480, 480):
            raise RuntimeError("expected canonical 480x480 RGB: {}".format(row["rgb_path"]))
        rel_plus, debug = generate_rel_plus(raw_depth, camera, return_debug=True)
        sample_dir = review_root / "sample_{:02d}_{}_{}".format(
            index, row["area"], Path(row["depth_path"]).stem
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        rel_path = sample_dir / "rel_plus.png"
        save_rel_plus_png(rel_path, rel_plus)
        montage_path = save_review_bundle(sample_dir, rgb, debug)

        depth_invalid = (raw_native == 0) | (raw_native == 65535)
        rel_valid = np.asarray(debug["valid_mask"], dtype=bool)
        invalid_triplet_ok = bool(np.all(rel_plus[~rel_valid] == 255))
        valid_count = int(np.count_nonzero(rel_valid))
        if valid_count == 0:
            raise RuntimeError("REL+ has no valid pixels: {}".format(row["sample_id"]))
        channel_stats = {}
        nondegenerate = True
        for channel_index, channel_name in enumerate(("EGVIA", "LOA", "ReD")):
            values = rel_plus[:, :, channel_index][rel_valid]
            channel_stats[channel_name] = {
                "min": int(values.min()),
                "max": int(values.max()),
                "mean": float(values.mean()),
                "unique_values": int(np.unique(values).size),
                "zero_ratio": float(np.mean(values == 0)),
                "value_255_ratio": float(np.mean(values == 255)),
            }
            nondegenerate = nondegenerate and np.unique(values).size > 1

        gravity = gravity_in_camera(camera.R_world_to_camera)
        gravity_field = gravity.reshape(1, 1, 3)
        aligned_gravity, _, _ = align_points_and_normals_to_gravity(
            gravity_field, gravity_field, gravity
        )
        gravity_error = float(
            np.max(np.abs(aligned_gravity[0, 0] - [0.0, 0.0, -1.0]))
        )
        numeric_pass = bool(
            rel_plus.shape == (480, 480, 3)
            and rel_plus.dtype == np.uint8
            and invalid_triplet_ok
            and nondegenerate
            and gravity_error <= 1e-6
        )
        stats = {
            "sample_id": row["sample_id"],
            "shape": list(rel_plus.shape),
            "dtype": str(rel_plus.dtype),
            "channel_order": ["EGVIA", "LOA", "ReD"],
            "invalid_triplet_ok": invalid_triplet_ok,
            "valid_pixel_count": valid_count,
            "rel_valid_ratio": float(np.mean(rel_valid)),
            "invalid_depth_ratio_native": float(np.mean(depth_invalid)),
            "gravity_alignment_max_abs_error": gravity_error,
            "channels": channel_stats,
            "numeric_review_status": "PASS" if numeric_pass else "FAIL",
        }
        (sample_dir / "numeric_stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if not numeric_pass:
            raise RuntimeError("numeric review failed: {}".format(row["sample_id"]))

        manifest = dict(row)
        manifest.pop("rotation", None)
        manifest.update(
            {
                "selection_index": index,
                "image_resolution": "native_depth={}x{}; canonical=480x480; rgb=480x480".format(
                    source_shape[1], source_shape[0]
                ),
                "k_source": "camera_k_matrix; canonical rows scaled by 480/native",
                "pose_source": "camera_rt_matrix explicit world-to-camera [R|t]",
                "invalid_depth_ratio": float(np.mean(depth_invalid)),
                "rel_valid_ratio": float(np.mean(rel_valid)),
                "review_directory": str(sample_dir),
                "rel_plus_path": str(rel_path),
                "montage_path": str(montage_path),
                "numeric_review_status": "PASS",
            }
        )
        manifest_rows.append(manifest)
        montage_paths.append(montage_path)
        labels.append(
            "{:02d} {} pitch={:.1f} roll={:.1f}".format(
                index, row["area"], row["w2c_pitch_deg"], row["w2c_roll_deg"]
            )
        )

    write_manifest(output_root / "real_samples_manifest.csv", manifest_rows)
    save_contact_sheet(
        montage_paths, labels, output_root / "review_samples_montage.png"
    )
    return manifest_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-root",
        default="/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480",
    )
    parser.add_argument(
        "--native-root",
        default="/data/zhuzhaoziao/datasets/Stanford2D3D/with_xyz",
    )
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = enumerate_candidates(Path(args.production_root), Path(args.native_root))
    selected = select_pose_diverse_ten(candidates)
    rows = generate_selected_samples(selected, output_root)
    summary = {
        "status": "PASS_REVIEW_SAMPLE_GENERATION",
        "candidate_count": len(candidates),
        "sample_count": len(rows),
        "areas": sorted({row["area"] for row in rows}),
        "distinct_cameras": len({row["camera"] for row in rows}),
        "pitch_range_deg": [
            min(row["w2c_pitch_deg"] for row in rows),
            max(row["w2c_pitch_deg"] for row in rows),
        ],
        "roll_range_deg": [
            min(row["w2c_roll_deg"] for row in rows),
            max(row["w2c_roll_deg"] for row in rows),
        ],
        "invalid_depth_ratio_range": [
            min(row["invalid_depth_ratio"] for row in rows),
            max(row["invalid_depth_ratio"] for row in rows),
        ],
        "visual_review_status": "PENDING_CODEX_INSPECTION",
    }
    (output_root / "review_generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
