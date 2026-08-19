#!/usr/bin/env python3
"""Select or generate a bounded 12-frame REL+ v2 review set."""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rel_plus.camera import gravity_in_camera, load_stanford_s2d_camera_geometry
from rel_plus.generator import generate_rel_plus_v2_1
from rel_plus.profiles import STANFORD_S2D_PROFILE
from rel_plus.geometry import GravityAlignmentSingularity
from rel_plus.stanford_s2d import load_canonical_frame
from rel_plus.storage import save_rel_plus_png
from visualize_rel_plus import save_contact_sheet, save_review_bundle


AREAS = ("area_1", "area_2", "area_3", "area_4", "area_5a", "area_5b", "area_6")
POSE_NAME = re.compile(r"^camera_([0-9a-f]{32})_(.+)_frame_([0-9]+)[.]json$")


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty table")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _semantic_ids(production_root):
    payload = json.loads((production_root / "class_mapping.json").read_text(encoding="utf-8"))
    return {name: int(index) for index, name in payload["stored_ids"].items()}


def enumerate_candidates(production_root, native_root):
    candidates = []
    for area in AREAS:
        pose_root = production_root / "Pose" / area
        if not pose_root.is_dir():
            continue
        for pose_path in sorted(pose_root.glob("*.json")):
            match = POSE_NAME.fullmatch(pose_path.name)
            if match is None:
                continue
            camera_id, room, frame_number = match.groups()
            stem = pose_path.stem
            paths = {
                "rgb_path": production_root / "RGB" / area / (stem + ".png"),
                "label_path": production_root / "Label" / area / (stem + ".png"),
                "depth_path": production_root / "Depth16" / area / (stem + ".png"),
            }
            if not all(path.is_file() for path in paths.values()):
                continue
            try:
                camera = load_stanford_s2d_camera_geometry(pose_path, (1080, 1080))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            gravity = gravity_in_camera(camera.R_world_to_camera)
            alignment_angle = float(
                np.degrees(np.arccos(np.clip(gravity @ [0.0, 0.0, -1.0], -1.0, 1.0)))
            )
            optical_world = camera.R_world_to_camera.T @ np.array([0.0, 0.0, 1.0])
            world_up_camera = camera.R_world_to_camera @ np.array([0.0, 0.0, 1.0])
            xyz_path = native_root / area / "data/global_xyz" / (stem + "_domain_global_xyz.exr")
            candidates.append(
                {
                    "sample_id": area + "/" + stem,
                    "area": area,
                    "room": room,
                    "camera": camera_id,
                    "frame": int(frame_number),
                    "rgb_path": str(paths["rgb_path"]),
                    "label_path": str(paths["label_path"]),
                    "depth_path": str(paths["depth_path"]),
                    "camera_metadata_path": str(pose_path),
                    "global_xyz_path": str(xyz_path) if xyz_path.is_file() else "",
                    "intrinsics_height": 1080,
                    "intrinsics_width": 1080,
                    "gravity_x": float(gravity[0]),
                    "gravity_y": float(gravity[1]),
                    "gravity_z": float(gravity[2]),
                    "gravity_alignment_angle_deg": alignment_angle,
                    "world_up_image_projection": float(np.hypot(world_up_camera[0], world_up_camera[1])),
                    "optical_world_x": float(optical_world[0]),
                    "optical_world_y": float(optical_world[1]),
                    "optical_world_z": float(optical_world[2]),
                    "geometry_oracle": "available" if xyz_path.is_file() else "unavailable",
                    "validation_level": "strong" if xyz_path.is_file() else "weak",
                }
            )
    return candidates


def select_twelve(candidates, production_root, limit=12):
    if limit != 12:
        raise ValueError("formal review selection is frozen at exactly 12 samples")
    semantic_ids = _semantic_ids(production_root)
    selected = []
    used_cameras = set()
    for area in AREAS:
        rows = [
            row for row in candidates
            if row["area"] == area and row["gravity_alignment_angle_deg"] < 179.0
        ]
        if not rows:
            raise RuntimeError("no non-singular candidates for {}".format(area))
        indices = np.linspace(0, len(rows) - 1, min(320, len(rows)), dtype=np.int64)
        pool = []
        for index in indices:
            row = dict(rows[int(index)])
            label = cv2.imread(row["label_path"], cv2.IMREAD_UNCHANGED)
            if label is None:
                continue
            row["floor_ratio"] = float(np.mean(label == semantic_ids["floor"]))
            row["ceiling_ratio"] = float(np.mean(label == semantic_ids["ceiling"]))
            pool.append(row)
        if not pool:
            raise RuntimeError("no readable labels for {}".format(area))
        first = max(
            pool,
            key=lambda row: (
                row["floor_ratio"] > 0.01,
                row["gravity_alignment_angle_deg"],
                row["floor_ratio"],
            ),
        )
        first["selection_reason"] = "area coverage; floor/tilt review"
        selected.append(first)
        used_cameras.add(first["camera"])
        alternatives = [
            row for row in pool
            if row["camera"] not in used_cameras and row["room"] != first["room"]
        ]
        if not alternatives:
            alternatives = [row for row in pool if row["camera"] != first["camera"]]
        second = max(
            alternatives,
            key=lambda row: (
                row["ceiling_ratio"] > 0.01,
                row["ceiling_ratio"],
                abs(row["optical_world_z"] - first["optical_world_z"]),
            ),
        )
        second["selection_reason"] = "area coverage; ceiling/optical-axis review"
        selected.append(second)
        used_cameras.add(second["camera"])

    area1 = [row for row in selected if row["area"] == "area_1"]
    strong_all = [row for row in selected if row["area"] != "area_1"]
    strong_ranked = sorted(
        strong_all,
        key=lambda row: (
            -row["gravity_alignment_angle_deg"],
            -row["world_up_image_projection"],
            row["sample_id"],
        ),
    )
    strong = []
    for area in AREAS[1:]:
        strong.append(next(row for row in strong_ranked if row["area"] == area))
    strong.extend(row for row in strong_ranked if row not in strong)
    strong = strong[:10]
    final = area1 + strong
    if len(final) != 12 or {row["area"] for row in final} != set(AREAS):
        raise RuntimeError("selection failed 12-sample/Area1-6 coverage contract")
    for index, row in enumerate(final):
        row["selection_index"] = index
    return final


def _save_raw_views(sample_dir, raw_native, raw_canonical, debug):
    files = {
        "raw_depth_native.png": raw_native,
        "canonical_depth16.png": raw_canonical,
        "depth_valid_mask.png": debug["depth_valid"].astype(np.uint8) * 255,
    }
    for name, value in files.items():
        if not cv2.imwrite(str(sample_dir / name), value):
            raise OSError("failed to write {}".format(sample_dir / name))


def generate_manifest_rows(rows, output_root):
    review_root = output_root / "review_samples"
    review_root.mkdir(parents=True, exist_ok=True)
    completed = []
    errors = []
    montage_paths = []
    labels = []
    for index, source in enumerate(rows):
        sample_id = source["sample_id"]
        try:
            raw_native = cv2.imread(source["depth_path"], cv2.IMREAD_UNCHANGED)
            raw_depth, camera, source_shape = load_canonical_frame(
                source["depth_path"],
                source["camera_metadata_path"],
                dataset_profile=STANFORD_S2D_PROFILE,
            )
            if tuple(source_shape) != (
                int(source["intrinsics_height"]), int(source["intrinsics_width"])
            ):
                raise ValueError("manifest K reference shape does not match native depth")
            rgb = cv2.imread(source["rgb_path"], cv2.IMREAD_COLOR)
            label = cv2.imread(source["label_path"], cv2.IMREAD_UNCHANGED)
            if rgb is None or rgb.shape[:2] != raw_depth.shape or label is None or label.shape != raw_depth.shape:
                raise ValueError("canonical RGB/label/depth shapes do not match")
            rel_plus, debug = generate_rel_plus_v2_1(
                raw_depth, camera, return_debug=True
            )
            sample_dir = review_root / "sample_{:02d}_{}".format(index, Path(source["depth_path"]).stem)
            sample_dir.mkdir(parents=True, exist_ok=True)
            rel_path = sample_dir / "rel_plus_v2.png"
            save_rel_plus_png(rel_path, rel_plus)
            _save_raw_views(sample_dir, raw_native, raw_depth, debug)
            montage = save_review_bundle(sample_dir, rgb, debug)

            valid = debug["encoding_valid_mask"]
            channel_stats = {}
            for channel_index, name in enumerate(("EGVIA", "LOA", "ReD")):
                values = rel_plus[:, :, channel_index][valid]
                channel_stats[name] = {
                    "min": int(values.min()),
                    "max": int(values.max()),
                    "mean": float(values.mean()),
                    "unique": int(np.unique(values).size),
                }
            stats = {
                "sample_id": sample_id,
                "invalid_depth_ratio": float(1.0 - np.mean(debug["depth_valid"])),
                "nonfinite_normal_ratio": debug["normal_invalid_ratio"],
                "zero_normal_ratio": debug["zero_normal_ratio"],
                "low_support_ratio": debug["low_support_ratio"],
                "gravity_vector": debug["gravity_camera"].tolist(),
                "gravity_alignment_angle_deg": float(source["gravity_alignment_angle_deg"]),
                "channel_order": ["EGVIA", "LOA", "ReD"],
                "channels": channel_stats,
                "geometry_oracle": source["geometry_oracle"],
                "validation_level": source["validation_level"],
                "pose_physical_status": "PENDING_VALIDATE_REAL_GEOMETRY",
            }
            (sample_dir / "numeric_stats.json").write_text(
                json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            result = dict(source)
            result.update(
                {
                    "review_directory": str(sample_dir),
                    "rel_plus_path": str(rel_path),
                    "montage_path": str(montage),
                    "generation_status": "PASS",
                }
            )
            completed.append(result)
            montage_paths.append(montage)
            labels.append("{:02d} {} {}".format(index, source["area"], source["room"]))
        except GravityAlignmentSingularity as error:
            errors.append({"sample_id": sample_id, "status": "FAIL", "error": str(error)})
        except Exception as error:
            errors.append({"sample_id": sample_id, "status": "FAIL", "error": repr(error)})

    if completed:
        _write_csv(output_root / "real_samples_manifest.csv", completed)
        save_contact_sheet(montage_paths, labels, output_root / "review_samples_montage.png")
    if errors:
        _write_csv(output_root / "review_errors.csv", errors)
    summary = {
        "status": "PASS" if len(completed) == len(rows) and not errors else "FAIL",
        "requested_count": len(rows),
        "completed_count": len(completed),
        "error_count": len(errors),
        "areas": sorted({row["area"] for row in completed}),
        "geometry_oracle_unavailable_count": sum(
            row["geometry_oracle"] == "unavailable" for row in completed
        ),
        "visual_review_status": "PENDING_CODEX_INSPECTION",
    }
    (output_root / "review_generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--production-root", default="/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480")
    parser.add_argument("--native-root", default="/data/zhuzhaoziao/datasets/Stanford2D3D/with_xyz")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        rows = _read_manifest(args.manifest)[: args.limit]
    else:
        candidates = enumerate_candidates(Path(args.production_root), Path(args.native_root))
        rows = select_twelve(candidates, Path(args.production_root), args.limit)
        _write_csv(output_root / "selected_manifest.csv", rows)
        (output_root / "selection_summary.json").write_text(
            json.dumps(
                {
                    "candidate_count": len(candidates),
                    "selected_count": len(rows),
                    "areas": sorted({row["area"] for row in rows}),
                    "geometry_oracle_unavailable_count": sum(
                        row["geometry_oracle"] == "unavailable" for row in rows
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
    if args.select_only:
        print(json.dumps({"status": "PASS_SELECTION", "count": len(rows)}, ensure_ascii=False))
        return 0
    summary = generate_manifest_rows(rows, output_root)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
