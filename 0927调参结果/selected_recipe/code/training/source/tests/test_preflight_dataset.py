import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from rel_plus.profiles import DatasetCameraProfile


TOOL = Path(__file__).resolve().parents[1] / "tools/preflight_dataset.py"
SPEC = importlib.util.spec_from_file_location("preflight_dataset_tool", str(TOOL))
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def write_case(tmp_path, declared_shape=(8, 8), rotation=None):
    raw = np.full((8, 8), 1024, dtype=np.uint16)
    depth_path = tmp_path / "depth.png"
    rgb_path = tmp_path / "rgb.png"
    label_path = tmp_path / "label.png"
    cv2.imwrite(str(depth_path), raw)
    cv2.imwrite(str(rgb_path), np.zeros((8, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(label_path), np.zeros((8, 8), dtype=np.uint8))
    rotation = np.eye(3) if rotation is None else rotation
    center = np.zeros(3)
    translation = -rotation @ center
    pose_path = tmp_path / "pose.json"
    pose_path.write_text(
        json.dumps(
            {
                "camera_k_matrix": [[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]],
                "camera_rt_matrix": np.column_stack([rotation, translation]).tolist(),
                "camera_location": center.tolist(),
            }
        ),
        encoding="utf-8",
    )
    return {
        "sample_id": "s",
        "rgb_path": str(rgb_path),
        "label_path": str(label_path),
        "depth_path": str(depth_path),
        "camera_metadata_path": str(pose_path),
        "intrinsics_height": str(declared_shape[0]),
        "intrinsics_width": str(declared_shape[1]),
    }


def profile(shape):
    return DatasetCameraProfile(
        "fixture", shape, shape, "json_half_pixel", "world_to_camera_3x4"
    )


def test_preflight_passes_matching_files_and_explicit_k_shape(tmp_path):
    result = PREFLIGHT.scan_row(write_case(tmp_path), profile((8, 8)))
    assert result["status"] == "PASS"


def test_preflight_reports_k_mismatch_and_blocks_full_cache(tmp_path):
    result = PREFLIGHT.scan_row(write_case(tmp_path), profile((16, 16)))
    assert result["status"] == "FAIL"
    assert "DEPTH_NATIVE_SHAPE_MISMATCH" in result["reasons"]


def test_preflight_lists_gravity_singularity_before_batch_generation(tmp_path):
    angle = np.pi
    rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]]
    )
    result = PREFLIGHT.scan_row(
        write_case(tmp_path, rotation=rotation), profile((8, 8))
    )
    assert result["status"] == "FAIL"
    assert "GRAVITY_SINGULARITY" in result["reasons"]
