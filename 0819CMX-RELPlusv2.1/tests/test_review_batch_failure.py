import importlib.util
from pathlib import Path

import cv2
import numpy as np

from rel_plus.geometry import GravityAlignmentSingularity


TOOL = Path(__file__).resolve().parents[1] / "tools/generate_review_samples.py"
SPEC = importlib.util.spec_from_file_location("review_samples_tool", str(TOOL))
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)


def test_batch_records_gravity_failure_without_writing_false_rel_png(tmp_path, monkeypatch):
    raw = np.full((8, 8), 1024, dtype=np.uint16)
    depth = tmp_path / "depth.png"
    rgb = tmp_path / "rgb.png"
    label = tmp_path / "label.png"
    cv2.imwrite(str(depth), raw)
    cv2.imwrite(str(rgb), np.zeros((8, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(label), np.zeros((8, 8), dtype=np.uint8))
    monkeypatch.setattr(
        REVIEW,
        "load_canonical_frame",
        lambda *_args, **_kwargs: (raw, object(), raw.shape),
    )

    def fail(*_args, **_kwargs):
        raise GravityAlignmentSingularity("sample_id=s gravity=[0,0,1] angle_deg=180")

    monkeypatch.setattr(REVIEW, "generate_rel_plus_v2_1", fail)
    source = {
        "sample_id": "s",
        "area": "area_2",
        "room": "r",
        "camera": "c",
        "depth_path": str(depth),
        "rgb_path": str(rgb),
        "label_path": str(label),
        "camera_metadata_path": str(tmp_path / "pose.json"),
        "intrinsics_height": 8,
        "intrinsics_width": 8,
        "gravity_alignment_angle_deg": 180.0,
        "geometry_oracle": "available",
        "validation_level": "strong",
    }
    summary = REVIEW.generate_manifest_rows([source], tmp_path / "out")
    assert summary["status"] == "FAIL" and summary["error_count"] == 1
    assert (tmp_path / "out/review_errors.csv").is_file()
    assert not list((tmp_path / "out").rglob("rel_plus_v2.png"))
