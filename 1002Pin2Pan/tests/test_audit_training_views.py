"""Training-view audit on synthetic Stanford-style pose files."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audit_training_views as audit  # noqa: E402
import cross_projection as cp  # noqa: E402


def write_pose(path, yaw, pitch, roll=0.0):
    camera_to_world = cp.crop_camera_to_world(yaw, pitch)
    c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    camera_to_world = camera_to_world @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotation = camera_to_world.T
    location = np.array([1.0, -2.0, 1.5])
    path.write_text(json.dumps({
        "camera_k_matrix": cp.crop_intrinsics(1080, 60.0).tolist(),
        "camera_rt_matrix": np.hstack([rotation, (-rotation @ location)[:, None]]).tolist(),
        "camera_location": location.tolist(),
    }))


def test_audit_recovers_pitch_roll_and_view_elevations(tmp_path, monkeypatch):
    views = [("train", 30, 0, 0), ("train", 200, 40, 5), ("test", 90, -20, 0)]
    rows = []
    for index, (split, yaw, pitch, roll) in enumerate(views):
        pose = tmp_path / "pose{}.json".format(index)
        write_pose(pose, yaw, pitch, roll)
        rows.append({"split": split, "camera_metadata_path": str(pose)})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "camera_metadata_path"])
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "views.json"
    monkeypatch.setattr(sys, "argv", [
        "audit_training_views.py", "--manifest", str(manifest), "--output", str(output),
    ])
    assert audit.main() == 0

    report = json.loads(output.read_text())
    train, test = report["splits"]["train"], report["splits"]["test"]
    assert np.isclose(train["pitch_deg_percentiles"]["min"], 0, atol=1e-6)
    assert np.isclose(train["pitch_deg_percentiles"]["max"], 40, atol=1e-6)
    assert np.isclose(train["abs_roll_deg_percentiles"]["max"], 5, atol=1e-6)
    assert np.isclose(test["pitch_deg_percentiles"]["median"], -20, atol=1e-6)
    assert train["fraction_within_10_deg_of_crop_pitch"]["0"] == 0.5
    assert train["fraction_within_10_deg_of_crop_pitch"]["45"] == 0.5
    assert train["pitch_histogram"]["counts"][6] == 1  # [0, 15)
    for split in (train, test):
        assert np.isclose(sum(split["pixel_share_by_elevation_band"]), 1.0)
    # A 60-degree camera pitched down 20 degrees sees elevations -50..10 only.
    assert test["pixel_share_by_elevation_band"][:2] == [0.0, 0.0]
    assert test["pixel_share_by_elevation_band"][-1] == 0.0
    assert np.isclose(sum(report["erp_grid_pixel_share_by_elevation_band"]), 1.0)
