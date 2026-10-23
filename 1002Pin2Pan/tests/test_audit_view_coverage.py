"""View-coverage audit on synthetic Stanford-style pose files."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audit_view_coverage as coverage  # noqa: E402
import cross_projection as cp  # noqa: E402


def write_pose(path, yaw, pitch, fov, location):
    rotation = cp.crop_camera_to_world(yaw, pitch).T
    location = np.asarray(location, dtype=float)
    path.write_text(json.dumps({
        "camera_k_matrix": cp.crop_intrinsics(1080, fov).tolist(),
        "camera_rt_matrix": np.hstack([rotation, (-rotation @ location)[:, None]]).tolist(),
        "camera_location": location.tolist(),
    }))


def test_sphere_directions_are_unit_and_balanced():
    directions = coverage.sphere_directions(5000)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert abs(directions.mean(axis=0)).max() < 0.01


def test_visible_share_matches_square_fov_solid_angle(tmp_path):
    # A square 90-degree pinhole spans 4*asin(sin^2 45) = 2*pi/3 sr, a sixth of the sphere.
    pose = tmp_path / "camera_ab_office_1_frame_0_domain_pose.json"
    write_pose(pose, 30, 10, 90.0, [0, 0, 0])
    cp.import_frozen_source(cp.DEFAULT_SOURCE_ROOT)
    from rel_plus.camera import load_stanford_s2d_camera_geometry

    camera = load_stanford_s2d_camera_geometry(pose, coverage.audit.NATIVE_SHAPE)
    seen = coverage.visible(camera, coverage.sphere_directions(20000))
    assert abs(seen.mean() - 1.0 / 6.0) < 0.005
    forward = cp.crop_camera_to_world(30, 10)[:, 2]
    assert coverage.visible(camera, np.stack([forward, -forward])).tolist() == [True, False]


def test_location_key_needs_stanford_name():
    assert coverage.location_key("/a/camera_0a9f_office_3_frame_12_domain_pose.json") == "0a9f"
    with pytest.raises(ValueError):
        coverage.location_key("/a/pose.json")


def test_greedy_drop_removes_redundant_views_first():
    masks = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=bool)
    groups = [np.array([0, 1, 2]), np.array([3])]
    keep = coverage.greedy_drop(masks, groups, 2, k=1)
    assert keep.tolist().count(True) == 2
    assert keep[3]


def test_audit_reports_pitch_locations_and_drop_rules(tmp_path, monkeypatch):
    # Location a: 24 level views every 15 degrees (redundant) plus two steep views.
    # Location b: four 90-degree level views, each direction seen about once.
    views = [("aa", yaw, 0, 60.0, [0, 0, 1.5]) for yaw in range(0, 360, 15)]
    views += [("aa", 0, 25, 60.0, [0, 0, 1.5]), ("aa", 180, -25, 60.0, [0, 0, 1.5])]
    views += [("bb", yaw, 0, 90.0, [4, 1, 1.5]) for yaw in (0, 90, 180, 270)]
    rows = []
    for frame, (uuid, yaw, pitch, fov, location) in enumerate(views):
        pose = tmp_path / "camera_{}_office_1_frame_{}_domain_pose.json".format(uuid, frame)
        write_pose(pose, yaw, pitch, fov, location)
        rows.append({"split": "train", "camera_metadata_path": str(pose)})
    rows.append({"split": "test", "camera_metadata_path": "unused"})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "camera_metadata_path"])
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "coverage.json"
    monkeypatch.setattr(sys, "argv", [
        "audit_view_coverage.py", "--manifest", str(manifest), "--output", str(output),
        "--drop-fraction", "0.2", "--directions", "2048",
    ])
    assert coverage.main() == 0

    report = json.loads(output.read_text())
    assert report["views"] == 30 and report["locations"] == 2
    assert report["views_per_location_percentiles"]["min"] == 4
    assert report["camera_centre_spread_m_percentiles"]["max"] < 1e-6
    assert np.isclose(report["pitch_deg_percentiles"]["max"], 25, atol=1e-6)
    assert np.isclose(report["hfov_deg_percentiles"]["max"], 90, atol=1e-6)
    assert report["drop_count"] == 6
    # Steepest first drops the two pitched views, and with them part of the
    # 25-60 degree bands that only they reach.
    steep = report["steepest_pitch_first"]
    assert steep["max_abs_pitch_kept_deg"] < 1e-6
    steep_upper = steep["lost_share_by_level"]["1"]["by_elevation_band"][1]
    assert steep_upper > 0.1
    # Random draws hit location b; greedy spends the drop on location a.
    assert report["random"]["hole_share_mean"] > 0.05
    greedy = report["greedy_coverage"]
    assert greedy["min_views_left_at_a_location"] == 4
    assert greedy["lost_share_by_level"]["1"]["by_elevation_band"][1] < 0.1 * steep_upper
    assert greedy["lost_share_by_level"]["1"]["whole_sphere"] < 0.5 * report["random"]["hole_share_mean"]
    assert report["rule"]["verdict"] == "COVERAGE_RULE_HAS_ROOM"
    level = report["coverage_all_views"]["by_elevation_band"][2]
    assert level["share_seen_by_at_least"]["1"] == 1.0


def test_decide_thresholds():
    assert coverage.decide(0.005, 0.0) == "REDUNDANT_RANDOM_SUFFICES"
    assert coverage.decide(0.04, 0.01) == "COVERAGE_RULE_HAS_ROOM"
    assert coverage.decide(0.04, 0.03) == "COVERAGE_RULE_NO_BETTER"
