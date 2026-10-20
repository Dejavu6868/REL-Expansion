"""Unit checks for the attention profile of level crops against the whole panorama."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import attention_profile as ap  # noqa: E402
import cross_projection as cp  # noqa: E402
import eval_pano_scope as es  # noqa: E402


def test_profile_bins_each_query_by_its_keys_angle():
    torch = pytest.importorskip("torch")
    from test_cross_projection_synthetic import render_erp_raw_depth

    height, width, pad, sr = 64, 128, 32, 2
    directions = es.padded_erp_directions(height, width, pad)
    range_m = cp.decode_erp_depth(render_erp_raw_depth(height, width))
    points = ap.with_wrap(ap.points_from_range(cp.erp_directions(height, width), range_m), pad)
    profile = ap.Profile(query_step=1)
    profile.set_input("erp", directions, points)
    probabilities = torch.zeros(1, 2, 16 * 48, 8 * 24)
    probabilities[..., 0] = 1  # every query attends to key 0 alone
    profile("rgb", 1, probabilities, 16, 48, sr)
    profile("rgb", 3, probabilities, 16, 48, sr)  # stages 3 and 4 are not profiled
    (row,) = profile.summary()

    queries = es.token_centres(directions, 16, 48)
    query_points = es.token_centres(points, 16, 48)
    elevation = np.degrees(np.arcsin(queries[:, 2]))
    chosen = (np.abs(elevation) <= ap.QUERY_BAND_DEG) & np.isfinite(query_points).all(axis=1)
    key = es.token_centres(directions, 16, 48, sr)[0]
    angle = np.degrees(np.arccos(np.clip(queries[chosen] @ key, -1, 1)))
    expected = np.histogram(angle, bins=ap.ANGLE_EDGES_DEG)[0] / chosen.sum()
    assert row["queries"] == chosen.sum() and (row["presentation"], row["stream"], row["stage"]) == ("erp", "rgb", 1)
    assert np.allclose(row["angle_mass"], expected)
    assert abs(row["mean_angle_deg"] - angle.mean()) < 1e-6
    assert abs(row["mass_beyond_cap"] - (angle > es.TRAINING_CORNER_DEG).mean()) < 1e-9
    distance = np.linalg.norm(query_points[chosen] - es.token_centres(points, 16, 48, sr)[0], axis=1)
    assert abs(row["mean_distance_m"] - distance.mean()) < 1e-6 and row["no_depth_mass"] == 0
    assert abs(sum(row["distance_mass"]) - 1) < 1e-9


def test_spaced_samples_cover_the_list():
    assert ap.spaced(list(range(10)), 3) == [0, 4, 9]
    assert ap.spaced(list(range(3)), 5) == [0, 1, 2]


def test_attention_profile_runs_end_to_end(monkeypatch, tmp_path):
    from test_eval_pano_scope import write_fake_eval

    shared = write_fake_eval(tmp_path)
    monkeypatch.setattr(sys, "argv", ["attention_profile.py"] + shared + [
        "--panoramas", "1", "--query-step", "1", "--output", str(tmp_path / "profile"),
    ])
    assert ap.main() == 0
    report = json.loads((tmp_path / "profile" / "attention_profile.json").read_text())
    keys = [(row["presentation"], row["stream"], row["stage"]) for row in report["profiles"]]
    assert keys == [(p, s, stage) for p in ("crop", "erp") for s in ("rgb", "x") for stage in (1, 2)]
    beyond_90 = ap.ANGLE_EDGES_DEG[:-1] >= 90
    for row in report["profiles"]:
        assert row["queries"] > 0 and abs(sum(row["angle_mass"]) - 1) < 1e-6
        assert abs(sum(row["distance_mass"]) + row["no_depth_mass"] - 1) < 1e-6
        if row["presentation"] == "crop":  # an 81-degree diagonal bounds a crop's key angles
            assert sum(np.asarray(row["angle_mass"])[beyond_90]) == 0
    assert report["reference_cap_deg"] == es.TRAINING_CORNER_DEG and len(report["samples"]) == 1
