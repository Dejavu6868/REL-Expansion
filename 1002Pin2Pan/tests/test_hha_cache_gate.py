"""Regression checks for the evidence that unlocks HHA transfer evaluation."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_hha_cache as check
import eval_pano_transfer as ev


def sample_result(candidate, cached, sample_id="sample"):
    comparison = check.compare(candidate, cached)
    return {
        "sample_id": sample_id,
        "native_then_resize": comparison,
        "resize_then_hha": comparison,
    }


def test_cache_gate_rejects_errors_hidden_by_pixel_medians():
    cached = np.zeros((10, 10, 3), dtype=np.uint8)
    candidate = cached.copy()
    candidate[:4] = 255
    result = sample_result(candidate, cached)
    assert all(c["median_abs"] == 0 for c in result["native_then_resize"]["as_stored"])
    assert check.summarize_results([result])["status"] == "NO_MATCH"


def test_cache_gate_requires_every_sample_to_match():
    cached = np.zeros((10, 10, 3), dtype=np.uint8)
    results = [sample_result(cached, cached, str(i)) for i in range(3)]
    results.append(sample_result(cached + 100, cached, "bad"))
    assert check.summarize_results(results)["status"] == "NO_MATCH"


def test_cache_gate_finds_one_exact_recipe_and_records_its_order():
    native = np.full((8, 8, 3), [20, 70, 110], dtype=np.uint8)
    cached = native[:, :, ::-1].copy()
    result = sample_result(native, cached)
    result["resize_then_hha"] = check.compare(native + 10, cached)
    report = check.summarize_results([result])
    assert report["status"] == "MATCH"
    assert report["best"]["variant"] == "native_then_resize"
    assert report["best"]["channel_order"] == "reversed"


def test_cache_gate_rejects_empty_evidence():
    with pytest.raises(ValueError, match="no HHA samples"):
        check.summarize_results([])


def test_evaluator_rejects_old_median_only_match_report(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"status": "MATCH", "best": {
        "variant": "native_then_resize", "channel_order": "as_stored",
        "median_of_worst_channel_median": 0.0,
    }}))
    with pytest.raises(ValueError, match="rerun"):
        ev.hha_cache_recipe(path)


def test_evaluator_checks_report_evidence_and_cache_identity(tmp_path):
    cached = np.full((8, 8, 3), [20, 70, 110], dtype=np.uint8)
    report = check.summarize_results([sample_result(cached, cached)])
    report["hha_root"] = str(tmp_path / "HHA")
    path = tmp_path / "match.json"
    path.write_text(json.dumps(report))
    recipe = ev.hha_cache_recipe(path, tmp_path / "HHA")
    assert recipe["variant"] == "native_then_resize"
    assert recipe["channel_order"] == [0, 1, 2]
    with pytest.raises(ValueError, match="cache root"):
        ev.hha_cache_recipe(path, tmp_path / "another_cache")
    report["best"]["channel_order"] = "unrecognised"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="evidence"):
        ev.hha_cache_recipe(path)


def test_cache_gate_reports_the_closest_recipe_when_none_is_exact():
    cached = np.zeros((10, 10, 3), dtype=np.uint8)
    close = cached.copy()
    close[0, 0] = 250
    far = cached + 30
    result = {"sample_id": "s", "native_then_resize": check.compare(close, cached),
              "resize_then_hha": check.compare(far, cached)}
    report = check.summarize_results([result])
    assert report["status"] == "NO_MATCH"
    assert report["best"]["variant"] == "native_then_resize"
    assert report["best"]["channel_order"] == "as_stored"


def test_cache_gate_can_unlock_a_non_nearest_resize_kernel():
    import cv2

    native = np.random.default_rng(0).integers(0, 256, (27, 27, 3), dtype=np.uint8)
    cached = cv2.resize(native, (12, 12), interpolation=cv2.INTER_AREA)
    result = {"sample_id": "s"}
    for variant in check.NATIVE_RESIZES:
        result[variant] = check.compare(check.resize_native_hha(native, (12, 12), variant), cached)
    result["resize_then_hha"] = check.compare(cached + 1, cached)
    report = check.summarize_results([result])
    assert report["status"] == "MATCH"
    assert report["best"]["variant"] == "native_then_resize_area"


def test_pixel_centre_nearest_differs_from_opencv_nearest():
    import cv2

    native = np.arange(9 * 9 * 3, dtype=np.uint8).reshape(9, 9, 3)
    centre = check.resize_native_hha(native, (4, 4), "native_then_resize_nearest_center")
    assert np.array_equal(centre, native[[1, 3, 5, 7]][:, [1, 3, 5, 7]])
    assert not np.array_equal(centre, cv2.resize(native, (4, 4), interpolation=cv2.INTER_NEAREST))


def near_result():
    cached = np.full((20, 20, 3), 100, dtype=np.uint8)
    candidate = cached.copy()
    candidate[:, :, 1] += np.uint8(1)
    candidate[0, 0, 0] = 230
    return {"sample_id": "s", "native_then_resize": check.compare(candidate, cached),
            "resize_then_hha": check.compare(cached + 30, cached)}


def test_cache_gate_reports_rounding_level_agreement_as_near_match():
    report = check.summarize_results([near_result()])
    assert report["status"] == "NEAR_MATCH"
    assert report["best"]["variant"] == "native_then_resize"


def test_near_match_rejects_two_loose_channels_or_a_wide_p95():
    cached = np.full((20, 20, 3), 100, dtype=np.uint8)
    two_loose = cached.copy()
    two_loose[0, 0, :2] = 230
    wide = cached.copy()
    wide[:2, :, 0] = 103
    for candidate in (two_loose, wide):
        result = {"sample_id": "s", "native_then_resize": check.compare(candidate, cached),
                  "resize_then_hha": check.compare(cached + 30, cached)}
        assert check.summarize_results([result])["status"] == "NO_MATCH"


def test_evaluator_accepts_near_match_only_when_asked(tmp_path):
    report = check.summarize_results([near_result()])
    path = tmp_path / "near.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="accept-hha-near-match"):
        ev.hha_cache_recipe(path)
    recipe = ev.hha_cache_recipe(path, accept_near_match=True)
    assert recipe["cache_status"] == "NEAR_MATCH"
    report["status"] = "MATCH"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="evidence"):
        ev.hha_cache_recipe(path, accept_near_match=True)
