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
