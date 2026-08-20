import ast
import csv
import importlib
import inspect
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch


REPRESENTATION_PROTOCOL_ID = "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"
INTEGRATION_PROTOCOL_ID = "CMX_RELPLUS_V2_3"


def _write_png(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), value)


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _artifact_fixture(tmp_path):
    cache_root = (tmp_path / "formal_cache").resolve()
    rel_root = cache_root / "RELPlus"
    mask_root = cache_root / "ValidMask"
    train_source = cache_root / "train.txt"
    eval_source = cache_root / "test.txt"
    manifest = (tmp_path / "full_manifest.csv").resolve()
    generation_resolved_manifest = (
        cache_root / "cache_manifest_resolved.csv"
    ).resolve()
    resolved_manifest = (cache_root / "audit" / "cache_manifest_resolved.csv").resolve()
    class_mapping = (tmp_path / "class_mapping.json").resolve()
    audit_report = (cache_root / "audit" / "cache_audit_summary.json").resolve()
    generation_report = (cache_root / "cache_generation_summary.json").resolve()
    preflight_report = (
        cache_root / "preflight" / "cmx_training_data_preflight_summary.json"
    ).resolve()
    rows = [
        {
            "sample_id": "area_1/train_a",
            "split": "train",
            "protocol_id": REPRESENTATION_PROTOCOL_ID,
        },
        {
            "sample_id": "area_2/train_b",
            "split": "train",
            "protocol_id": REPRESENTATION_PROTOCOL_ID,
        },
        {
            "sample_id": "area_5a/test_a",
            "split": "test",
            "protocol_id": REPRESENTATION_PROTOCOL_ID,
        },
    ]
    _write_manifest(manifest, rows)
    _write_manifest(generation_resolved_manifest, rows)
    _write_manifest(resolved_manifest, rows)
    train_source.parent.mkdir(parents=True, exist_ok=True)
    train_source.write_text("area_1/train_a\narea_2/train_b\n", encoding="utf-8")
    eval_source.write_text("area_5a/test_a\n", encoding="utf-8")
    class_mapping.write_text(
        json.dumps(
            {
                "stored_ids": {str(index): "class-{}".format(index) for index in range(14)},
                "loader_transform": "stored label - 1; 0 becomes 255",
            }
        ),
        encoding="utf-8",
    )
    identity = {
        "integration_protocol_id": INTEGRATION_PROTOCOL_ID,
        "representation_protocol_id": REPRESENTATION_PROTOCOL_ID,
        "cache_root": str(cache_root),
        "rel_plus_root": str(rel_root),
        "valid_mask_root": str(mask_root),
        "manifest_path": str(manifest),
        "resolved_manifest_path": str(resolved_manifest),
        "train_source": str(train_source),
        "eval_source": str(eval_source),
        "manifest_count": 3,
        "train_count": 2,
        "test_count": 1,
        "failure_count": 0,
    }
    audit_report.parent.mkdir(parents=True, exist_ok=True)
    generation_report.write_text(
        json.dumps(
            dict(
                identity,
                status="PASS",
                resolved_manifest_path=str(generation_resolved_manifest),
                selected_count=3,
                generated_or_resumed_count=3,
                full_cache_generated=True,
                dry_run=False,
            )
        ),
        encoding="utf-8",
    )
    audit_report.write_text(
        json.dumps(
            dict(
                identity,
                status="PASS",
                regeneration_count=70,
                regeneration_failure_count=0,
            )
        ),
        encoding="utf-8",
    )
    preflight_report.parent.mkdir(parents=True, exist_ok=True)
    preflight_report.write_text(
        json.dumps(
            dict(
                identity,
                status="PASS",
                sample_count=3,
                class_mapping=str(class_mapping),
                all_samples_decoded_this_run=True,
            )
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        training_authorized=True,
        source_compatible_invalid_accepted=True,
        integration_protocol_id=INTEGRATION_PROTOCOL_ID,
        representation_protocol_id=REPRESENTATION_PROTOCOL_ID,
        formal_cache_root=str(cache_root),
        x_root_folder=str(rel_root),
        x_valid_root_folder=str(mask_root),
        full_manifest=str(manifest),
        cache_generation_report=str(generation_report),
        generation_resolved_manifest_path=str(generation_resolved_manifest),
        resolved_manifest_path=str(resolved_manifest),
        train_source=str(train_source),
        eval_source=str(eval_source),
        class_mapping=str(class_mapping),
        cache_audit_report=str(audit_report),
        training_data_preflight_report=str(preflight_report),
        num_train_imgs=2,
        num_eval_imgs=1,
    )
    return config, audit_report, preflight_report


def test_auditor_duplicate_detection_is_linear_for_70496_ids():
    from tools.audit_full_relplus_cache import find_duplicate_sample_ids

    sample_ids = ["sample-{:05d}".format(index) for index in range(70496)]
    sample_ids.extend(("sample-00017", "sample-70000"))
    started = time.perf_counter()
    duplicates = find_duplicate_sample_ids(sample_ids)
    elapsed = time.perf_counter() - started
    assert duplicates == ["sample-00017", "sample-70000"]
    assert elapsed < 2.0
    source = inspect.getsource(find_duplicate_sample_ids)
    assert ".count(" not in source


def test_risk_regeneration_selects_ten_unique_rows_per_area():
    from tools.audit_full_relplus_cache import select_regeneration_rows

    rows = []
    areas = ("area_1", "area_2", "area_3", "area_4", "area_5a", "area_5b", "area_6")
    for area_index, area in enumerate(areas):
        for index in range(30):
            rows.append(
                {
                    "sample_id": "{}/sample_{:02d}".format(area, index),
                    "area": area,
                    "area_group": "area_5" if area in ("area_5a", "area_5b") else area,
                    "room": "room_{}".format(index % 6),
                    "camera": "camera_{}".format(index % 8),
                    "depth_invalid_ratio": str(index / 100.0),
                    "normal_quality_ratio": str(1.0 - index / 1000.0),
                    "gravity_alignment_angle_deg": str(70 + area_index + index),
                }
            )
    selected = select_regeneration_rows(rows, 70, seed=2303)
    assert len(selected) == 70
    assert len({row["sample_id"] for row in selected}) == 70
    assert Counter(row["area"] for row in selected) == Counter(
        {area: 10 for area in areas}
    )
    reasons = {row["selection_reason"] for row in selected}
    assert {"invalid_high", "normal_quality_low", "gravity_tilt_large"}.issubset(reasons)
    for area in areas:
        area_rows = [row for row in selected if row["area"] == area]
        diversity = next(
            row for row in area_rows if row["selection_reason"] == "room_camera_diversity"
        )
        risk_pairs = {
            (row["room"], row["camera"])
            for row in area_rows
            if row["selection_reason"] not in ("room_camera_diversity", "fixed_random")
        }
        assert (diversity["room"], diversity["camera"]) not in risk_pairs

    incomplete = [dict(row) for row in rows]
    incomplete[0].pop("normal_quality_ratio")
    with pytest.raises(ValueError, match="normal_quality_ratio"):
        select_regeneration_rows(incomplete, 70, seed=2303)


def test_full_cache_state_requires_all_rows_and_zero_failures():
    from tools.generate_full_relplus_cache import is_full_cache_generated

    manifest = [{"sample_id": str(index)} for index in range(40)]
    assert is_full_cache_generated(
        dry_run=False,
        failures=[],
        selected_rows=manifest,
        manifest_rows=manifest,
        generated_or_verified_rows=manifest,
    )
    assert not is_full_cache_generated(
        dry_run=False,
        failures=[{"sample_id": "3"}],
        selected_rows=manifest,
        manifest_rows=manifest,
        generated_or_verified_rows=manifest,
    )
    assert not is_full_cache_generated(
        dry_run=False,
        failures=[],
        selected_rows=manifest,
        manifest_rows=manifest,
        generated_or_verified_rows=manifest[:-1],
    )
    assert not is_full_cache_generated(
        dry_run=True,
        failures=[],
        selected_rows=manifest,
        manifest_rows=manifest,
        generated_or_verified_rows=manifest,
    )


def test_resume_validation_rejects_corruption(tmp_path):
    from tools.generate_full_relplus_cache import validate_cached_pair

    rel = tmp_path / "rel.png"
    mask = tmp_path / "mask.png"
    _write_png(rel, np.zeros((480, 480, 3), dtype=np.uint8))
    _write_png(mask, np.full((480, 480), 255, dtype=np.uint8))
    assert validate_cached_pair(rel, mask) == []
    _write_png(rel, np.zeros((12, 12, 3), dtype=np.uint8))
    assert "rel_plus_shape_or_channels" in validate_cached_pair(rel, mask)
    _write_png(rel, np.zeros((480, 480, 3), dtype=np.uint8))
    _write_png(mask, np.full((480, 480), 127, dtype=np.uint8))
    assert "valid_mask_binary" in validate_cached_pair(rel, mask)


def test_full_cache_resume_skips_valid_and_regenerates_corruption(
    tmp_path, monkeypatch
):
    import tools.generate_full_relplus_cache as generator

    sample_id = "area_1/resume_sample"
    row = {
        "sample_id": sample_id,
        "protocol_id": REPRESENTATION_PROTOCOL_ID,
        "depth_path": "synthetic_depth",
        "camera_metadata_path": "synthetic_camera",
    }
    rel = tmp_path / "RELPlus" / (sample_id + ".png")
    mask = tmp_path / "ValidMask" / (sample_id + ".png")
    _write_png(rel, np.zeros((480, 480, 3), dtype=np.uint8))
    _write_png(mask, np.full((480, 480), 255, dtype=np.uint8))

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("valid resume must not invoke the generator")

    monkeypatch.setattr(generator, "load_canonical_frame", unexpected_load)
    skipped = generator.generate_one(row, tmp_path, resume=True)
    assert skipped["cache_status"] == "RESUMED_VALID"

    raw_depth = np.ones((480, 480), dtype=np.uint16)
    monkeypatch.setattr(
        generator,
        "load_canonical_frame",
        lambda *_args, **_kwargs: (raw_depth, object(), None),
    )
    monkeypatch.setattr(
        generator,
        "generate_rel_plus_v2_1",
        lambda *_args, **_kwargs: (
            np.zeros((480, 480, 3), dtype=np.uint8),
            {"depth_valid": np.ones((480, 480), dtype=bool)},
        ),
    )
    rel.write_bytes(b"corrupt")
    regenerated = generator.generate_one(row, tmp_path, resume=True)
    assert regenerated["cache_status"] == "GENERATED"
    assert generator.validate_cached_pair(rel, mask) == []

    _write_png(mask, np.zeros((12, 12), dtype=np.uint8))
    regenerated_shape = generator.generate_one(row, tmp_path, resume=True)
    assert regenerated_shape["cache_status"] == "GENERATED"
    assert generator.validate_cached_pair(rel, mask) == []


def test_training_gate_binds_audit_preflight_cache_manifest_and_splits(tmp_path):
    from utils.training_protocol import assert_training_ready

    config, audit_path, preflight_path = _artifact_fixture(tmp_path)
    reports = assert_training_ready(config)
    assert reports["cache_generation"]["full_cache_generated"] is True
    assert reports["cache_audit"]["status"] == "PASS"
    assert reports["training_data_preflight"]["status"] == "PASS"

    generation_path = Path(config.cache_generation_report)
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation["full_cache_generated"] = False
    generation_path.write_text(json.dumps(generation), encoding="utf-8")
    with pytest.raises(RuntimeError, match="full_cache_generated"):
        assert_training_ready(config)
    generation["full_cache_generated"] = True
    generation_path.write_text(json.dumps(generation), encoding="utf-8")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["cache_root"] = str(tmp_path / "other_cache")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(RuntimeError, match="cache_root"):
        assert_training_ready(config)

    audit["cache_root"] = config.formal_cache_root
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    original_manifest = config.full_manifest
    config.full_manifest = str(tmp_path / "other_manifest.csv")
    with pytest.raises(RuntimeError, match="manifest_path"):
        assert_training_ready(config)
    config.full_manifest = original_manifest

    original_protocol = config.representation_protocol_id
    config.representation_protocol_id = "RELPLUS_WRONG_PROTOCOL"
    with pytest.raises(RuntimeError, match="representation_protocol_id"):
        assert_training_ready(config)
    config.representation_protocol_id = original_protocol

    Path(config.train_source).write_text(
        "area_2/train_b\narea_1/train_a\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="ordered sample IDs"):
        assert_training_ready(config)

    Path(config.train_source).write_text(
        "area_1/train_a\narea_2/train_b\n", encoding="utf-8"
    )
    Path(config.eval_source).write_text("area_5a/different_test\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="test ordered sample IDs"):
        assert_training_ready(config)

    Path(config.eval_source).write_text("area_5a/test_a\n", encoding="utf-8")
    preflight_path.unlink()
    with pytest.raises(RuntimeError, match="training-data preflight"):
        assert_training_ready(config)


@pytest.mark.parametrize(
    "fault,expected_reason",
    [
        ("rgb", "rgb_decode_shape_dtype"),
        ("label", "label_stored_ids"),
        ("rel", "rel_plus_decode_shape_dtype"),
        ("mask", "valid_mask_binary"),
    ],
)
def test_cmx_preflight_rejects_corrupt_modalities(tmp_path, fault, expected_reason):
    from tools.preflight_cmx_training_data_v2_3 import audit_row

    sample_id = "area_1/sample"
    rgb = tmp_path / "RGB" / (sample_id + ".png")
    label = tmp_path / "Label" / (sample_id + ".png")
    rel = tmp_path / "cache" / "RELPlus" / (sample_id + ".png")
    mask = tmp_path / "cache" / "ValidMask" / (sample_id + ".png")
    _write_png(rgb, np.zeros((480, 480, 3), dtype=np.uint8))
    _write_png(label, np.ones((480, 480), dtype=np.uint8))
    _write_png(rel, np.zeros((480, 480, 3), dtype=np.uint8))
    _write_png(mask, np.full((480, 480), 255, dtype=np.uint8))
    if fault == "rgb":
        rgb.write_bytes(b"broken")
    elif fault == "label":
        _write_png(label, np.full((480, 480), 99, dtype=np.uint8))
    elif fault == "rel":
        _write_png(rel, np.zeros((12, 12, 3), dtype=np.uint8))
    else:
        _write_png(mask, np.full((480, 480), 127, dtype=np.uint8))
    row = {
        "sample_id": sample_id,
        "rgb_path": str(rgb),
        "label_path": str(label),
        "depth_path": "unused",
        "camera_metadata_path": "unused",
        "dataset_profile": "stanford2d3d_s2d",
        "protocol_id": REPRESENTATION_PROTOCOL_ID,
        "split": "train",
    }
    result = audit_row(row, tmp_path / "cache", set(range(14)))
    assert result["status"] == "FAIL"
    assert expected_reason in result["reasons"]


def test_checkpoint_payload_epoch_must_match_filename_epoch(tmp_path):
    from tools.eval_rel_plus_v2_3_full import load_checkpoint_once

    network = torch.nn.Linear(2, 2)
    checkpoint = tmp_path / "epoch_150.pth"
    torch.save({"model": network.state_dict(), "epoch": 145}, checkpoint)
    with pytest.raises(RuntimeError, match="expected epoch 150"):
        load_checkpoint_once(network, checkpoint, expected_epoch=150)
    torch.save({"model": network.state_dict()}, checkpoint)
    with pytest.raises(RuntimeError, match="missing payload epoch"):
        load_checkpoint_once(network, checkpoint, expected_epoch=150)
    torch.save({"model": network.state_dict(), "epoch": 150}, checkpoint)
    assert load_checkpoint_once(network, checkpoint, expected_epoch=150) == 150


def test_metrics_emit_fraction_and_percent_units():
    from engine.relplus_evaluator import metrics_from_confusion

    metrics = metrics_from_confusion(np.array([[8, 2], [1, 9]], dtype=np.int64))
    assert metrics["metric_unit"] == "fraction_0_to_1"
    assert metrics["mIoU_percent"] == pytest.approx(metrics["mIoU"] * 100)
    assert metrics["pixel_accuracy_percent"] == pytest.approx(
        metrics["pixel_accuracy"] * 100
    )
    assert metrics["mean_accuracy_percent"] == pytest.approx(
        metrics["mean_accuracy"] * 100
    )
    np.testing.assert_allclose(
        np.asarray(metrics["per_class_iou_percent"]),
        np.asarray(metrics["per_class_iou"]) * 100,
    )


def test_sweep_builds_one_eight_rank_launch_per_checkpoint(tmp_path):
    from tools.eval_checkpoint_sweep_v2_3 import build_evaluator_command

    checkpoint = tmp_path / "epoch_100.pth"
    command = build_evaluator_command(
        python="/env/bin/python",
        evaluator=Path("tools/eval_rel_plus_v2_3_full.py"),
        config_module="configs.formal",
        checkpoint=checkpoint,
        output=tmp_path / "evaluation",
        epoch=100,
        nproc_per_node=8,
        launcher="torch.distributed.launch",
    )
    joined = " ".join(str(value) for value in command)
    assert "torch.distributed.launch" in joined
    assert "--nproc_per_node" in command
    assert "8" in command
    assert "--expected-epoch" in command
    assert "100" in command
    assert "eval_checkpoint_sweep_v2_3.py" not in joined


def test_v2_3_formal_config_is_import_safe_and_fail_closed():
    module = importlib.import_module(
        "configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_3_formal"
    )
    config = module.config
    assert config.integration_protocol_id == INTEGRATION_PROTOCOL_ID
    assert config.representation_protocol_id == REPRESENTATION_PROTOCOL_ID
    assert config.training_authorized is False
    assert config.full_cache_authorized is False
    assert config.source_compatible_invalid_accepted is False
    assert config.train_source.endswith("formal_cache/train.txt")
    assert config.eval_source.endswith("formal_cache/test.txt")
    assert config.training_data_preflight_report.endswith(
        "cmx_training_data_preflight_summary.json"
    )


def test_real_ddp_smoke_contains_step_save_restore_and_parameter_groups():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools" / "run_ddp_optimizer_smoke_v2_3.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"backward", "step", "save", "load", "barrier"}.issubset(calls)
    source = path.read_text(encoding="utf-8")
    for name in ("rgb_encoder", "x_encoder", "fusion", "decoder"):
        assert name in source
    for evidence in (
        "parameters_match_after_restore",
        "parameter_groups_changed_after_resume",
        "lr_continuous_across_restore",
        "lr_updated_after_resume",
        "pretrained_model_loaded",
    ):
        assert evidence in source
    assert "DISPOSABLE_DDP_SMOKE" in source


def test_three_arm_rawdepth_contract_surfaces_uint16_decision():
    from tools.audit_three_arm_x_modalities_v2_3 import classify_raw_depth_contract

    result = classify_raw_depth_contract(
        {"file_count": 70496, "dtype_counts": {"uint16": 70496, "uint8": 0}}
    )
    assert result["status"] == "RGBD_INPUT_CONTRACT_REQUIRES_DECISION"
    assert "IMREAD_GRAYSCALE" in result["reason"]
