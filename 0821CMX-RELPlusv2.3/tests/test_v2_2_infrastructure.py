import builtins
import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from dataloader.RGBXDataset import RGBXDataset
from rel_plus.integration.cmx_preprocess import sample_spatial_transform


def _write_png(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), value)


def _dataset_setting(root, source):
    return {
        "rgb_root": str(root / "RGB"),
        "rgb_format": ".png",
        "gt_root": str(root / "Label"),
        "gt_format": ".png",
        "transform_gt": True,
        "x_root": str(root / "RELPlus"),
        "x_format": ".png",
        "x_single_channel": False,
        "x_mode": "rel_plus_v2_1",
        "x_valid_root": str(root / "ValidMask"),
        "x_valid_format": ".png",
        "channel_order": ("EGVIA", "LOA", "ReD"),
        "train_source": str(source),
        "eval_source": str(source),
        "class_names": ["class-{}".format(index) for index in range(13)],
    }


def test_dataset_keeps_actual_length_and_signed_label_mapping(tmp_path, monkeypatch):
    sample_ids = ["area_1/a", "area_1/b"]
    source = tmp_path / "train.txt"
    source.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
    label = np.array([[0, 1], [13, 255]], dtype=np.uint8)
    for sample_id in sample_ids:
        _write_png(tmp_path / "RGB" / (sample_id + ".png"), np.zeros((2, 2, 3), np.uint8))
        _write_png(tmp_path / "Label" / (sample_id + ".png"), label)
        _write_png(tmp_path / "RELPlus" / (sample_id + ".png"), np.zeros((2, 2, 3), np.uint8))
        _write_png(tmp_path / "ValidMask" / (sample_id + ".png"), np.full((2, 2), 255, np.uint8))

    dataset = RGBXDataset(_dataset_setting(tmp_path, source), "train", file_length=8)
    assert len(dataset) == 2
    monkeypatch.setattr(torch, "randperm", lambda *args, **kwargs: pytest.fail("per-item randperm"))
    sample = dataset[0]
    assert sample["label"].tolist() == [[255, 0], [12, 255]]


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_fixed_length_sampler_is_reproducible_rank_partition(world_size):
    from dataloader.samplers import FixedLengthDistributedSampler

    dataset = list(range(7))
    logical = 8
    by_rank = []
    for rank in range(world_size):
        sampler = FixedLengthDistributedSampler(
            dataset,
            logical_samples_per_epoch=logical,
            num_replicas=world_size,
            rank=rank,
            seed=41,
        )
        sampler.set_epoch(0)
        first = list(sampler)
        sampler.set_epoch(0)
        assert first == list(sampler)
        assert len(first) == logical // world_size
        by_rank.append(first)
    flattened = [index for rank_values in by_rank for index in rank_values]
    counts = Counter(flattened)
    assert set(counts) == set(range(7))
    assert sum(value - 1 for value in counts.values()) == 1

    epoch_zero = FixedLengthDistributedSampler(
        dataset, logical_samples_per_epoch=logical, seed=41
    )
    epoch_one = FixedLengthDistributedSampler(
        dataset, logical_samples_per_epoch=logical, seed=41
    )
    epoch_zero.set_epoch(0)
    epoch_one.set_epoch(1)
    assert list(epoch_zero) != list(epoch_one)


def test_no_pad_eval_sampler_and_confusion_merge_are_exact():
    from dataloader.samplers import DistributedEvalSamplerNoPad
    from engine.relplus_evaluator import merge_confusion_matrices

    dataset = list(range(11))
    parts = [
        list(DistributedEvalSamplerNoPad(dataset, num_replicas=rank_count, rank=rank))
        for rank_count in (1, 2, 8)
        for rank in range(rank_count)
    ]
    assert parts[0] == list(range(11))
    two_rank = parts[1:3]
    eight_rank = parts[3:11]
    assert sorted(value for part in two_rank for value in part) == list(range(11))
    assert sorted(value for part in eight_rank for value in part) == list(range(11))
    assert len({value for part in eight_rank for value in part}) == 11

    matrices = [np.eye(3, dtype=np.int64) * index for index in (1, 2, 3)]
    np.testing.assert_array_equal(
        merge_confusion_matrices(matrices), np.eye(3, dtype=np.int64) * 6
    )


def test_v2_2_formal_config_import_has_no_file_io(monkeypatch):
    module_name = "configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_2_formal"
    sys.modules.pop(module_name, None)

    def reject_open(*args, **kwargs):
        raise AssertionError("formal config performed import-time file I/O")

    monkeypatch.setattr(builtins, "open", reject_open)
    formal = importlib.import_module(module_name).config
    assert formal.dataset_split == "Stanford2D3D_S2D_official_train_test"
    assert formal.dataset_fold is None
    assert formal.integration_protocol_id == "CMX_RELPLUS_V2_2"
    assert formal.representation_protocol_id == "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"
    assert formal.num_train_imgs == 52903
    assert formal.num_eval_imgs == 17593
    assert formal.logical_samples_per_epoch == 52904
    assert formal.niters_per_epoch == 6613
    assert formal.primary_endpoint == "epoch_200"
    assert formal.secondary_endpoint == "test_selected_best"
    assert formal.training_authorized is False
    assert formal.full_cache_authorized is False
    assert formal.data_ready is False
    assert formal.cache_audit_report.endswith("cache_audit_summary.json")


def test_v2_2_formal_config_covers_train_fields_and_three_arm_controls():
    import ast

    from configs.stanford2d3d_s2d.cmx_mit_b2_hha_v2_2_formal import config as hha
    from configs.stanford2d3d_s2d.cmx_mit_b2_rgbd_v2_2_formal import config as rgbd
    from configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_2_formal import config as relplus
    from configs.stanford2d3d_s2d.comparison_v2_2 import frozen_control_fields

    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "train.py").read_text(encoding="utf-8"))
    fields = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
    }
    assert not sorted(field for field in fields if not hasattr(relplus, field))
    assert frozen_control_fields(rgbd) == frozen_control_fields(hha)
    assert frozen_control_fields(hha) == frozen_control_fields(relplus)
    assert {rgbd.x_mode, hha.x_mode, relplus.x_mode} == {
        "standard", "rel_plus_v2_1"
    }
    assert len({rgbd.x_root_folder, hha.x_root_folder, relplus.x_root_folder}) == 3
    assert all(arm.training_authorized is False for arm in (rgbd, hha, relplus))


def test_training_readiness_is_derived_from_cache_audit(tmp_path):
    from utils.training_protocol import assert_training_ready

    report = tmp_path / "cache_audit_summary.json"
    cfg = SimpleNamespace(
        training_authorized=True,
        data_ready=True,
        cache_audit_report=str(report),
        integration_protocol_id="CMX_RELPLUS_V2_2",
        representation_protocol_id="RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT",
        num_train_imgs=2,
        num_eval_imgs=1,
    )
    with pytest.raises(RuntimeError, match="audit report"):
        assert_training_ready(cfg)
    report.write_text(
        json.dumps(
            {
                "status": "PASS",
                "integration_protocol_id": cfg.integration_protocol_id,
                "representation_protocol_id": cfg.representation_protocol_id,
                "manifest_count": 3,
                "train_count": 2,
                "test_count": 1,
                "failure_count": 0,
            }
        ),
        encoding="utf-8",
    )
    assert_training_ready(cfg)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["failure_count"] = 1
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="failure_count"):
        assert_training_ready(cfg)


def test_full_cache_scope_gate_and_cached_pair_validation(tmp_path):
    from tools.generate_full_relplus_cache import (
        resolve_generation_scope,
        validate_cached_pair,
    )

    rows = [{"sample_id": "sample-{}".format(index)} for index in range(40)]
    assert len(resolve_generation_scope(rows, limit=36, dry_run=False, authorized=False)) == 36
    with pytest.raises(RuntimeError, match="full_cache_authorized"):
        resolve_generation_scope(rows, limit=None, dry_run=False, authorized=False)

    rel_path = tmp_path / "rel.png"
    mask_path = tmp_path / "mask.png"
    _write_png(rel_path, np.zeros((480, 480, 3), dtype=np.uint8))
    _write_png(mask_path, np.full((480, 480), 255, dtype=np.uint8))
    assert validate_cached_pair(rel_path, mask_path, (480, 480)) == []
    _write_png(mask_path, np.zeros((12, 12), dtype=np.uint8))
    assert "valid_mask_shape" in validate_cached_pair(rel_path, mask_path, (480, 480))


def test_cache_auditor_detects_structural_failures(tmp_path):
    from tools.audit_full_relplus_cache import audit_cache_rows

    sample_id = "area_1/sample"
    row = {
        "sample_id": sample_id,
        "split": "train",
        "protocol_id": "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT",
    }
    rel_path = tmp_path / "RELPlus" / (sample_id + ".png")
    mask_path = tmp_path / "ValidMask" / (sample_id + ".png")
    _write_png(rel_path, np.full((480, 480, 3), 17, dtype=np.uint8))
    _write_png(mask_path, np.full((480, 480), 255, dtype=np.uint8))
    summary, failures = audit_cache_rows([row], tmp_path, expected_shape=(480, 480))
    assert summary["status"] == "PASS"
    assert failures == []

    rel_path.unlink()
    summary, failures = audit_cache_rows([row], tmp_path, expected_shape=(480, 480))
    assert any(item["reason"] == "rel_plus_missing" for item in failures)
    _write_png(rel_path, np.full((480, 480, 3), 17, dtype=np.uint8))

    mask_path.write_bytes(b"not a png")
    summary, failures = audit_cache_rows([row], tmp_path, expected_shape=(480, 480))
    assert summary["status"] == "FAIL"
    assert any(item["reason"] == "valid_mask_decode" for item in failures)
    _write_png(mask_path, np.full((12, 12), 255, dtype=np.uint8))
    _, failures = audit_cache_rows([row], tmp_path, expected_shape=(480, 480))
    assert any(item["reason"] == "valid_mask_shape" for item in failures)
    _write_png(mask_path, np.full((480, 480), 65535, dtype=np.uint16))
    _, failures = audit_cache_rows([row], tmp_path, expected_shape=(480, 480))
    assert any(item["reason"] == "valid_mask_dtype" for item in failures)
    _write_png(mask_path, np.full((480, 480, 3), 255, dtype=np.uint8))
    _, failures = audit_cache_rows([row], tmp_path, expected_shape=(480, 480))
    assert any(item["reason"] == "valid_mask_shape" for item in failures)
    _write_png(mask_path, np.full((480, 480), 255, dtype=np.uint8))
    _write_png(tmp_path / "RELPlus" / "extra.png", np.zeros((480, 480, 3), np.uint8))
    _, failures = audit_cache_rows([row], tmp_path, expected_shape=(480, 480))
    assert any(item["reason"] == "extra_rel_plus" for item in failures)


def test_checkpoint_endpoint_selection_is_explicit():
    from tools.eval_checkpoint_sweep_v2_2 import select_endpoints

    rows = [
        {"epoch": 105, "mIoU": 0.50},
        {"epoch": 200, "mIoU": 0.49},
        {"epoch": 100, "mIoU": 0.45},
    ]
    ordered, primary, secondary = select_endpoints(rows)
    assert [row["epoch"] for row in ordered] == [100, 105, 200]
    assert primary["epoch"] == 200
    assert secondary["epoch"] == 105
    assert secondary["endpoint"] == "test_selected_best"
    with pytest.raises(ValueError, match="epoch 200"):
        select_endpoints(rows[:-2])


def test_rng_contract_accepts_only_python_random_and_numpy_generator():
    import random

    sample_spatial_transform((8, 9), [1.0], (6, 7), random.Random(1))
    sample_spatial_transform((8, 9), [1.0], (6, 7), np.random.default_rng(1))
    with pytest.raises(TypeError, match="RandomState"):
        sample_spatial_transform((8, 9), [1.0], (6, 7), np.random.RandomState(1))
    with pytest.raises(TypeError, match="np.random module"):
        sample_spatial_transform((8, 9), [1.0], (6, 7), np.random)


def test_engine_defaults_and_logger_path(tmp_path, monkeypatch):
    from engine.engine import Engine
    from engine.logger import get_logger

    monkeypatch.setattr(sys, "argv", ["test"])
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    engine = Engine()
    assert engine.local_rank == 0
    assert engine.world_size == 1

    log_file = tmp_path / "logs" / "train.log"
    logger = get_logger(str(log_file.parent), str(log_file))
    logger.info("v2.2 startup")
    for handler in logger.handlers:
        handler.flush()
    assert "v2.2 startup" in log_file.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="self-referential"):
        engine.validate_log_link_paths(str(log_file.parent), str(log_file.parent))


def test_loss_accumulation_detaches_graph():
    from utils.training_runtime import accumulate_loss

    total = 0.0
    for _ in range(5):
        loss = (torch.ones(1, requires_grad=True) * 2).sum()
        total = accumulate_loss(total, loss)
    assert isinstance(total, float)
    assert total == 10.0


def test_shared_training_runtime_and_smoke_forbid_steps():
    import ast

    from utils.training_runtime import build_training_runtime

    assert callable(build_training_runtime)
    root = Path(__file__).resolve().parents[1]
    smoke = root / "tools" / "validate_formal_startup_no_step_v2_2.py"
    tree = ast.parse(smoke.read_text(encoding="utf-8"))
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"step", "save_checkpoint", "save_and_link_checkpoint"}:
                forbidden.append((node.func.attr, node.lineno))
    assert forbidden == []
