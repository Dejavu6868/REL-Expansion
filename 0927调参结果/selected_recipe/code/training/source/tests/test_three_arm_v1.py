import json
import random
import copy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch


def _write_png(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), value)


def _dataset_setting(tmp_path, *, x_mode, x_value, x_single_channel):
    sample_id = "area_1/sentinel"
    rgb_root = tmp_path / "RGB"
    label_root = tmp_path / "Label"
    x_root = tmp_path / "X"
    _write_png(
        rgb_root / (sample_id + ".png"),
        np.zeros((3, 4, 3), dtype=np.uint8),
    )
    _write_png(
        label_root / (sample_id + ".png"),
        np.ones((3, 4), dtype=np.uint8),
    )
    _write_png(x_root / (sample_id + ".png"), x_value)
    source = tmp_path / "split.txt"
    source.write_text(sample_id + "\n", encoding="utf-8")
    return {
        "rgb_root": str(rgb_root),
        "rgb_format": ".png",
        "gt_root": str(label_root),
        "gt_format": ".png",
        "transform_gt": True,
        "x_root": str(x_root),
        "x_format": ".png",
        "x_single_channel": x_single_channel,
        "x_mode": x_mode,
        "x_valid_root": None,
        "x_valid_format": None,
        "channel_order": None,
        "train_source": str(source),
        "eval_source": str(source),
        "class_names": ["class-{}".format(index) for index in range(13)],
    }


def test_rawdepth_uint8_loader_is_explicit_and_repeat3(tmp_path):
    from dataloader.RGBXDataset import RGBXDataset

    raw = np.arange(12, dtype=np.uint8).reshape(3, 4)
    setting = _dataset_setting(
        tmp_path,
        x_mode="rawdepth_uint8_repeat3",
        x_value=raw,
        x_single_channel=True,
    )
    item = RGBXDataset(setting, "val")[0]
    assert item["modal_x"].dtype == np.uint8
    assert item["modal_x"].shape == (3, 4, 3)
    for channel in range(3):
        np.testing.assert_array_equal(item["modal_x"][..., channel], raw)


def test_rawdepth_loader_rejects_uint16_instead_of_silent_compression(tmp_path):
    from dataloader.RGBXDataset import RGBXDataset

    raw = (np.arange(12, dtype=np.uint16).reshape(3, 4) << 8)
    setting = _dataset_setting(
        tmp_path,
        x_mode="rawdepth_uint8_repeat3",
        x_value=raw,
        x_single_channel=True,
    )
    with pytest.raises(TypeError, match="uint8"):
        RGBXDataset(setting, "val")[0]


def test_hha_loader_preserves_frozen_executable_channel_order(tmp_path):
    from dataloader.RGBXDataset import RGBXDataset

    hha = np.zeros((3, 4, 3), dtype=np.uint8)
    hha[..., 0] = 11
    hha[..., 1] = 22
    hha[..., 2] = 33
    setting = _dataset_setting(
        tmp_path,
        x_mode="hha_frozen_cache",
        x_value=hha,
        x_single_channel=False,
    )
    item = RGBXDataset(setting, "val")[0]
    np.testing.assert_array_equal(item["modal_x"][0, 0], [11, 22, 33])


def test_hha_loader_rejects_non_three_channel_cache(tmp_path):
    from dataloader.RGBXDataset import RGBXDataset

    setting = _dataset_setting(
        tmp_path,
        x_mode="hha_frozen_cache",
        x_value=np.zeros((3, 4), dtype=np.uint8),
        x_single_channel=False,
    )
    with pytest.raises(ValueError, match="three-channel"):
        RGBXDataset(setting, "val")[0]


def test_transform_trace_is_emitted_only_for_explicit_audit_mode(tmp_path):
    from dataloader.RGBXDataset import RGBXDataset

    setting = _dataset_setting(
        tmp_path,
        x_mode="rawdepth_uint8_repeat3",
        x_value=np.zeros((3, 4), dtype=np.uint8),
        x_single_channel=True,
    )

    class TracedPreprocess:
        last_transform = SimpleNamespace(
            scale=0.5,
            scaled_height=2,
            scaled_width=3,
            crop_top=0,
            crop_left=0,
            output_height=3,
            output_width=4,
        )

        def __call__(self, rgb, gt, modal_x):
            return rgb, gt, modal_x

    ordinary = RGBXDataset(setting, "train", TracedPreprocess())[0]
    assert "transform_trace" not in ordinary
    assert "worker_seed" not in ordinary

    setting["emit_transform_trace"] = True
    setting["train_horizontal_flip"] = False
    traced = RGBXDataset(setting, "train", TracedPreprocess())[0]
    assert traced["transform_trace"] == {
        "scale": 0.5,
        "scaled_height": 2,
        "scaled_width": 3,
        "crop_top": 0,
        "crop_left": 0,
        "output_height": 3,
        "output_width": 4,
        "pad_top": 0,
        "pad_bottom": 1,
        "pad_left": 0,
        "pad_right": 1,
        "horizontal_flip": False,
    }
    assert traced["worker_seed"].isdigit()


def test_three_arm_formal_configs_freeze_training_controls():
    from configs.stanford2d3d_s2d.cmx_mit_b2_hha_three_arm_v1 import (
        config as hha,
    )
    from configs.stanford2d3d_s2d.cmx_mit_b2_rgbd_three_arm_v1 import (
        config as rgbd,
    )

    frozen = (
        "train_source",
        "eval_source",
        "rgb_root_folder",
        "gt_root_folder",
        "class_mapping",
        "backbone",
        "decoder",
        "seed",
        "batch_size",
        "num_workers",
        "nepochs",
        "niters_per_epoch",
        "logical_samples_per_epoch",
        "criterion",
        "focal_gamma",
        "loss_reduction",
        "optimizer",
        "lr",
        "weight_decay",
        "scheduler",
        "warm_up_epoch",
        "augmentation_profile",
        "train_horizontal_flip",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "checkpoint_epochs",
        "primary_endpoint",
    )
    for field in frozen:
        assert getattr(rgbd, field) == getattr(hha, field), field
    assert rgbd.integration_protocol_id == "CMX_S2D_THREE_ARM_V1"
    assert hha.integration_protocol_id == "CMX_S2D_THREE_ARM_V1"
    assert rgbd.x_mode == "rawdepth_uint8_repeat3"
    assert hha.x_mode == "hha_frozen_cache"
    assert rgbd.training_authorized is False
    assert hha.training_authorized is False


def test_sampler_reseeds_reproducibly_across_required_epochs():
    from dataloader.samplers import FixedLengthDistributedSampler

    dataset = list(range(17))
    observed = {}
    for epoch in (1, 2, 3, 10, 11):
        sampler = FixedLengthDistributedSampler(
            dataset,
            logical_samples_per_epoch=24,
            num_replicas=8,
            rank=0,
            seed=12345,
        )
        sampler.set_epoch(epoch)
        first = list(sampler)
        sampler.set_epoch(epoch)
        assert list(sampler) == first
        observed[epoch] = first
    assert len({tuple(values) for values in observed.values()}) == len(observed)


def test_recovery_checkpoint_uses_two_atomic_slots_and_runtime_state(tmp_path):
    from engine.engine import Engine

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sampler = SimpleNamespace(seed=12345, epoch=5)
    engine = Engine.__new__(Engine)
    engine.distributed = False
    engine.local_rank = 0
    engine.world_size = 1
    engine.state = SimpleNamespace(
        epoch=5,
        iteration=10,
        model=model,
        optimizer=optimizer,
        sampler=sampler,
    )
    rank_states = engine.collect_rank_runtime_states()
    first = engine.save_recovery_checkpoint(
        str(tmp_path), step=5, rank_runtime_states=rank_states
    )
    assert Path(first).name == "recovery_epoch_odd.pth"
    payload = torch.load(first, map_location="cpu")
    assert payload["epoch"] == 5
    assert payload["rank_runtime_states"][0]["sampler"] == {
        "seed": 12345,
        "epoch": 5,
    }
    assert "python_rng_state" in payload["rank_runtime_states"][0]
    assert "numpy_rng_state" in payload["rank_runtime_states"][0]
    assert "torch_rng_state" in payload["rank_runtime_states"][0]
    assert not list(tmp_path.glob("*.tmp-*"))

    engine.state.epoch = 10
    sampler.epoch = 10
    second = engine.save_recovery_checkpoint(
        str(tmp_path), step=5, rank_runtime_states=engine.collect_rank_runtime_states()
    )
    assert Path(second).name == "recovery_epoch_even.pth"
    assert Path(first).is_file()
    assert Path(second).is_file()


def test_three_arm_launcher_freezes_workers_rng_and_builds_narrow_overlay():
    from configs.stanford2d3d_s2d.cmx_mit_b2_rgbd_three_arm_v1 import (
        config as rgbd,
    )
    from tools.launch_three_arm_training_v1 import (
        build_resolved_payload,
        validate_frozen_controls,
    )

    validate_frozen_controls(rgbd)
    broken = copy.deepcopy(rgbd)
    broken.num_workers = 8
    with pytest.raises(RuntimeError, match="num_workers"):
        validate_frozen_controls(broken)

    payload = build_resolved_payload(
        rgbd,
        launch_id="unit-test",
        launcher="torch.distributed.launch",
        nproc_per_node=8,
    )
    assert payload["integration_protocol_id"] == "CMX_S2D_THREE_ARM_V1"
    assert payload["comparison_arm"] == "rgbd"
    assert payload["runtime_overrides"]["training_authorized"] is True
    assert (
        payload["runtime_overrides"]["input_audit_report"]
        == rgbd.input_audit_report
    )
    assert payload["runtime_overrides"]["source_compatible_invalid_accepted"] is False


def test_three_arm_launcher_requires_all_shared_prerequisite_reports(tmp_path):
    from tools.launch_three_arm_training_v1 import validate_shared_prerequisites

    reports = {
        "equivalence": {
            "status": "PASS",
            "decision": "PASS_WITH_EXPLAINED_CUDA_BACKWARD_NONDETERMINISM",
            "steps": 200,
            "single_gpu": True,
            "forward_and_loss_map_bitwise_exact": True,
            "formal_training_flags_unchanged": True,
            "envelope_checks": {"loss": True},
            "numerical_source_files": [{"path": "models", "exact": True}],
        },
        "initialization": {
            "status": "PASS",
            "seed_reset_before_each_model": True,
            "state_dict_order_exact": True,
            "state_dict_shapes_exact": True,
            "all_state_tensors_bitwise_exact": True,
            "parameter_count_exact": True,
            "missing_keys_identical": True,
            "unexpected_keys_identical": True,
            "cross_arm_mismatch_count": 0,
        },
        "sampler": {
            "status": "PASS",
            "epochs": [1, 2, 3, 10, 11],
            "world_size": 8,
            "sample_ids_per_rank_epoch": 6613,
            "element_comparison_count": 264520,
            "all_ordered_sample_ids_exact": True,
            "mismatch_count": 0,
        },
        "dataloader": {
            "status": "PASS",
            "epochs": [1, 2, 3, 10, 11],
            "rank": 0,
            "world_size": 8,
            "num_workers": 16,
            "head_count_per_epoch": 50,
            "tail_count_per_epoch": 50,
            "trace_count_per_arm": 500,
            "all_trace_fields_exact": True,
            "horizontal_flip_all_false": True,
            "mismatch_count": 0,
            "constructed_trace_copy_used": False,
        },
        "evaluator": {
            "status": "PASS",
            "reference_world_size": 8,
            "candidate_world_size": 8,
            "confusion_matrix_exact": True,
            "per_class_iou_exact": True,
            "metric_exact": {"mIoU": True},
            "expected_metric_exact": {"mIoU_percent": True},
            "protocol_exact": {"class_count": True},
            "manifest": {
                "reference_count": 17593,
                "candidate_count": 17593,
                "reference_unique_count": 17593,
                "candidate_unique_count": 17593,
                "sample_id_sets_exact": True,
            },
            "evaluator_source_files": [{"path": "models", "exact": True}],
        },
        "rank_evaluator": {
            "status": "PASS",
            "reference_world_size": 8,
            "candidate_world_size": 1,
            "confusion_matrix_exact": True,
            "per_class_iou_exact": True,
            "metric_exact": {"mIoU": True},
            "expected_metric_exact": {"mIoU_percent": True},
            "protocol_exact": {"class_count": True},
            "manifest": {
                "reference_count": 17593,
                "candidate_count": 17593,
                "reference_unique_count": 17593,
                "candidate_unique_count": 17593,
                "sample_id_sets_exact": True,
            },
            "evaluator_source_files": [{"path": "models", "exact": True}],
        },
    }
    paths = {}
    for name, payload in reports.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = str(path)
    config = SimpleNamespace(
        training_code_equivalence_report=paths["equivalence"],
        model_initialization_report=paths["initialization"],
        sampler_sequence_report=paths["sampler"],
        dataloader_trace_report=paths["dataloader"],
        evaluator_equivalence_report=paths["evaluator"],
        evaluator_rank_consistency_report=paths["rank_evaluator"],
    )
    validated = validate_shared_prerequisites(config)
    assert set(validated) == {
        "training_code_equivalence",
        "model_initialization",
        "sampler_sequence",
        "dataloader_trajectory",
        "evaluator_equivalence",
        "evaluator_rank_consistency",
    }

    reports["dataloader"]["num_workers"] = 8
    Path(paths["dataloader"]).write_text(
        json.dumps(reports["dataloader"]), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="num_workers"):
        validate_shared_prerequisites(config)
