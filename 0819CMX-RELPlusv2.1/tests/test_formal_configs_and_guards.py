import ast
import filecmp
import os
from pathlib import Path

import pytest

from configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_1_formal import (
    config as formal,
)
from configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_1_pilot import (
    config as pilot,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = Path(
    os.environ.get("RELPLUS_V21_FROZEN_ROOT", "/home/zhuzhaoziao/RELPlus/RELPlusv2.1")
)
ORIGINAL_CMX_ROOT = Path(
    os.environ.get("ORIGINAL_CMX_ROOT", "/home/zhuzhaoziao/RELPlus/CMX")
)


def _files_below(root):
    return sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )


def test_pilot_and_formal_config_states_are_separate_and_fail_closed():
    assert pilot.num_train_imgs == pilot.num_eval_imgs == 36
    assert pilot.nepochs == 0
    assert pilot.training_authorized is False
    assert pilot.data_ready is True

    assert formal.num_train_imgs == 52903
    assert formal.num_eval_imgs == 17593
    assert formal.num_train_imgs + formal.num_eval_imgs == 70496
    assert formal.training_authorized is False
    assert formal.data_ready is False
    assert formal.criterion == "Focal"
    assert formal.focal_gamma == 2
    assert formal.loss_reduction == "none_then_mean"
    assert formal.backbone == "mit_b2"
    assert formal.decoder == "MLPDecoder"
    assert formal.using_gate is False
    assert formal.using_smmf is False
    assert formal.using_dymm is False
    assert formal.using_sga is False
    assert formal.representation_protocol_id == "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"
    assert formal.augmentation_profile == "S2D_RELPLUS_COMPARISON_NO_FLIP"
    assert formal.channel_order == ("EGVIA", "LOA", "ReD")
    assert formal.optimizer == "AdamW"
    assert formal.scheduler == "WarmUpPolyLR"
    assert formal.batch_size == 8
    assert formal.num_workers == 16
    assert formal.nepochs == 200
    assert formal.eval_crop_size == [480, 480]
    assert formal.eval_scale_array == [1]
    assert formal.eval_flip is False
    assert formal.data_setting["x_mode"] == "rel_plus_v2_1"
    assert sum(1 for _ in open(formal.train_source, encoding="utf-8")) == 52903
    assert sum(1 for _ in open(formal.eval_source, encoding="utf-8")) == 17593


def test_train_guard_runs_before_engine_model_optimizer_or_tensorboard():
    source = (ROOT / "train.py").read_text(encoding="utf-8")
    guard = source.index("assert_training_ready(config)")
    assert guard < source.index("with Engine")
    assert guard < source.index("SummaryWriter(")
    assert guard < source.index("segmodel(")
    assert guard < source.index("build_author_optimizer(")
    epoch_loop = source.index("for epoch in range")
    epoch_seed = source.index("set_author_seed(", epoch_loop)
    sampler_epoch = source.index("train_sampler.set_epoch(epoch)", epoch_loop)
    assert epoch_loop < epoch_seed < sampler_epoch


def test_single_batch_tool_cannot_step_optimizer_or_scheduler():
    tree = ast.parse(
        (ROOT / "tools/validate_single_batch_v2_1.py").read_text(encoding="utf-8")
    )
    forbidden_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "step":
                forbidden_calls.append(node.lineno)
    assert forbidden_calls == []


def test_legacy_eval_entries_reject_relplus_and_legacy_launchers_are_guarded():
    for relative in ("eval.py", "tools/eval_fold1.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "rejects REL+ v2.1" in source
        assert "build_data_setting" in source
    for relative in (
        "tools/prepare_and_launch_fold1.sh",
        "tools/run_fold1_arm.sh",
        "tools/run_fold1_suite.sh",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "CMX_LEGACY_FOLD1_EXPLICITLY_AUTHORIZED" in source


@pytest.mark.live_source
def test_original_cmx_model_tree_and_relplus_generator_core_are_unchanged():
    if not ORIGINAL_CMX_ROOT.is_dir() or not FROZEN_ROOT.is_dir():
        pytest.skip("server source trees are unavailable")

    ours_models = ROOT / "models"
    reference_models = ORIGINAL_CMX_ROOT / "models"
    assert _files_below(ours_models) == _files_below(reference_models)
    for relative in _files_below(ours_models):
        assert filecmp.cmp(
            str(ours_models / relative), str(reference_models / relative), shallow=False
        ), str(relative)

    frozen_core = (
        "camera.py", "constants.py", "depth.py", "encoding.py", "generator.py",
        "geometry.py", "normal_diagnostics.py", "profiles.py", "source_helpers.py",
        "stanford_s2d.py", "storage.py",
    )
    for name in frozen_core:
        assert filecmp.cmp(
            str(ROOT / "rel_plus" / name),
            str(FROZEN_ROOT / "rel_plus" / name),
            shallow=False,
        ), name
    assert filecmp.cmp(
        str(ROOT / "third_party/rel_original/hha_util.py"),
        str(FROZEN_ROOT / "cmx_integration/third_party/rel_original/hha_util.py"),
        shallow=False,
    )
