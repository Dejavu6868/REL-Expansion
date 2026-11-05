"""Supervised panorama arm: the REL+ v3.1new panorama cache, the training config and the SMMF eval's X folder.

Tests marked by the ``smmf`` fixture import Max's original Pano_MA_Seg code and are skipped unless SMMF_ROOT
points at it.
"""

import importlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_supervised_pano_cache as bc  # noqa: E402
import cross_projection as cp  # noqa: E402
import eval_smmf_gate as eg  # noqa: E402
import relplus_variant as rv  # noqa: E402
import train_supervised_pano as tsp  # noqa: E402
from test_cross_projection_synthetic import render_erp_raw_depth  # noqa: E402
from test_eval_smmf_gate import jit, stub_module, EasyDict  # noqa: E402

SIZE = (64, 128)


@pytest.fixture
def restore_v2_1():
    yield
    rv.apply_variant("v2_1")


def fake_stanford(tmp_path, names):
    stanford, data = tmp_path / "Stanford2D3D", tmp_path / "Stanford2D3DPano"
    raw = render_erp_raw_depth(*SIZE)
    for name in names:
        area, stem = name.split("/")
        path = stanford / area / "pano" / "depth" / (stem + "_depth.png")
        path.parent.mkdir(parents=True)
        assert cv2.imwrite(str(path), raw)
    data.mkdir()
    (data / "fold1_train.txt").write_text(names[0] + "\n")
    (data / "fold1_test.txt").write_text("\n".join(names[1:]) + "\n")
    return stanford, data, raw


def test_cache_holds_the_v3_1new_erp_input_eval_feeds(tmp_path, monkeypatch, restore_v2_1):
    monkeypatch.setattr(bc, "SIZE", SIZE)
    names = ["area_1/camera_a_office_1_frame_equirectangular_domain", "area_5a/camera_b_office_2_frame_equirectangular_domain"]
    stanford, data, raw = fake_stanford(tmp_path, names)
    out = tmp_path / "cache"
    assert bc.main(["--stanford-root", str(stanford), "--data-root", str(data), "--output", str(out),
                    "--workers", "1"]) == 0

    assert rv.read_marker(out)["variant"] == "v3_1new"
    assert json.loads((out / "manifest.json").read_text())["panoramas"] == 2
    expected = rv.get_rel_banded_metric(cp.decode_erp_depth(raw), variant="v3_1new")
    assert not np.array_equal(expected, rv.get_rel_banded_metric(cp.decode_erp_depth(raw), variant="v3_1"))
    for name in names:  # Max's rle names: <area>/<stem>_hha.png, read back as his loader does
        area, stem = name.split("/")
        written = cv2.imread(str(out / "X" / area / (stem + "_hha.png")), cv2.COLOR_BGR2RGB)
        assert np.array_equal(written, expected)

    with pytest.raises(FileExistsError):
        bc.main(["--stanford-root", str(stanford), "--data-root", str(data), "--output", str(out)])


def test_cache_refuses_a_depth_of_the_wrong_size(tmp_path, restore_v2_1):
    stanford, data, _ = fake_stanford(tmp_path, ["area_1/a", "area_5a/b"])  # 64x128, not 2048x4096
    with pytest.raises(ValueError, match="2048x4096"):
        bc.main(["--stanford-root", str(stanford), "--data-root", str(data), "--output", str(tmp_path / "c"),
                 "--workers", "1"])


def marked_x_root(tmp_path, variant="v3_1new"):
    x_root = tmp_path / "cache" / "X"
    x_root.mkdir(parents=True)
    if variant:
        (x_root.parent / rv.MARKER_NAME).write_text(json.dumps({"variant": variant}))
    return x_root


@pytest.mark.parametrize("marker, x_variant, ok", [
    ("v3_1new", "v3_1new", True), (None, "v3_1new", False), ("v3_1", "v3_1new", False),
    (None, "rel", True), ("v3_1new", "rel", False),
])
def test_x_root_must_carry_the_marker_of_its_variant(tmp_path, marker, x_variant, ok):
    x_root = marked_x_root(tmp_path, marker)
    if ok:
        tsp.check_x_root(x_root, x_variant)
    else:
        with pytest.raises(ValueError):
            tsp.check_x_root(x_root, x_variant)


def test_training_refuses_an_unmarked_x_root_before_running(tmp_path):
    pretrained = tmp_path / "mit_b2.pth"
    pretrained.write_bytes(b"")
    x_root = marked_x_root(tmp_path, None)
    with pytest.raises(ValueError, match="v3_1new marker"):
        tsp.main(["--smmf-root", str(tmp_path), "--data-root", str(tmp_path), "--x-root", str(x_root),
                  "--output", str(tmp_path / "run"), "--pretrained", str(pretrained)])
    assert not (tmp_path / "run").exists()


def test_smmf_eval_refuses_an_unmarked_x_root(tmp_path, monkeypatch):
    x_root = marked_x_root(tmp_path, None)
    monkeypatch.setattr(sys, "argv", ["eval_smmf_gate.py", "--smmf-root", str(tmp_path), "--data-root", str(tmp_path),
                                      "--checkpoint", str(tmp_path / "m.pth"), "--output", str(tmp_path / "e"),
                                      "--x-root", str(x_root)])
    with pytest.raises(SystemExit):
        eg.main()
    assert eg.data_setting(tmp_path, x_root)["x_root"] == str(x_root)
    assert eg.data_setting(tmp_path)["x_root"] == str(tmp_path / "rle")


@pytest.fixture
def smmf_config(monkeypatch):
    root = os.environ.get("SMMF_ROOT")
    if not root:
        pytest.skip("set SMMF_ROOT to the original Pano_MA_Seg folder")
    stubs = {
        "easydict": stub_module("easydict", EasyDict=EasyDict),
        "thop": stub_module("thop", profile=None, clever_format=None),
        "numba": stub_module("numba", jit=jit),
    }
    for name, module in stubs.items():
        try:
            importlib.import_module(name)
        except ImportError:
            monkeypatch.setitem(sys.modules, name, module)
    for name in list(sys.modules):  # the frozen CMX source has packages of the same names
        if name.split(".")[0] in ("config", "dataloader", "engine", "models", "utils"):
            monkeypatch.delitem(sys.modules, name)
    root = Path(root).resolve()
    path = list(sys.path)
    yield root, tsp.import_config(root)
    for name, module in list(sys.modules.items()):
        if (getattr(module, "__file__", None) or "").startswith(str(root)):
            del sys.modules[name]
    sys.path[:] = path


def test_config_gets_the_stanford_recipe_back_from_the_matterport_default(smmf_config, tmp_path):
    root, config = smmf_config
    assert config.dataset_name == "Matterport" and config.batch_size == 16  # the server copy's default
    x_root = marked_x_root(tmp_path)
    config = tsp.configure(config, tmp_path / "data", x_root, tmp_path / "run", tmp_path / "mit_b2.pth", 12345)
    assert (config.image_height, config.batch_size, config.nepochs, config.niters_per_epoch) == (1080, 8, 200, 131)
    assert (config.epoch_hard, config.checkpoint_start_epoch, config.checkpoint_step) == (100, 100, 5)
    assert config.num_classes == 13 and config.backbone == "gate_mit_b2" and config.lr == 6e-5
    assert config.data_setting["x_root"] == str(x_root)
    assert config.data_setting["train_source"] == str(tmp_path / "data" / "fold1_train.txt")
    assert config.window_positions == [0, 484, 968] and config.eval_stride_rate == 2 / 3
    smoke = tsp.configure(config, tmp_path / "data", x_root, tmp_path / "run", tmp_path / "p", 1, smoke_iterations=3)
    assert (smoke.nepochs, smoke.niters_per_epoch) == (1, 3)


def test_max_s_loader_reads_x_from_the_cache(smmf_config, tmp_path, monkeypatch):
    root, config = smmf_config
    data, x_root = tmp_path / "data", marked_x_root(tmp_path)
    rng = np.random.default_rng(0)
    for split, name in (("train", "area_1/a"), ("test", "area_5a/b")):
        for folder, suffix, image in (
            (data / "image", "_rgb.png", rng.integers(0, 256, SIZE + (3,), dtype=np.uint8)),
            (data / "label", "_semantic.png", rng.integers(1, 14, SIZE, dtype=np.uint8)),
            (x_root, "_hha.png", rng.integers(0, 256, SIZE + (3,), dtype=np.uint8)),
        ):
            path = folder / (name + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(path), image)
        (data / "fold1_{}.txt".format(split)).write_text(name + "\n")
    config = tsp.configure(config, data, x_root, tmp_path / "run", tmp_path / "p", 12345)
    monkeypatch.setattr(tsp, "FOLD1_COUNTS", (1, 1))
    assert tsp.check_files(config.data_setting) == [1, 1]

    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.dataloader import ValPre

    sample = RGBXDataset(config.data_setting, "val", ValPre())[0]
    assert np.array_equal(sample["modal_x"], cv2.imread(str(x_root / "area_5a" / "b_hha.png")))
    (x_root / "area_5a" / "b_hha.png").unlink()
    with pytest.raises(FileNotFoundError):
        tsp.check_files(config.data_setting)
