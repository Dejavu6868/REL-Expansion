"""Checks for the gated SMMF evaluation (zone layout, row composition, gate counts).

The tests marked by the ``smmf`` fixture import the original Pano_MA_Seg code and are skipped
unless SMMF_ROOT points at it (SMMF_CKPT_KEYS at its smmf_ckpt_keys.json for the key check).
"""

import importlib
import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eval_smmf_gate as eg  # noqa: E402


def test_reference_plan_is_smmfs_layout():
    (smmf,) = eg.zone_plan(2048, 4096, [0, 484, 968], 652, 2 / 3)
    assert smmf["name"] == "smmf" and smmf["scale"] == 1
    assert smmf["crop"] == (1080, 1080) and smmf["pad"] == 652 and smmf["rows"] == [0, 484, 968]
    assert smmf["columns"] == [3444, 68, 788, 1508, 2228, 2948, 3668]
    assert eg.row_runs(smmf["keep"]) == [[0, 2048]]


def test_polar_plans_halve_the_outer_rows_and_keep_the_middle_row_native():
    polar, middle = eg.zone_plan(2048, 4096, [0, 484, 968], 652, 2 / 3, polar_scale=0.5)
    assert polar["input_size"] == [1024, 2048] and polar["crop"] == (540, 540) and polar["pad"] == 326
    assert polar["rows"] == [0, 484] and polar["columns"] == [1722, 34, 394, 754, 1114, 1474, 1834]
    assert eg.row_runs(polar["keep"]) == [[0, 484], [1564, 2048]]
    assert middle["scale"] == 1 and middle["crop"] == (1080, 1080) and middle["rows"] == [484]
    assert middle["columns"] == [3444, 68, 788, 1508, 2228, 2948, 3668]
    assert eg.row_runs(middle["keep"]) == [[484, 1564]]

    polar, middle = eg.zone_plan(2048, 4096, [0, 484, 968], 652, 2 / 3, polar_scale=0.5, middle_height=1366)
    assert polar["rows"] == [0, 484] and eg.row_runs(polar["keep"]) == [[0, 341], [1707, 2048]]
    assert middle["crop"] == (1366, 1080) and middle["rows"] == [341]
    assert eg.row_runs(middle["keep"]) == [[341, 1707]]


def test_whole_scale_and_five_row_plans():
    (whole,) = eg.zone_plan(2048, 4096, [0, 484, 968], 652, 2 / 3, whole_scale=0.5)
    assert whole["name"] == "whole" and whole["rows"] == [0, 242, 484] and whole["crop"] == (540, 540)
    assert whole["columns"] == [1722, 34, 394, 754, 1114, 1474, 1834]

    rows = [0, 242, 484, 726, 968]
    (smmf,) = eg.zone_plan(2048, 4096, rows, 652, 0.4)
    assert len(smmf["columns"]) == 11
    polar, middle = eg.zone_plan(2048, 4096, rows, 652, 0.4, polar_scale=0.5)
    assert polar["rows"] == [0, 121, 363, 484] and middle["rows"] == [484]
    assert polar["columns"] == [c // 2 for c in smmf["columns"]]


@pytest.mark.parametrize("kwargs", [
    dict(polar_scale=0.5, whole_scale=0.5),
    dict(polar_scale=1.0),
    dict(whole_scale=0.0),
    dict(middle_height=1366),
    dict(polar_scale=0.5, middle_height=1367),
    dict(polar_scale=0.5, middle_height=2048),
    dict(polar_scale=0.5, middle_height=1000),
    dict(polar_scale=0.3),
    dict(rows=[0, 400, 968], polar_scale=0.5),
    dict(rows=[0]),
    dict(rows=[0, 484, 1000]),
])
def test_bad_plans_are_rejected(kwargs):
    rows = kwargs.pop("rows", [0, 484, 968])
    with pytest.raises(ValueError):
        eg.zone_plan(2048, 4096, rows, 652, 2 / 3, **kwargs)


class FakeSegmentor:
    """Each call returns its zone's index everywhere; one gate choice per window, cycling 0..4."""

    def __init__(self, config, plan, gate_log):
        self.config, self.plan, self.gate_log, self.calls = config, plan, gate_log, []

    def scale_process_rgbX_smmf(self, image, modal_x, lat, ori_shape, crop, stride_rate, device):
        index = len(self.calls)
        spec = self.plan[index]
        assert image.shape[:2] == modal_x.shape[:2] == lat.shape == tuple(spec["input_size"])
        assert (self.config.smmf_pad_h, self.config.window_positions, crop) == (spec["pad"], spec["rows"], spec["crop"])
        self.calls.append(index)
        self.gate_log.extend(k % 5 for k in range(len(spec["rows"]) * len(spec["columns"])))
        return np.full(tuple(ori_shape) + (3,), float(index + 1), dtype=np.float32)


@pytest.mark.parametrize("middle_height", [None, 40])
def test_fused_scores_take_each_row_from_its_zone(middle_height):
    plan = eg.zone_plan(64, 128, [0, 16, 32], 20, 2 / 3, polar_scale=0.5, middle_height=middle_height,
                        crop=(32, 32))
    config = types.SimpleNamespace()
    gate_log = []
    counts = [np.zeros((len(z["rows"]), 5), dtype=np.int64) for z in plan]
    segmentor = FakeSegmentor(config, plan, gate_log)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (64, 128, 3), dtype=np.uint8)
    lat = np.cos(np.linspace(-1.5, 1.5, 64))[:, None].repeat(128, 1)
    fused = eg.fused_scores(segmentor, config, plan, image, image.copy(), lat, "cpu", gate_log, counts)
    start = 16 if middle_height is None else 12
    expected = np.ones(64)
    expected[start:64 - start] = 2
    assert segmentor.calls == [0, 1]
    assert (fused == expected[:, None, None]).all()
    for z, count in zip(plan, counts):
        columns = len(z["columns"])
        assert count.sum() == len(z["rows"]) * columns
        assert count[0].tolist() == np.bincount(np.arange(columns) % 5, minlength=5).tolist()


def stub_module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


class EasyDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


def jit(*args, **kwargs):
    return args[0] if len(args) == 1 and callable(args[0]) and not kwargs else (lambda function: function)


@pytest.fixture
def smmf(monkeypatch):
    root = os.environ.get("SMMF_ROOT")
    if not root:
        pytest.skip("set SMMF_ROOT to the original Pano_MA_Seg folder")
    stubs = {
        "easydict": stub_module("easydict", EasyDict=EasyDict),
        "thop": stub_module("thop", profile=None, clever_format=None),
        "numba": stub_module("numba", jit=jit),
    }
    for name, module in stubs.items():  # only what this container lacks; the server has them
        try:
            importlib.import_module(name)
        except ImportError:
            monkeypatch.setitem(sys.modules, name, module)
    root = str(Path(root).resolve())
    path = list(sys.path)
    yield eg.import_smmf(root)
    for name, module in list(sys.modules.items()):
        if (getattr(module, "__file__", None) or "").startswith(root):
            del sys.modules[name]
    sys.path[:] = path


def normalized(image):
    return (image.astype(np.float64) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]


@pytest.mark.parametrize("middle_height", [None, 40])
def test_original_smmf_fusion_composes_the_zones(smmf, monkeypatch, middle_height):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    weights = np.random.default_rng(1).normal(size=(6, 13))

    def logits(rgb, modal_x):  # pixelwise, (h, w, 13)
        return np.concatenate([rgb, modal_x], axis=2) @ weights

    gate_log = []

    def network(segmentor, input_data, input_modal_x, lat, device):
        gate_log.append(len(gate_log) % 5)
        chw = logits(input_data.transpose(1, 2, 0), input_modal_x.transpose(1, 2, 0))
        return torch.from_numpy(np.exp(chw).transpose(2, 0, 1).astype(np.float32))

    config = smmf["config"]
    eg.configure(config, 2 / 3)
    segmentor = smmf["Evaluator"].__new__(smmf["Evaluator"])
    segmentor.class_num = 13
    segmentor.norm_mean = segmentor.norm_mean_x = config.norm_mean
    segmentor.norm_std = segmentor.norm_std_x = config.norm_std
    monkeypatch.setattr(smmf["Evaluator"], "val_func_process_rgbX", network)

    plan = eg.zone_plan(64, 128, [0, 16, 32], 20, 2 / 3, polar_scale=0.5, middle_height=middle_height,
                        crop=(32, 32))
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (64, 128, 3), dtype=np.uint8)
    modal_x = rng.integers(0, 256, (64, 128, 3), dtype=np.uint8)
    lat = np.cos(np.linspace(-1.5, 1.5, 64))[:, None].repeat(128, 1)
    counts = [np.zeros((len(z["rows"]), 5), dtype=np.int64) for z in plan]
    fused = eg.fused_scores(segmentor, config, plan, image, modal_x, lat, "cpu", gate_log, counts)

    expected = np.zeros_like(fused)
    for z in plan:  # windows counted from the plan's own geometry, then the pixelwise scores
        scale = z["scale"]
        rgb, x = (cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale != 1 else a
                  for a in (image, modal_x))
        count = np.zeros(rgb.shape[:2])
        for row in z["rows"]:
            for column in z["columns"]:
                count[row:row + z["crop"][0], (column + np.arange(z["crop"][1])) % rgb.shape[1]] += 1
        score = (count[:, :, None] * np.exp(logits(normalized(rgb), normalized(x)))).astype(np.float32)
        score = cv2.resize(score, (128, 64), interpolation=cv2.INTER_LINEAR)
        expected[z["keep"]] = score[z["keep"]]
    np.testing.assert_allclose(fused, expected, rtol=1e-5)
    assert [c.sum() for c in counts] == [len(z["rows"]) * len(z["columns"]) for z in plan]


def test_original_model_matches_the_checkpoint_keys(smmf, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    keys_path = os.environ.get("SMMF_CKPT_KEYS")
    if not keys_path:
        pytest.skip("set SMMF_CKPT_KEYS to smmf_ckpt_keys.json")
    network = eg.build_network(smmf, 2 / 3)
    expected = json.loads(Path(keys_path).read_text())
    assert {name: list(t.shape) for name, t in network.state_dict().items()} == expected

    gate_log = []
    network.backbone.Gate.register_forward_hook(
        lambda module, inputs, weight: gate_log.append(int(weight.argmax(1)[0]))
    )
    monkeypatch.chdir(tmp_path)  # the original forward appends gate logs to the working directory
    with torch.no_grad():
        out = network(torch.zeros(1, 3, 96, 64), torch.zeros(1, 3, 96, 64), 0)
    assert out.shape == (1, 13, 96, 64) and len(gate_log) == 1
    assert (tmp_path / "gate_selection.txt").read_text() == "Grid0:tensor([{}])\n".format(gate_log[0])
