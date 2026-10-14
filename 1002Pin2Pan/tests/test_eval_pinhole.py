"""Checks for the pinhole evaluation of arbitrary checkpoints (eval_pinhole.py)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eval_pinhole as ep  # noqa: E402

CHANNELS = [3, 4, 7, 8, 9, 10, 11, 12]


def test_eight_class_metrics_ignore_other_ground_truth_rows():
    confusion = np.zeros((13, 13), dtype=np.int64)
    confusion[11, 11] = 60  # wall correct
    confusion[11, 5] = 20  # wall read as clutter: a miss for wall
    confusion[11, 7] = 20  # wall read as door: a miss for wall, a false positive for door
    confusion[7, 7] = 30  # door correct
    confusion[5, 11] = 1000  # clutter ground truth: ignored entirely
    confusion[0, 7] = 1000  # beam ground truth: ignored entirely
    metrics = ep.eight_class_metrics(confusion, CHANNELS)
    iou = metrics["per_class_iou_percent"]
    assert iou["wall"] == pytest.approx(60.0)
    assert iou["door"] == pytest.approx(60.0)
    assert np.isnan(iou["sofa"])
    assert metrics["mIoU_percent"] == pytest.approx(60.0)
    assert metrics["pixel_accuracy_percent"] == pytest.approx(90 / 130 * 100)
    assert metrics["predicted_as_other_classes_percent"] == pytest.approx(20 / 130 * 100)
    assert metrics["evaluated_pixels"] == 130


def _frozen():
    modules = ep.import_frozen(ep.DEFAULT_SOURCE_ROOT)
    return modules["tools.eval_rel_plus_v2_3_full"], modules["engine.relplus_evaluator"]


class _FixedLogits:
    def __init__(self, logits):
        self.logits = logits

    def eval(self):
        return self

    def __call__(self, rgb, modal_x):
        return self.logits


def test_restricted_evaluate_only_changes_pixels_won_by_other_classes():
    torch = pytest.importorskip("torch")
    _, core = _frozen()
    logits = torch.zeros(1, 13, 1, 3)
    logits[0, 5, 0, 0], logits[0, 12, 0, 0] = 9.0, 1.0  # clutter wins; window is best of the 8
    logits[0, 0, 0, 1], logits[0, 4, 0, 1] = 9.0, 1.0  # beam wins; chair is best of the 8
    logits[0, 10, 0, 2] = 2.0  # table wins either way
    prepared = core.PreparedEvalSample(
        rgb=torch.zeros(1, 3, 1, 3), modal_x=torch.zeros(1, 3, 1, 3),
        label=np.array([[12, 4, 10]], dtype=np.uint8), valid_mask=None, sample_id="x",
    )
    keywords = dict(class_num=13, ignore_index=255, device="cpu")
    frozen_result = core.evaluate_prepared_sample(_FixedLogits(logits), prepared, **keywords)
    restricted = ep.eight_class_evaluate(core, CHANNELS)(_FixedLogits(logits), prepared, **keywords)
    assert frozen_result["prediction"].tolist() == [[5, 0, 10]]
    assert restricted["prediction"].tolist() == [[12, 4, 10]]
    assert restricted["correct"] == 3 and frozen_result["correct"] == 1
    assert set(restricted) == set(frozen_result)

    logits[0, 1, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        ep.eight_class_evaluate(core, CHANNELS)(_FixedLogits(logits), prepared, **keywords)


class _TinyDataset:
    """Stands in for the frozen RGBXDataset: three 64x64 samples, no cache files."""

    def __init__(self, setting, split, preprocess):
        rng = np.random.RandomState(0)
        self.items = []
        for index in range(3):
            label = rng.randint(0, 13, size=(64, 64)).astype(np.uint8)
            label[:4] = 255
            item = {
                "fn": "area_5a/frame_{}".format(index),
                "data": rng.randint(0, 256, size=(64, 64, 3)).astype(np.uint8),
                "modal_x": rng.randint(0, 256, size=(64, 64, 3)).astype(np.uint8),
                "label": label,
            }
            if setting["x_mode"] == "rel_plus_v2_1":
                item["modal_x_valid_mask"] = np.ones((64, 64), dtype=bool)
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


@pytest.mark.parametrize("arm", ["hha", "relplus"])
def test_runs_the_frozen_evaluator_end_to_end(monkeypatch, tmp_path, arm):
    torch = pytest.importorskip("torch")
    frozen, core = _frozen()
    import dataloader.RGBXDataset as dataset_module
    import dataloader.data_setting as data_setting
    import models.builder as builder
    import utils.training_protocol as protocol

    for target, name in ((core, "prepare_eval_sample"), (core, "evaluate_prepared_sample"),
                         (frozen, "_unbatch")):
        monkeypatch.setattr(target, name, getattr(target, name))  # restored after the test
    monkeypatch.setattr(protocol, "assert_runtime_dataset_contract", lambda config, **_: {})
    monkeypatch.setattr(data_setting, "build_data_setting",
                        lambda config, split: {"x_mode": config.x_mode})
    monkeypatch.setattr(dataset_module, "RGBXDataset", _TinyDataset)

    config = json.loads(
        (ep.DEFAULT_SOURCE_ROOT.parents[3] / "configs" / (arm + ".json")).read_text(encoding="utf-8")
    )
    config.update(num_eval_imgs=3, num_workers=0, eval_crop_size=[64, 64],
                  x_root_folder=str(tmp_path / "cache" / "x"))
    config_path = tmp_path / (arm + ".json")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    network = builder.EncoderDecoder(cfg=ep.load_config(config_path), criterion=None,
                                     norm_layer=torch.nn.BatchNorm2d)
    state = network.state_dict()
    state["decode_head.linear_pred.bias"][5] = 100.0  # clutter wins every pixel unless restricted
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"epoch": 200, "model": state}, checkpoint)

    reports = {}
    for name, extra in (("plain", []), ("eight", ["--eight-class-predictions"])):
        monkeypatch.setattr(sys, "argv", [
            "eval_pinhole.py", "--config", str(config_path), "--checkpoint", str(checkpoint),
            "--output", str(tmp_path / name), "--device", "cpu",
        ] + extra)
        assert ep.main() == 0
        reports[name] = json.loads((tmp_path / name / "eval_pinhole.json").read_text())
        confusion = np.loadtxt(tmp_path / name / "evaluation" / "confusion_matrix.csv",
                               delimiter=",", skiprows=1, dtype=np.int64)[:, 1:]
        reports[name]["confusion"] = confusion

    plain, eight = reports["plain"], reports["eight"]
    labelled = 3 * 60 * 64
    assert plain["confusion"].sum() == eight["confusion"].sum() == labelled
    assert plain["confusion"][:, 5].sum() == labelled  # frozen argmax: clutter everywhere
    assert eight["confusion"][:, [0, 1, 2, 5, 6]].sum() == 0
    assert plain["prediction_classes"][5] == "clutter"
    assert eight["prediction_classes"] == list(ep.EIGHT_CLASS_NAMES)
    assert eight["classes_8"] == ep.eight_class_metrics(eight["confusion"], CHANNELS)
    assert plain["classes_8"]["predicted_as_other_classes_percent"] == pytest.approx(100.0)
    assert eight["classes_8"]["predicted_as_other_classes_percent"] == 0.0
    assert (arm == "hha") == (eight["hha_input_adapter"] is not None)
    assert eight["relplus_cache_variant"] == (None if arm == "hha" else "v2_1")

    monkeypatch.setattr(sys, "argv", [
        "eval_pinhole.py", "--config", str(config_path), "--checkpoint", str(checkpoint),
        "--output", str(tmp_path / "plain"), "--device", "cpu",
    ])
    with pytest.raises(FileExistsError):
        ep.main()
