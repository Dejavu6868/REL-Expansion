from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from dataloader.RGBXDataset import RGBXDataset
from dataloader.data_setting import build_data_setting
from dataloader.dataloader import TrainPre, ValPre
from dataloader.profiles import S2D_RELPLUS_COMPARISON_NO_FLIP


def _config(**updates):
    values = {
        "rgb_root_folder": "/dataset/RGB",
        "rgb_format": ".png",
        "gt_root_folder": "/dataset/Label",
        "gt_format": ".png",
        "gt_transform": True,
        "x_root_folder": "/cache/RELPlus",
        "x_format": ".png",
        "x_is_single_channel": False,
        "x_mode": "rel_plus_v2_1",
        "x_valid_root_folder": "/cache/ValidMask",
        "x_valid_format": ".png",
        "train_source": "/dataset/train.txt",
        "eval_source": "/dataset/test.txt",
        "class_names": ["class-0"],
        "representation_protocol_id": "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), array)


def test_build_data_setting_is_shared_and_relplus_fail_loud():
    config = _config()
    train = build_data_setting(config, split="train")
    val = build_data_setting(config, split="val")

    assert train == val
    assert train["x_mode"] == "rel_plus_v2_1"
    assert train["x_valid_root"] == "/cache/ValidMask"
    assert train["x_valid_format"] == ".png"
    assert train["channel_order"] == ("EGVIA", "LOA", "ReD")

    missing_mode = _config()
    delattr(missing_mode, "x_mode")
    with pytest.raises(ValueError, match="x_mode"):
        build_data_setting(missing_mode, split="train")

    missing_root = _config()
    delattr(missing_root, "x_valid_root_folder")
    with pytest.raises(ValueError, match="x_valid_root"):
        build_data_setting(missing_root, split="val")

    missing_format = _config()
    delattr(missing_format, "x_valid_format")
    with pytest.raises(ValueError, match="x_valid_format"):
        build_data_setting(missing_format, split="val")


def test_standard_data_setting_and_valpre_regression():
    config = _config(
        x_mode="standard",
        representation_protocol_id="CMX_STANDARD",
    )
    delattr(config, "x_valid_root_folder")
    delattr(config, "x_valid_format")
    setting = build_data_setting(config, split="val")
    assert setting["x_mode"] == "standard"
    assert setting["x_valid_root"] is None

    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    label = np.zeros((3, 4), dtype=np.uint8)
    x = np.ones((3, 4, 3), dtype=np.uint8)
    result = ValPre(x_mode="standard")(rgb, label, x)
    assert len(result) == 3
    np.testing.assert_array_equal(result[2], x)


def test_relplus_valpre_requires_matching_mask():
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    label = np.zeros((3, 4), dtype=np.uint8)
    x = np.ones((3, 4, 3), dtype=np.uint8)
    pre = ValPre(x_mode="rel_plus_v2_1")

    with pytest.raises(ValueError, match="requires.*valid mask"):
        pre(rgb, label, x)
    with pytest.raises(ValueError, match="shape mismatch"):
        pre(rgb, label, x, np.ones((2, 4), dtype=bool))

    mask = np.ones((3, 4), dtype=bool)
    result = pre(rgb, label, x, mask)
    assert len(result) == 4
    np.testing.assert_array_equal(result[3], mask)


def test_dataset_relplus_channel_and_rgb_sentinel(tmp_path):
    sample_id = "area_1/sentinel"
    rgb = np.full((4, 5, 3), [7, 13, 29], dtype=np.uint8)
    rel_plus = np.full((4, 5, 3), [11, 22, 33], dtype=np.uint8)
    label = np.ones((4, 5), dtype=np.uint8)
    valid = np.full((4, 5), 255, dtype=np.uint8)
    valid[0, 0] = 0

    _write(tmp_path / "RGB" / (sample_id + ".png"), rgb)
    _write(tmp_path / "Label" / (sample_id + ".png"), label)
    _write(tmp_path / "RELPlus" / (sample_id + ".png"), rel_plus)
    _write(tmp_path / "ValidMask" / (sample_id + ".png"), valid)
    split = tmp_path / "split.txt"
    split.write_text(sample_id + "\n", encoding="utf-8")

    setting = {
        "rgb_root": str(tmp_path / "RGB"),
        "rgb_format": ".png",
        "gt_root": str(tmp_path / "Label"),
        "gt_format": ".png",
        "transform_gt": True,
        "x_root": str(tmp_path / "RELPlus"),
        "x_format": ".png",
        "x_single_channel": False,
        "x_mode": "rel_plus_v2_1",
        "x_valid_root": str(tmp_path / "ValidMask"),
        "x_valid_format": ".png",
        "channel_order": ("EGVIA", "LOA", "ReD"),
        "train_source": str(split),
        "eval_source": str(split),
        "class_names": ["class-0"],
    }
    sample = RGBXDataset(
        setting, "val", ValPre(x_mode="rel_plus_v2_1")
    )[0]
    np.testing.assert_array_equal(sample["data"][1, 1], [7, 13, 29])
    np.testing.assert_array_equal(sample["modal_x"][1, 1], [11, 22, 33])
    assert sample["modal_x"].dtype == np.uint8
    assert sample["modal_x_valid_mask"].dtype == np.bool_
    assert not bool(sample["modal_x_valid_mask"][0, 0])

    Path(tmp_path / "ValidMask" / (sample_id + ".png")).unlink()
    with pytest.raises(FileNotFoundError):
        RGBXDataset(setting, "val", ValPre(x_mode="rel_plus_v2_1"))[0]


def test_train_and_validation_dataloaders_keep_relplus_contract(tmp_path):
    sample_id = "area_1/loader_sentinel"
    rgb = np.full((4, 5, 3), [7, 13, 29], dtype=np.uint8)
    rel_plus = np.full((4, 5, 3), [11, 22, 33], dtype=np.uint8)
    label = np.ones((4, 5), dtype=np.uint8)
    valid = np.full((4, 5), 255, dtype=np.uint8)
    _write(tmp_path / "RGB" / (sample_id + ".png"), rgb)
    _write(tmp_path / "Label" / (sample_id + ".png"), label)
    _write(tmp_path / "RELPlus" / (sample_id + ".png"), rel_plus)
    _write(tmp_path / "ValidMask" / (sample_id + ".png"), valid)
    split = tmp_path / "split.txt"
    split.write_text(sample_id + "\n", encoding="utf-8")
    setting = {
        "rgb_root": str(tmp_path / "RGB"),
        "rgb_format": ".png",
        "gt_root": str(tmp_path / "Label"),
        "gt_format": ".png",
        "transform_gt": True,
        "x_root": str(tmp_path / "RELPlus"),
        "x_format": ".png",
        "x_single_channel": False,
        "x_mode": "rel_plus_v2_1",
        "x_valid_root": str(tmp_path / "ValidMask"),
        "x_valid_format": ".png",
        "channel_order": ("EGVIA", "LOA", "ReD"),
        "train_source": str(split),
        "eval_source": str(split),
        "class_names": ["class-0"],
    }
    cfg = SimpleNamespace(
        x_mode="rel_plus_v2_1",
        augmentation_profile=S2D_RELPLUS_COMPARISON_NO_FLIP,
        train_horizontal_flip=False,
        train_vertical_flip=False,
        train_arbitrary_rotation=False,
        train_perspective_warp=False,
        train_scale_array=[1.0],
        image_height=4,
        image_width=5,
    )
    train = RGBXDataset(
        setting,
        "train",
        TrainPre(
            np.array([0.485, 0.456, 0.406]),
            np.array([0.229, 0.224, 0.225]),
            cfg=cfg,
        ),
    )
    train_batch = next(iter(torch.utils.data.DataLoader(train, batch_size=1)))
    assert train_batch["data"].shape == (1, 3, 4, 5)
    assert train_batch["modal_x"].shape == (1, 3, 4, 5)
    assert train_batch["data"].dtype == torch.float32
    assert train_batch["modal_x"].dtype == torch.float32
    assert train_batch["label"].dtype == torch.int64
    assert train_batch["modal_x_valid_mask"].dtype == torch.bool
    expected_rel = (
        np.array([11.0, 22.0, 33.0]) / 255.0
        - np.array([0.485, 0.456, 0.406])
    ) / np.array([0.229, 0.224, 0.225])
    np.testing.assert_allclose(
        train_batch["modal_x"][0, :, 1, 1].numpy(), expected_rel, rtol=1e-6
    )

    validation = RGBXDataset(
        setting, "val", ValPre(x_mode="rel_plus_v2_1")
    )
    val_batch = next(iter(torch.utils.data.DataLoader(validation, batch_size=1)))
    assert val_batch["modal_x"].shape == (1, 4, 5, 3)
    assert val_batch["modal_x_valid_mask"].shape == (1, 4, 5)
