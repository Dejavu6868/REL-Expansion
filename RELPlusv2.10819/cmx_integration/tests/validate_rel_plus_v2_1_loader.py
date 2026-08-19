#!/usr/bin/env python3
"""Exercise REL+ v2.1 bytes through the real RGBXDataset/TrainPre/DataLoader."""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RELPLUS_ROOT = REPO_ROOT.parent
sys.path[:0] = [str(REPO_ROOT), str(RELPLUS_ROOT)]

from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import TrainPre


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), array):
        raise OSError("failed to write {}".format(path))


def main():
    with tempfile.TemporaryDirectory(prefix="relplus_v2_1_loader_") as temporary:
        root = Path(temporary)
        sample_id = "area_1/sentinel"
        rgb = np.full((4, 5, 3), [7, 13, 29], dtype=np.uint8)
        rel_plus = np.full((4, 5, 3), [11, 22, 33], dtype=np.uint8)
        label = np.ones((4, 5), dtype=np.uint8)
        valid = np.ones((4, 5), dtype=np.uint8) * 255
        valid[0, 0] = 0
        _write(root / "RGB" / (sample_id + ".png"), rgb)
        _write(root / "Label" / (sample_id + ".png"), label)
        _write(root / "RELPlus" / (sample_id + ".png"), rel_plus)
        _write(root / "ValidMask" / (sample_id + ".png"), valid)
        split = root / "train.txt"
        split.write_text(sample_id + "\n", encoding="utf-8")

        cfg = SimpleNamespace(
            x_mode="rel_plus_v2_1",
            train_horizontal_flip=False,
            train_scale_array=[1.0],
            image_height=4,
            image_width=5,
        )
        preprocess = TrainPre(
            np.array([0.485, 0.456, 0.406]),
            np.array([0.229, 0.224, 0.225]),
            cfg=cfg,
            rng=np.random.default_rng(7),
        )
        setting = {
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
            "train_source": str(split),
            "eval_source": str(split),
            "class_names": ["sentinel"],
        }
        dataset = RGBXDataset(setting, "train", preprocess)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )
        batch = next(iter(loader))
        expected_modal = (
            np.array([11.0, 22.0, 33.0], dtype=np.float32) / 255.0
            - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        expected_rgb = (
            np.array([7.0, 13.0, 29.0], dtype=np.float32) / 255.0
            - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        np.testing.assert_allclose(batch["modal_x"][0, :, 1, 1].numpy(), expected_modal)
        np.testing.assert_allclose(batch["data"][0, :, 1, 1].numpy(), expected_rgb)
        assert batch["modal_x"].shape == (1, 3, 4, 5)
        assert batch["data"].shape == (1, 3, 4, 5)
        assert batch["modal_x"].dtype == torch.float32
        assert batch["data"].dtype == torch.float32
        assert not bool(batch["modal_x_valid_mask"][0, 0, 0])
        assert preprocess.last_transform is not None
        report = {
            "status": "PASS",
            "protocol_id": "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT",
            "loader": "RGBXDataset -> TrainPre -> DataLoader",
            "modal_x_channel_sentinel": [11, 22, 33],
            "rgb_byte_sentinel": [7, 13, 29],
            "rgb_cvtColor_called": False,
            "modal_x_cvtColor_called": False,
            "train_horizontal_flip": False,
            "modal_x_shape": list(batch["modal_x"].shape),
            "modal_x_dtype": str(batch["modal_x"].dtype),
            "diagnostic_mask_in_batch": True,
            "diagnostic_mask_model_channels": 0,
        }
        print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
