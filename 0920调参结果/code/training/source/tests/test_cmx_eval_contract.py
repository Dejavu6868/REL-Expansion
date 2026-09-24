from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from engine.relplus_evaluator import (
    evaluate_prepared_sample,
    prepare_eval_sample,
    save_prediction_pair,
)


class TinyNetwork(nn.Module):
    def forward(self, rgb, modal_x):
        batch, _, height, width = rgb.shape
        logits = torch.zeros(batch, 2, height, width, device=rgb.device)
        logits[:, 1] = modal_x[:, 0]
        return logits


def _config():
    return SimpleNamespace(
        x_mode="rel_plus_v2_1",
        eval_flip=False,
        eval_scale_array=[1],
        eval_crop_size=[4, 5],
        norm_mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
        norm_std=np.array([0.229, 0.224, 0.225], dtype=np.float32),
        num_classes=2,
        background=255,
    )


def test_eval_preparation_keeps_relplus_channel_order_and_mask_out_of_model():
    sample = {
        "data": np.full((4, 5, 3), [7, 13, 29], dtype=np.uint8),
        "modal_x": np.full((4, 5, 3), [11, 22, 33], dtype=np.uint8),
        "label": np.ones((4, 5), dtype=np.uint8),
        "modal_x_valid_mask": np.ones((4, 5), dtype=bool),
        "fn": "area_1/sentinel",
    }
    prepared = prepare_eval_sample(sample, _config())
    expected = (
        np.array([11.0, 22.0, 33.0], dtype=np.float32) / 255.0
        - _config().norm_mean
    ) / _config().norm_std
    np.testing.assert_allclose(prepared.modal_x[0, :, 1, 1].numpy(), expected)
    assert prepared.rgb.shape == (1, 3, 4, 5)
    assert prepared.modal_x.shape == (1, 3, 4, 5)
    assert prepared.valid_mask.shape == (4, 5)

    result = evaluate_prepared_sample(
        TinyNetwork(), prepared, class_num=2, ignore_index=255, device="cpu"
    )
    assert result["prediction"].shape == (4, 5)
    assert result["hist"].shape == (2, 2)
    assert result["logits_finite"] is True
    assert result["diagnostic_mask_passed_to_model"] is False


def test_eval_rejects_flip_and_save_path_has_real_imports(tmp_path):
    config = _config()
    config.eval_flip = True
    sample = {
        "data": np.zeros((4, 5, 3), dtype=np.uint8),
        "modal_x": np.zeros((4, 5, 3), dtype=np.uint8),
        "label": np.zeros((4, 5), dtype=np.uint8),
        "modal_x_valid_mask": np.ones((4, 5), dtype=bool),
        "fn": "sentinel",
    }
    with pytest.raises(ValueError, match="eval_flip"):
        prepare_eval_sample(sample, config)

    prediction = np.arange(20, dtype=np.uint8).reshape(4, 5) % 2
    paths = save_prediction_pair(
        prediction,
        "area_1/sentinel",
        tmp_path,
        class_colors=[[0, 0, 0], [255, 0, 0]],
    )
    assert cv2.imread(str(paths["raw"]), cv2.IMREAD_UNCHANGED).shape == (4, 5)
    assert paths["color"].is_file()
