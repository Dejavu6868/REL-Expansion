import random
from types import SimpleNamespace

import numpy as np
import pytest

from dataloader.dataloader import TrainPre
from dataloader.profiles import (
    S2D_RELPLUS_COMPARISON_NO_FLIP,
    trace_comparison_profile,
)


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _config(x_mode, flip=False):
    return SimpleNamespace(
        x_mode=x_mode,
        augmentation_profile=S2D_RELPLUS_COMPARISON_NO_FLIP,
        train_horizontal_flip=flip,
        train_scale_array=[0.75, 1.0, 1.25],
        image_height=6,
        image_width=7,
    )


def _run(x_mode):
    rgb = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    x = (rgb + np.uint8(17)).astype(np.uint8)
    label = np.arange(8 * 9, dtype=np.uint8).reshape(8, 9) % 3
    mask = np.ones((8, 9), dtype=bool)
    pre = TrainPre(MEAN, STD, cfg=_config(x_mode), rng=random.Random(91))
    if x_mode == "rel_plus_v2_1":
        output = pre(rgb, label, x, mask)
    else:
        output = pre(rgb, label, x)
    return pre.last_transform, output


def test_three_arms_share_identical_spatial_parameters_and_no_flip():
    runs = dict((mode, _run(mode)) for mode in ("raw_depth", "standard", "rel_plus_v2_1"))
    assert runs["raw_depth"][0] == runs["standard"][0] == runs["rel_plus_v2_1"][0]
    for _, output in runs.values():
        assert output[0].shape == (3, 6, 7)
        assert output[1].shape == (6, 7)
        assert output[2].shape == (3, 6, 7)
    assert len(runs["rel_plus_v2_1"][1]) == 4
    assert len(runs["standard"][1]) == 3


def test_comparison_profile_rejects_flip_for_every_arm():
    for mode in ("raw_depth", "standard", "rel_plus_v2_1"):
        with pytest.raises(ValueError, match="no-flip"):
            TrainPre(MEAN, STD, cfg=_config(mode, flip=True), rng=random.Random(1))


def test_first_fifty_transform_traces_match_across_arms():
    sample_ids = ["sample_{:03d}".format(index) for index in range(50)]
    traces = trace_comparison_profile(
        sample_ids,
        input_shape=(480, 480),
        output_shape=(480, 480),
        scales=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
        base_seed=12345,
        epoch=7,
        rank=3,
        x_modes=("rgbd", "hha", "rel_plus_v2_1"),
    )
    assert len(traces["rgbd"]) == 50
    assert traces["rgbd"] == traces["hha"] == traces["rel_plus_v2_1"]
    assert set(traces["rgbd"][0]) == {
        "sample_id", "epoch", "rank", "scale", "crop_top", "crop_left",
        "pad_top", "pad_bottom", "pad_left", "pad_right",
    }
