import numpy as np
import pytest

from rel_plus.policy import validate_rel_plus_augmentation_policy
from rel_plus.storage import load_rel_plus_png, save_rel_plus_png


def test_channel_sentinel_round_trip_has_no_colour_conversion(tmp_path):
    sentinel = np.zeros((3, 4, 3), dtype=np.uint8)
    sentinel[..., 0] = 11
    sentinel[..., 1] = 22
    sentinel[..., 2] = 33
    path = tmp_path / "sentinel.png"
    save_rel_plus_png(path, sentinel)
    loaded = load_rel_plus_png(path)
    np.testing.assert_array_equal(loaded, sentinel)


def test_horizontal_flip_is_rejected_explicitly():
    with pytest.raises(ValueError, match="horizontal flip"):
        validate_rel_plus_augmentation_policy(horizontal_flip=True)
    validate_rel_plus_augmentation_policy(horizontal_flip=False)

