import numpy as np

from rel_plus.integration.cmx_preprocess import SpatialTransform, apply_cmx_compatible_preprocess
from rel_plus.storage import load_rel_plus_png, save_rel_plus_png


def test_storage_and_model_input_channel_sentinel(tmp_path):
    sentinel = np.zeros((3, 4, 3), dtype=np.uint8)
    sentinel[..., 0] = 11
    sentinel[..., 1] = 22
    sentinel[..., 2] = 33
    path = tmp_path / "sentinel.png"
    save_rel_plus_png(path, sentinel)
    loaded = load_rel_plus_png(path)
    np.testing.assert_array_equal(loaded, sentinel)

    rgb = np.zeros_like(sentinel)
    label = np.zeros(sentinel.shape[:2], dtype=np.uint8)
    output = apply_cmx_compatible_preprocess(
        rgb, loaded, label, SpatialTransform(1.0, 0, 0, 3, 4)
    )
    expected = (np.array([11.0, 22.0, 33.0]) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np.testing.assert_allclose(output.rel_plus_chw[:, 0, 0], expected)
