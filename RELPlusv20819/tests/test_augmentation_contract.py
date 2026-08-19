import numpy as np
import pytest

from rel_plus.integration.cmx_preprocess import (
    SpatialTransform,
    apply_cmx_compatible_preprocess,
    sample_spatial_transform,
)
from rel_plus.policy import validate_rel_plus_augmentation_policy


def coordinate_triplet(shape=(6, 8)):
    rows, columns = np.indices(shape)
    marker = (rows * 10 + columns).astype(np.uint8)
    rgb = np.dstack([marker, marker + 1, marker + 2])
    rel = np.dstack([marker + 11, marker + 22, marker + 33])
    label = marker.copy()
    return rgb, rel, label


def test_one_spatial_transform_applies_to_rgb_rel_and_label():
    rgb, rel, label = coordinate_triplet()
    transform = SpatialTransform(1.0, 1, 2, 4, 5)
    output = apply_cmx_compatible_preprocess(rgb, rel, label, transform)
    np.testing.assert_array_equal(output.label, label[1:5, 2:7])
    expected_rel = rel[1:5, 2:7].astype(np.float64) / 255.0
    expected_rel = (expected_rel - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np.testing.assert_allclose(output.rel_plus_chw, expected_rel.transpose(2, 0, 1))


def test_scale_uses_linear_for_rgb_and_rel_nearest_for_label():
    rgb, rel, label = coordinate_triplet((4, 4))
    transform = SpatialTransform(1.5, 0, 0, 6, 6)
    output = apply_cmx_compatible_preprocess(rgb, rel, label, transform)
    assert output.rgb_chw.shape == (3, 6, 6)
    assert output.rel_plus_chw.shape == (3, 6, 6)
    assert set(np.unique(output.label)).issubset(set(np.unique(label)))


def test_photometric_augmentation_changes_rgb_only():
    rgb, rel, label = coordinate_triplet()
    transform = SpatialTransform(1.0, 0, 0, 6, 8)
    plain = apply_cmx_compatible_preprocess(rgb, rel, label, transform)
    changed = apply_cmx_compatible_preprocess(
        rgb, rel, label, transform, photometric_augmentation=lambda value: np.zeros_like(value)
    )
    assert not np.array_equal(plain.rgb_chw, changed.rgb_chw)
    np.testing.assert_array_equal(plain.rel_plus_chw, changed.rel_plus_chw)
    np.testing.assert_array_equal(plain.label, changed.label)


def test_seeded_transform_sampling_is_reproducible():
    a = sample_spatial_transform((480, 480), [0.5, 1.0, 1.5], (320, 320), np.random.default_rng(7))
    b = sample_spatial_transform((480, 480), [0.5, 1.0, 1.5], (320, 320), np.random.default_rng(7))
    assert a == b


@pytest.mark.parametrize("name", ["horizontal_flip", "vertical_flip", "arbitrary_rotation", "perspective_warp"])
def test_forbidden_spatial_policies_fail(name):
    with pytest.raises(ValueError, match=name):
        validate_rel_plus_augmentation_policy(**{name: True})
