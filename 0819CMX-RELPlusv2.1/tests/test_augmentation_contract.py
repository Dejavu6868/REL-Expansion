import cv2
import numpy as np
import pytest

from rel_plus.integration.cmx_preprocess import (
    SpatialTransform,
    apply_cmx_compatible_preprocess,
    sample_spatial_transform,
)
from rel_plus.policy import validate_rel_plus_augmentation_policy
from utils.transforms import normalize, random_crop_pad_to_shape


def coordinate_triplet(shape=(6, 8)):
    rows, columns = np.indices(shape)
    marker = (rows * 10 + columns).astype(np.uint8)
    rgb = np.dstack([marker, marker + 1, marker + 2])
    rel = np.dstack([marker + 11, marker + 22, marker + 33])
    label = marker.copy()
    valid = np.ones(shape, dtype=bool)
    return rgb, rel, label, valid


def make_transform(source_shape, scale, crop_top, crop_left, output_shape):
    return SpatialTransform(
        source_shape[0],
        source_shape[1],
        scale,
        max(1, int(source_shape[0] * scale)),
        max(1, int(source_shape[1] * scale)),
        crop_top,
        crop_left,
        output_shape[0],
        output_shape[1],
    )


def cmx_source_reference(rgb, rel, label, valid, transform):
    size = (transform.scaled_width, transform.scaled_height)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
    rel = cv2.resize(rel, size, interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
    valid = cv2.resize(
        valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = normalize(rgb, mean, std)
    rel = normalize(rel, mean, std)
    crop_pos = (transform.crop_top, transform.crop_left)
    crop_size = (transform.output_height, transform.output_width)

    rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
    rel, _ = random_crop_pad_to_shape(rel, crop_pos, crop_size, 0)
    label, _ = random_crop_pad_to_shape(label, crop_pos, crop_size, 255)
    valid, _ = random_crop_pad_to_shape(
        valid.astype(np.uint8), crop_pos, crop_size, 0
    )

    return (
        np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32),
        np.ascontiguousarray(rel.transpose(2, 0, 1), dtype=np.float32),
        label,
        valid.astype(bool),
    )


def test_one_spatial_transform_applies_to_rgb_rel_and_label():
    rgb, rel, label, valid = coordinate_triplet()
    transform = make_transform(rgb.shape[:2], 1.0, 1, 2, (4, 5))
    output = apply_cmx_compatible_preprocess(rgb, rel, label, valid, transform)
    np.testing.assert_array_equal(output.label, label[1:5, 2:7])
    expected_rel = normalize(
        rel[1:5, 2:7],
        np.array([0.485, 0.456, 0.406], np.float32),
        np.array([0.229, 0.224, 0.225], np.float32),
    ).astype(np.float32)
    np.testing.assert_allclose(output.modal_x, expected_rel.transpose(2, 0, 1))
    np.testing.assert_array_equal(output.modal_x_valid_mask, valid[1:5, 2:7])


def test_scale_uses_linear_for_rgb_and_rel_nearest_for_label():
    rgb, rel, label, valid = coordinate_triplet((4, 4))
    transform = make_transform(rgb.shape[:2], 1.5, 0, 0, (6, 6))
    output = apply_cmx_compatible_preprocess(rgb, rel, label, valid, transform)
    assert output.rgb.shape == (3, 6, 6)
    assert output.modal_x.shape == (3, 6, 6)
    assert set(np.unique(output.label)).issubset(set(np.unique(label)))


@pytest.mark.parametrize(
    "transform",
    [
        make_transform((6, 8), 0.75, 0, 0, (6, 6)),
        make_transform((6, 8), 1.25, 1, 2, (6, 6)),
    ],
)
def test_shared_adapter_matches_independent_cmx_array_chain(transform):
    rgb, rel, label, valid = coordinate_triplet()
    expected_rgb, expected_rel, expected_label, expected_valid = cmx_source_reference(
        rgb, rel, label, valid, transform
    )
    output = apply_cmx_compatible_preprocess(
        rgb, rel, label, valid, transform
    )
    np.testing.assert_allclose(output.rgb, expected_rgb, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(output.modal_x, expected_rel, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(output.label, expected_label)
    np.testing.assert_array_equal(output.modal_x_valid_mask, expected_valid)


def test_photometric_augmentation_changes_rgb_only():
    rgb, rel, label, valid = coordinate_triplet()
    transform = make_transform(rgb.shape[:2], 1.0, 0, 0, (6, 8))
    plain = apply_cmx_compatible_preprocess(rgb, rel, label, valid, transform)
    changed = apply_cmx_compatible_preprocess(
        rgb, rel, label, valid, transform,
        photometric_augmentation=lambda value: np.zeros_like(value)
    )
    assert not np.array_equal(plain.rgb, changed.rgb)
    np.testing.assert_array_equal(plain.modal_x, changed.modal_x)
    np.testing.assert_array_equal(plain.label, changed.label)
    np.testing.assert_array_equal(
        plain.modal_x_valid_mask, changed.modal_x_valid_mask
    )


def test_seeded_transform_sampling_is_reproducible():
    a = sample_spatial_transform((480, 480), [0.5, 1.0, 1.5], (320, 320), np.random.default_rng(7))
    b = sample_spatial_transform((480, 480), [0.5, 1.0, 1.5], (320, 320), np.random.default_rng(7))
    assert a == b


@pytest.mark.parametrize("name", ["horizontal_flip", "vertical_flip", "arbitrary_rotation", "perspective_warp"])
def test_forbidden_spatial_policies_fail(name):
    with pytest.raises(ValueError, match=name):
        validate_rel_plus_augmentation_policy(**{name: True})
