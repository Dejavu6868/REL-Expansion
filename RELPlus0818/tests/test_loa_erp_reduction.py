import numpy as np

from rel_plus.encoding import erp_tangent_field, perspective_tangent_field


def test_perspective_tangent_reduces_to_original_erp_tangent():
    height, width = 3, 64
    phi = (np.arange(width, dtype=np.float64) / width) * 2.0 * np.pi - np.pi
    radial = np.stack([np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
    points = np.broadcast_to(radial[None, ...], (height, width, 3)).copy()

    perspective_tangent, radius = perspective_tangent_field(points)
    erp_tangent, _ = erp_tangent_field(height, width)

    assert float(np.min(radius)) > 0.0
    assert float(np.max(np.abs(perspective_tangent - erp_tangent))) < 1e-5


def test_horizontal_mirror_negates_continuous_hcos_not_plain_uint8_pixels():
    normal = np.array([0.6, 0.8, 0.0], dtype=np.float64)
    tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    mirrored_normal = np.array([-normal[0], normal[1], normal[2]])
    mirrored_tangent = np.array([tangent[0], -tangent[1], tangent[2]])
    hcos = float(normal @ tangent)
    mirrored_hcos = float(mirrored_normal @ mirrored_tangent)
    assert abs(hcos) > 0.1
    np.testing.assert_allclose(mirrored_hcos, -hcos, atol=1e-12)
    angle = np.degrees(np.arccos(hcos))
    mirrored_angle = np.degrees(np.arccos(mirrored_hcos))
    np.testing.assert_allclose(mirrored_angle, 180.0 - angle, atol=1e-12)


def test_horizontal_axis_singularity_has_frozen_loa_of_90_degrees():
    points = np.array([[[0.0, 0.0, 2.0]]], dtype=np.float64)
    tangent, radius = perspective_tangent_field(points)
    np.testing.assert_array_equal(tangent, np.zeros((1, 1, 3)))
    np.testing.assert_array_equal(radius, [[0.0]])
    hcos = float(np.array([0.2, -0.4, 0.8]) @ tangent[0, 0])
    assert int(np.degrees(np.arccos(hcos))) == 90
