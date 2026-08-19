import numpy as np
import pytest

from rel_plus.encoding import encode_rel_channels, perspective_tangent_field


def encode(points, normals, valid):
    tangent, _ = perspective_tangent_field(points)
    return encode_rel_channels(points, normals, valid, tangent=tangent)[0]


def source_cm_reference(points_m, normals, valid):
    return encode(points_m * 100.0, normals, valid)


def test_dense_nondegenerate_encoding_is_scale_invariant():
    rows, columns = np.indices((7, 9), dtype=np.float64)
    points_m = np.stack([0.2 + columns, 0.3 + rows, 1.0 + rows * 0.1], axis=-1)
    normals = np.dstack([np.zeros_like(rows), np.zeros_like(rows), np.ones_like(rows)])
    valid = np.ones(rows.shape, dtype=bool)
    np.testing.assert_array_equal(encode(points_m, normals, valid), source_cm_reference(points_m, normals, valid))


@pytest.mark.parametrize("case", ["all_invalid", "single", "constant_height", "constant_radius"])
def test_degenerate_unit_branches_match_source_centimetre_path(case):
    points_m = np.zeros((3, 4, 3), dtype=np.float64)
    normals = np.zeros_like(points_m)
    normals[..., 2] = 1.0
    valid = np.ones((3, 4), dtype=bool)
    if case == "all_invalid":
        valid[:] = False
    elif case == "single":
        valid[:] = False
        valid[1, 2] = True
        points_m[1, 2] = [0.01, 0.02, 0.03]
    elif case == "constant_height":
        points_m[..., 0] = np.arange(4)[None, :] * 0.01
        points_m[..., 1] = np.arange(3)[:, None] * 0.01
        points_m[..., 2] = 0.02
    else:
        phi = np.linspace(0.0, 1.0, 12).reshape(3, 4)
        points_m[..., 0] = np.cos(phi) * 0.02
        points_m[..., 1] = np.sin(phi) * 0.02
        points_m[..., 2] = np.arange(12).reshape(3, 4) * 0.01
    expected = source_cm_reference(points_m, normals, valid)
    actual = encode(points_m * 100.0, normals, valid)
    np.testing.assert_array_equal(actual, expected)


def test_nan_and_zero_normal_source_semantics_are_channel_local():
    points_cm = np.array([[[1.0, 0.0, 1.0], [2.0, 0.0, 2.0], [3.0, 0.0, 3.0]]])
    normals = np.array([[[np.nan, np.nan, np.nan], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    actual = encode(points_cm, normals, np.ones((1, 3), bool))
    assert actual[0, 0, 0] == 255 and actual[0, 0, 1] == 90 and actual[0, 0, 2] != 255
    assert actual[0, 1, 0] == 127 and actual[0, 1, 1] == 90 and actual[0, 1, 2] != 255
