import numpy as np
import pytest

from rel_plus.camera import CameraGeometry, backproject_z_depth, resize_camera_geometry


def make_camera(shape):
    return CameraGeometry.from_json_k(
        np.array([[800.0, 0.0, 540.0], [0.0, 810.0, 540.0], [0.0, 0.0, 1.0]]),
        shape,
        np.eye(3),
        np.zeros(3),
    )


@pytest.mark.parametrize("shape", [(1080, 1080), (480, 480)])
def test_matching_depth_and_intrinsics_shape_pass(shape):
    camera = make_camera(shape)
    camera.assert_matches_image_shape(shape)


@pytest.mark.parametrize(
    "depth_shape,k_shape", [((480, 480), (1080, 1080)), ((1080, 1080), (480, 480))]
)
def test_mismatched_depth_and_intrinsics_shape_fail(depth_shape, k_shape):
    with pytest.raises(ValueError, match="intrinsics_shape"):
        make_camera(k_shape).assert_matches_image_shape(depth_shape)


def test_resize_uses_bound_source_shape_and_rebinds_destination():
    camera = make_camera((1080, 1080))
    resized = resize_camera_geometry(camera, (480, 480))
    assert resized.intrinsics_shape == (480, 480)
    np.testing.assert_allclose(resized.K_json[0], camera.K_json[0] * (480.0 / 1080.0))
    np.testing.assert_allclose(resized.K_json[1], camera.K_json[1] * (480.0 / 1080.0))


def test_half_pixel_backprojection_and_zero_based_constructor():
    zero_k = np.array([[100.0, 0.0, 2.0], [0.0, 120.0, 1.0], [0.0, 0.0, 1.0]])
    camera = CameraGeometry.from_zero_based_center_k(
        zero_k, (3, 4), np.eye(3), np.zeros(3)
    )
    np.testing.assert_allclose(camera.K_json[:2, 2], [2.5, 1.5])
    depth = np.full((3, 4), 2.0, dtype=np.float32)
    points = backproject_z_depth(depth, np.ones_like(depth, bool), camera.K_json)
    rows, columns = np.indices(depth.shape)
    u = camera.K_json[0, 0] * points[..., 0] / points[..., 2] + camera.K_json[0, 2]
    v = camera.K_json[1, 1] * points[..., 1] / points[..., 2] + camera.K_json[1, 2]
    np.testing.assert_allclose(u, columns + 0.5, atol=1e-12)
    np.testing.assert_allclose(v, rows + 0.5, atol=1e-12)
