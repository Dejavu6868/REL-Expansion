import numpy as np

from rel_plus.camera import CameraGeometry
from rel_plus.encoding import encode_rel_channels, perspective_tangent_field
from rel_plus.generator import generate_rel_plus


def test_generate_rel_plus_returns_frozen_shape_dtype_order_and_debug_fields():
    height = width = 32
    raw = np.full((height, width), 1024, dtype=np.uint16)
    raw[0, 0] = 0
    camera = CameraGeometry.from_json_k(
        np.array([[40.0, 0.0, 16.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]]),
        np.eye(3),
        np.zeros(3),
    )

    rel_plus, debug = generate_rel_plus(raw, camera, normal_radius=2, return_debug=True)

    assert rel_plus.shape == (height, width, 3)
    assert rel_plus.dtype == np.uint8
    np.testing.assert_array_equal(rel_plus[0, 0], [255, 255, 255])
    required = {
        "depth_m", "valid_mask", "points_camera", "normals_camera",
        "gravity_camera", "gravity_alignment_rotation", "points_aligned",
        "normals_aligned", "red_raw", "red_encoded", "height_raw",
        "height_normalized", "egvia_angle_before_blend", "egvia_encoded",
        "horizontal_radius", "tangent", "hcos", "loa_encoded", "rel_plus",
    }
    assert required.issubset(debug)


def test_synthetic_floor_wall_ceiling_angle_responses_are_distinct():
    points = np.array([[[1.0, 0.0, 1.0], [1.0, 0.0, 2.0], [1.0, 0.0, 3.0]]])
    normals = np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]])
    valid = np.ones((1, 3), dtype=bool)
    tangent, _ = perspective_tangent_field(points)
    _, debug = encode_rel_channels(points, normals, valid, tangent=tangent, lam=1.0)
    angles = debug["egvia_angle_before_blend"][0]
    np.testing.assert_allclose(angles, [255.0, 127.5, 0.0], atol=1e-4)

