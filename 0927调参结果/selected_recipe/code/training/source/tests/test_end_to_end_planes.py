import numpy as np
import pytest

from rel_plus.camera import CameraGeometry
from rel_plus.generator import generate_rel_plus_v2
from tests.helpers import (
    independent_plane_depth,
    look_rotation,
    rotation_x,
    rotation_y,
    rotation_z,
)


CASES = [
    # One degree avoids the deliberately rejected exact anti-parallel gravity
    # singularity while remaining a frontal floor view.
    ("floor_front", [0, 0, -1], [0, 0, -1], rotation_x(1), -1.0),
    ("ceiling_front", [0, 0, 1], [0, 0, 1], np.eye(3), 1.0),
    ("wall_front", [1, 0, 0], [1, 0, 0], np.eye(3), 0.0),
    ("floor_pitch_pos", [0, 0, -1], [0, 0, -1], rotation_x(30), -1.0),
    ("ceiling_pitch_neg", [0, 0, 1], [0, 0, 1], rotation_x(-30), 1.0),
    ("wall_roll_pos", [1, 0, 0], [1, 0, 0], rotation_z(20), 0.0),
    ("wall_roll_neg", [1, 0, 0], [1, 0, 0], rotation_z(-20), 0.0),
    ("room_pitch_roll", [0, 0, -1], [0, 0, -1], rotation_y(18) @ rotation_z(14), -1.0),
]


@pytest.mark.parametrize("name,forward,normal,local_rotation,expected_normal_z", CASES)
def test_full_depth_to_rel_pipeline_on_analytic_planes(name, forward, normal, local_rotation, expected_normal_z):
    shape = (48, 48)
    k = np.array([[70.0, 0.0, 24.0], [0.0, 70.0, 24.0], [0.0, 0.0, 1.0]])
    rotation = local_rotation @ look_rotation(forward)
    raw, valid, _world_points, camera_points = independent_plane_depth(
        shape, k, rotation, normal, 2.0
    )
    assert float(np.mean(valid)) > 0.95, name
    raw[0, 0] = 0
    camera = CameraGeometry.from_json_k(k, shape, rotation, np.zeros(3), sample_id=name)
    rel_plus, debug = generate_rel_plus_v2(raw, camera, return_debug=True)
    quantization_tolerance = 3.0 / 512.0
    compare = debug["depth_valid"] & valid
    assert float(np.quantile(np.abs(debug["points_camera_m"][compare] - camera_points[compare]), 0.95)) < quantization_tolerance
    median_normal_z = float(np.nanmedian(debug["normals_aligned"][:, :, 2][compare]))
    assert abs(median_normal_z - expected_normal_z) < 0.08
    np.testing.assert_array_equal(rel_plus[0, 0], [255, 255, 255])
    assert np.unique(rel_plus[:, :, 2][compare]).size > 1
    assert np.all((rel_plus[:, :, 1][compare] >= 0) & (rel_plus[:, :, 1][compare] <= 180))
    if expected_normal_z < -0.5:
        assert float(np.median(rel_plus[:, :, 0][compare])) < 20
    elif expected_normal_z > 0.5:
        assert float(np.median(rel_plus[:, :, 0][compare])) > 235
