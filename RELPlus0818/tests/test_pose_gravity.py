import json

import numpy as np
import pytest

from rel_plus.camera import CameraGeometry, gravity_in_camera, load_stanford_s2d_camera_geometry
from rel_plus.source_helpers import align_points_and_normals_to_gravity


def _rotation_x(degrees):
    angle = np.radians(degrees)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]]
    )


@pytest.mark.parametrize("degrees", [-35.0, -12.0, 18.0, 41.0])
def test_source_rotation_aligns_pose_gravity_to_world_down(degrees):
    rotation = _rotation_x(degrees)
    gravity = gravity_in_camera(rotation)
    vectors = gravity.reshape(1, 1, 3)
    aligned, _, matrix = align_points_and_normals_to_gravity(vectors, vectors, gravity)
    np.testing.assert_allclose(aligned[0, 0], [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(matrix @ gravity, [0.0, 0.0, -1.0], atol=1e-6)


def test_real_parser_contract_is_explicit_world_to_camera_and_checks_camera_center(tmp_path):
    rotation = _rotation_x(23.0)
    center = np.array([1.2, -0.4, 2.5])
    translation = -rotation @ center
    payload = {
        "camera_k_matrix": [[500.0, 0.0, 540.0], [0.0, 500.0, 540.0], [0.0, 0.0, 1.0]],
        "camera_rt_matrix": np.column_stack([rotation, translation]).tolist(),
        "camera_location": center.tolist(),
    }
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    camera = load_stanford_s2d_camera_geometry(path)
    assert isinstance(camera, CameraGeometry)
    np.testing.assert_allclose(camera.R_world_to_camera, rotation)
    np.testing.assert_allclose(camera.t_world_to_camera, translation)
    np.testing.assert_allclose(-rotation.T @ translation, center)

