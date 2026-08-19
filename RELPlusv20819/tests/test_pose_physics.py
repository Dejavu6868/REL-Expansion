import numpy as np

from rel_plus.camera import CameraGeometry
from rel_plus.validation.pose_physics import validate_pose_physics
from tests.helpers import rotation_y


def test_self_consistent_transposed_pose_passes_algebra_but_fails_physics():
    shape = (40, 50)
    k = np.array([[80.0, 0.0, 25.0], [0.0, 82.0, 20.0], [0.0, 0.0, 1.0]])
    rotation_true = rotation_y(27.0)
    center = np.array([1.2, -0.4, 0.7])
    translation_true = -rotation_true @ center
    camera_true = CameraGeometry.from_json_k(k, shape, rotation_true, translation_true)
    world = np.array(
        [[x, y, 4.0 + 0.2 * x] for y in np.linspace(-1.0, 1.0, 20) for x in np.linspace(-1.0, 1.0, 25)]
    )
    expected_camera = world @ rotation_true.T + translation_true
    u = k[0, 0] * expected_camera[:, 0] / expected_camera[:, 2] + k[0, 2]
    v = k[1, 1] * expected_camera[:, 1] / expected_camera[:, 2] + k[1, 2]
    pixels = np.column_stack([u, v])
    assert validate_pose_physics(
        camera_true, world_points=world, camera_points=expected_camera, pixel_coordinates=pixels
    ).status == "PASS"

    rotation_wrong = rotation_true.T
    translation_wrong = -rotation_wrong @ center
    camera_wrong = CameraGeometry.from_json_k(k, shape, rotation_wrong, translation_wrong)
    np.testing.assert_allclose(rotation_wrong.T @ rotation_wrong, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation_wrong), 1.0, atol=1e-12)
    np.testing.assert_allclose(-rotation_wrong.T @ translation_wrong, center, atol=1e-12)
    result = validate_pose_physics(
        camera_wrong, world_points=world, camera_points=expected_camera, pixel_coordinates=pixels
    )
    assert result.status == "FAIL"


def test_absent_semantic_class_is_not_a_pose_failure():
    camera = CameraGeometry.from_json_k(np.eye(3), (4, 4), np.eye(3), np.zeros(3))
    labels = np.zeros((4, 4), dtype=np.uint8)
    normals = np.zeros((4, 4, 3), dtype=np.float64)
    result = validate_pose_physics(
        camera,
        labels=labels,
        normals_aligned=normals,
        semantic_ids={"floor": 9, "ceiling": 4, "wall": 12},
        minimum_semantic_pixels=2,
    )
    assert result.status == "NOT_APPLICABLE"


def test_height_deciles_are_warning_only_weak_evidence():
    camera = CameraGeometry.from_json_k(np.eye(3), (4, 4), np.eye(3), np.zeros(3))
    points = np.zeros((4, 4, 3), dtype=np.float64)
    points[:, :, 2] = np.arange(16).reshape(4, 4)
    normals = np.zeros_like(points)
    normals[:, :, 2] = np.linspace(-1.0, 1.0, 16).reshape(4, 4)
    result = validate_pose_physics(
        camera, normals_aligned=normals, points_aligned_m=points
    )
    assert result.status == "NOT_APPLICABLE"
    assert "weak_low_height_normal_z_median" in result.metrics
    assert "weak_high_height_normal_z_median" in result.metrics
