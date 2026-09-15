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
    ).status == "PASS_STRONG"

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


def test_absent_semantic_class_requires_review_instead_of_passing():
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
    assert result.status == "REVIEW_REQUIRED"


def test_height_deciles_are_warning_only_weak_evidence():
    camera = CameraGeometry.from_json_k(np.eye(3), (4, 4), np.eye(3), np.zeros(3))
    points = np.zeros((4, 4, 3), dtype=np.float64)
    points[:, :, 2] = np.arange(16).reshape(4, 4)
    normals = np.zeros_like(points)
    normals[:, :, 2] = np.linspace(-1.0, 1.0, 16).reshape(4, 4)
    result = validate_pose_physics(
        camera, normals_aligned=normals, points_aligned_m=points
    )
    assert result.status == "REVIEW_REQUIRED"
    assert "weak_low_height_normal_z_median" in result.metrics
    assert "weak_high_height_normal_z_median" in result.metrics


def test_semantic_only_correct_pose_is_weak_and_wrong_pose_fails():
    shape = (12, 12)
    camera = CameraGeometry.from_json_k(
        np.array([[20.0, 0.0, 6.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]]),
        shape, np.eye(3), np.zeros(3),
    )
    labels = np.zeros(shape, dtype=np.uint8)
    labels[:4] = 9
    labels[4:8] = 4
    labels[8:] = 12
    normals = np.zeros(shape + (3,), dtype=np.float64)
    normals[:4, :, 2] = -1.0
    normals[4:8, :, 2] = 1.0
    normals[8:, :, 0] = 1.0
    semantic_valid = np.ones(shape, dtype=bool)
    semantic_ids = {"floor": 9, "ceiling": 4, "wall": 12}
    result = validate_pose_physics(
        camera,
        labels=labels,
        normals_aligned=normals,
        semantic_valid_mask=semantic_valid,
        semantic_ids=semantic_ids,
        minimum_semantic_pixels=8,
    )
    assert result.status == "PASS_WEAK"
    assert result.evidence_level == "weak"
    assert "floor_angle_deg" in result.metrics
    assert "ceiling_angle_deg" in result.metrics
    assert "wall_angle_from_horizontal_deg" in result.metrics

    rotation_true = rotation_y(27.0)
    center = np.array([1.2, -0.4, 0.7])
    rotation_wrong = rotation_true.T
    camera_wrong = CameraGeometry.from_json_k(
        camera.K_json, shape, rotation_wrong, -rotation_wrong @ center
    )
    wrong_normals = normals.copy()
    wrong_normals[..., 2] = 0.0
    wrong = validate_pose_physics(
        camera_wrong,
        labels=labels,
        normals_aligned=wrong_normals,
        semantic_valid_mask=semantic_valid,
        semantic_ids=semantic_ids,
        minimum_semantic_pixels=8,
    )
    assert wrong.status == "FAIL"


def test_semantic_quality_mask_excludes_zero_and_low_support_normals():
    shape = (6, 6)
    camera = CameraGeometry.from_json_k(np.eye(3), shape, np.eye(3), np.zeros(3))
    labels = np.full(shape, 9, dtype=np.uint8)
    normals = np.zeros(shape + (3,), dtype=np.float64)
    normals[..., 2] = 1.0
    semantic_valid = np.zeros(shape, dtype=bool)
    semantic_valid[1:5, 1:5] = True
    normals[semantic_valid, 2] = -1.0
    result = validate_pose_physics(
        camera,
        labels=labels,
        normals_aligned=normals,
        semantic_valid_mask=semantic_valid,
        semantic_ids={"floor": 9},
        minimum_semantic_pixels=8,
    )
    assert result.status == "PASS_WEAK"
    assert result.metrics["floor_count"] == 16
