import json

import cv2
import numpy as np
import pytest

from rel_plus.camera import CameraGeometry
from rel_plus.profiles import DatasetCameraProfile, STANFORD_S2D_PROFILE
from rel_plus.stanford_s2d import load_canonical_frame


def _write_frame(tmp_path, shape, principal):
    depth_path = tmp_path / "depth.png"
    pose_path = tmp_path / "pose.json"
    assert cv2.imwrite(str(depth_path), np.full(shape, 1024, dtype=np.uint16))
    payload = {
        "camera_k_matrix": [
            [800.0, 0.0, float(principal[0])],
            [0.0, 810.0, float(principal[1])],
            [0.0, 0.0, 1.0],
        ],
        "camera_rt_matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        "camera_location": [0.0, 0.0, 0.0],
    }
    pose_path.write_text(json.dumps(payload), encoding="utf-8")
    return depth_path, pose_path


def test_stanford_profile_is_frozen():
    assert STANFORD_S2D_PROFILE == DatasetCameraProfile(
        name="stanford2d3d_s2d",
        native_image_shape=(1080, 1080),
        canonical_image_shape=(480, 480),
        k_convention="json_half_pixel",
        pose_convention="world_to_camera_3x4",
    )


def test_native_depth_and_native_k_pass(tmp_path):
    depth, pose = _write_frame(tmp_path, (1080, 1080), (540.0, 540.0))
    raw, camera, source_shape = load_canonical_frame(
        depth, pose, dataset_profile=STANFORD_S2D_PROFILE
    )
    assert source_shape == (1080, 1080)
    assert raw.shape == (480, 480)
    assert camera.intrinsics_shape == (480, 480)


def test_canonical_depth_and_canonical_k_pass_with_explicit_profile(tmp_path):
    profile = DatasetCameraProfile(
        "canonical_fixture", (480, 480), (480, 480),
        "json_half_pixel", "world_to_camera_3x4",
    )
    depth, pose = _write_frame(tmp_path, (480, 480), (240.0, 240.0))
    raw, camera, source_shape = load_canonical_frame(
        depth, pose, dataset_profile=profile
    )
    assert source_shape == raw.shape == camera.intrinsics_shape == (480, 480)


@pytest.mark.parametrize(
    "shape,principal",
    [
        ((480, 480), (540.0, 540.0)),
        ((1080, 1080), (240.0, 240.0)),
    ],
)
def test_cross_resolution_k_is_rejected(tmp_path, shape, principal):
    profile = DatasetCameraProfile(
        "explicit_fixture", shape, shape,
        "json_half_pixel", "world_to_camera_3x4",
    )
    depth, pose = _write_frame(tmp_path, shape, principal)
    with pytest.raises(ValueError, match="principal point"):
        load_canonical_frame(depth, pose, dataset_profile=profile)


def test_wrong_profile_is_rejected_before_camera_use(tmp_path):
    depth, pose = _write_frame(tmp_path, (1080, 1080), (540.0, 540.0))
    wrong = DatasetCameraProfile(
        "wrong", (720, 720), (480, 480),
        "json_half_pixel", "world_to_camera_3x4",
    )
    with pytest.raises(ValueError, match="native_image_shape"):
        load_canonical_frame(depth, pose, dataset_profile=wrong)


def test_nonzero_skew_and_obvious_principal_point_are_rejected():
    with pytest.raises(ValueError, match="zero skew"):
        CameraGeometry.from_json_k(
            np.array([[800.0, 0.1, 540.0], [0.0, 810.0, 540.0], [0.0, 0.0, 1.0]]),
            (1080, 1080), np.eye(3), np.zeros(3),
        )
    with pytest.raises(ValueError, match="principal point"):
        CameraGeometry.from_json_k(
            np.array([[800.0, 0.0, 5000.0], [0.0, 810.0, 540.0], [0.0, 0.0, 1.0]]),
            (1080, 1080), np.eye(3), np.zeros(3),
        )
