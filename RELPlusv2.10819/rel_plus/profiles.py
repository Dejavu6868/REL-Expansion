"""Trusted dataset camera profiles for REL+ v2.1 production entry points."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DatasetCameraProfile:
    name: str
    native_image_shape: Tuple[int, int]
    canonical_image_shape: Tuple[int, int]
    k_convention: str
    pose_convention: str

    def __post_init__(self):
        for field_name in ("native_image_shape", "canonical_image_shape"):
            shape = tuple(getattr(self, field_name))
            if (
                len(shape) != 2
                or any(isinstance(value, bool) for value in shape)
                or any(int(value) != value or int(value) <= 0 for value in shape)
            ):
                raise ValueError("{} must contain two positive integers".format(field_name))
            object.__setattr__(self, field_name, tuple(int(value) for value in shape))
        if self.k_convention != "json_half_pixel":
            raise ValueError("unsupported k_convention: {}".format(self.k_convention))
        if self.pose_convention != "world_to_camera_3x4":
            raise ValueError("unsupported pose_convention: {}".format(self.pose_convention))
        if not str(self.name):
            raise ValueError("profile name must not be empty")

    def assert_native_image_shape(self, image_shape):
        shape = tuple(int(value) for value in tuple(image_shape)[:2])
        if shape != self.native_image_shape:
            raise ValueError(
                "depth shape {} does not match dataset profile native_image_shape {}".format(
                    shape, self.native_image_shape
                )
            )

    def assert_camera_reference(self, camera):
        """Reject a plausibly cross-resolution K before any resize is applied."""
        camera.assert_matches_image_shape(self.native_image_shape)
        height, width = self.native_image_shape
        cx = float(camera.K_json[0, 2])
        cy = float(camera.K_json[1, 2])
        # Stanford S2D principal points are near the image centre. This narrow
        # profile-level guard catches a canonical K presented as native (and
        # vice versa) while CameraGeometry retains the general image-bound guard.
        if not (0.25 * width <= cx <= 0.75 * width):
            raise ValueError(
                "principal point cx={} is inconsistent with profile width {}".format(cx, width)
            )
        if not (0.25 * height <= cy <= 0.75 * height):
            raise ValueError(
                "principal point cy={} is inconsistent with profile height {}".format(cy, height)
            )


STANFORD_S2D_PROFILE = DatasetCameraProfile(
    name="stanford2d3d_s2d",
    native_image_shape=(1080, 1080),
    canonical_image_shape=(480, 480),
    k_convention="json_half_pixel",
    pose_convention="world_to_camera_3x4",
)
