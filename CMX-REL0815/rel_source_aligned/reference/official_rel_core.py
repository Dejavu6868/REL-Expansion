from dataclasses import dataclass

import numpy as np


# This is the actual np.stack order in getREL.py at authority commit 16c1267.
OFFICIAL_CHANNEL_ORDER = (
    "EGVIA_source_code",
    "LOA_source_code",
    "ReD_source_code",
)


@dataclass(frozen=True)
class SourceExactRELResult:
    rel: np.ndarray
    channel_order: tuple
    red: np.ndarray
    egvia: np.ndarray
    loa: np.ndarray
    height_normalized: np.ndarray
    inclination: np.ndarray
    valid_mask: np.ndarray


def erp_azimuth(shape):
    height, width = shape
    phi = (np.arange(width) / width) * 2 * np.pi - np.pi
    return np.broadcast_to(phi[np.newaxis, :], (height, width))


class SourceExactRELCore:
    """The post-geometry body of author getREL.py, without semantic repairs."""

    def encode(self, pc_rot, normals_rot, azimuth, missing_mask, alpha=45, lam=0.5):
        pc_rot = np.asarray(pc_rot)
        normals_rot = np.asarray(normals_rot)
        azimuth = np.asarray(azimuth)
        missing_mask = np.asarray(missing_mask, dtype=bool)
        expected_image_shape = pc_rot.shape[:2]
        if pc_rot.shape != normals_rot.shape or pc_rot.ndim != 3 or pc_rot.shape[2] != 3:
            raise ValueError("pc_rot and normals_rot must both be HxWx3")
        if azimuth.shape != expected_image_shape or missing_mask.shape != expected_image_shape:
            raise ValueError("azimuth and missing_mask must match the image shape")

        cos_theta = np.cos(azimuth)
        sin_theta = np.sin(azimuth)
        hcos = normals_rot[:, :, 0] * cos_theta - normals_rot[:, :, 1] * sin_theta
        hcos = np.nan_to_num(hcos, nan=0)
        hcos = np.clip(hcos, -1.0, 1.0)
        loa = (np.arccos(hcos) * 180 / np.pi).astype(np.uint8)

        red = np.hypot(pc_rot[:, :, 0], pc_rot[:, :, 1])
        red_min = red.min()
        red_max = red.max()
        if red_max > red_min:
            red = (red - red_min) * 255.0 / (red_max - red_min)
        red = np.clip(red, 0, 255).astype(np.uint8)

        height = pc_rot[:, :, 2]
        height_min = np.percentile(height, 1)
        height_max = np.percentile(height, 99)
        if height_max > height_min:
            height = (height - height_min) * 255.0 / (height_max - height_min)
        height = np.clip(height, 0, 255).astype(np.float32)

        normal_z = -normals_rot[:, :, 2]
        normal_z = np.clip(normal_z, -1.0, 1.0)
        inclination = (np.arccos(normal_z, dtype=np.float32) / np.pi) * 255.0
        inclination = np.clip(inclination, 0, 255).astype(np.float32)
        egvia = inclination.copy()

        angle_threshold = alpha * 255.0 / 180.0
        is_horizontal = (egvia <= angle_threshold) | (
            egvia >= 255.0 - angle_threshold
        )
        egvia[~is_horizontal] = (
            lam * egvia[~is_horizontal] + (1 - lam) * height[~is_horizontal]
        )

        rel = np.stack([egvia, loa, red], axis=2).astype(np.float32)
        rel = np.nan_to_num(rel, nan=255.0)
        rel[missing_mask, :] = 255.0
        rel = np.clip(rel, 0, 255).astype(np.uint8)

        return SourceExactRELResult(
            rel=rel,
            channel_order=OFFICIAL_CHANNEL_ORDER,
            red=red,
            egvia=egvia,
            loa=loa,
            height_normalized=height,
            inclination=inclination,
            valid_mask=~missing_mask,
        )

