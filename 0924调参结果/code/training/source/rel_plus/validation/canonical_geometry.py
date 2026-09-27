"""Native and canonical external-XYZ geometry validation."""

import cv2
import numpy as np

from ..camera import backproject_z_depth, resize_camera_geometry
from ..depth import decode_stanford_s2d_depth, resize_raw_depth_nearest
from .geometry_oracle import (
    component_statistics,
    evenly_spaced_joint_pixels,
    project_camera_points,
    scalar_statistics,
)


NATIVE_COMPONENT_P95_TOLERANCE_M = 1.0 / 512.0
NATIVE_REPROJECTION_P95_TOLERANCE_PIXELS = 0.05
CANONICAL_REPROJECTION_P95_TOLERANCE_PIXELS = 1.0
CANONICAL_REPROJECTION_MAX_TOLERANCE_PIXELS = 1.5


def _stage_metrics(raw_depth, xyz_world, camera, maximum_count):
    camera.assert_matches_image_shape(raw_depth.shape)
    depth_m, depth_valid = decode_stanford_s2d_depth(raw_depth)
    depth_points = backproject_z_depth(depth_m, depth_valid, camera.K_json)
    xyz = np.asarray(xyz_world, dtype=np.float64)
    if xyz.shape != depth_points.shape:
        raise ValueError("global XYZ shape does not match depth")
    xyz_valid = np.all(np.isfinite(xyz), axis=2) & ~np.all(xyz == 0.0, axis=2)
    rows, columns = evenly_spaced_joint_pixels(
        depth_valid & xyz_valid, maximum_count=maximum_count
    )
    xyz_camera = (
        xyz[rows, columns] @ camera.R_world_to_camera.T
        + camera.t_world_to_camera
    )
    expected = depth_points[rows, columns]
    component_error = np.abs(xyz_camera - expected)
    euclidean_error = np.linalg.norm(xyz_camera - expected, axis=1)
    projected_u, projected_v = project_camera_points(xyz_camera, camera.K_json)
    reprojection = np.hypot(
        projected_u - (columns + 0.5), projected_v - (rows + 0.5)
    )
    return {
        "probe_count": int(len(rows)),
        "component_error_m": component_statistics(component_error),
        "euclidean_error_m": scalar_statistics(euclidean_error),
        "reprojection_pixels": scalar_statistics(reprojection),
        "depth_m_p95": float(np.quantile(depth_m[rows, columns], 0.95)),
    }


def validate_canonical_geometry(
    raw_depth,
    xyz_world,
    camera,
    destination_shape=(480, 480),
    *,
    maximum_count=4096
):
    """Use identical nearest mapping for depth and external XYZ, then compare."""
    native = _stage_metrics(raw_depth, xyz_world, camera, maximum_count)
    canonical_raw = resize_raw_depth_nearest(raw_depth, destination_shape)
    canonical_xyz = cv2.resize(
        np.asarray(xyz_world),
        (int(destination_shape[1]), int(destination_shape[0])),
        interpolation=cv2.INTER_NEAREST,
    )
    canonical_camera = resize_camera_geometry(camera, destination_shape)
    canonical = _stage_metrics(
        canonical_raw, canonical_xyz, canonical_camera, maximum_count
    )

    native_component_p95 = max(
        native["component_error_m"][axis]["p95"] for axis in ("x", "y", "z")
    )
    canonical_component_p95 = max(
        canonical["component_error_m"][axis]["p95"] for axis in ("x", "y", "z")
    )
    # Nearest resize selects a native ray whose projected canonical centre may
    # differ by less than one pixel. Convert that frozen pixel tolerance to a
    # metric allowance at the observed P95 depth; keep depth quantization too.
    min_focal = min(canonical_camera.K_json[0, 0], canonical_camera.K_json[1, 1])
    canonical_component_tolerance = (
        NATIVE_COMPONENT_P95_TOLERANCE_M
        + canonical["depth_m_p95"]
        * CANONICAL_REPROJECTION_P95_TOLERANCE_PIXELS
        / min_focal
    )
    passed = bool(
        native_component_p95 <= NATIVE_COMPONENT_P95_TOLERANCE_M
        and native["reprojection_pixels"]["p95"]
        <= NATIVE_REPROJECTION_P95_TOLERANCE_PIXELS
        and canonical["component_error_m"]["z"]["p95"]
        <= NATIVE_COMPONENT_P95_TOLERANCE_M
        and canonical_component_p95 <= canonical_component_tolerance
        and canonical["reprojection_pixels"]["p95"]
        <= CANONICAL_REPROJECTION_P95_TOLERANCE_PIXELS
        and canonical["reprojection_pixels"]["max"]
        <= CANONICAL_REPROJECTION_MAX_TOLERANCE_PIXELS
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "native": native,
        "canonical": canonical,
        "thresholds": {
            "native_component_p95_m": NATIVE_COMPONENT_P95_TOLERANCE_M,
            "native_reprojection_p95_pixels": NATIVE_REPROJECTION_P95_TOLERANCE_PIXELS,
            "canonical_component_p95_m": float(canonical_component_tolerance),
            "canonical_reprojection_p95_pixels": CANONICAL_REPROJECTION_P95_TOLERANCE_PIXELS,
            "canonical_reprojection_max_pixels": CANONICAL_REPROJECTION_MAX_TOLERANCE_PIXELS,
        },
    }
