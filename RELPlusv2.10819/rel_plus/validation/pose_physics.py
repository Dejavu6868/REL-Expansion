"""Physical pose checks kept separate from the generation path."""

from dataclasses import dataclass

import numpy as np

from .geometry_oracle import project_camera_points


@dataclass(frozen=True)
class PosePhysicsResult:
    status: str
    evidence_level: str
    metrics: dict
    warnings: tuple

    def as_dict(self):
        return {
            "status": self.status,
            "evidence_level": self.evidence_level,
            "metrics": self.metrics,
            "warnings": list(self.warnings),
        }


def _angle_degrees_from_target(normal_z, target_z):
    return float(np.degrees(np.arccos(np.clip(normal_z * target_z, -1.0, 1.0))))


def _semantic_normal_metrics(
    labels, normals, semantic_valid_mask, semantic_ids, minimum_pixels
):
    metrics = {}
    checks = []
    rules = {
        "floor": (-1.0, 0.8),
        "ceiling": (1.0, 0.8),
        "wall": (0.0, 0.3),
    }
    for name, (target, tolerance) in rules.items():
        if name not in semantic_ids:
            continue
        mask = (
            (labels == semantic_ids[name])
            & semantic_valid_mask
            & np.all(np.isfinite(normals), axis=2)
        )
        count = int(np.count_nonzero(mask))
        metrics[name + "_count"] = count
        if count < minimum_pixels:
            continue
        median = float(np.median(normals[:, :, 2][mask]))
        metrics[name + "_median_normal_z"] = median
        if name == "wall":
            metrics["wall_median_abs_normal_z"] = float(
                np.median(np.abs(normals[:, :, 2][mask]))
            )
            metrics["wall_angle_from_horizontal_deg"] = float(
                np.degrees(np.arcsin(np.clip(abs(median), 0.0, 1.0)))
            )
            checks.append(abs(median) <= tolerance)
        else:
            metrics[name + "_angle_deg"] = _angle_degrees_from_target(median, target)
            checks.append(abs(median - target) <= tolerance)
    return metrics, checks


def validate_pose_physics(
    camera,
    *,
    world_points=None,
    camera_points=None,
    pixel_coordinates=None,
    labels=None,
    normals_aligned=None,
    points_aligned_m=None,
    semantic_valid_mask=None,
    semantic_ids=None,
    component_p95_tolerance_m=1.0 / 512.0,
    reprojection_p95_tolerance_pixels=1.0,
    minimum_semantic_pixels=50
):
    """Classify pose evidence without treating absent evidence as a pass."""
    metrics = {}
    warnings = []
    strong_checks = []
    if world_points is not None and camera_points is not None:
        world = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
        expected = np.asarray(camera_points, dtype=np.float64).reshape(-1, 3)
        if world.shape != expected.shape or world.shape[0] == 0:
            raise ValueError("world_points and camera_points must be matching nonempty Nx3")
        transformed = world @ camera.R_world_to_camera.T + camera.t_world_to_camera
        component_error = np.abs(transformed - expected)
        p95 = np.quantile(component_error, 0.95, axis=0)
        metrics["component_p95_m"] = [float(value) for value in p95]
        strong_checks.append(bool(np.all(p95 <= component_p95_tolerance_m)))
        if pixel_coordinates is not None:
            pixels = np.asarray(pixel_coordinates, dtype=np.float64).reshape(-1, 2)
            if pixels.shape[0] != transformed.shape[0]:
                raise ValueError("pixel_coordinates must match point count")
            u, v = project_camera_points(transformed, camera.K_json)
            reprojection = np.hypot(u - pixels[:, 0], v - pixels[:, 1])
            p95_reprojection = float(np.quantile(reprojection, 0.95))
            metrics["reprojection_p95_pixels"] = p95_reprojection
            strong_checks.append(
                p95_reprojection <= reprojection_p95_tolerance_pixels
            )

    semantic_checks = []
    semantic_supplied = (
        labels is not None and normals_aligned is not None and semantic_ids is not None
    )
    if semantic_supplied:
        label_array = np.asarray(labels)
        normal_array = np.asarray(normals_aligned)
        if normal_array.shape != label_array.shape + (3,):
            raise ValueError("labels and normals_aligned shapes do not match")
        if semantic_valid_mask is None:
            warnings.append("semantic_valid_mask is required for semantic pose evidence")
        else:
            semantic_valid = np.asarray(semantic_valid_mask, dtype=bool)
            if semantic_valid.shape != label_array.shape:
                raise ValueError("semantic_valid_mask must match labels")
            semantic_metrics, semantic_checks = _semantic_normal_metrics(
                label_array,
                normal_array,
                semantic_valid,
                semantic_ids,
                minimum_semantic_pixels,
            )
            metrics.update(semantic_metrics)
            if not semantic_checks:
                warnings.append("semantic classes lacked enough quality normal pixels")

    if strong_checks:
        return PosePhysicsResult(
            "PASS_STRONG" if all(strong_checks) else "FAIL",
            "strong",
            metrics,
            tuple(warnings),
        )

    if semantic_checks:
        return PosePhysicsResult(
            "PASS_WEAK" if all(semantic_checks) else "FAIL",
            "weak",
            metrics,
            tuple(warnings),
        )

    if normals_aligned is not None and points_aligned_m is not None:
        normal_array = np.asarray(normals_aligned)
        point_array = np.asarray(points_aligned_m)
        if point_array.shape != normal_array.shape:
            raise ValueError("points_aligned_m and normals_aligned shapes do not match")
        finite = np.all(np.isfinite(normal_array), axis=2) & np.all(
            np.isfinite(point_array), axis=2
        )
        if np.any(finite):
            heights = point_array[:, :, 2][finite]
            normal_z = normal_array[:, :, 2][finite]
            low, high = np.quantile(heights, [0.10, 0.90])
            metrics["weak_low_height_normal_z_median"] = float(
                np.median(normal_z[heights <= low])
            )
            metrics["weak_high_height_normal_z_median"] = float(
                np.median(normal_z[heights >= high])
            )
        warnings.append("height-decile normal evidence is warning-only")
    elif normals_aligned is not None:
        finite = np.all(np.isfinite(normals_aligned), axis=2)
        if np.any(finite):
            metrics["weak_normal_z_median"] = float(
                np.median(np.asarray(normals_aligned)[:, :, 2][finite])
            )
        warnings.append("normal evidence without height is warning-only")
    if semantic_supplied or normals_aligned is not None or points_aligned_m is not None:
        return PosePhysicsResult("REVIEW_REQUIRED", "weak", metrics, tuple(warnings))
    return PosePhysicsResult(
        "NOT_APPLICABLE", "none", metrics, ("no physical pose evidence supplied",)
    )
