"""End-to-end frozen REL+ v2 generation with a byte-identical v2.1 entry."""

from .camera import backproject_z_depth, gravity_in_camera
from .constants import (
    REL_PLUS_V2_ALPHA,
    REL_PLUS_V2_LAMBDA,
    REL_PLUS_V2_NORMAL_RADIUS,
)
from .depth import decode_stanford_s2d_depth
from .encoding import encode_rel_channels, perspective_tangent_field
from .geometry import align_points_and_normals_to_gravity
from .source_helpers import (
    estimate_source_perspective_normals,
)


def generate_rel_plus_v2(
    raw_depth,
    camera,
    *,
    return_debug=False,
):
    """Generate frozen HxWx3 uint8 [EGVIA, LOA, ReD]."""
    camera.assert_matches_image_shape(raw_depth.shape)
    depth_m, depth_valid = decode_stanford_s2d_depth(raw_depth)
    points_camera_m = backproject_z_depth(depth_m, depth_valid, camera.K_json)
    normals_camera, normal_diagnostics = estimate_source_perspective_normals(
        depth_m, depth_valid, camera.K_json, radius=REL_PLUS_V2_NORMAL_RADIUS
    )
    encoding_valid_mask = depth_valid
    gravity_camera = gravity_in_camera(camera.R_world_to_camera)
    points_aligned_m, normals_aligned, alignment = (
        align_points_and_normals_to_gravity(
            points_camera_m,
            normals_camera,
            gravity_camera,
            sample_id=camera.sample_id,
        )
    )
    points_for_encoding_cm = points_aligned_m * 100.0
    tangent, _ = perspective_tangent_field(points_for_encoding_cm)
    rel_plus, encoding_debug = encode_rel_channels(
        points_for_encoding_cm,
        normals_aligned,
        encoding_valid_mask,
        tangent=tangent,
        alpha=REL_PLUS_V2_ALPHA,
        lam=REL_PLUS_V2_LAMBDA,
    )
    if not return_debug:
        return rel_plus
    debug = {
        "depth_m": depth_m,
        "depth_valid": depth_valid,
        "encoding_valid_mask": encoding_valid_mask,
        "points_camera_m": points_camera_m,
        "normals_camera": normals_camera,
        "normal_finite_mask": normal_diagnostics.finite_mask,
        "normal_nonzero_mask": normal_diagnostics.nonzero_mask,
        "normal_support_count": normal_diagnostics.support_count,
        "normal_quality_mask": normal_diagnostics.quality_mask,
        "gravity_camera": gravity_camera,
        "gravity_alignment_rotation": alignment,
        "points_aligned_m": points_aligned_m,
        "points_for_encoding_cm": points_for_encoding_cm,
        "normals_aligned": normals_aligned,
    }
    debug.update(normal_diagnostics.ratios(depth_valid))
    debug.update(encoding_debug)
    debug["rel_plus"] = rel_plus
    return rel_plus, debug


def generate_rel_plus_v2_1(raw_depth, camera, *, return_debug=False):
    """Expose the v2.1 interface without changing any representation byte."""
    return generate_rel_plus_v2(raw_depth, camera, return_debug=return_debug)
