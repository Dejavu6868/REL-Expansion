"""End-to-end REL+ v1 generation from Stanford2D3D S2D depth and pose."""

from .camera import backproject_z_depth, gravity_in_camera
from .depth import decode_stanford_s2d_depth
from .encoding import encode_rel_channels, perspective_tangent_field
from .source_helpers import (
    align_points_and_normals_to_gravity,
    estimate_source_perspective_normals,
)


def generate_rel_plus(
    raw_depth,
    camera,
    *,
    normal_radius=2,
    alpha=45.0,
    lam=0.5,
    return_debug=False,
):
    """Generate frozen HxWx3 uint8 [EGVIA, LOA, ReD]."""
    if normal_radius != 2:
        raise ValueError("REL+ v1 freezes normal_radius at 2")
    depth_m, depth_valid = decode_stanford_s2d_depth(raw_depth)
    points_camera = backproject_z_depth(depth_m, depth_valid, camera.K_json)
    normals_camera, normal_valid = estimate_source_perspective_normals(
        depth_m, depth_valid, camera.K_json, radius=normal_radius
    )
    valid_mask = depth_valid & normal_valid
    gravity_camera = gravity_in_camera(camera.R_world_to_camera)
    points_aligned, normals_aligned, alignment = (
        align_points_and_normals_to_gravity(
            points_camera, normals_camera, gravity_camera
        )
    )
    tangent, _ = perspective_tangent_field(points_aligned)
    rel_plus, encoding_debug = encode_rel_channels(
        points_aligned,
        normals_aligned,
        valid_mask,
        tangent=tangent,
        alpha=alpha,
        lam=lam,
    )
    if not return_debug:
        return rel_plus
    debug = {
        "depth_m": depth_m,
        "valid_mask": valid_mask,
        "points_camera": points_camera,
        "normals_camera": normals_camera,
        "gravity_camera": gravity_camera,
        "gravity_alignment_rotation": alignment,
        "points_aligned": points_aligned,
        "normals_aligned": normals_aligned,
    }
    debug.update(encoding_debug)
    debug["rel_plus"] = rel_plus
    return rel_plus, debug
