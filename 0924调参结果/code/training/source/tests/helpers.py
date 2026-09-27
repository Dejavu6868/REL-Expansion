import numpy as np


def rotation_x(degrees):
    angle = np.radians(degrees)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]]
    )


def rotation_y(degrees):
    angle = np.radians(degrees)
    return np.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )


def rotation_z(degrees):
    angle = np.radians(degrees)
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )


def look_rotation(forward_world):
    forward = np.asarray(forward_world, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    reference_up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ reference_up)) > 0.9:
        reference_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, reference_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=0)


def independent_plane_depth(shape, k_json, rotation_world_to_camera, plane_normal_world, plane_offset):
    rows, columns = np.indices(shape, dtype=np.float64)
    rays_camera = np.stack(
        [
            (columns + 0.5 - k_json[0, 2]) / k_json[0, 0],
            (rows + 0.5 - k_json[1, 2]) / k_json[1, 1],
            np.ones(shape, dtype=np.float64),
        ],
        axis=-1,
    )
    rays_world = rays_camera @ rotation_world_to_camera
    normal = np.asarray(plane_normal_world, dtype=np.float64)
    denominator = rays_world @ normal
    z_depth = plane_offset / denominator
    valid = np.isfinite(z_depth) & (z_depth > 0.0) & (z_depth < 127.0)
    raw = np.zeros(shape, dtype=np.uint16)
    raw[valid] = np.rint(z_depth[valid] * 512.0).astype(np.uint16)
    points_world = rays_world * z_depth[..., None]
    points_camera = rays_camera * z_depth[..., None]
    return raw, valid, points_world, points_camera
