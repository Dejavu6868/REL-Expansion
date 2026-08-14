"""Compatibility dependency omitted by the author REL repository.

The author source imports ``utils.hha_util`` but does not track that file. This
module preserves the helper behavior from the clean remote CMX-REL reproduction
at commit 9d614e2. It is reference plumbing, not a replacement REL definition.
"""

import cv2
import numpy as np


def filterItChopOff(f, r, sp):
    f = np.nan_to_num(f, copy=False)
    height, width, depth = f.shape
    kernel = np.ones((2 * r + 1, 2 * r + 1), dtype=np.float32)
    min_sp = cv2.erode(sp, kernel, iterations=1)
    max_sp = cv2.dilate(sp, kernel, iterations=1)
    edge_indices = np.where(np.logical_or(min_sp != sp, max_sp != sp))
    if len(edge_indices[0]) == 0:
        return cv2.filter2D(f, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    sp_expanded = np.pad(sp, ((r, r), (r, r)), mode="constant", constant_values=-1)
    delta = np.zeros_like(f)
    for x, y in zip(*edge_indices):
        neighborhood_sp = sp_expanded[x : x + 2 * r + 1, y : y + 2 * r + 1]
        mask = (neighborhood_sp != sp[x, y])[r:-r, r:-r]
        x_start = max(0, x - r)
        x_end = min(height, x + r + 1)
        y_start = max(0, y - r)
        y_end = min(width, y + r + 1)
        neighborhood_f = f[x_start:x_end, y_start:y_end, :]
        valid_mask = mask[: neighborhood_f.shape[0], : neighborhood_f.shape[1]]
        valid_mask_3d = np.repeat(valid_mask[:, :, np.newaxis], depth, axis=2)
        delta[x, y, :] = np.sum(
            neighborhood_f[valid_mask_3d].reshape(-1, depth), axis=0
        )
    return cv2.filter2D(f, -1, kernel, borderType=cv2.BORDER_CONSTANT) - delta


def mutiplyIt(ata_inverse, atb):
    result = np.zeros_like(atb)
    result[..., 0] = (
        ata_inverse[..., 0] * atb[..., 0]
        + ata_inverse[..., 1] * atb[..., 1]
        + ata_inverse[..., 2] * atb[..., 2]
    )
    result[..., 1] = (
        ata_inverse[..., 1] * atb[..., 0]
        + ata_inverse[..., 3] * atb[..., 1]
        + ata_inverse[..., 4] * atb[..., 2]
    )
    result[..., 2] = (
        ata_inverse[..., 2] * atb[..., 0]
        + ata_inverse[..., 4] * atb[..., 1]
        + ata_inverse[..., 5] * atb[..., 2]
    )
    return result


def invertIt(ata):
    a, b, c, d, e, f_value = [ata[..., index] for index in range(6)]
    height, width, _ = ata.shape
    inverse = np.empty((height, width, 6))
    inverse[..., 0] = d * f_value - e * e
    inverse[..., 1] = -b * f_value + c * e
    inverse[..., 2] = b * e - c * d
    inverse[..., 3] = a * f_value - c * c
    inverse[..., 4] = -a * e + b * c
    inverse[..., 5] = a * d - b * b
    determinant = a * inverse[..., 0] + b * inverse[..., 1] + c * inverse[..., 2]
    return inverse, determinant


def getRMatrix(initial_axis, final_axis):
    if np.isscalar(final_axis):
        axis = initial_axis / np.linalg.norm(initial_axis)
        angle = final_axis
    else:
        initial_axis = initial_axis / np.linalg.norm(initial_axis)
        final_axis = final_axis / np.linalg.norm(final_axis)
        axis = np.cross(initial_axis.T, final_axis.T).T
        axis = axis / np.linalg.norm(axis)
        angle = np.degrees(np.arccos(np.dot(initial_axis.T, final_axis)))
    if abs(angle) <= 0.1:
        return np.eye(3)
    angle = angle * (np.pi / 180)
    axis = axis.flatten()
    skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * np.dot(skew, skew)


def rotatePC(point_cloud, rotation):
    if np.allclose(rotation, np.eye(3)):
        return point_cloud
    return np.einsum("ij,klj->kli", rotation, point_cloud)


def getGDir(normals, angle_thresholds, iterations, initial_gravity):
    gravity = initial_gravity.copy()
    for index in range(len(angle_thresholds)):
        gravity = _get_gravity_direction(
            normals,
            gravity,
            np.radians(angle_thresholds[index]),
            iterations[index],
        )
    return gravity


def _get_gravity_direction(normals, initial_gravity, threshold, num_iterations):
    normal_matrix = np.moveaxis(normals, -1, 0).reshape(3, -1)
    normal_matrix = normal_matrix[:, ~np.isnan(normal_matrix[0])]
    if normal_matrix.shape[1] < 10:
        return initial_gravity
    gravity = initial_gravity.copy()
    cos_threshold = np.cos(threshold)
    sin_threshold = np.sin(threshold)
    for _ in range(num_iterations):
        similarity = np.dot(gravity, normal_matrix)
        floor = np.abs(similarity) > cos_threshold
        wall = np.abs(similarity) < sin_threshold
        if np.sum(floor) < 5 or np.sum(wall) < 5:
            break
        floor_normals = normal_matrix[:, floor]
        wall_normals = normal_matrix[:, wall]
        matrix = np.dot(wall_normals, wall_normals.T) - np.dot(
            floor_normals, floor_normals.T
        )
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            new_gravity = eigenvectors[:, np.argmin(eigenvalues)]
            if np.dot(gravity, new_gravity) < 0:
                new_gravity = -new_gravity
            if np.linalg.norm(gravity - new_gravity) < 1e-3:
                break
            gravity = new_gravity
        except (ValueError, np.linalg.LinAlgError):
            break
    return gravity

