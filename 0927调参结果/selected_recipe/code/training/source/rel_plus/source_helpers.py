"""Minimal adaptations of public rgbd_util.py and hha_util.py REL helpers.

The audited source paths and exact function mapping are recorded in
``SOURCE_AUDIT.md``.
"""

import cv2
import numpy as np

from .camera import json_k_to_rel_helper_k
from .normal_diagnostics import build_normal_diagnostics


def filter_it_chop_off(values, radius, superpixels):
    """Source-equivalent square-window accumulator."""
    values = np.nan_to_num(values, copy=False)
    height, width, _ = values.shape
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.float32)

    minimum = cv2.erode(superpixels, kernel, iterations=1)
    maximum = cv2.dilate(superpixels, kernel, iterations=1)
    edge_indices = np.where((minimum != superpixels) | (maximum != superpixels))
    if len(edge_indices[0]) == 0:
        return cv2.filter2D(
            values, -1, kernel, borderType=cv2.BORDER_CONSTANT
        )

    expanded = np.pad(
        superpixels,
        ((radius, radius), (radius, radius)),
        mode="constant",
        constant_values=-1,
    )
    delta = np.zeros_like(values)
    for row, column in zip(*edge_indices):
        neighborhood_labels = expanded[
            row : row + 2 * radius + 1,
            column : column + 2 * radius + 1,
        ]
        different = neighborhood_labels != superpixels[row, column]
        row_start = max(0, row - radius)
        row_end = min(height, row + radius + 1)
        column_start = max(0, column - radius)
        column_end = min(width, column + radius + 1)
        neighborhood = values[row_start:row_end, column_start:column_end]
        mask = different[
            radius - (row - row_start) : radius + (row_end - row),
            radius - (column - column_start) : radius + (column_end - column),
        ]
        delta[row, column] = np.sum(neighborhood[mask], axis=0)

    filtered = cv2.filter2D(
        values, -1, kernel, borderType=cv2.BORDER_CONSTANT
    )
    return filtered - delta


def multiply_symmetric_inverse(inverse_terms, right_hand_side):
    result = np.zeros_like(right_hand_side)
    result[..., 0] = (
        inverse_terms[..., 0] * right_hand_side[..., 0]
        + inverse_terms[..., 1] * right_hand_side[..., 1]
        + inverse_terms[..., 2] * right_hand_side[..., 2]
    )
    result[..., 1] = (
        inverse_terms[..., 1] * right_hand_side[..., 0]
        + inverse_terms[..., 3] * right_hand_side[..., 1]
        + inverse_terms[..., 4] * right_hand_side[..., 2]
    )
    result[..., 2] = (
        inverse_terms[..., 2] * right_hand_side[..., 0]
        + inverse_terms[..., 4] * right_hand_side[..., 1]
        + inverse_terms[..., 5] * right_hand_side[..., 2]
    )
    return result


def invert_symmetric_terms(matrix_terms):
    a_value = matrix_terms[..., 0]
    b_value = matrix_terms[..., 1]
    c_value = matrix_terms[..., 2]
    d_value = matrix_terms[..., 3]
    e_value = matrix_terms[..., 4]
    f_value = matrix_terms[..., 5]
    inverse_terms = np.empty(matrix_terms.shape, dtype=np.float64)
    inverse_terms[..., 0] = d_value * f_value - e_value * e_value
    inverse_terms[..., 1] = -b_value * f_value + c_value * e_value
    inverse_terms[..., 2] = b_value * e_value - c_value * d_value
    inverse_terms[..., 3] = a_value * f_value - c_value * c_value
    inverse_terms[..., 4] = -a_value * e_value + b_value * c_value
    inverse_terms[..., 5] = a_value * d_value - b_value * b_value
    determinant = (
        a_value * inverse_terms[..., 0]
        + b_value * inverse_terms[..., 1]
        + c_value * inverse_terms[..., 2]
    )
    return inverse_terms, determinant


def get_point_cloud_from_z(depth_cm, helper_k):
    """Original helper projection with one-based pixel coordinates."""
    height, width = depth_cm.shape
    xx, yy = np.meshgrid(np.arange(width) + 1, np.arange(height) + 1)
    principal = helper_k[0:2, 2]
    focal = np.diag(helper_k[0:2, 0:2])
    x_value = (xx - principal[0]) * depth_cm / focal[0]
    y_value = (yy - principal[1]) * depth_cm / focal[1]
    return x_value, y_value, depth_cm


def compute_normals_square_support(
    depth_m, missing_mask, radius, helper_k, superpixels
):
    """Original perspective REL normal estimator at a frozen support radius."""
    depth_cm = depth_m * 100.0
    x_value, y_value, z_value = get_point_cloud_from_z(depth_cm, helper_k)
    points_full = np.stack((x_value, y_value, z_value), axis=2)

    x_value = np.where(missing_mask, np.nan, x_value)
    y_value = np.where(missing_mask, np.nan, y_value)
    z_value = np.where(missing_mask, np.nan, z_value)
    reciprocal_z = (1.0 / z_value)[..., np.newaxis]
    x_over_z = x_value / z_value
    y_over_z = y_value / z_value
    one = np.where(np.isnan(z_value), np.nan, 1.0)
    z_squared = z_value * z_value

    raw_matrix = np.concatenate(
        (
            (x_over_z * x_over_z)[..., np.newaxis],
            (x_over_z * y_over_z)[..., np.newaxis],
            x_over_z[..., np.newaxis],
            (y_over_z * y_over_z)[..., np.newaxis],
            y_over_z[..., np.newaxis],
            one[..., np.newaxis],
        ),
        axis=2,
    )
    raw_rhs = np.concatenate(
        (
            (x_value / z_squared)[..., np.newaxis],
            (y_value / z_squared)[..., np.newaxis],
            reciprocal_z,
        ),
        axis=2,
    )
    accumulated = filter_it_chop_off(
        np.concatenate((raw_matrix, raw_rhs), axis=2), radius, superpixels
    )
    matrix_terms = accumulated[:, :, : raw_matrix.shape[2]]
    right_hand_side = accumulated[:, :, raw_matrix.shape[2] :]
    inverse_terms, determinant = invert_symmetric_terms(matrix_terms)
    normals = multiply_symmetric_inverse(inverse_terms, right_hand_side)

    magnitude = np.sqrt(np.sum(normals * normals, axis=2))
    magnitude = np.where(magnitude == 0, 1.0, magnitude)
    offset = -determinant / magnitude
    normals = normals / magnitude[..., np.newaxis]

    sign_z = np.sign(normals[:, :, 2])
    sign_z[sign_z == 0] = 1
    normals = normals * sign_z[..., np.newaxis]
    offset = offset * sign_z
    facing_sign = np.sign(np.sum(normals * points_full, axis=2))
    facing_sign[np.isnan(facing_sign)] = 1
    facing_sign[facing_sign == 0] = 1
    normals = normals * facing_sign[..., np.newaxis]
    offset = offset * facing_sign
    return normals, offset


def estimate_source_perspective_normals(depth_m, valid_mask, k_json, radius=2):
    """Return unmodified source normals plus non-authoritative diagnostics."""
    if radius != 2:
        raise ValueError("REL+ v2 freezes normal_radius at 2")
    helper_k = json_k_to_rel_helper_k(k_json)
    with np.errstate(divide="ignore", invalid="ignore"):
        normals, _ = compute_normals_square_support(
            np.asarray(depth_m),
            ~np.asarray(valid_mask, dtype=bool),
            radius,
            helper_k,
            np.ones(np.asarray(depth_m).shape, dtype=np.float32),
        )
    normals = normals.astype(np.float64, copy=False)
    diagnostics = build_normal_diagnostics(normals, valid_mask, radius)
    return normals, diagnostics


def get_r_matrix(initial_axis, final_axis):
    """Numerically faithful form of the source getRMatrix axis alignment."""
    initial = np.asarray(initial_axis, dtype=np.float64)
    final = np.asarray(final_axis, dtype=np.float64)
    initial = initial / np.linalg.norm(initial)
    final = final / np.linalg.norm(final)
    dot_value = float(np.clip(np.dot(initial.T, final), -1.0, 1.0))
    degrees = float(np.degrees(np.arccos(dot_value)))
    if abs(degrees) <= 0.1:
        return np.eye(3)
    axis = np.cross(initial.T, final.T).T
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        raise ValueError("source gravity alignment is singular for opposite axes")
    axis = (axis / axis_norm).flatten()
    radians = degrees * np.pi / 180.0
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + np.sin(radians) * skew
        + (1.0 - np.cos(radians)) * (skew @ skew)
    )


def rotate_pc(point_field, rotation):
    if np.allclose(rotation, np.eye(3)):
        return point_field.copy()
    return np.einsum("ij,klj->kli", rotation, point_field)
