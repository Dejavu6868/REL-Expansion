"""Shared numeric summaries for independent global-XYZ checks."""

import numpy as np


def evenly_spaced_joint_pixels(mask, maximum_count=4096):
    rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        raise ValueError("no joint-valid geometry pixels")
    count = min(int(maximum_count), int(rows.size))
    selected = np.linspace(0, rows.size - 1, count, dtype=np.int64)
    return rows[selected], columns[selected]


def scalar_statistics(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("cannot summarize an empty array")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def component_statistics(component_error):
    error = np.asarray(component_error, dtype=np.float64)
    if error.ndim != 2 or error.shape[1] != 3:
        raise ValueError("component_error must have shape Nx3")
    return {
        axis: scalar_statistics(error[:, index])
        for index, axis in enumerate(("x", "y", "z"))
    }


def project_camera_points(points_camera, k_json):
    points = np.asarray(points_camera, dtype=np.float64)
    k_matrix = np.asarray(k_json, dtype=np.float64)
    u = k_matrix[0, 0] * points[:, 0] / points[:, 2] + k_matrix[0, 2]
    v = k_matrix[1, 1] * points[:, 1] / points[:, 2] + k_matrix[1, 2]
    return u, v
