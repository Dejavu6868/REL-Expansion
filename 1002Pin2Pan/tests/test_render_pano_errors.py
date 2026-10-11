"""Checks for saved panorama predictions and the error renderer's counts."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eval_pano_transfer as ev  # noqa: E402
import render_pano_errors as rpe  # noqa: E402


def test_saved_prediction_round_trips(tmp_path):
    prediction = np.random.default_rng(0).integers(0, 13, size=(8, 16)).astype(np.uint8)
    path = ev.save_prediction(tmp_path, "area_5a/camera_x_pano", prediction)
    assert path == tmp_path / "area_5a" / "camera_x_pano.png"
    assert np.array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), prediction)


def test_error_counts_split_floor_as_table_and_wall_errors():
    f, t, w = rpe.FLOOR, rpe.TABLE, rpe.WALL
    door = rpe.CLASS_NAMES.index("door")
    label = np.array([[f, f, w, w, door, 255]], dtype=np.uint8)
    prediction = np.array([[t, f, door, w, w, w]], dtype=np.uint8)
    counts = rpe.error_counts(prediction, label)
    assert counts["valid_pixels"] == 5
    assert counts["wrong_pixels"] == 3
    assert counts["floor_as_table"] == 1
    assert counts["wall_errors"] == 2  # wall read as door, door read as wall; ignore pixel skipped
    assert counts["wall_as_door"] == 1 and counts["door_as_wall"] == 1


def test_error_map_marks_each_error_kind():
    f, t, w = rpe.FLOOR, rpe.TABLE, rpe.WALL
    label = np.array([[f, w, 0, 0]], dtype=np.uint8)
    prediction = np.array([[t, 0, 1, 0]], dtype=np.uint8)
    out = rpe.error_map(np.full((1, 4, 3), 200, dtype=np.uint8), prediction, label)
    assert tuple(out[0, 0]) == rpe.ERROR_COLOURS["floor_as_table"]
    assert tuple(out[0, 1]) == rpe.ERROR_COLOURS["wall"]
    assert tuple(out[0, 2]) == rpe.ERROR_COLOURS["wrong"]
    assert tuple(out[0, 3]) == (70, 70, 70)  # correct pixels are dimmed RGB


def test_egvia_draws_missing_depth_black():
    rel = np.zeros((2, 2, 3), dtype=np.uint8)
    rel[0, 0, 0] = 255
    out = rpe.egvia_image(rel)
    assert tuple(out[0, 0]) == (0, 0, 0)
    assert tuple(out[1, 1]) != (0, 0, 0)
