"""Synthetic box-room check of the perspective and ERP REL code paths."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import cross_projection as cp  # noqa: E402


ROOM_MIN = np.array([-2.5, -2.0, -1.4])  # camera 1.4 m above the floor
ROOM_MAX = np.array([3.5, 4.0, 1.6])
TABLE_MIN = np.array([0.6, 1.2, -1.4])
TABLE_MAX = np.array([1.8, 2.2, -0.65])


def _ray_box_interior(directions):
    with np.errstate(divide="ignore", invalid="ignore"):
        bounds = np.where(directions > 0, ROOM_MAX, ROOM_MIN)
        hits = np.where(directions != 0, bounds / directions, np.inf)
    return hits.min(axis=-1)


def _ray_box_exterior(directions):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = TABLE_MIN / directions
        t2 = TABLE_MAX / directions
    near = np.nanmax(np.minimum(t1, t2), axis=-1)
    far = np.nanmin(np.maximum(t1, t2), axis=-1)
    hit = (far >= near) & (near > 0)
    return np.where(hit, near, np.inf)


def render_erp_raw_depth(height, width):
    directions = cp.erp_directions(height, width)
    range_m = np.minimum(_ray_box_interior(directions), _ray_box_exterior(directions))
    return np.clip(np.rint(range_m * 512.0) - 1, 0, 65534).astype(np.uint16)


@pytest.fixture(scope="module")
def frozen():
    return cp.import_frozen_source(cp.DEFAULT_SOURCE_ROOT)


@pytest.fixture(scope="module")
def report(frozen):
    raw = render_erp_raw_depth(512, 1024)
    return cp.check_panorama(
        raw,
        frozen,
        yaws=(0.0, 90.0, 225.0),
        pitches=(-20.0, 0.0, 20.0),
        size=160,
        fov=90.0,
        thresholds=cp.DEFAULT_THRESHOLDS,
    )


def test_every_crop_compares_most_pixels(report):
    for crop in report["crops"]:
        assert crop["compared_pixels"] > 0.5 * crop["crop_valid_pixels"]


def test_raw_geometry_agrees_across_projections(report):
    assert report["status"] == "PASS", report["failures"]


def test_erp_gravity_estimate_is_vertical_for_aligned_room(report):
    alpha, beta = report["erp_estimated_gravity_rotation_deg"]
    assert abs(alpha) < 1.0 and abs(beta) < 1.0


def test_check_detects_a_wrong_gravity(frozen):
    """Negative control: a crop camera that ignores its 20-degree pitch."""
    real = frozen["CameraGeometry"]

    class LevelCamera:
        @staticmethod
        def from_json_k(k, shape, rotation, **keywords):
            return real.from_json_k(k, shape, cp.crop_camera_to_world(0.0, 0.0).T, **keywords)

    broken = dict(frozen, CameraGeometry=LevelCamera)
    result = cp.check_panorama(
        render_erp_raw_depth(512, 1024),
        broken,
        yaws=(0.0,),
        pitches=(20.0,),
        size=160,
        fov=90.0,
        thresholds=cp.DEFAULT_THRESHOLDS,
    )
    assert result["status"] == "FAIL"
