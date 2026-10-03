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


def test_erp_hha_matches_pinhole_hha_on_crops(frozen):
    result = cp.check_panorama_hha(
        cp.decode_erp_depth(render_erp_raw_depth(512, 1024)),
        frozen,
        yaws=(0.0, 225.0),
        pitches=(-20.0, 20.0),
        size=160,
        fov=90.0,
    )
    assert result["status"] == "PASS", result["failures"]
    assert result["erp_gravity_error_deg"] < 1.0
    for crop in result["crops"]:
        assert crop["crop_gravity_error_deg"] < 1.0
        # Angle and height bytes agree; disparity cannot (z-depth vs range).
        assert crop["encoded_bytes"]["angle"]["median"] <= 1
        assert crop["encoded_bytes"]["height"]["median"] <= 2


def test_gravity_accepts_real_eigenvectors_returned_as_complex(monkeypatch):
    import hha

    real_eig = np.linalg.eig

    def complex_eig(matrix):
        values, vectors = real_eig(matrix)
        return values.astype(complex), vectors.astype(complex)

    monkeypatch.setattr(np.linalg, "eig", complex_eig)
    normals = np.broadcast_to([0.0, 1.0, 0.0], (8, 8, 3))
    gravity = hha.getYDir(
        normals, hha.ANGLE_THRESHOLDS, hha.GRAVITY_ITERATIONS, np.array([0.0, 1.0, 0.0])
    )
    assert np.isrealobj(gravity)
    np.testing.assert_allclose(gravity, [0.0, 1.0, 0.0])


def test_nonfinite_geometry_cannot_pass(frozen, report, monkeypatch):
    from copy import deepcopy

    crop = deepcopy(report["crops"][0])
    crop["raw"]["height_cm"] = cp.summarize([0.0, np.nan])
    monkeypatch.setattr(cp, "compare_crop", lambda *args: crop)
    result = cp.check_panorama(
        render_erp_raw_depth(64, 128), frozen, yaws=(0.0,), pitches=(0.0,),
        size=32, fov=90.0, thresholds=cp.DEFAULT_THRESHOLDS,
    )
    assert result["status"] == "FAIL"
