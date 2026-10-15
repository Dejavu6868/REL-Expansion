"""check_input_defects.py: each Phase 0 check on the synthetic room."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_input_defects as cid  # noqa: E402
import cross_projection as cp  # noqa: E402
from test_check_egvia_rules import CLASSES, erp_labels, write_frame  # noqa: E402
from test_cross_projection_synthetic import render_erp_raw_depth  # noqa: E402


@pytest.fixture(scope="module")
def frozen():
    return cp.import_frozen_source(cp.DEFAULT_SOURCE_ROOT)


@pytest.fixture(scope="module")
def room():
    raw = render_erp_raw_depth(256, 512)
    return raw, cp.decode_erp_depth(raw)


def crop_frame(frozen, depth, holes):
    """A 480 pinhole frame of the room through the frozen generator, optionally with a hole."""
    from rel_plus.camera import gravity_in_camera

    k_json, camera_to_world, _, _, raw = cp.render_crop(depth, 35.0, -20.0, 480, 62.0)
    if holes:
        raw[200:240, 100:180] = 0
        raw[300:310, 300:400] = 65535
    camera = frozen["CameraGeometry"].from_json_k(k_json, (480, 480), camera_to_world.T, sample_id="crop")
    rel_plus, debug = frozen["generate_rel_plus_v2_1"](raw, camera, return_debug=True)
    return raw, k_json, gravity_in_camera(camera.R_world_to_camera), rel_plus, debug


@pytest.mark.parametrize("holes", [False, True])
def test_red_shift_only_when_invalid_pixels_pull_the_minimum(frozen, room, holes):
    _, _, _, _, debug = crop_frame(frozen, room[1], holes)
    valid = np.asarray(debug["encoding_valid_mask"], dtype=bool)
    stats, histogram = cid.red_statistics(debug["red_raw_cm"], debug["red_encoded"], valid)
    assert histogram.sum() == valid.sum()
    assert stats["red_min_valid_cm"] > 50
    if holes:
        assert stats["red_min_all_cm"] == 0 and stats["shift_median"] > 5
    else:
        assert stats["red_min_all_cm"] == stats["red_min_valid_cm"] and stats["shift_p95"] == 0


@pytest.mark.parametrize("holes", [False, True])
def test_mirrored_frame_is_the_flipped_frame_with_loa_remapped(frozen, room, holes):
    raw, k_json, gravity, rel_plus, _ = crop_frame(frozen, room[1], holes)
    stats = cid.flip_statistics(raw, k_json, gravity, rel_plus)
    share = {key: value / stats["compared_pixels"] for key, value in stats.items() if "_exact" in key or "_over_1" in key}
    assert stats["valid_mask_mismatch"] == 0
    for variant in ("v2_1", "v3_1"):
        for channel in ("EGVIA", "ReD"):
            assert share["{}_{}_over_1".format(variant, channel)] < 0.002
        # Truncation: floor(180 - a) = 179 - floor(a) unless a is a whole number of degrees.
        assert share["{}_LOA_179_exact".format(variant)] > 0.95
        assert share["{}_LOA_179_over_1".format(variant)] < 0.002


def test_flip_check_refuses_a_wrong_encoder(frozen, room):
    raw, k_json, gravity, rel_plus, _ = crop_frame(frozen, room[1], False)
    with pytest.raises(AssertionError):
        cid.flip_statistics(raw, k_json, -gravity, rel_plus)


def test_rotation_between_recovers_the_gravity_rotation():
    rng = np.random.default_rng(0)
    points = rng.uniform(-300, 300, (40, 50, 3))
    angle = np.radians(1.3)
    m = np.array([[1, 0, 0], [0, np.cos(angle), np.sin(angle)], [0, -np.sin(angle), np.cos(angle)]])
    assert np.allclose(cid.rotation_between(points, points @ m, np.ones(points.shape[:2], bool)), m)


def test_level_panorama(frozen, tmp_path):
    raw = render_erp_raw_depth(512, 1024)  # at 256x512 the crop normals miss the 2 degree bar
    raw[:4, :8] = 0  # read as a valid 2 mm point by the ERP decoder
    raw[-4:, :8] = 65535
    path = tmp_path / "p_depth.png"
    cv2.imwrite(str(path), raw)
    cid._JOB.update(
        size=raw.shape, crop_fov=60.0, crop_size=160, frozen=frozen,
        getPointCloud_ERP=frozen["getPointCloud_ERP"], processDepthImage_ERP=frozen["processDepthImage_ERP"],
    )
    result = cid.panorama(({"sample_id": "area_1/p", "depth": path}, True))
    row = result["row"]
    assert row["native_raw0_share"] == row["raw0_share"] == pytest.approx(32 / raw.size)
    assert row["native_raw65535_share"] == row["missing_share"] == pytest.approx(32 / raw.size)
    assert row["red_min_valid_cm"] < 1  # the raw-0 pixels sit at the camera
    assert row["tilt_deg"] < 0.5
    assert row["egvia_v3_1_diff_p95"] <= 1
    crops = result["crops"]
    assert len(crops) == len(cid.CROP_YAWS) * len(cid.CROP_PITCHES)
    assert result["crop_geometry_status"] == "PASS"
    # Same rays, same geometry: the distance agrees, the per-image ReD byte does not.
    assert np.median([c["distance_cm_diff_p50"] for c in crops]) < 1
    assert np.median([c["red_byte_diff_p50"] for c in crops]) > 5


def test_blend_counts_only_pixels_next_to_holes(frozen):
    pytest.importorskip("torch")
    from dataloader.profiles import sample_comparison_transform
    import random

    cid._JOB["classes"] = 13
    x = np.full((480, 480, 3), 100, np.uint8)
    valid = np.ones((480, 480), bool)
    label = np.full((480, 480), 11, np.uint8)
    transform = sample_comparison_transform((480, 480), [1.25], (480, 480), random.Random(0))
    clean = cid.blend_statistics(x, valid, label, transform)
    assert clean["affected_pixels"] == 0 and clean["labelled_pixels"] == 480 * 480
    valid[100:140, 100:140] = False
    x[~valid] = 255
    holed = cid.blend_statistics(x, valid, label, transform)
    assert 0 < holed["affected_labelled_pixels"] < 0.02 * holed["labelled_pixels"]
    assert holed["affected_by_class"][11] == holed["affected_labelled_pixels"]


def test_synthetic_run(tmp_path, room):
    raw, depth = room
    label = erp_labels(depth)
    (tmp_path / "Label").mkdir()
    manifest = "sample_id,split,depth_path,camera_metadata_path\n"
    for sample_id, yaw, pitch in (("a", 35.0, -20.0), ("b", 200.0, 10.0)):
        manifest += write_frame(tmp_path, sample_id, depth, label, yaw, pitch, 62.0)
    (tmp_path / "manifest.csv").write_text(manifest)
    (tmp_path / "config.json").write_text(json.dumps({
        "class_names": CLASSES, "gt_root_folder": str(tmp_path / "Label"), "gt_format": ".png",
        "train_scale_array": [0.5, 1.0, 1.75], "image_height": 480, "image_width": 480,
    }))
    for area in ("area_1", "area_5a"):
        folder = tmp_path / "s2d" / area / "pano"
        for kind, image in (("rgb", np.zeros(raw.shape + (3,), np.uint8)), ("depth", raw), ("semantic", np.zeros(raw.shape + (3,), np.uint8))):
            (folder / kind).mkdir(parents=True)
            cv2.imwrite(str(folder / kind / "p_{}.png".format(kind)), image)
    subprocess.run(
        [sys.executable, str(TOOLS / "check_input_defects.py"), "--manifest", str(tmp_path / "manifest.csv"),
         "--config", str(tmp_path / "config.json"), "--stanford-root", str(tmp_path / "s2d"),
         "--output", str(tmp_path / "out"), "--skip-blend", "--areas", "area_1", "area_5a",
         "--crop-panoramas", "1", "--height", "256", "--width", "512", "--workers", "2"],
        check=True, capture_output=True,
    )
    report = json.loads((tmp_path / "out" / "input_defects.json").read_text())
    assert report["frames"] == 2 and report["flip_frames"] == 2
    assert report["pinhole"]["flip"]["valid_mask_mismatch_pixels"] == 0
    assert set(report["panorama"]["by_area"]) == {"area_1", "area_5a"}
    assert report["panorama"]["crops"]["crops"] == 12  # training areas only
    assert "phase1_metric_red" in report["plan_gates"] and "phase2_resize_fix" not in report["plan_gates"]
    for name in ("pinhole_frames.csv", "panoramas.csv", "crops.csv"):
        assert (tmp_path / "out" / name).is_file()
