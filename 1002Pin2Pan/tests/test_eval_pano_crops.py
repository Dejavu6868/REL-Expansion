"""Unit checks for the crop-tiled panorama evaluation."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import cross_projection as cp  # noqa: E402
import eval_pano_crops as ec  # noqa: E402


def regions(directions):
    """Twelve labels: eight 45-degree azimuth sectors, two elevation bands per pole."""
    phi = np.arctan2(directions[..., 0], directions[..., 1])
    elevation = np.degrees(np.arcsin(np.clip(directions[..., 2], -1.0, 1.0)))
    label = ((phi + np.pi) // (np.pi / 4)).astype(np.int64) % 8
    label[elevation > 30] = 8
    label[elevation < -30] = 9
    label[elevation > 70] = 10
    label[elevation < -70] = 11
    return label


@pytest.mark.parametrize("fov", [60.0, ec.CROP_FOV_DEG, 75.0])
@pytest.mark.parametrize("eval_size", [(64, 128), (1024, 2048)])
def test_default_layout_covers_every_erp_pixel(eval_size, fov):
    grids, inside = ec.stitch_grids(eval_size, fov=fov)
    assert grids.shape == (len(ec.CROP_LAYOUT),) + eval_size + (2,)
    assert inside.sum(axis=0).min() >= 1
    assert np.abs(grids).max() <= 1 + 1e-6


def test_stitched_crops_reproduce_the_panorama_labels():
    torch = pytest.importorskip("torch")

    rgb_erp = np.zeros((256, 512, 3), dtype=np.uint8)
    rgb_erp[:, :, 0] = regions(cp.erp_directions(256, 512)) * 20 + 10
    size = 96
    k_json = cp.crop_intrinsics(size, ec.CROP_FOV_DEG)
    rgb_crops = np.asarray([
        ec.sample_erp_rgb(rgb_erp, cp.crop_rays(size, k_json, cp.crop_camera_to_world(*crop))[1])
        for crop in ec.CROP_LAYOUT
    ])

    class ColourToLabel(torch.nn.Module):
        def forward(self, rgb, modal_x):
            centres = (torch.arange(12, dtype=torch.float32) * 20 + 10) / 255
            return -1000 * (rgb[:, :1] - centres[None, :, None, None]) ** 2

    def normalize(image):
        return torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255)[None]

    grids, inside = ec.stitch_grids((64, 128))
    prediction = ec.predict_panorama(
        ColourToLabel(), rgb_crops, np.zeros_like(rgb_crops),
        torch.from_numpy(grids), torch.from_numpy(inside.astype(np.float32)), normalize, "cpu",
    ).numpy()
    expected = regions(cp.erp_directions(64, 128))
    accuracy = (prediction == expected).mean()
    assert accuracy > 0.99
    assert (prediction[0] == 10).all() and (prediction[-1] == 11).all()
    for offset in ((5.0, 0.0), (0.0, 5.0)):
        wrong_layout = tuple((yaw + offset[0], pitch + offset[1]) for yaw, pitch in ec.CROP_LAYOUT)
        wrong_grids, wrong_inside = ec.stitch_grids((64, 128), layout=wrong_layout)
        wrong = ec.predict_panorama(
            ColourToLabel(), rgb_crops, np.zeros_like(rgb_crops),
            torch.from_numpy(wrong_grids), torch.from_numpy(wrong_inside.astype(np.float32)),
            normalize, "cpu",
        ).numpy()
        assert (wrong == expected).mean() < accuracy - 0.01


def test_old_layout_leaves_gaps_at_training_fov():
    old_layout = tuple((float(yaw), 0.0) for yaw in range(0, 360, 45)) + tuple(
        (float(yaw), pitch) for pitch in (45.0, -45.0) for yaw in range(0, 360, 90)
    )
    _, inside = ec.stitch_grids((64, 128), layout=old_layout)
    assert inside.sum(axis=0).min() == 0
    assert not inside[:, 0].any() and not inside[:, -1].any()


@pytest.fixture(scope="module")
def frozen():
    frozen = cp.import_frozen_source(cp.DEFAULT_SOURCE_ROOT)
    from rel_plus.camera import resize_camera_geometry
    from rel_plus.depth import resize_raw_depth_nearest

    frozen.update(
        open_image=cv2.imread, rgb_flag=cv2.IMREAD_COLOR,
        resize_camera_geometry=resize_camera_geometry,
        resize_raw_depth_nearest=resize_raw_depth_nearest,
    )
    return frozen


@pytest.fixture
def panorama(tmp_path):
    from test_cross_projection_synthetic import render_erp_raw_depth

    raw = render_erp_raw_depth(64, 128)
    sample = {"sample_id": "area_5a/synthetic"}
    sample.update({kind: tmp_path / (kind + ".png") for kind in ("rgb", "depth", "semantic")})
    assert cv2.imwrite(str(sample["rgb"]), np.full((64, 128, 3), 100, dtype=np.uint8))
    assert cv2.imwrite(str(sample["depth"]), raw)
    assert cv2.imwrite(str(sample["semantic"]), np.zeros((64, 128, 3), dtype=np.uint8))
    return sample, cp.decode_erp_depth(raw)


LAYOUT = ((30.0, 0.0), (200.0, 45.0))


def test_polar_crops_use_nonsingular_frozen_rel_plus_geometry(frozen, panorama):
    sample, _ = panorama
    polar_layout = tuple(crop for crop in ec.CROP_LAYOUT if abs(crop[1]) > 70)
    assert len(polar_layout) == 2
    rgb, modal_x, _ = ec.prepare_panorama(
        sample, frozen, "rel_plus_v2_1", None, np.array([0], dtype=np.uint8),
        (32, 64), 32, native_size=72, layout=polar_layout,
    )
    assert rgb.shape == modal_x.shape == (2, 32, 32, 3)
    assert modal_x.dtype == np.uint8
    assert (modal_x != 255).any()


def test_rel_plus_crops_use_the_canonical_recipe(frozen, panorama):
    sample, depth = panorama
    rgb, modal_x, label = ec.prepare_panorama(
        sample, frozen, "rel_plus_v2_1", None, np.array([0], dtype=np.uint8),
        (32, 64), 32, native_size=72, layout=LAYOUT,
    )
    assert rgb.shape == modal_x.shape == (2, 32, 32, 3)
    assert label.shape == (32, 64) and (label == 255).all()
    for crop, actual in zip(LAYOUT, modal_x):
        k_json, camera_to_world, _, _, raw = cp.render_crop(depth, *crop, 72, ec.CROP_FOV_DEG)
        camera = frozen["CameraGeometry"].from_json_k(k_json, (72, 72), camera_to_world.T)
        expected = frozen["generate_rel_plus_v2_1"](
            frozen["resize_raw_depth_nearest"](raw, (32, 32)),
            frozen["resize_camera_geometry"](camera, (32, 32)),
        )
        assert np.array_equal(actual, expected)


@pytest.mark.parametrize("variant", ["native_then_resize_linear", "resize_then_hha"])
def test_hha_crops_use_the_verified_cache_recipe(frozen, panorama, variant):
    import hha
    from check_hha_cache import decode

    sample, depth = panorama
    recipe = {"variant": variant, "channel_order": [2, 1, 0]}
    _, modal_x, _ = ec.prepare_panorama(
        sample, frozen, "hha_frozen_cache", recipe, np.array([0], dtype=np.uint8),
        (32, 64), 32, native_size=72, layout=LAYOUT,
    )
    for crop, actual in zip(LAYOUT, modal_x):
        k_json, _, _, _, raw = cp.render_crop(depth, *crop, 72, ec.CROP_FOV_DEG)
        if variant == "resize_then_hha":
            raw = cv2.resize(raw, (32, 32), interpolation=cv2.INTER_NEAREST)
            k_json = cp.crop_intrinsics(32, ec.CROP_FOV_DEG)
        expected, _ = hha.pinhole_hha(*decode(raw), k_json)
        if variant != "resize_then_hha":
            expected = cv2.resize(expected, (32, 32), interpolation=cv2.INTER_LINEAR)
        assert np.array_equal(actual, expected[:, :, ::-1])


def test_level_layout_sees_the_horizon_band_and_not_the_poles():
    height, width = 1024, 2048
    _, inside = ec.stitch_grids((height, width), layout=ec.LAYOUTS["level"])
    elevation = 90.0 - np.arange(height) * 180.0 / height
    seen_rows = inside.any(axis=0).all(axis=1)
    assert seen_rows[np.abs(elevation) <= 25].all()
    assert not inside.any(axis=0)[np.abs(elevation) > 35].any()


# --- ERP-tile control (--projection erp) ---------------------------------------------------------

def test_erp_tiles_match_the_level_crop_windows():
    height, width = 1024, 2048
    source = ec.erp_tile_source_size(480)
    assert source == (1382, 2764)
    windows = ec.erp_tile_windows(ec.LAYOUTS["level"], source, 480)
    grids, inside = ec.erp_tile_grids((height, width), windows, source, 480)
    assert np.abs(grids).max() <= 1 + 1e-6
    elevation = 90.0 - np.arange(height) * 180.0 / height
    assert inside.any(axis=0)[np.abs(elevation) <= 25].all()
    assert not inside.any(axis=0)[np.abs(elevation) > 32].any()
    _, crops = ec.stitch_grids((height, width), layout=ec.LAYOUTS["level"])
    equator = height // 2
    for tile, crop in zip(inside[:, equator], crops[:, equator]):  # same yaw window as its crop
        assert (tile & crop).sum() / (tile | crop).sum() > 0.98
    with pytest.raises(ValueError):
        ec.erp_tile_windows(((0.0, 45.0),), source, 480)


def test_stitched_erp_tiles_reproduce_the_panorama_labels():
    torch = pytest.importorskip("torch")

    size = 96
    source = ec.erp_tile_source_size(size)
    rgb_erp = np.zeros(source + (3,), dtype=np.uint8)
    rgb_erp[:, :, 0] = regions(cp.erp_directions(*source)) * 20 + 10

    class ColourToLabel(torch.nn.Module):
        def forward(self, rgb, modal_x):
            centres = (torch.arange(12, dtype=torch.float32) * 20 + 10) / 255
            return -1000 * (rgb[:, :1] - centres[None, :, None, None]) ** 2

    def normalize(image):
        return torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255)[None]

    def stitched(windows, cut_windows=None):
        tiles = ec.cut_erp_tiles(rgb_erp, cut_windows or windows, size)
        grids, inside = ec.erp_tile_grids((64, 128), windows, source, size)
        prediction = ec.predict_panorama(
            ColourToLabel(), tiles, np.zeros_like(tiles), torch.from_numpy(grids),
            torch.from_numpy(inside.astype(np.float32)), normalize, "cpu",
        ).numpy()
        seen = inside.any(axis=0)
        return (prediction == regions(cp.erp_directions(64, 128)))[seen].mean()

    windows = ec.erp_tile_windows(ec.LAYOUTS["level"], source, size)
    assert any(left + size > source[1] for _, left in windows)  # one tile crosses the seam
    accuracy = stitched(windows)
    assert accuracy > 0.99
    for shift in ((0, 8), (8, 0)):  # tiles cut 5 degrees off the grid they are stitched with
        shifted = [(top + shift[0], (left + shift[1]) % source[1]) for top, left in windows]
        assert stitched(windows, shifted) < accuracy - 0.01


def test_erp_tiles_are_cut_from_the_whole_panorama_inputs(frozen, panorama):
    import eval_pano_transfer as ev

    sample, _ = panorama
    lookup = np.array([0], dtype=np.uint8)
    windows = [(8, 56), (8, 4)]  # the first wraps across the seam
    rgb, modal_x, label = ec.prepare_erp_tiles(
        sample, frozen, "rel_plus_v2_1", None, lookup, (16, 32), 16, windows, (32, 64),
    )
    whole_rgb, whole_x, _ = ev.load_sample(sample, (32, 64), lookup, frozen, "rel_plus_v2_1")
    assert rgb.shape == modal_x.shape == (2, 16, 16, 3) and label.shape == (16, 32)
    assert np.array_equal(modal_x[0], np.concatenate([whole_x[8:24, 56:], whole_x[8:24, :8]], axis=1))
    assert np.array_equal(rgb[1], whole_rgb[8:24, 4:20]) and np.array_equal(modal_x[1], whole_x[8:24, 4:20])


def test_eight_class_predictions_only_pick_kept_channels():
    torch = pytest.importorskip("torch")

    class DroppedClassWins(torch.nn.Module):
        def forward(self, rgb, modal_x):
            logits = torch.zeros(rgb.shape[0], 13, *rgb.shape[2:])
            logits[:, 0] = 5.0  # beam, not one of the 8
            logits[:, 9] = 3.0  # sofa
            return logits

    crops = np.zeros((8, 24, 24, 3), dtype=np.uint8)
    grids, inside = ec.stitch_grids((16, 32), layout=ec.LAYOUTS["level"])
    seen = inside.any(axis=0)

    def normalize(image):
        return torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))[None]

    def predict(channels):
        return ec.predict_panorama(
            DroppedClassWins(), crops, crops, torch.from_numpy(grids),
            torch.from_numpy(inside.astype(np.float32)), normalize, "cpu", channels,
        ).numpy()[seen]

    assert (predict(None) == 0).all()
    assert (predict([3, 4, 7, 8, 9, 10, 11, 12]) == 9).all()


def test_erp_projection_needs_the_level_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "eval_pano_crops.py", "--stanford-root", str(tmp_path), "--semantic-labels", str(tmp_path / "l.json"),
        "--config", str(tmp_path / "c.json"), "--checkpoint", str(tmp_path / "c.pth"),
        "--output", str(tmp_path / "out"), "--projection", "erp",
    ])
    with pytest.raises(SystemExit) as error:
        ec.main()
    assert error.value.code == 2


def test_erp_tile_eval_runs_end_to_end(monkeypatch, tmp_path):
    import json

    torch = pytest.importorskip("torch")
    import eval_pano_transfer as ev
    from test_cross_projection_synthetic import render_erp_raw_depth

    config = json.loads((ev.REPO_ROOT / "0927调参结果" / "configs" / "relplus.json").read_text(encoding="utf-8"))
    config.update(backbone="mit_b0", decoder_embed_dim=32, image_height=96, image_width=96)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    frozen = ec.import_frozen_crops(ev.DEFAULT_SOURCE_ROOT)
    network = frozen["EncoderDecoder"](
        cfg=ev.load_config(tmp_path / "config.json"), criterion=None, norm_layer=torch.nn.BatchNorm2d
    )
    torch.save({"epoch": 200, "model": network.state_dict()}, tmp_path / "checkpoint.pth")

    names = ["<UNK>_0"] + [name + "_1" for name in ev.EXPECTED_CLASSES[1:]]
    (tmp_path / "semantic_labels.json").write_text(json.dumps(names))
    folder = tmp_path / "s2d" / "area_5a" / "pano"
    for kind in ("rgb", "depth", "semantic"):
        (folder / kind).mkdir(parents=True)
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(names), size=(64, 128)).astype(np.uint32)
    cv2.imwrite(str(folder / "rgb" / "c_rgb.png"), rng.integers(0, 256, (64, 128, 3), dtype=np.uint8))
    cv2.imwrite(str(folder / "depth" / "c_depth.png"), render_erp_raw_depth(64, 128))
    cv2.imwrite(str(folder / "semantic" / "c_semantic.png"),
                np.dstack([indices % 256, indices // 256 % 256, indices // 65536]).astype(np.uint8))
    monkeypatch.setattr(sys, "argv", [
        "eval_pano_crops.py", "--stanford-root", str(tmp_path / "s2d"),
        "--semantic-labels", str(tmp_path / "semantic_labels.json"), "--areas", "area_5a",
        "--config", str(tmp_path / "config.json"), "--checkpoint", str(tmp_path / "checkpoint.pth"),
        "--height", "64", "--width", "128", "--device", "cpu", "--layout", "level",
        "--projection", "erp", "--eight-class-predictions", "--output", str(tmp_path / "eval"),
    ])
    assert ec.main() == 0
    report = json.loads((tmp_path / "eval" / "metrics.json").read_text())
    assert report["projection"] == "erp" and report["erp_tile_source_size"] == [276, 552]
    assert abs(report["erp_tile_fov_deg"] - ec.CROP_FOV_DEG) < 0.5
    assert report["prediction_classes"] == list(ev.EIGHT_CLASS_NAMES)
    assert report["crop_x_input"].startswith("whole-panorama X at 276x552")
    confusion = np.loadtxt(tmp_path / "eval" / "confusion_matrix.csv", delimiter=",", dtype=np.int64)
    assert confusion.sum() > 0 and confusion[:, [0, 1, 2, 5, 6]].sum() == 0
