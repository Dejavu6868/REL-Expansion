"""Unit checks for the panorama evaluation helpers."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eval_pano_transfer as ev  # noqa: E402


def test_semantic_decoding_matches_s2d_label_convention(tmp_path):
    names = ["<UNK>_0", "beam_1_a"]
    names += [c + "_9_z" for c in ev.EXPECTED_CLASSES[2:12]]  # board..table
    names += ["wall_2_b", "window_3_c", "wall_4_d"]
    path = tmp_path / "semantic_labels.json"
    path.write_text(json.dumps(names))
    lookup = ev.label_lookup(path)

    first_wall = names.index("wall_2_b")
    indices = np.array(
        [[0, 1, first_wall], [len(names) - 1, len(names), 855309]], dtype=np.uint32
    )
    bgr = np.stack(
        [indices % 256, (indices // 256) % 256, indices // 65536], axis=2
    ).astype(np.uint8)
    label = ev.decode_semantic(bgr, lookup)
    wall = ev.EXPECTED_CLASSES.index("wall") - 1
    assert label.tolist() == [[255, 0, wall], [wall, 255, 255]]


def test_wrap_padding_is_cropped_back():
    torch = pytest.importorskip("torch")

    class PixelwiseNet(torch.nn.Module):
        def forward(self, rgb, modal_x):
            return rgb[:, :2] + modal_x[:, :2]

    rgb = torch.randn(1, 3, 8, 32)
    modal_x = torch.randn(1, 3, 8, 32)
    padded = ev.predict_with_wrap(PixelwiseNet(), rgb, modal_x, 4, "cpu")
    plain = ev.predict_with_wrap(PixelwiseNet(), rgb, modal_x, 0, "cpu")
    assert padded.shape == plain.shape == (1, 2, 8, 32)
    assert torch.equal(padded, plain)


def test_infer_scale_logits_return_to_the_label_size():
    torch = pytest.importorskip("torch")

    class PixelwiseNet(torch.nn.Module):
        def forward(self, rgb, modal_x):
            return rgb[:, :2] + modal_x[:, :2]

    rgb = torch.randn(1, 3, 10, 40)
    modal_x = torch.randn(1, 3, 10, 40)
    net = PixelwiseNet()
    same = ev.predict_with_wrap(net, rgb, modal_x, 4, "cpu", output_size=(10, 40))
    assert torch.equal(same, ev.predict_with_wrap(net, rgb, modal_x, 4, "cpu"))
    resized = ev.predict_with_wrap(net, rgb, modal_x, 5, "cpu", output_size=(8, 32))
    assert resized.shape == (1, 2, 8, 32)


def test_infer_scale_resizes_inputs_but_not_labels(tmp_path):
    import cv2
    import cross_projection as cp
    from test_cross_projection_synthetic import render_erp_raw_depth

    sample = {kind: tmp_path / (kind + ".png") for kind in ("rgb", "depth", "semantic")}
    assert cv2.imwrite(str(sample["rgb"]), np.full((64, 128, 3), 100, dtype=np.uint8))
    assert cv2.imwrite(str(sample["depth"]), render_erp_raw_depth(64, 128))
    assert cv2.imwrite(str(sample["semantic"]), np.zeros((64, 128, 3), dtype=np.uint8))
    frozen = cp.import_frozen_source(cp.DEFAULT_SOURCE_ROOT)
    frozen.update(open_image=cv2.imread, rgb_flag=cv2.IMREAD_COLOR)
    recipe = {"variant": "native_then_resize_linear", "channel_order": [2, 1, 0]}
    rgb, modal_x, label = ev.load_sample(
        sample, (40, 80), np.array([0], dtype=np.uint8), frozen,
        "hha_frozen_cache", recipe, label_size=(32, 64),
    )
    assert rgb.shape[:2] == modal_x.shape[:2] == (40, 80)
    assert label.shape == (32, 64)


@pytest.mark.parametrize("margin, middle, polar", [(0, 704, 96), (32, 768, 128), (64, 832, 128)])
def test_polar_strip_rows_follow_the_band_rule(margin, middle, polar):
    rows = ev.polar_strip_rows(1024, margin)
    elevation = 90.0 - np.arange(1024) * 180.0 / 1024
    assert rows["top_end"] == ((elevation <= 90) & (elevation > 60)).sum() == 171
    assert rows["bottom_start"] == 1024 - ((elevation <= -60) & (elevation > -90)).sum() == 854
    start, end = rows["middle"]
    assert end - start == middle  # the sizes measured in results/half_res_poles_2026-10-25/strip_flops.py
    assert start <= 171 - margin and end >= 854 + margin
    assert rows["top"] == [0, polar] and rows["bottom"] == [512 - polar, 512]
    assert polar / 0.5 >= 171 + margin and polar / 0.5 >= 170 + margin


def test_polar_strips_reproduce_the_two_whole_passes_for_a_pixelwise_net():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    class PixelwiseNet(torch.nn.Module):
        def forward(self, rgb, modal_x):
            return rgb[:, :2] + modal_x[:, :2]

    torch.manual_seed(0)
    full = [torch.randn(1, 3, 192, 384) for _ in range(2)]
    polar = [F.interpolate(image, size=(96, 192), mode="bilinear", align_corners=False) for image in full]
    rows = ev.polar_strip_rows(192, 8)
    net = PixelwiseNet()
    strips = ev.predict_polar_strips(net, full, polar, rows, 16, 8, "cpu")
    whole = ev.predict_with_wrap(net, *full, 16, "cpu")
    half = ev.predict_with_wrap(net, *polar, 8, "cpu", output_size=(192, 384))
    top_end, bottom_start = rows["top_end"], rows["bottom_start"]
    assert strips.shape == whole.shape
    assert torch.allclose(strips[:, :, :top_end], half[:, :, :top_end], atol=1e-6)
    assert torch.equal(strips[:, :, top_end:bottom_start], whole[:, :, top_end:bottom_start])
    assert torch.allclose(strips[:, :, bottom_start:], half[:, :, bottom_start:], atol=1e-6)


@pytest.mark.parametrize("extra", [["--polar-strips", "-1"], ["--polar-strips", "32", "--infer-scale", "0.5"]])
def test_bad_polar_strips_are_rejected_before_loading(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, "argv", [
        "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
        "--semantic-labels", "unused.json", "--config", "unused.json",
        "--checkpoint", "unused.pth", "--output", str(tmp_path / "output"),
    ] + extra)
    with pytest.raises(SystemExit) as error:
        ev.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_erp_tile_windows_cover_the_panorama_and_reject_bad_grids():
    windows = ev.erp_tile_windows(1024, 2048, 3, 8, 384, 384)
    assert len(windows) == 24
    assert sorted({top for top, _ in windows}) == [0, 320, 640]
    assert sorted({left for _, left in windows}) == list(range(0, 2048, 256))
    count = np.zeros((1024, 2048), dtype=int)
    for top, left in windows:
        count[top:top + 384, np.arange(left, left + 384) % 2048] += 1
    assert count.min() >= 1
    windows = ev.erp_tile_windows(2048, 4096, 3, 7, 1080, 1080)  # SMMF's 3 x 7 layout: 4096 / 7 is not whole
    assert sorted({top for top, _ in windows}) == [0, 484, 968]
    assert sorted({left for _, left in windows}) == [0, 585, 1170, 1755, 2340, 2925, 3510]
    count = np.zeros((2048, 4096), dtype=int)
    for top, left in windows:
        count[top:top + 1080, np.arange(left, left + 1080) % 4096] += 1
    assert count.min() >= 1
    for grid in [(3, 4, 384, 384), (3, 8, 300, 384), (3, 8, 384, 200), (1, 8, 384, 384), (0, 8, 384, 384)]:
        with pytest.raises(ValueError):
            ev.erp_tile_windows(1024, 2048, *grid)


def test_erp_tiles_reproduce_the_whole_pass_for_a_pixelwise_net():
    torch = pytest.importorskip("torch")

    class PixelwiseNet(torch.nn.Module):
        def forward(self, rgb, modal_x):
            return rgb + modal_x

    torch.manual_seed(0)
    rgb, modal_x = torch.randn(1, 3, 64, 128), torch.randn(1, 3, 64, 128)
    windows = ev.erp_tile_windows(64, 128, 3, 8, 32, 32)  # the last column wraps across the seam
    whole = ev.predict_with_wrap(PixelwiseNet(), rgb, modal_x, 0, "cpu")
    for channels in (None, [0, 2]):
        tiled = ev.predict_erp_tiles(PixelwiseNet(), rgb, modal_x, windows, (32, 32), "cpu", channels)
        assert torch.allclose(tiled, ev.class_probabilities(whole, channels), atol=1e-6)
        assert torch.equal(ev.predict_classes(tiled, channels), ev.predict_classes(whole, channels))
    assert torch.allclose(ev.class_probabilities(whole, [0, 2]).sum(dim=1), torch.ones(1, 64, 128))
    assert (ev.class_probabilities(whole, [0, 2])[:, 1] == 0).all()


@pytest.mark.parametrize("extra", [
    ["--erp-tiles", "3", "4"],
    ["--erp-tiles", "3", "8", "--polar-strips", "32"],
    ["--erp-tiles", "3", "8", "--infer-scale", "0.5"],
])
def test_bad_erp_tiles_are_rejected_before_loading(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, "argv", [
        "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
        "--semantic-labels", "unused.json", "--config", "unused.json",
        "--checkpoint", "unused.pth", "--output", str(tmp_path / "output"),
    ] + extra)
    with pytest.raises(SystemExit) as error:
        ev.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_polar_caps_turn_the_poles_onto_the_equator_and_cover_the_caps():
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(100, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    assert np.allclose(ev.from_cap_frame(ev.to_cap_frame(directions)), directions)
    poles = ev.longitude_latitude(ev.to_cap_frame(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])))
    assert np.allclose(np.degrees(poles[0]), [-90, 90]) and np.allclose(poles[1], 0)
    for height, size in ((1024, 384), (2048, 768)):
        caps = ev.polar_cap_grids(height, 2 * height, size)
        assert (caps["top_end"], caps["bottom_start"]) == ev.polar_band_rows(height)
        assert caps["tiles"].shape == (2, size, size, 2) and np.abs(caps["tiles"]).max() <= 1
        assert caps["north"].shape == (caps["top_end"], 2 * height, 2)
        assert caps["south"].shape == (height - caps["bottom_start"], 2 * height, 2)
        for back in (caps["north"], caps["south"]):  # every polar pixel lies inside its tile, with context
            assert np.abs(back).max() < 30.1 / 33.75
    caps = ev.polar_cap_grids(1024, 2048, 384)
    assert caps["top_end"] == 171 and caps["bottom_start"] == 854


def test_polar_caps_reproduce_a_smooth_whole_pass_for_a_pixelwise_net():
    torch = pytest.importorskip("torch")

    class PixelwiseNet(torch.nn.Module):
        def forward(self, rgb, modal_x):
            return 4 * rgb + modal_x

    height, width, size = 128, 256, 64
    longitude, latitude = np.meshgrid(
        (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi,
        np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi,
    )
    smooth = torch.as_tensor(ev.unit_directions(longitude, latitude), dtype=torch.float32).permute(2, 0, 1)[None]
    steps = torch.as_tensor(np.digitize(longitude, [-1, 1]), dtype=torch.float32)[None, None].repeat(1, 3, 1, 1)
    caps = ev.polar_cap_grids(height, width, size)
    rgb_tiles, x_tiles = ev.sample_cap_tiles(smooth, steps, caps, "cpu")
    assert rgb_tiles.shape == x_tiles.shape == (2, 3, size, size)
    assert x_tiles.unique().tolist() == [0.0, 1.0, 2.0]  # nearest: X values are never blended
    whole = ev.predict_with_wrap(PixelwiseNet(), smooth, smooth, 0, "cpu")
    top_end, bottom_start = caps["top_end"], caps["bottom_start"]
    for channels in (None, [0, 2]):
        expected = ev.class_probabilities(whole, channels)
        north, south = ev.predict_polar_caps(PixelwiseNet(), smooth, smooth, caps, "cpu", channels)
        assert torch.allclose(north, expected[:, :, :top_end], atol=0.01)  # a mirrored cap is off by 0.14
        assert torch.allclose(south, expected[:, :, bottom_start:], atol=0.01)
        assert torch.allclose(north.sum(dim=1), torch.ones(1, top_end, width), atol=1e-5)


@pytest.mark.parametrize("extra", [
    ["--polar-caps", "0"],
    ["--polar-caps", "400"],
    ["--polar-caps", "384", "--polar-strips", "32"],
    ["--polar-caps", "384", "--infer-scale", "0.5"],
])
def test_bad_polar_caps_are_rejected_before_loading(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, "argv", [
        "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
        "--semantic-labels", "unused.json", "--config", "unused.json",
        "--checkpoint", "unused.pth", "--output", str(tmp_path / "output"),
    ] + extra)
    with pytest.raises(SystemExit) as error:
        ev.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def frozen_attention_class():
    ev.import_frozen(ev.DEFAULT_SOURCE_ROOT)
    from models.encoders.dual_segformer import Attention

    return Attention


@pytest.mark.parametrize("sr_ratio", [1, 2])
def test_chunked_attention_matches_the_frozen_forward(sr_ratio):
    torch = pytest.importorskip("torch")
    attention_class = frozen_attention_class()
    torch.manual_seed(0)
    module = attention_class(dim=16, num_heads=2, sr_ratio=sr_ratio).eval()
    x = torch.randn(1, 6 * 10, 16)
    keys = 60 // sr_ratio ** 2
    with torch.no_grad():
        frozen = module(x, 6, 10)
        chunked = ev._chunked_attention_forward(module, 7 * 2 * keys * 4, x, 6, 10)  # 7 queries a chunk
    assert torch.allclose(chunked, frozen, atol=1e-6)


def test_chunk_attention_covers_both_streams_and_keeps_the_logits(tmp_path):
    torch = pytest.importorskip("torch")
    frozen = ev.import_frozen(ev.DEFAULT_SOURCE_ROOT)
    config = json.loads((ev.REPO_ROOT / "0927调参结果" / "configs" / "relplus.json").read_text(encoding="utf-8"))
    config.update(backbone="mit_b0", decoder_embed_dim=32)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    torch.manual_seed(0)
    network = frozen["EncoderDecoder"](
        cfg=ev.load_config(config_path), criterion=None, norm_layer=torch.nn.BatchNorm2d
    ).eval()
    rgb, modal_x = torch.randn(1, 3, 64, 128), torch.randn(1, 3, 64, 128)
    with torch.no_grad():
        plain = network(rgb, modal_x)
        assert ev.chunk_attention(network, 1 / 1024) == 2 * 8  # mit_b0 depths 2, 2, 2, 2, two streams
        chunked = network(rgb, modal_x)
    assert torch.allclose(chunked, plain, atol=1e-5)


@pytest.mark.parametrize("scale", ["0", "-1", "nan"])
def test_bad_infer_scale_is_rejected_before_loading(monkeypatch, tmp_path, scale):
    monkeypatch.setattr(sys, "argv", [
        "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
        "--semantic-labels", "unused.json", "--config", "unused.json",
        "--checkpoint", "unused.pth", "--output", str(tmp_path / "output"),
        "--infer-scale", scale,
    ])
    with pytest.raises(SystemExit) as error:
        ev.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_negative_attention_chunk_is_rejected_before_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
        "--semantic-labels", "unused.json", "--config", "unused.json",
        "--checkpoint", "unused.pth", "--output", str(tmp_path / "output"),
        "--attention-chunk-mb", "-1",
    ])
    with pytest.raises(SystemExit) as error:
        ev.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_eight_class_predictions_only_pick_span8_channels():
    torch = pytest.importorskip("torch")

    config = json.loads((ev.REPO_ROOT / "0927调参结果" / "configs" / "relplus.json").read_text(encoding="utf-8"))
    channels = ev.eight_class_channels(config["class_names"])
    assert channels == [3, 4, 7, 8, 9, 10, 11, 12]
    logits = torch.zeros(1, 13, 1, 3)
    logits[0, 5, 0, 0], logits[0, 12, 0, 0] = 9.0, 1.0  # clutter wins; window is the best of the 8
    logits[0, 0, 0, 1], logits[0, 4, 0, 1] = 9.0, 1.0  # beam wins; chair is the best of the 8
    logits[0, 10, 0, 2] = 2.0  # table wins either way
    assert ev.predict_classes(logits)[0, 0].tolist() == [5, 0, 10]
    assert ev.predict_classes(logits, channels)[0, 0].tolist() == [12, 4, 10]
    with pytest.raises(ValueError, match="sofa"):
        ev.eight_class_channels([name for name in config["class_names"] if name != "sofa"])


def test_every_requested_area_must_have_samples(tmp_path):
    for kind in ("rgb", "depth", "semantic"):
        folder = tmp_path / "area_5a" / "pano" / kind
        folder.mkdir(parents=True)
        (folder / ("sample_" + kind + ".png")).touch()
    with pytest.raises(FileNotFoundError, match="area_5b"):
        ev.list_panoramas(tmp_path, ["area_5a", "area_5b"])


def test_duplicate_areas_cannot_double_count_samples(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        ev.list_panoramas(tmp_path, ["area_5a", "area_5a"])


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_nonpositive_smoke_limit_is_rejected_before_loading(monkeypatch, tmp_path, limit):
    monkeypatch.setattr(sys, "argv", [
        "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
        "--semantic-labels", "unused.json", "--config", "unused.json",
        "--checkpoint", "unused.pth", "--output", str(tmp_path / "output"),
        "--limit", limit,
    ])
    with pytest.raises(SystemExit) as error:
        ev.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "variant", ["native_then_resize", "native_then_resize_linear", "resize_then_hha"]
)
def test_hha_input_uses_the_verified_resolution_recipe(tmp_path, variant):
    import cv2
    import cross_projection as cp
    import hha
    from test_cross_projection_synthetic import render_erp_raw_depth

    raw = render_erp_raw_depth(64, 128)
    sample = {kind: tmp_path / (kind + ".png") for kind in ("rgb", "depth", "semantic")}
    assert cv2.imwrite(str(sample["rgb"]), np.full((64, 128, 3), 100, dtype=np.uint8))
    assert cv2.imwrite(str(sample["depth"]), raw)
    assert cv2.imwrite(str(sample["semantic"]), np.zeros((64, 128, 3), dtype=np.uint8))
    frozen = cp.import_frozen_source(cp.DEFAULT_SOURCE_ROOT)
    frozen.update(open_image=cv2.imread, rgb_flag=cv2.IMREAD_COLOR)
    recipe = {"variant": variant, "channel_order": [2, 1, 0]}
    _, actual, _ = ev.load_sample(
        sample, (32, 64), np.array([0], dtype=np.uint8), frozen,
        "hha_frozen_cache", recipe,
    )
    generation_depth = raw if variant.startswith("native_then_resize") else cv2.resize(
        raw, (64, 32), interpolation=cv2.INTER_NEAREST
    )
    expected, _ = hha.erp_hha(cp.decode_erp_depth(generation_depth), frozen)
    interpolation = cv2.INTER_LINEAR if variant.endswith("linear") else cv2.INTER_NEAREST
    expected = cv2.resize(expected, (64, 32), interpolation=interpolation)
    assert np.array_equal(actual, expected[:, :, ::-1])


def test_elevation_bands_split_rows_and_skip_empty_bands(tmp_path):
    # 12 rows at 90, 75, ..., -75 degrees; each band keeps its top edge: [90,60) [60,25) ...
    label = np.ones((12, 4), dtype=np.uint8)
    label[10:] = 255  # the bottom band's rows (-60, -75) are unlabelled
    prediction = np.ones((12, 4), dtype=np.uint8)
    prediction[0] = 0

    def hist_info(n, pred, gt):
        keep = gt < n
        return np.bincount(n * gt[keep].astype(int) + pred[keep], minlength=n * n).reshape(n, n), 0, 0

    bands = ev.elevation_band_confusions(2, prediction, label, hist_info)
    assert bands.sum(axis=(1, 2)).tolist() == [8, 12, 4, 8, 8, 0]
    assert bands[0].tolist() == [[0, 0], [4, 4]]

    def metrics(confusion):
        iou = (np.diag(confusion) / confusion.sum()).tolist()
        return {"mIoU_percent": 1.0, "pixel_accuracy_percent": 2.0, "per_class_iou_percent": iou}

    summary = ev.write_elevation_bands(tmp_path, bands, metrics, ["a", "b"])
    assert [band["elevation_deg"] for band in summary][0] == [90, 60]
    assert summary[-1] == {
        "elevation_deg": [-60, -90], "valid_pixels": 0, "mIoU_percent": None,
        "pixel_accuracy_percent": None, "per_class_iou_percent": None,
    }
    assert np.array_equal(np.load(tmp_path / "elevation_band_confusion.npy"), bands)
    rows = (tmp_path / "elevation_band_metrics.csv").read_text().splitlines()
    assert len(rows) == 7 and rows[-1] == "-60,-90,0,,,,"
