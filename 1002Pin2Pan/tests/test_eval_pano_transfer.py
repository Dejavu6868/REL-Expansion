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
    windows = ev.erp_tile_windows(2048, 4096, 3, 7, 1080, 1080, 720)  # REL-SF4PASS's stated stride, wrapping
    assert sorted({top for top, _ in windows}) == [0, 484, 968]
    assert [left for top, left in windows if top == 0] == [0, 720, 1440, 2160, 2880, 3600, 224]
    count = np.zeros((2048, 4096), dtype=int)
    for top, left in windows:
        count[top:top + 1080, np.arange(left, left + 1080) % 4096] += 1
    assert count.min() >= 1
    windows = ev.erp_tile_windows(2048, 4096, 3, 7, 1080, 1080, 720, -652)  # SMMF's code: 652 px cyclic pad
    assert [left for top, left in windows if top == 0] == [3444, 68, 788, 1508, 2228, 2948, 3668]
    count = np.zeros((2048, 4096), dtype=int)
    for top, left in windows:
        count[top:top + 1080, np.arange(left, left + 1080) % 4096] += 1
    assert count.min() >= 1
    windows = ev.erp_tile_windows(1536, 3072, 3, 8, 576, 576)  # the 3 x 8 grid at --infer-scale 0.75 of 2048 x 4096
    assert sorted({top for top, _ in windows}) == [0, 480, 960]
    assert sorted({left for _, left in windows}) == list(range(0, 3072, 384))
    count = np.zeros((1536, 3072), dtype=int)
    for top, left in windows:
        count[top:top + 576, np.arange(left, left + 576) % 3072] += 1
    assert count.min() >= 1
    for stride in (1081, 500):  # wider than a tile; 7 x 500 columns leave a gap
        with pytest.raises(ValueError):
            ev.erp_tile_windows(2048, 4096, 3, 7, 1080, 1080, stride)
    for grid in [(3, 4, 384, 384), (3, 8, 300, 384), (3, 8, 384, 200), (1, 8, 384, 384), (0, 8, 384, 384)]:
        with pytest.raises(ValueError):
            ev.erp_tile_windows(1024, 2048, *grid)


def cmx_sliding_windows(rows, cols, crop, stride_rate):
    """(top, left) of each window in CMX's scale_process_rgbX loop, copied from Max's Pano_MA_Seg evaluator."""
    stride = (int(np.ceil(crop[0] * stride_rate)), int(np.ceil(crop[1] * stride_rate)))
    r_grid = int(np.ceil((rows - crop[0]) / stride[0])) + 1
    c_grid = int(np.ceil((cols - crop[1]) / stride[1])) + 1
    windows = []
    for grid_yidx in range(r_grid):
        for grid_xidx in range(c_grid):
            s_x = grid_xidx * stride[0]
            s_y = grid_yidx * stride[1]
            e_x = min(s_x + crop[0], cols)
            e_y = min(s_y + crop[1], rows)
            windows.append((e_y - crop[1], e_x - crop[0]))
    return windows


def test_sliding_windows_match_cmx_sliding_evaluation():
    windows = ev.sliding_windows(2048, 4096, 1080, 1080, 720)  # REL-SF4PASS's 1080^2 regions, stride 720
    assert len(windows) == 18
    assert sorted({top for top, _ in windows}) == [0, 720, 968]
    assert sorted({left for _, left in windows}) == [0, 720, 1440, 2160, 2880, 3016]
    assert all(left + 1080 <= 4096 for _, left in windows)  # nothing wraps across the seam
    count = np.zeros((2048, 4096), dtype=int)
    for top, left in windows:
        count[top:top + 1080, left:left + 1080] += 1
    assert count.min() >= 1
    for rows, cols, crop, rate in [(2048, 4096, 1080, 2 / 3), (2048, 4096, 1080, 0.4), (1024, 2048, 540, 2 / 3),
                                   (64, 128, 32, 0.5), (1080, 1080, 1080, 2 / 3)]:
        stride = int(np.ceil(crop * rate))
        assert ev.sliding_windows(rows, cols, crop, crop, stride) == cmx_sliding_windows(rows, cols, (crop, crop), rate)
    for bad in [(2048, 4096, 1080, 1080, 0), (2048, 4096, 1080, 1080, 1081), (1024, 2048, 1080, 1080, 720)]:
        with pytest.raises(ValueError):
            ev.sliding_windows(*bad)


def test_sliding_tiles_fuse_their_own_columns_without_wrapping():
    torch = pytest.importorskip("torch")

    class RowMeanNet(torch.nn.Module):  # depends on the whole tile width, so each tile's own columns matter
        def forward(self, rgb, modal_x):
            return rgb + modal_x + rgb.mean(dim=3, keepdim=True)

    torch.manual_seed(0)
    rgb, modal_x = torch.randn(1, 3, 64, 128), torch.randn(1, 3, 64, 128)
    windows = ev.sliding_windows(64, 128, 32, 32, 24)  # rows 0, 24, 32; columns 0, 24, ..., 96
    net = RowMeanNet()
    total, count = torch.zeros(1, 3, 64, 128), torch.zeros(1, 1, 64, 128)
    for top, left in windows:
        area = (slice(None), slice(None), slice(top, top + 32), slice(left, left + 32))
        total[area] += net(rgb[area], modal_x[area]).exp()
        count[area] += 1
    fused = ev.predict_erp_tiles(net, rgb, modal_x, windows, (32, 32), "cpu", None, True)
    assert torch.allclose(fused, total / count, rtol=1e-5, atol=1e-6)


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
        summed = ev.predict_erp_tiles(PixelwiseNet(), rgb, modal_x, windows, (32, 32), "cpu", channels, True)
        assert torch.allclose(summed, whole.exp(), rtol=1e-5)
        assert torch.equal(ev.predict_classes(summed, channels), ev.predict_classes(whole, channels))
    assert torch.allclose(ev.class_probabilities(whole, [0, 2]).sum(dim=1), torch.ones(1, 64, 128))
    assert (ev.class_probabilities(whole, [0, 2])[:, 1] == 0).all()


@pytest.mark.parametrize("extra", [
    ["--erp-tiles", "3", "4"],
    ["--erp-tile-column-stride", "720"],
    ["--erp-tiles", "3", "7", "--erp-tile-column-offset", "-652"],
    ["--erp-tile-exp-sum"],
    ["--erp-tiles", "3", "8", "--polar-strips", "32"],
    ["--erp-tiles", "3", "8", "--erp-tile-size", "768", "768", "--infer-scale", "0.25"],  # 768 > 256 input rows
    ["--erp-tile-sliding-stride", "0"],
    ["--erp-tile-sliding-stride", "500"],  # wider than the 384 px tile
    ["--erp-tile-sliding-stride", "720", "--erp-tile-size", "1080", "1080"],  # 1080 > 1024 rows
    ["--erp-tiles", "3", "8", "--erp-tile-sliding-stride", "256"],
    ["--erp-tile-sliding-stride", "256", "--polar-strips", "32"],
    ["--erp-tile-sliding-stride", "256", "--erp-tile-polar-scale", "0.5"],
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


@pytest.mark.parametrize("middle_height", [32, 48])
def test_polar_scaled_tiles_keep_each_zone_to_its_own_rows(middle_height):
    torch = pytest.importorskip("torch")

    class PixelwiseNet(torch.nn.Module):
        def forward(self, rgb, modal_x):
            return rgb + modal_x

    torch.manual_seed(0)
    full = torch.randn(1, 3, 64, 128), torch.randn(1, 3, 64, 128)
    polar = torch.randn(1, 3, 32, 64), torch.randn(1, 3, 32, 64)  # unrelated, so each row shows its source
    windows = ev.erp_tile_windows(64, 128, 3, 8, 32, 32)  # rows 0, 16, 32; columns wrap across the seam
    start = (64 - middle_height) // 2
    for exp_sum in (False, True):
        fused = ev.predict_polar_scaled_tiles(
            PixelwiseNet(), full, polar, windows, (32, 32), (start, start + middle_height), "cpu", None, exp_sum,
        )
        to_fused = (lambda logits: logits.exp()) if exp_sum else ev.class_probabilities
        native = to_fused(full[0] + full[1])
        small = torch.nn.functional.interpolate(  # the polar tiles reproduce the small whole pass, resized back
            to_fused(polar[0] + polar[1]), size=(64, 128), mode="bilinear", align_corners=False
        )
        rows = torch.zeros(64, dtype=torch.bool)
        rows[start:start + middle_height] = True
        assert torch.allclose(fused[:, :, rows], native[:, :, rows], rtol=1e-5, atol=1e-6)
        assert torch.allclose(fused[:, :, ~rows], small[:, :, ~rows], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("wrap_pad", [0, 8])
def test_polar_strips_run_each_polar_row_as_one_wrapped_pass(wrap_pad):
    torch = pytest.importorskip("torch")
    F = torch.nn.functional

    class RowMeanNet(torch.nn.Module):  # depends on the whole input width, so a strip differs from its tiles
        def forward(self, rgb, modal_x):
            return rgb + modal_x + rgb.mean(dim=3, keepdim=True)

    torch.manual_seed(0)
    full = torch.randn(1, 3, 64, 128), torch.randn(1, 3, 64, 128)
    polar = torch.randn(1, 3, 32, 64), torch.randn(1, 3, 32, 64)
    windows = ev.erp_tile_windows(64, 128, 3, 8, 32, 32)  # rows 0, 16, 32 -> polar strips at rows 0 and 16
    net = RowMeanNet()
    fused = ev.predict_polar_scaled_tiles(net, full, polar, windows, (32, 32), (16, 48), "cpu", None, True, wrap_pad)

    def strip(rows):
        images = [F.pad(image[:, :, rows], (wrap_pad, wrap_pad, 0, 0), mode="circular") for image in polar]
        return net(*images)[:, :, :, wrap_pad:wrap_pad + 64].exp()

    small = torch.cat([strip(slice(0, 16)), strip(slice(16, 32))], dim=2)
    expected = F.interpolate(small, size=(64, 128), mode="bilinear", align_corners=False)
    rows = torch.zeros(64, dtype=torch.bool)
    rows[16:48] = True
    assert torch.allclose(fused[:, :, ~rows], expected[:, :, ~rows], rtol=1e-5, atol=1e-6)
    tiles =ev.predict_polar_scaled_tiles(net, full, polar, windows, (32, 32), (16, 48), "cpu", None, True)
    assert torch.equal(fused[:, :, rows], tiles[:, :, rows])  # the middle row is unchanged
    assert not torch.allclose(fused[:, :, ~rows], tiles[:, :, ~rows])  # strips are not the tiles


@pytest.mark.parametrize("scale", [0.5, 1])
def test_polar_squeeze_runs_each_polar_row_as_one_tile_sized_input(scale):
    torch = pytest.importorskip("torch")
    F = torch.nn.functional

    class RowMeanNet(torch.nn.Module):  # depends on the whole input width; records its input sizes
        sizes = []

        def forward(self, rgb, modal_x):
            self.sizes.append(tuple(rgb.shape[2:]))
            return rgb + modal_x + rgb.mean(dim=3, keepdim=True)

    torch.manual_seed(0)
    full = torch.randn(1, 3, 64, 128), torch.randn(1, 3, 64, 128)
    polar = full if scale == 1 else (torch.randn(1, 3, 32, 64), torch.randn(1, 3, 32, 64))
    windows = ev.erp_tile_windows(64, 128, 3, 8, 32, 32)  # rows 0, 16, 32
    tile = int(32 * scale)
    net = RowMeanNet()
    fused = ev.predict_polar_scaled_tiles(net, full, polar, windows, (32, 32), (16, 48), "cpu", None, True,
                                          squeeze=True)
    assert net.sizes[:2] == [(tile, tile)] * 2  # one tile-sized input per polar row, then the middle tiles

    def squeezed(row):
        images = [F.interpolate(image[:, :, row:row + tile], size=(tile, tile), mode="area") for image in polar]
        logits = F.interpolate(net(*images), size=(tile, polar[0].shape[3]), mode="bilinear", align_corners=False)
        return logits.exp()

    small = torch.zeros(1, 3, *polar[0].shape[2:])
    count = torch.zeros(1, 1, *polar[0].shape[2:])
    for row in (0, polar[0].shape[2] - tile):
        small[:, :, row:row + tile] += squeezed(row)
        count[:, :, row:row + tile] += 1
    expected = F.interpolate(small / count, size=(64, 128), mode="bilinear", align_corners=False)
    rows = torch.zeros(64, dtype=torch.bool)
    rows[16:48] = True
    assert torch.allclose(fused[:, :, ~rows], expected[:, :, ~rows], rtol=1e-5, atol=1e-6)
    tiles = ev.predict_polar_scaled_tiles(net, full, polar, windows, (32, 32), (16, 48), "cpu", None, True)
    assert torch.equal(fused[:, :, rows], tiles[:, :, rows])  # the middle row is unchanged
    assert not torch.allclose(fused[:, :, ~rows], tiles[:, :, ~rows])  # one squeezed input is not the tiles


def test_pooled_polar_squeeze_matches_the_pooled_zone_control_on_a_panorama_constant_along_longitude():
    torch = pytest.importorskip("torch")

    class PixelNet(torch.nn.Module):
        def forward(self, rgb, modal_x):
            return rgb - 0.5 * modal_x

    torch.manual_seed(0)
    full = [column.expand(1, 3, 64, 128).clone() for column in (torch.randn(1, 3, 64, 1), torch.randn(1, 3, 64, 1))]
    windows = ev.erp_tile_windows(64, 128, 3, 8, 32, 32)
    net = ev.input_pool_network(PixelNet(), "avg", "avg")
    zones = ev.predict_polar_scaled_tiles(net, full, full, windows, (32, 32), (16, 48), "cpu", None, True)
    squeeze = ev.predict_polar_scaled_tiles(net, full, full, windows, (32, 32), (16, 48), "cpu", None, True,
                                            squeeze=True)
    assert torch.allclose(squeeze, zones, rtol=1e-5, atol=1e-6)  # squeezing a constant row changes nothing


def test_smmf_layout_with_polar_scale_passes_the_argument_checks(monkeypatch, tmp_path):
    for middle in ([], ["--erp-tile-middle-height", "1366"], ["--erp-tile-polar-strips"],
                   ["--erp-tile-polar-squeeze"]):
        output = tmp_path / "output{}".format(len(middle))
        monkeypatch.setattr(sys, "argv", [
            "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
            "--semantic-labels", "unused.json", "--config", str(tmp_path / "missing.json"),
            "--checkpoint", "unused.pth", "--output", str(output), "--height", "2048", "--width", "4096",
            "--erp-tiles", "3", "7", "--erp-tile-size", "1080", "1080", "--erp-tile-column-stride", "720",
            "--erp-tile-column-offset", "-652", "--erp-tile-exp-sum", "--erp-tile-polar-scale", "0.5",
        ] + middle)
        with pytest.raises(Exception):  # gets past the parser (SystemExit is not an Exception) to the config
            ev.main()
        assert output.exists()


def test_sliding_layout_passes_the_argument_checks(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
        "--semantic-labels", "unused.json", "--config", str(tmp_path / "missing.json"),
        "--checkpoint", "unused.pth", "--output", str(tmp_path / "output"), "--height", "2048", "--width", "4096",
        "--erp-tile-size", "1080", "1080", "--erp-tile-sliding-stride", "720", "--erp-tile-exp-sum",
    ])
    with pytest.raises(Exception):  # gets past the parser (SystemExit is not an Exception) to the config
        ev.main()
    assert (tmp_path / "output").exists()


@pytest.mark.parametrize("extra", [
    ["--erp-tile-polar-scale", "0.5"],
    ["--erp-tiles", "3", "8", "--erp-tile-middle-height", "512"],
    ["--erp-tiles", "2", "8", "--erp-tile-size", "512", "384", "--erp-tile-polar-scale", "0.5"],
    ["--erp-tiles", "3", "8", "--erp-tile-polar-scale", "0.5", "--infer-scale", "0.5"],
    ["--erp-tiles", "3", "8", "--erp-tile-polar-scale", "0.5", "--polar-caps", "384"],
    ["--erp-tiles", "3", "8", "--erp-tile-polar-scale", "1"],
    ["--erp-tiles", "3", "8", "--erp-tile-polar-scale", "0.3"],  # 384 x 0.3 is not a whole tile
    ["--erp-tiles", "3", "8", "--erp-tile-size", "400", "384", "--erp-tile-polar-scale", "0.5"],  # rows don't meet
    ["--erp-tiles", "3", "8", "--erp-tile-polar-scale", "0.5", "--erp-tile-middle-height", "383"],
    ["--erp-tiles", "3", "8", "--erp-tile-polar-scale", "0.5", "--erp-tile-middle-height", "385"],
    ["--erp-tiles", "3", "8", "--erp-tile-polar-strips"],  # strips need the polar scale
    ["--erp-tiles", "3", "8", "--erp-tile-polar-scale", "0.5", "--erp-tile-polar-strips", "--wrap-pad", "3"],
    ["--erp-tiles", "3", "8", "--erp-tile-polar-squeeze"],  # the squeeze needs the polar scale
    ["--erp-tiles", "3", "8", "--erp-tile-polar-scale", "0.5", "--erp-tile-polar-squeeze", "--erp-tile-polar-strips"],
])
def test_bad_polar_scaled_tiles_are_rejected_before_loading(monkeypatch, tmp_path, extra):
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


def test_pole_patches_cover_the_caps_without_mirroring():
    for size, focal in ((540, 294.25), (1080, 588.50)):
        patches = ev.pole_patch_grids(2048, 4096, size, (484, 1564))  # SMMF's middle row: caps beyond +-47.46 deg
        assert abs(patches["cap_radius_deg"] - 42.539) < 1e-3 and abs(patches["focal_px"] - focal) < 0.01
        assert patches["tiles"].shape == (2, size, size, 2) and np.abs(patches["tiles"]).max() <= 1
        assert patches["north"].shape == (484, 4096, 2) and patches["south"].shape == (484, 4096, 2)
        for back in (patches["north"], patches["south"]):  # every polar pixel lies inside the inscribed circle
            assert np.hypot(back[..., 0], back[..., 1]).max() < 1
    patches = ev.pole_patch_grids(256, 512, 64, (60, 196))
    centre = -patches["tiles"][:, 31:33, 31:33, 1].mean(axis=(1, 2)) * 90  # latitude read at the patch centre
    assert centre[0] > 88 and centre[1] < -88
    for name, row in (("north", 5), ("south", -6)):  # ERP right/down map to a patch turn, not a mirror
        back = patches[name]
        jacobian = np.stack([back[row, 101] - back[row, 100], back[row + 1, 100] - back[row, 100]], axis=1)
        assert np.linalg.det(jacobian) > 0
    with pytest.raises(ValueError):
        ev.pole_patch_grids(256, 512, 64, (60, 190))


def test_pole_patches_reproduce_a_smooth_whole_pass_and_keep_p47s_middle_row():
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
    windows = ev.erp_tile_windows(height, width, 3, 8, 64, 64)  # rows 0, 32, 64: the middle row is 32..96
    patches = ev.pole_patch_grids(height, width, size, (32, 96))
    whole = ev.predict_with_wrap(PixelwiseNet(), smooth, smooth, 0, "cpu")
    rows = torch.zeros(height, dtype=torch.bool)
    rows[32:96] = True
    for channels in (None, [0, 2]):
        fused = ev.predict_pole_patches(PixelwiseNet(), (smooth, smooth), windows, (64, 64), (32, 96), patches,
                                        "cpu", channels)
        expected = ev.class_probabilities(whole, channels)
        assert torch.allclose(fused[:, :, ~rows], expected[:, :, ~rows], atol=0.01)
        half = torch.nn.functional.interpolate(smooth, scale_factor=0.5, mode="bilinear", align_corners=False)
        p47 = ev.predict_polar_scaled_tiles(PixelwiseNet(), (smooth, smooth), (half, half), windows, (64, 64),
                                            (32, 96), "cpu", channels)
        assert torch.equal(fused[:, :, rows], p47[:, :, rows])  # the same native middle windows


def test_smmf_layout_with_pole_patches_passes_the_argument_checks(monkeypatch, tmp_path):
    for size in ("540", "1080"):
        output = tmp_path / ("output" + size)
        monkeypatch.setattr(sys, "argv", [
            "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
            "--semantic-labels", "unused.json", "--config", str(tmp_path / "missing.json"),
            "--checkpoint", "unused.pth", "--output", str(output), "--height", "2048", "--width", "4096",
            "--erp-tiles", "3", "7", "--erp-tile-size", "1080", "1080", "--erp-tile-column-stride", "720",
            "--erp-tile-column-offset", "-652", "--erp-tile-exp-sum", "--erp-tile-pole-patches", size,
        ])
        with pytest.raises(Exception):  # gets past the parser (SystemExit is not an Exception) to the config
            ev.main()
        assert output.exists()


@pytest.mark.parametrize("extra", [
    ["--erp-tile-pole-patches", "384"],
    ["--erp-tiles", "2", "8", "--erp-tile-size", "512", "384", "--erp-tile-pole-patches", "384"],
    ["--erp-tiles", "3", "8", "--erp-tile-pole-patches", "0"],
    ["--erp-tiles", "3", "8", "--erp-tile-pole-patches", "384", "--erp-tile-polar-scale", "0.5"],
    ["--erp-tiles", "3", "8", "--erp-tile-pole-patches", "384", "--polar-caps", "384"],
    ["--erp-tiles", "3", "8", "--erp-tile-pole-patches", "384", "--infer-scale", "0.5"],
    ["--erp-tile-size", "384", "384", "--erp-tile-sliding-stride", "320", "--erp-tile-pole-patches", "384"],
])
def test_bad_pole_patches_are_rejected_before_loading(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, "argv", [
        "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
        "--semantic-labels", "unused.json", "--config", "unused.json",
        "--checkpoint", "unused.pth", "--output", str(tmp_path / "output"),
    ] + extra)
    with pytest.raises(SystemExit) as error:
        ev.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_sphere_views_are_the_crop_tools_views_and_see_every_pixel():
    pytest.importorskip("cv2")
    import cross_projection as cp
    import eval_pano_crops as crops

    assert ev.SPHERE_VIEW_LAYOUT == crops.CROP_LAYOUT and ev.SPHERE_VIEW_FOV_DEG == crops.CROP_FOV_DEG
    height, width, size = 128, 256, 48
    views = ev.sphere_view_grids(height, width, size)
    assert views["tiles"].shape == (22, size, size, 2) and views["views_per_pixel"]["min"] >= 1
    assert abs(views["focal_px"] - 24 / np.tan(np.radians(ev.SPHERE_VIEW_FOV_DEG / 2))) < 1e-9
    k_json = cp.crop_intrinsics(size, ev.SPHERE_VIEW_FOV_DEG)
    for tile, (yaw, pitch) in zip(views["tiles"], ev.SPHERE_VIEW_LAYOUT):  # same rays as eval_pano_crops' crops
        _, rays = cp.crop_rays(size, k_json, cp.crop_camera_to_world(yaw, pitch))
        ours = ev.unit_directions(tile[..., 0] * np.pi * (width + 2) / width, -tile[..., 1] * np.pi / 2)
        assert np.abs(ours - np.stack([rays[..., 1], rays[..., 0], rays[..., 2]], axis=-1)).max() < 1e-5
    for (inside, where), tile in zip(views["back"], views["tiles"]):  # each seen pixel reads its own ray back
        assert np.abs(where).max() <= 1
        rows, columns = np.divmod(inside, width)
        longitude = (columns + 0.5) / width * 2 * np.pi - np.pi
        latitude = np.pi / 2 - (rows + 0.5) / height * np.pi
        assert np.all(np.hypot(where[:, 0], where[:, 1]) <= np.sqrt(2))
        pick = np.abs(where).max(axis=1) < 1 - 2 / size
        j = np.rint((where[pick, 0] + 1) / 2 * size - 0.5).astype(int)
        i = np.rint((where[pick, 1] + 1) / 2 * size - 0.5).astype(int)
        read = ev.unit_directions(tile[i, j, 0] * np.pi * (width + 2) / width, -tile[i, j, 1] * np.pi / 2)
        assert np.degrees(np.arccos(np.clip(
            (read * ev.unit_directions(longitude[pick], latitude[pick])).sum(axis=-1), -1, 1))).max() < 2.0
    level = views["tiles"][0]  # pitch 0: right is +longitude and down is -latitude (not mirrored)
    assert level[24, 40, 0] > level[24, 8, 0] and level[40, 24, 1] > level[8, 24, 1]


def test_sphere_views_reproduce_a_smooth_whole_pass():
    torch = pytest.importorskip("torch")

    class PixelwiseNet(torch.nn.Module):
        def forward(self, rgb, modal_x):
            return 4 * rgb + modal_x

    height, width = 128, 256
    longitude, latitude = np.meshgrid(
        (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi,
        np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi,
    )
    smooth = torch.as_tensor(ev.unit_directions(longitude, latitude), dtype=torch.float32).permute(2, 0, 1)[None]
    views = ev.sphere_view_grids(height, width, 64)
    whole = ev.predict_with_wrap(PixelwiseNet(), smooth, smooth, 0, "cpu")
    for channels in (None, [0, 2]):
        fused = ev.predict_sphere_views(PixelwiseNet(), (smooth, smooth), views, "cpu", channels)
        assert fused.shape == (1, 3, height, width)
        assert torch.allclose(fused, ev.class_probabilities(whole, channels), atol=0.01)
    fused = ev.predict_sphere_views(PixelwiseNet(), (smooth, smooth), views, "cpu", exp_sum=True)
    assert torch.allclose(fused, whole.exp(), rtol=0.03)


def test_tile_views_are_centred_on_the_tiles_and_as_wide_as_them():
    height, width, size = 128, 256, 64
    windows = ev.erp_tile_windows(height, width, 3, 7, 64, 64, 40, -24)  # SMMF-like, wraps the seam
    layout, fov_deg = ev.tile_view_layout(height, width, windows, (64, 64))
    assert fov_deg == 90 and [pitch for _, pitch in layout] == [45.0] * 7 + [0.0] * 7 + [-45.0] * 7
    views = ev.sphere_view_grids(height, width, size, layout, fov_deg)
    assert views["views_per_pixel"]["min"] >= 2
    for (top, left), (inside, _) in zip(windows, views["back"]):
        rows, columns = np.divmod(inside, width)
        middle = (left + 31) % width  # half a pixel left of the tile's centre line
        assert np.array_equal(np.sort(rows[columns == middle]), np.arange(top, top + 64))
        if top == 32:  # pitch 0: rows within +-33 deg (clear of the view's corners) span exactly the tile's columns
            for row in (40, 63, 87):
                assert np.array_equal(np.sort(columns[rows == row]), np.sort(np.arange(left, left + 64) % width))


def test_pooled_tile_views_reproduce_a_smooth_whole_pass():
    torch = pytest.importorskip("torch")

    class PixelwiseNet(torch.nn.Module):
        def forward(self, rgb, modal_x):
            return 4 * rgb + modal_x

    height, width = 128, 256
    longitude, latitude = np.meshgrid(
        (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi,
        np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi,
    )
    smooth = torch.as_tensor(ev.unit_directions(longitude, latitude), dtype=torch.float32).permute(2, 0, 1)[None]
    windows = ev.erp_tile_windows(height, width, 3, 7, 64, 64, 40, -24)
    views = ev.sphere_view_grids(height, width, 64, *ev.tile_view_layout(height, width, windows, (64, 64)))
    whole = ev.predict_with_wrap(PixelwiseNet(), smooth, smooth, 0, "cpu")
    pooled = ev.input_pool_network(PixelwiseNet(), "avg", "avg")
    fused = ev.predict_sphere_views(pooled, (smooth, smooth), views, "cpu", exp_sum=True)
    assert torch.allclose(fused, whole.exp(), rtol=0.05)


def test_middle_row_views_keep_the_polar_tiles():
    torch = pytest.importorskip("torch")

    class PixelwiseNet(torch.nn.Module):
        def forward(self, rgb, modal_x):
            return 4 * rgb + modal_x

    height, width = 128, 256
    longitude, latitude = np.meshgrid(
        (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi,
        np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi,
    )
    smooth = torch.as_tensor(ev.unit_directions(longitude, latitude), dtype=torch.float32).permute(2, 0, 1)[None]
    windows = ev.erp_tile_windows(height, width, 3, 7, 64, 64, 40, -24)  # rows 0, 32, 64
    middle = [window for window in windows if window[0] == 32]
    polar = [window for window in windows if window[0] != 32]
    views = ev.sphere_view_grids(height, width, 64, *ev.tile_view_layout(height, width, middle, (64, 64)),
                                 require_full=False)
    assert views["views_per_pixel"]["min"] == 0  # the middle-row views do not reach the poles
    pooled = ev.input_pool_network(PixelwiseNet(), "avg", "avg")
    fused = ev.predict_tiles_and_views(pooled, (smooth, smooth), polar, (64, 64), views, "cpu", exp_sum=True)
    pooled_smmf = ev.predict_erp_tiles(pooled, smooth, smooth, windows, (64, 64), "cpu", exp_sum=True)
    rows = torch.zeros(height, dtype=torch.bool)
    rows[32:96] = True  # the middle tile row; outside it only the same polar tiles see a pixel
    assert torch.allclose(fused[:, :, ~rows], pooled_smmf[:, :, ~rows], atol=1e-6)
    whole = ev.predict_with_wrap(PixelwiseNet(), smooth, smooth, 0, "cpu")  # where views and tiles mix
    assert torch.allclose(fused[:, :, rows], whole.exp()[:, :, rows], rtol=0.05)


def test_pole_views_are_the_pole_patches_widened_and_cover_the_sphere_with_the_middle_row():
    torch = pytest.importorskip("torch")

    class PixelwiseNet(torch.nn.Module):
        def forward(self, rgb, modal_x):
            return 4 * rgb + modal_x

    patches = ev.pole_patch_grids(128, 256, 64, (32, 96))  # same orientation as the pole patches
    same = ev.sphere_view_grids(128, 256, 64, ev.POLE_VIEWS, 2 * patches["cap_radius_deg"], require_full=False)
    assert np.allclose(same["tiles"], patches["tiles"], atol=1e-5)
    height, width = 256, 512
    windows = ev.erp_tile_windows(height, width, 3, 7, 136, 136, 90, -54)  # as in the gated end-to-end test
    layout, fov_deg = ev.tile_view_layout(height, width, [w for w in windows if w[0] == 60], (136, 136))
    views = ev.sphere_view_grids(height, width, 136, layout + ev.POLE_VIEWS, fov_deg)  # raises if a pixel is unseen
    assert views["views_per_pixel"]["min"] >= 1 and len(views["layout"]) == 9
    with pytest.raises(ValueError):  # the pole patches' own width (the cap beyond the middle row) leaves a gap
        ev.sphere_view_grids(height, width, 136, layout + ev.POLE_VIEWS, 2 * 60 / height * 180)
    longitude, latitude = np.meshgrid(
        (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi,
        np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi,
    )
    smooth = torch.as_tensor(ev.unit_directions(longitude, latitude), dtype=torch.float32).permute(2, 0, 1)[None]
    whole = ev.predict_with_wrap(PixelwiseNet(), smooth, smooth, 0, "cpu")
    pooled = ev.input_pool_network(PixelwiseNet(), "avg", "avg")
    fused = ev.predict_sphere_views(pooled, (smooth, smooth), views, "cpu", exp_sum=True)
    assert torch.allclose(fused, whole.exp(), rtol=0.05)


def test_sphere_views_at_the_native_size_pass_the_argument_checks(monkeypatch, tmp_path):
    smmf = ["--erp-tiles", "3", "7", "--erp-tile-size", "1080", "1080", "--erp-tile-column-stride", "720",
            "--erp-tile-column-offset", "-652"]
    for name, extra in (("views22", ["--sphere-views", "480"]),
                        ("tileviews", smmf + ["--sphere-views", "1080"]),
                        ("pooledtileviews", smmf + ["--sphere-views", "1080", "--input-pool", "avg", "avg"]),
                        ("pooledmiddleviews", smmf + ["--sphere-views", "1080", "--sphere-views-middle-row",
                                                      "--input-pool", "avg", "avg"]),
                        ("pooledpoleviews", smmf + ["--sphere-views", "1080", "--sphere-views-poles",
                                                    "--input-pool", "avg", "avg"])):
        output = tmp_path / name
        monkeypatch.setattr(sys, "argv", [
            "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
            "--semantic-labels", "unused.json", "--config", str(tmp_path / "missing.json"),
            "--checkpoint", "unused.pth", "--output", str(output), "--height", "2048", "--width", "4096",
            "--erp-tile-exp-sum",
        ] + extra)
        with pytest.raises(Exception):  # gets past the parser (SystemExit is not an Exception) to the config
            ev.main()
        assert output.exists()


@pytest.mark.parametrize("extra", [
    ["--sphere-views", "0"],
    ["--sphere-views", "384", "--erp-tiles", "3", "8", "--erp-tile-size", "384", "512"],  # not square
    ["--sphere-views", "384", "--erp-tiles", "1", "4", "--erp-tile-size", "1024", "1024"],  # 180 degrees wide
    ["--sphere-views", "384", "--erp-tiles", "3", "8", "--erp-tile-size", "512", "512", "--erp-tile-polar-scale",
     "0.5"],
    ["--sphere-views", "384", "--erp-tiles", "3", "8", "--erp-tile-size", "512", "512", "--erp-tile-pole-patches",
     "384"],
    ["--sphere-views", "384", "--input-pool", "avg", "avg"],  # the 22-view layout has no pooled mode
    ["--sphere-views", "385", "--erp-tiles", "3", "8", "--input-pool", "avg", "avg"],  # odd view size
    ["--sphere-views-middle-row", "--erp-tiles", "3", "8", "--erp-tile-size", "512", "512"],
    ["--sphere-views", "384", "--sphere-views-middle-row"],  # the 22-view layout has no tile rows
    ["--sphere-views", "384", "--sphere-views-middle-row", "--erp-tiles", "2", "8", "--erp-tile-size", "512", "512"],
    ["--sphere-views-poles", "--erp-tiles", "3", "8", "--erp-tile-size", "512", "512"],
    ["--sphere-views", "384", "--sphere-views-poles"],
    ["--sphere-views", "384", "--erp-tiles", "3", "8", "--erp-tile-size", "512", "512", "--sphere-views-poles",
     "--sphere-views-middle-row"],
    ["--sphere-views", "384", "--erp-tiles", "3", "7", "--erp-tile-size", "512", "512", "--sphere-views-poles"],  # gap
    ["--sphere-views", "384", "--erp-tile-size", "384", "384", "--erp-tile-sliding-stride", "320"],
    ["--sphere-views", "384", "--polar-strips", "32"],
    ["--sphere-views", "384", "--polar-caps", "384"],
    ["--sphere-views", "384", "--infer-scale", "0.5"],
])
def test_bad_sphere_views_are_rejected_before_loading(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, "argv", [
        "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
        "--semantic-labels", "unused.json", "--config", "unused.json",
        "--checkpoint", "unused.pth", "--output", str(tmp_path / "output"),
    ] + extra)
    with pytest.raises(SystemExit) as error:
        ev.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_input_pool_modes_match_the_resizes():
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    import torch.nn.functional as F

    image = np.random.default_rng(0).random((6, 10, 3)).astype(np.float32)
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None]

    def hwc(t):
        return t[0].permute(1, 2, 0).numpy()

    avg = ev.pool_half(tensor, "avg")  # SMMF050's RGB: cv2.INTER_LINEAR by 0.5; also bilinear by 0.5
    assert np.allclose(hwc(avg), cv2.resize(image, (5, 3), interpolation=cv2.INTER_LINEAR), atol=1e-6)
    assert torch.allclose(avg, F.interpolate(tensor, scale_factor=0.5, mode="bilinear", align_corners=False), atol=1e-6)
    assert np.array_equal(hwc(ev.pool_half(tensor, "nearest")), cv2.resize(image, (5, 3), interpolation=cv2.INTER_NEAREST))
    assert np.array_equal(hwc(ev.pool_half(tensor, "max")), image.reshape(3, 2, 5, 2, 3).max(axis=(1, 3)))
    with pytest.raises(ValueError):
        ev.pool_half(tensor, "bilinear")


def test_input_pool_network_pools_in_front_and_upsamples_behind():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    class PixelwiseNet(torch.nn.Module):
        def forward(self, rgb, modal_x, *rest):
            self.rest = rest
            return 4 * rgb + modal_x

    net = PixelwiseNet()
    net.gate_log = [3, 4]
    pooled = ev.input_pool_network(net, "avg", "nearest")
    rgb, modal_x = torch.randn(1, 3, 8, 12), torch.randn(1, 3, 8, 12)
    expected = F.interpolate(4 * F.avg_pool2d(rgb, 2) + modal_x[:, :, ::2, ::2], size=(8, 12), mode="bilinear",
                             align_corners=False)
    assert torch.allclose(pooled(rgb, modal_x, "lat"), expected) and net.rest == ("lat",)
    assert pooled.gate_log is net.gate_log
    with pytest.raises(ValueError):
        pooled(torch.randn(1, 3, 7, 12), torch.randn(1, 3, 7, 12))


def test_pooled_smmf_is_smmf_on_the_pooled_panorama():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    class PixelwiseNet(torch.nn.Module):
        def forward(self, rgb, modal_x):
            return 4 * rgb + modal_x

    height, width = 128, 256
    longitude, latitude = np.meshgrid(
        (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi,
        np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi,
    )
    smooth = torch.as_tensor(ev.unit_directions(longitude, latitude), dtype=torch.float32).permute(2, 0, 1)[None]
    windows = ev.erp_tile_windows(height, width, 3, 7, 64, 64, 40, -24)  # SMMF-like: even origins, wraps the seam
    half_windows = ev.erp_tile_windows(height // 2, width // 2, 3, 7, 32, 32, 20, -12)
    assert half_windows == [(top // 2, left // 2) for top, left in windows]
    small = ev.pool_half(smooth, "avg")
    for (top, left), (half_top, half_left) in zip(windows, half_windows):  # pooling a window = cutting the pooled one
        window = smooth[:, :, top:top + 64, torch.arange(left, left + 64) % width]
        assert torch.equal(ev.pool_half(window, "avg"),
                           small[:, :, half_top:half_top + 32, torch.arange(half_left, half_left + 32) % (width // 2)])
    pooled = ev.input_pool_network(PixelwiseNet(), "avg", "avg")
    fused = ev.predict_erp_tiles(pooled, smooth, smooth, windows, (64, 64), "cpu", exp_sum=True)
    smmf050 = F.interpolate(ev.predict_erp_tiles(PixelwiseNet(), small, small, half_windows, (32, 32), "cpu",
                                                 exp_sum=True), size=(height, width), mode="bilinear", align_corners=False)
    # Only the order of upsampling and fusion differs: each window's logits are upsampled inside the window
    # (exp after upsampling) instead of the fused exp map across window edges. Coarse here (32 px windows).
    relative = (fused - smmf050).abs() / smmf050
    assert relative.max() < 0.08 and relative.median() < 0.005
    assert (fused.argmax(1) == smmf050.argmax(1)).float().mean() > 0.999


def test_pooled_pole_patches_and_pooled_zones_share_the_middle_row():
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
    windows = ev.erp_tile_windows(height, width, 3, 8, 64, 64)  # rows 0, 32, 64: the middle row is 32..96
    patches = ev.pole_patch_grids(height, width, size, (32, 96))
    pooled = ev.input_pool_network(PixelwiseNet(), "avg", "avg")
    rows = torch.zeros(height, dtype=torch.bool)
    rows[32:96] = True
    expected = ev.class_probabilities(ev.predict_with_wrap(PixelwiseNet(), smooth, smooth, 0, "cpu"))
    warped = ev.predict_pole_patches(pooled, (smooth, smooth), windows, (64, 64), (32, 96), patches, "cpu")
    zones = ev.predict_polar_scaled_tiles(pooled, (smooth, smooth), (smooth, smooth), windows, (64, 64), (32, 96),
                                          "cpu")  # --erp-tile-polar-scale 1: ERP polar rows at the evaluation size
    assert torch.equal(warped[:, :, rows], zones[:, :, rows])  # the same pooled middle windows
    for fused in (warped, zones):  # on a smooth panorama both reproduce the whole pass outside the middle row
        assert torch.allclose(fused[:, :, ~rows], expected[:, :, ~rows], atol=0.03)  # 2x upsampling at the edges


def test_smmf_layout_with_input_pool_passes_the_argument_checks(monkeypatch, tmp_path):
    for modes, polar in ((("avg", "avg"), []), (("max", "max"), []), (("avg", "nearest"), []),
                         (("avg", "avg"), ["--erp-tile-pole-patches", "1080"]),
                         (("avg", "avg"), ["--erp-tile-polar-scale", "1"]),
                         (("avg", "avg"), ["--erp-tile-polar-scale", "1", "--erp-tile-polar-squeeze"])):
        output = tmp_path / ("output" + "_".join(modes) + "_".join(polar))
        monkeypatch.setattr(sys, "argv", [
            "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
            "--semantic-labels", "unused.json", "--config", str(tmp_path / "missing.json"),
            "--checkpoint", "unused.pth", "--output", str(output), "--height", "2048", "--width", "4096",
            "--erp-tiles", "3", "7", "--erp-tile-size", "1080", "1080", "--erp-tile-column-stride", "720",
            "--erp-tile-column-offset", "-652", "--erp-tile-exp-sum", "--input-pool", *modes, *polar,
        ])
        with pytest.raises(Exception):  # gets past the parser (SystemExit is not an Exception) to the config
            ev.main()
        assert output.exists()


@pytest.mark.parametrize("extra", [
    ["--input-pool", "avg", "avg"],
    ["--erp-tiles", "3", "8", "--input-pool", "bilinear", "avg"],
    ["--erp-tiles", "3", "8", "--input-pool", "avg", "avg", "--infer-scale", "0.5"],
    ["--erp-tiles", "3", "8", "--input-pool", "avg", "avg", "--erp-tile-polar-scale", "0.5"],
    ["--erp-tiles", "3", "8", "--input-pool", "avg", "avg", "--erp-tile-pole-patches", "385"],  # odd patch
    ["--erp-tiles", "3", "8", "--input-pool", "avg", "avg", "--erp-tile-polar-scale", "1", "--erp-tile-polar-strips"],
    ["--erp-tiles", "3", "8", "--input-pool", "avg", "avg", "--erp-tile-polar-scale", "0.5",
     "--erp-tile-polar-squeeze"],
    ["--erp-tiles", "3", "8", "--erp-tile-polar-scale", "1.5"],
    ["--erp-tiles", "3", "8", "--input-pool", "avg", "avg", "--polar-caps", "384"],
    ["--erp-tiles", "3", "7", "--input-pool", "avg", "avg"],  # columns start at 0, 292, 585 ...: odd
    ["--erp-tile-size", "384", "384", "--erp-tile-sliding-stride", "320", "--input-pool", "avg", "avg"],
])
def test_bad_input_pools_are_rejected_before_loading(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, "argv", [
        "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
        "--semantic-labels", "unused.json", "--config", "unused.json",
        "--checkpoint", "unused.pth", "--output", str(tmp_path / "output"),
    ] + extra)
    with pytest.raises(SystemExit) as error:
        ev.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


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


def test_gores_are_equal_area_and_read_back_their_own_pixels():
    height, width = 128, 256
    gores = ev.gore_grids(height, width, 7)
    assert gores["tiles"].shape == (7, 128, 42, 2) and gores["masks"].all()  # 38 columns plus 2 of context a side
    assert gores["labels"] == ["gore"] * 7 and gores["starts"] == [0] * 7 and gores["gores_per_pixel"]["min"] >= 1
    assert gores["centres_deg"][0] == 0 and abs(gores["centres_deg"][1] - 360 / 7) < 1e-9
    longitude = gores["tiles"][..., 0] * np.pi * (width + 2) / width
    latitude = -gores["tiles"][..., 1] * np.pi / 2
    erp_latitude = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    assert np.allclose(latitude[:, :, 0], erp_latitude[None], atol=1e-6)  # a gore row keeps its ERP latitude
    for row in (0, 30, 64, 100):  # a gore row is the ERP row compressed by cos(latitude): equal area everywhere
        step = np.diff(np.unwrap(longitude[0, row]))
        assert np.allclose(step, 2 * np.pi / (width * np.cos(erp_latitude[row])), rtol=1e-5)
    assert longitude[0, 64, 20] < 0 < longitude[0, 64, 21]  # the central meridian splits the two middle columns
    for kwargs in ({}, {"fill": "petal"}, {"hemisphere_margin_rows": 8, "offset_set": True}):
        grids = ev.gore_grids(height, width, 7, **kwargs)
        for tile, (inside, where) in zip(grids["tiles"], grids["back"]):  # each covered pixel reads its own ray
            assert np.abs(where).max() <= 1
            rows, columns = np.divmod(inside, width)
            pixel = ev.unit_directions((columns + 0.5) / width * 2 * np.pi - np.pi, erp_latitude[rows])
            u = (where[:, 0] + 1) / 2 * grids["columns"] - 0.5
            v = (where[:, 1] + 1) / 2 * grids["rows"] - 0.5
            pick = (u > 0) & (u < grids["columns"] - 1)
            i, j = np.rint(v[pick]).astype(int), np.rint(u[pick]).astype(int)
            read = ev.unit_directions(tile[i, j, 0] * np.pi * (width + 2) / width, -tile[i, j, 1] * np.pi / 2)
            apart = np.arccos(np.clip((read * pixel[pick]).sum(axis=-1), -1, 1))
            assert np.all(apart <= 0.75 * 2 * np.pi / (width * np.cos(erp_latitude[rows[pick]])) + 1e-6)
    petals = ev.gore_grids(height, width, 7, fill="petal")
    assert petals["gores_per_pixel"] == {"min": 1, "max": 1, "mean": 1.0}  # the petals tile the sphere
    assert 2 / np.pi < petals["masks"].mean() < 0.8  # a petal plus its 3-pixel margin, out of the rectangle
    assert petals["masks"][0, 64].all() and not petals["masks"][0, 20].all()  # whole at the equator, cut at 61 deg
    halves = ev.gore_grids(height, width, 7, hemisphere_margin_rows=8, offset_set=True)
    assert halves["tiles"].shape == (28, 72, 42, 2) and halves["gores_per_pixel"]["min"] >= 2
    assert halves["labels"] == ["north petal"] * 14 + ["south petal"] * 14 and halves["starts"] == [0] * 14 + [56] * 14
    assert len(halves["centres_deg"]) == 14 and abs(halves["centres_deg"][7] - 180 / 7) < 1e-9
    assert ev.gate_rows(None, gores=halves) == halves["labels"]
    assert ev.gore_rectangle(2048, 4096, 7) == (2048, 590) and ev.gore_rectangle(2048, 4096, 7, 114) == (1138, 590)
    with pytest.raises(ValueError):
        ev.gore_grids(height, width, 1)
    with pytest.raises(ValueError):
        ev.gore_grids(127, width, 7, hemisphere_margin_rows=8)
    with pytest.raises(ValueError):
        ev.gore_grids(height, width, 7, fill="half")


def test_gores_reproduce_a_smooth_whole_pass():
    torch = pytest.importorskip("torch")

    class PixelwiseNet(torch.nn.Module):
        def forward(self, rgb, modal_x):
            return 4 * rgb + modal_x

    height, width = 128, 256
    longitude, latitude = np.meshgrid(
        (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi,
        np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi,
    )
    smooth = torch.as_tensor(ev.unit_directions(longitude, latitude), dtype=torch.float32).permute(2, 0, 1)[None]
    whole = ev.predict_with_wrap(PixelwiseNet(), smooth, smooth, 0, "cpu")
    for kwargs in ({}, {"fill": "petal"}, {"hemisphere_margin_rows": 8}, {"offset_set": True, "fill": "petal"}):
        gores = ev.gore_grids(height, width, 7, **kwargs)
        for channels in (None, [0, 2]):
            fused = ev.predict_gores(PixelwiseNet(), (smooth, smooth), gores, "cpu", channels)
            assert fused.shape == (1, 3, height, width)
            assert torch.allclose(fused, ev.class_probabilities(whole, channels), atol=0.02)
        fused = ev.predict_gores(PixelwiseNet(), (smooth, smooth), gores, "cpu", exp_sum=True)
        assert torch.allclose(fused, whole.exp(), rtol=0.05)
    pooled = ev.input_pool_network(PixelwiseNet(), "avg", "avg")  # even gore sizes: 128 x 42 and 72 x 42
    for kwargs in ({}, {"hemisphere_margin_rows": 8, "fill": "petal"}):
        fused = ev.predict_gores(pooled, (smooth, smooth), ev.gore_grids(height, width, 7, **kwargs), "cpu",
                                 exp_sum=True)
        assert torch.allclose(fused[:, :, 8:-8], whole.exp()[:, :, 8:-8], rtol=0.05)
        assert torch.allclose(fused, whole.exp(), rtol=0.15)  # 38-column gores: a pooled pixel at 84 deg spans 60 deg


def test_gore_runs_pass_the_argument_checks(monkeypatch, tmp_path):
    for name, extra in (("gores", ["--erp-gores", "7"]),
                        ("petals", ["--erp-gores", "7", "--erp-gore-hemispheres", "10", "--erp-gore-fill", "petal"]),
                        ("pooledgores", ["--erp-gores", "7", "--input-pool", "avg", "avg"]),
                        ("pooledoffsetpetals", ["--erp-gores", "7", "--erp-gore-hemispheres", "11.25",  # 64 rows
                                                "--erp-gore-offset-set", "--input-pool", "avg", "avg"])):
        output = tmp_path / name
        monkeypatch.setattr(sys, "argv", [
            "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
            "--semantic-labels", "unused.json", "--config", str(tmp_path / "missing.json"),
            "--checkpoint", "unused.pth", "--output", str(output), "--height", "1024", "--width", "2048",
            "--erp-tile-exp-sum",
        ] + extra)
        with pytest.raises(Exception):  # gets past the parser (SystemExit is not an Exception) to the config
            ev.main()
        assert output.exists()


@pytest.mark.parametrize("extra", [
    ["--erp-gores", "1"],
    ["--erp-gores", "7", "--erp-tiles", "3", "7", "--erp-tile-size", "512", "512"],
    ["--erp-gores", "7", "--erp-tile-size", "384", "384", "--erp-tile-sliding-stride", "320"],
    ["--erp-gores", "7", "--sphere-views", "384"],
    ["--erp-gores", "7", "--polar-strips", "32"],
    ["--erp-gores", "7", "--polar-caps", "384"],
    ["--erp-gores", "7", "--infer-scale", "0.5"],
    ["--erp-gore-hemispheres", "10"],
    ["--erp-gore-fill", "petal"],
    ["--erp-gore-offset-set"],
    ["--erp-gores", "7", "--erp-gore-hemispheres", "90"],
    ["--erp-gores", "7", "--erp-gore-hemispheres", "10", "--height", "1023", "--width", "2046"],
    ["--erp-gores", "7", "--erp-gore-hemispheres", "0.1", "--height", "1028", "--width", "2056", "--input-pool",
     "avg", "avg"],  # petals of 515 rows cannot be pooled
])
def test_bad_gores_are_rejected_before_loading(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, "argv", [
        "eval_pano_transfer.py", "--stanford-root", str(tmp_path),
        "--semantic-labels", "unused.json", "--config", "unused.json",
        "--checkpoint", "unused.pth", "--output", str(tmp_path / "output"),
    ] + extra)
    with pytest.raises(SystemExit) as error:
        ev.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()
