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
