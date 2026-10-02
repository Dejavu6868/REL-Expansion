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
