"""Unit checks for the training-free attention cap, tangent offsets and coplanar error share."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import cross_projection as cp  # noqa: E402
import eval_pano_scope as es  # noqa: E402
import eval_pano_transfer as ev  # noqa: E402


def frozen_attention_class():
    ev.import_frozen(ev.DEFAULT_SOURCE_ROOT)  # also keeps bytecode out of the frozen tree
    from models.encoders.dual_segformer import Attention

    return Attention


def test_training_corner_angle():
    assert abs(es.TRAINING_CORNER_DEG - 40.6) < 0.05


def test_token_centres_follow_the_convolution_arithmetic():
    rows, columns = np.indices((32, 64))
    pixels = np.stack([rows, columns], axis=-1)
    tokens = es.token_centres(pixels, 8, 16).reshape(8, 16, 2)  # stride 4: token i on pixel 4i
    assert (tokens[..., 0] == 4 * np.arange(8)[:, None]).all()
    assert (tokens[..., 1] == 4 * np.arange(16)[None, :]).all()
    keys = es.token_centres(pixels, 8, 16, 4).reshape(2, 4, 2)  # 4x4 cells centred on token 1.5
    assert (keys[..., 0] == 16 * np.arange(2)[:, None] + 6).all()
    assert (keys[..., 1] == 16 * np.arange(4)[None, :] + 6).all()
    with pytest.raises(ValueError):
        es.token_centres(pixels, 7, 16)


def test_attention_cap_keeps_the_keys_within_the_radius():
    torch = pytest.importorskip("torch")
    height, width, pad, stride, sr = 64, 128, 32, 4, 2
    outside = es.AttentionCap(height, width, pad, 30.0)(16, 48, sr, "cpu").numpy()
    directions = cp.erp_directions(height, width)

    def direction(row, padded_column):
        return directions[row, (padded_column - pad) % width]

    queries = np.array([direction(stride * r, stride * c) for r in range(16) for c in range(48)])
    keys = np.array([direction(stride * (sr * r) + stride // 2, stride * (sr * c) + stride // 2)
                     for r in range(8) for c in range(24)])
    angle = np.degrees(np.arccos(np.clip(queries @ keys.T, -1, 1)))
    far = np.abs(angle - 30.0) > 1e-3
    assert (outside[far] == (angle[far] > 30.0)).all()
    assert (~outside).any(axis=1).all() and outside.any()
    assert es.AttentionCap(height, width, pad, 180.0)(16, 48, sr, "cpu") is None
    assert torch.is_tensor(outside) is False


def test_capped_attention_matches_the_frozen_attention():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    attention = frozen_attention_class()(32, num_heads=2, sr_ratio=2).eval()
    x = torch.randn(1, 8 * 12, 32)
    with torch.no_grad():
        expected = attention(x, 8, 12)
        plain = es._attention_forward(attention, "rgb", 1, None, None, x, 8, 12)
        keep = torch.arange(8 * 12) % 24  # one kept key per query
        outside = torch.ones(8 * 12, 24, dtype=torch.bool)
        outside[torch.arange(8 * 12), keep] = False
        masked = es._attention_forward(attention, "rgb", 1, lambda *_: outside, None, x, 8, 12)
        reduced = attention.norm(attention.sr(x.permute(0, 2, 1).reshape(1, 32, 8, 12)).reshape(1, 32, -1).permute(0, 2, 1))
        value = attention.kv(reduced)[..., 32:]
    assert torch.equal(plain, expected)
    assert torch.allclose(masked, attention.proj(value[:, keep]), atol=1e-6)


def tangent_tap_directions(conv, map_size, erp_size, wrap_pad):
    """Independent reference: inverse gnomonic projection (Snyder) of each tap's plane point."""
    height, width = erp_size
    stride = height // map_size[0]
    k, t, p = conv.kernel_size[0], conv.stride[0], conv.padding[0]
    out_h = (map_size[0] + 2 * p - k) // t + 1
    out_w = (map_size[1] + 2 * p - k) // t + 1
    centre = (k - 1) / 2
    directions = np.zeros((out_h, out_w, k, k, 3))
    for i in range(out_h):
        theta0 = (0.5 - stride * (t * i - p + centre) / height) * np.pi
        for j in range(out_w):
            phi0 = (stride * (t * j - p + centre) - wrap_pad) / width * 2 * np.pi - np.pi
            for ky in range(k):
                for kx in range(k):
                    x = np.tan((kx - centre) * stride * 2 * np.pi / width)
                    y = -np.tan((ky - centre) * stride * np.pi / height)
                    rho = np.hypot(x, y)
                    if rho == 0:
                        theta, phi = theta0, phi0
                    else:
                        c = np.arctan(rho)
                        theta = np.arcsin(np.cos(c) * np.sin(theta0) + y * np.sin(c) * np.cos(theta0) / rho)
                        phi = phi0 + np.arctan2(
                            x * np.sin(c), rho * np.cos(theta0) * np.cos(c) - y * np.sin(theta0) * np.sin(c))
                    directions[i, j, ky, kx] = [np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi), np.sin(theta)]
    return directions


@pytest.mark.parametrize("kernel, stride, padding, map_stride", [(3, 1, 1, 4), (3, 2, 1, 4), (2, 2, 0, 4), (7, 4, 3, 1)])
def test_tangent_taps_sample_the_tangent_plane(kernel, stride, padding, map_stride):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    height, width, pad = 64, 128, 32
    map_size = (height // map_stride, (width + 2 * pad) // map_stride)
    conv = torch.nn.Conv2d(1, kernel * kernel, kernel, stride=stride, padding=padding, bias=False)
    with torch.no_grad():  # output channel n reads tap n alone
        conv.weight.copy_(torch.eye(kernel * kernel).reshape(kernel * kernel, 1, kernel, kernel))
    probe = np.array([0.3, -0.5, 0.8])  # a smooth function of direction, periodic across the seam
    cells = es.padded_erp_directions(height, width, pad)[::map_stride, ::map_stride]
    signal = torch.from_numpy((cells @ probe).astype(np.float32))[None, None]
    es.patch_tangent_convs(type("Net", (), {"backbone": torch.nn.Sequential(conv)})(),
                           es.TangentOffsets(height, width, pad))
    with torch.no_grad():
        sampled = conv(signal)[0].numpy().transpose(1, 2, 0).reshape(-1, kernel, kernel)
    expected = (tangent_tap_directions(conv, map_size, (height, width), pad) @ probe).reshape(sampled.shape)
    out_h = sampled.shape[0] // ((map_size[1] + 2 * padding - kernel) // stride + 1)
    centre_rows = (stride * np.arange(out_h) - padding + (kernel - 1) / 2) * map_stride
    temperate = np.repeat(np.abs(90 - centre_rows * 180 / height) <= 55, sampled.shape[0] // out_h)
    cell_deg = map_stride * 360 / width
    assert np.abs(sampled - expected)[temperate].max() < 0.02 * max(1.0, cell_deg / 11.25) ** 2


def test_tangent_offsets_vanish_on_the_kernel_axes_at_the_equator():
    torch = pytest.importorskip("torch")
    height, width, pad = 64, 128, 32
    conv = torch.nn.Conv2d(4, 4, 3, padding=1, groups=4)
    offsets = es.tangent_offsets(conv, (16, 48), (height, width), pad)[0].reshape(3, 3, 2, 16, 48)
    kept = slice(pad // 4 + 1, pad // 4 + width // 4 - 1)  # away from the seam and the padding
    equator = height // 2 // 4
    axes = [(1, 0), (1, 2), (0, 1), (2, 1), (1, 1)]
    for ky, kx in axes:
        assert np.abs(offsets[ky, kx, :, equator, kept]).max() < 1e-4
    assert np.abs(offsets[0, 0, 0, equator, kept]).min() > 1e-3  # diagonal taps bend off the ERP grid
    polar = offsets[1, 2, 1, 2, pad // 4:pad // 4 + width // 4 - 3]  # east tap at 67.5 degrees, seam excluded
    reach = np.degrees(np.arctan2(np.tan(np.radians(11.25)), np.cos(np.radians(67.5)))) / 11.25
    assert np.abs(polar - (reach - 1)).max() < 1e-4 and reach > 2


def test_patching_covers_the_wide_encoder_convs_and_every_attention():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    config = json.loads((ev.REPO_ROOT / "0927调参结果" / "configs" / "relplus.json").read_text(encoding="utf-8"))
    config.update(backbone="mit_b0", decoder_embed_dim=32, image_height=96, image_width=96)
    from easydict import EasyDict

    network = ev.import_frozen(ev.DEFAULT_SOURCE_ROOT)["EncoderDecoder"](
        cfg=EasyDict(config), criterion=None, norm_layer=torch.nn.BatchNorm2d)
    wide = [name for name, module in network.backbone.named_modules()
            if isinstance(module, torch.nn.Conv2d) and max(module.kernel_size) > 1]
    names = es.patch_tangent_convs(network, es.TangentOffsets(64, 128, 32))
    assert names == wide
    assert any(".sr" in name for name in names) and any("FFMs" in name for name in names)
    assert any("extra_patch_embed1" in name for name in names) and any("dwconv" in name for name in names)
    assert not any(isinstance(module, torch.nn.Conv2d) and "forward" in vars(module)
                   for module in network.decode_head.modules())
    assert es.patch_attention(network) == 2 * sum(network.backbone.depths)


def test_geometry_blind_marks_flat_walls_and_not_depth_edges():
    height, width = 128, 256
    directions = cp.erp_directions(height, width)
    ahead = directions[..., 1]
    with np.errstate(divide="ignore"):
        range_m = np.where(ahead > 0.2, 2.0 / ahead, 0.0)  # a wall 2 m ahead
    door = (np.abs(directions[..., 0] / np.maximum(ahead, 1e-6)) < 0.25) & (np.abs(directions[..., 2] / np.maximum(ahead, 1e-6)) < 0.4)
    range_m = np.where(door & (ahead > 0.2), 2.1 / np.maximum(ahead, 1e-6), range_m)  # recessed 10 cm
    rows, columns = np.nonzero(range_m > 0)
    full, blind = es.geometry_blind(range_m.astype(np.float32), 7, 0.005, (rows, columns))
    blind_map = np.zeros((height, width), bool)
    blind_map[rows[blind], columns[blind]] = True
    edge = door & ~cv2.erode(door.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    centre = (height // 2, width // 2)
    assert blind_map[centre]  # door interior: a plane, no depth edge
    assert blind_map[height // 2, width // 2 + 40]  # plain wall
    assert not blind_map[edge].any()
    assert full[(rows == centre[0]) & (columns == centre[1])].all()


def write_fake_eval(tmp_path, config_name="relplus.json"):
    """mit_b0 checkpoint, config, labels and one 64x128 panorama; returns the shared CLI arguments."""
    torch = pytest.importorskip("torch")
    import eval_pano_crops as ec
    from test_cross_projection_synthetic import render_erp_raw_depth

    config = json.loads((ev.REPO_ROOT / "0927调参结果" / "configs" / config_name).read_text(encoding="utf-8"))
    config.update(backbone="mit_b0", decoder_embed_dim=32, image_height=96, image_width=96)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    frozen = ec.import_frozen_crops(ev.DEFAULT_SOURCE_ROOT)
    torch.manual_seed(0)
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
    return [
        "--stanford-root", str(tmp_path / "s2d"), "--semantic-labels", str(tmp_path / "semantic_labels.json"),
        "--areas", "area_5a", "--config", str(tmp_path / "config.json"),
        "--checkpoint", str(tmp_path / "checkpoint.pth"), "--height", "64", "--width", "128",
        "--wrap-pad", "32", "--device", "cpu",
    ]


def test_scope_probes_run_end_to_end(monkeypatch, tmp_path):
    pytest.importorskip("torchvision")
    shared = write_fake_eval(tmp_path)
    monkeypatch.setattr(sys, "argv", ["eval_pano_transfer.py"] + shared + ["--output", str(tmp_path / "plain")])
    assert ev.main() == 0
    plain = np.loadtxt(tmp_path / "plain" / "confusion_matrix.csv", delimiter=",")

    assert es.main(shared + ["--attention-cap-deg", "180", "--output", str(tmp_path / "open")]) == 0
    assert (np.loadtxt(tmp_path / "open" / "confusion_matrix.csv", delimiter=",") == plain).all()
    assert ev.predict_with_wrap.__module__ == "eval_pano_transfer"  # the plain functions are back

    # 128x256: stage-1 key cells of 45 degrees, so every query keeps a key within 40.6 degrees
    assert es.main(shared + ["--height", "128", "--width", "256", "--attention-cap-deg", "--tangent-offsets",
                             "--coplanar-share", "--output", str(tmp_path / "probes")]) == 0
    report = json.loads((tmp_path / "probes" / "metrics.json").read_text())
    probes = report["scope_probes"]
    assert probes["attention_cap_deg"] == es.TRAINING_CORNER_DEG and probes["wrap_pad"] == 32
    assert probes["erp_size"] == [128, 256] and len(probes["tangent_offset_convs"]) > 0
    assert "attention cap 40.6" in report["setting"]
    label = cv2.imread(str(tmp_path / "s2d" / "area_5a" / "pano" / "semantic" / "c_semantic.png"))
    lookup = ev.label_lookup(tmp_path / "semantic_labels.json")
    label = ev.decode_semantic(cv2.resize(label, (256, 128), interpolation=cv2.INTER_NEAREST), lookup)
    counts = probes["coplanar_share"]["counts"]
    for name in es.COPLANAR_CLASSES:
        groups = counts[name]
        assert groups["all"]["with_depth"] <= int((label == ev.EXPECTED_CLASSES.index(name) - 1).sum())
        assert groups["predicted_wall"]["with_depth"] + groups["correct"]["with_depth"] <= groups["all"]["with_depth"]
        assert all(values["blind"] <= values["with_depth"] for values in groups.values())
    assert counts["door"]["all"]["with_depth"] > 0


def test_scope_needs_a_probe():
    with pytest.raises(SystemExit) as error:
        es.main(["--output", "unused"])
    assert error.value.code == 2
