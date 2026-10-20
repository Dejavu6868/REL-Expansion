#!/usr/bin/env python3
"""Training-free probes of the whole-panorama evaluation: attention cap, tangent offsets, coplanar errors.

Runs ``eval_pano_transfer.py`` unchanged, with every argument not listed here passed to it, and
adds up to three eval-only changes. Their settings and results go into its ``metrics.json``
under ``scope_probes``.

* ``--attention-cap-deg R``: each MiT self-attention query, in both streams and all four
  stages, sees only keys within R degrees of geodesic angle, as a pinhole frame at the
  training FOV bounds what a query sees in training. Given without a value, R is 40.6 degrees,
  the centre-to-corner angle of the median 62.48-degree training frame. A token at cumulative
  stride S sits on input pixel S*i (the convolution arithmetic), and a spatial-reduction key
  of ratio r on the centre of its r x r tokens. Keys in the wrap padding are the duplicates
  the plain evaluation also sees. R = 180 masks nothing and reproduces the plain evaluation.
* ``--tangent-offsets``: each encoder convolution wider than 1x1 (patch embeddings, Mix-FFN
  depthwise convs, spatial-reduction convs, FFM depthwise convs) samples its taps on the
  tangent plane at its output's direction (Tateno et al., ECCV 2018), through
  ``torchvision.ops.deform_conv2d`` with fixed offsets and the trained weights. A tap one cell
  from the centre sits tan(one cell's equatorial angle) away on the plane, so taps on the
  kernel axes are unchanged at the equator. Taps that cross the 180-degree seam read the kept
  panorama columns, and taps past a pole cross it.
* ``--coplanar-share``: for door and window pixels with depth around them, the share that is
  geometry-blind (one plane fits the camera-centred points of the surrounding window to within
  a fraction of their range, so no depth edge or shape change is near), among all of them,
  those predicted as wall and those predicted correctly. Predictions are unchanged.

Left global: the FRM channel weights pool over the whole map and the FFM cross-attention
builds a whole-map channel context, so the cap does not reach them. The decoder has 1x1
convolutions only.
"""

import argparse
import functools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import cross_projection as cp
import eval_pano_crops as ec
import eval_pano_transfer as ev


# Centre-to-corner angle of a square pinhole frame at the training median FOV.
TRAINING_CORNER_DEG = float(np.degrees(np.arctan(np.sqrt(2) * np.tan(np.radians(ec.CROP_FOV_DEG / 2)))))
CAP_SCOPE = "MiT self-attention, both streams, stages 1-4 (FRM channel pooling and FFM channel context stay global)"
COPLANAR_CLASSES = ("door", "window")


def padded_erp_directions(height, width, wrap_pad):
    """Unit direction of each pixel of the circularly padded ERP network input."""
    directions = cp.erp_directions(height, width)
    if wrap_pad:
        directions = np.concatenate([directions[:, -wrap_pad:], directions, directions[:, :wrap_pad]], axis=1)
    return directions


def token_centres(pixels, height, width, reduction=1):
    """``pixels`` (one row per input pixel) at the centre of each token of a height x width map,
    or of each reduction x reduction cell of it, in row-major order."""
    stride = pixels.shape[0] // height
    if stride * height != pixels.shape[0] or stride * width != pixels.shape[1]:
        raise ValueError("a {}x{} token map does not tile the {}x{} input".format(
            height, width, pixels.shape[0], pixels.shape[1]))
    if height % reduction or width % reduction or stride * (reduction - 1) % 2:
        raise ValueError("reduction {} has no centre pixel on a stride-{} map".format(reduction, stride))
    step = stride * reduction
    offset = stride * (reduction - 1) // 2
    rows = np.arange(height // reduction) * step + offset
    columns = np.arange(width // reduction) * step + offset
    return pixels[rows][:, columns].reshape(-1, pixels.shape[2])


def attention_modules(backbone):
    """(stream, stage, Attention) of every MiT block; an RGB-only backbone has no X stream."""
    for stage in range(1, 5):
        for stream, prefix in (("rgb", "block"), ("x", "extra_block")):
            for block in getattr(backbone, prefix + str(stage), ()):
                yield stream, stage, block.attn


def _attention_forward(module, stream, stage, outside, record, x, H, W):
    """The frozen ``Attention.forward`` with optional key masking and a look at the probabilities."""
    B, N, C = x.shape
    q = module.q(x).reshape(B, N, module.num_heads, C // module.num_heads).permute(0, 2, 1, 3)
    if module.sr_ratio > 1:
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_ = module.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
        x_ = module.norm(x_)
        kv = module.kv(x_).reshape(B, -1, 2, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
    else:
        kv = module.kv(x).reshape(B, -1, 2, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]

    attn = (q @ k.transpose(-2, -1)) * module.scale
    mask = outside(H, W, module.sr_ratio, attn.device) if outside else None
    if mask is not None:
        attn.masked_fill_(mask, float("-inf"))
    attn = attn.softmax(dim=-1)
    if record:
        record(stream, stage, attn, H, W, module.sr_ratio)
    attn = module.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = module.proj(x)
    x = module.proj_drop(x)
    return x


def patch_attention(network, outside=None, record=None):
    """Swap every MiT attention's forward for the frozen one with two hooks; returns the count.

    ``outside(H, W, sr, device)`` gives the (queries, keys) boolean mask of keys to drop, or None;
    ``record(stream, stage, probabilities, H, W, sr)`` sees each softmax output.
    """
    count = 0
    for stream, stage, attention in attention_modules(network.backbone):
        attention.forward = functools.partial(_attention_forward, attention, stream, stage, outside, record)
        count += 1
    return count


class AttentionCap:
    """Per token map, the keys beyond ``cap_deg`` of each query on the padded ERP input."""

    def __init__(self, height, width, wrap_pad, cap_deg):
        self.directions = padded_erp_directions(height, width, wrap_pad)
        self.cosine = np.cos(np.radians(cap_deg))
        self.masks = {}

    def __call__(self, H, W, sr, device):
        key = (H, W, sr, str(device))
        if key not in self.masks:
            import torch

            queries = torch.from_numpy(token_centres(self.directions, H, W)).float().to(device)
            keys = torch.from_numpy(token_centres(self.directions, H, W, sr)).float().to(device)
            outside = (queries @ keys.T) < self.cosine
            if bool(outside.all(dim=1).any()):
                raise ValueError("the attention cap leaves a query with no key")
            self.masks[key] = outside if bool(outside.any()) else None
        return self.masks[key]


def tangent_offsets(conv, map_size, erp_size, wrap_pad):
    """``deform_conv2d`` offsets (1, 2*kh*kw, out_h, out_w) that put ``conv``'s taps on the tangent plane.

    ``map_size`` is the conv input's (rows, columns); the network input is the ERP at ``erp_size``
    with ``wrap_pad`` columns of circular padding on each side.
    """
    height, width = erp_size
    rows, columns = map_size
    stride = height // rows  # input pixels per cell of the conv input
    if stride * rows != height or stride * columns != width + 2 * wrap_pad:
        raise ValueError("a {}x{} map does not tile the padded {}x{} ERP".format(rows, columns, height, width))
    if wrap_pad < stride or wrap_pad % stride:
        raise ValueError("tangent offsets need a wrap pad that is a positive multiple of {}".format(stride))
    (kh, kw), (sh, sw), (ph, pw), (dh, dw) = conv.kernel_size, conv.stride, conv.padding, conv.dilation
    out_h = (rows + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (columns + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    tap_y = np.arange(kh) * dh - (kh - 1) * dh / 2  # tap offsets from the kernel centre, in cells
    tap_x = np.arange(kw) * dw - (kw - 1) * dw / 2
    centre_y = np.arange(out_h) * sh - ph + (kh - 1) * dh / 2
    centre_x = np.arange(out_w) * sw - pw + (kw - 1) * dw / 2

    # Tangent-plane taps at longitude 0 for each output row; longitude is added per column below.
    theta = (0.5 - stride * centre_y / height) * np.pi
    east = np.tan(tap_x * stride * 2 * np.pi / width)[None, None, :]  # one cell's angle at the equator
    north = -np.tan(tap_y * stride * np.pi / height)[None, :, None]  # image down is south
    sin0, cos0 = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
    px = np.broadcast_to(east, (out_h, kh, kw))
    py = cos0 - north * sin0
    pz = sin0 + north * cos0
    tap_theta = np.arcsin(pz / np.sqrt(px ** 2 + py ** 2 + pz ** 2))
    tap_phi = np.arctan2(px, py)

    target_y = np.clip((0.5 - tap_theta / np.pi) * height / stride, 0, rows - 1)  # (out_h, kh, kw)
    u = stride * centre_x - wrap_pad  # ERP column of each output's centre
    target_u = np.mod(u[None, :, None, None] + tap_phi[:, None] / (2 * np.pi) * width, width)
    target_x = (target_u + wrap_pad) / stride  # always in the kept panorama columns
    regular_y = (np.arange(out_h) * sh - ph)[:, None, None, None] + np.arange(kh)[None, None, :, None] * dh
    regular_x = (np.arange(out_w) * sw - pw)[None, :, None, None] + np.arange(kw)[None, None, None, :] * dw
    offset_y = target_y[:, None] - regular_y
    offset_x = target_x - regular_x
    offsets = np.stack(np.broadcast_arrays(offset_y, offset_x), axis=-1)  # (out_h, out_w, kh, kw, 2)
    return offsets.reshape(out_h, out_w, 2 * kh * kw).transpose(2, 0, 1)[None].astype(np.float32)


class TangentOffsets:
    """Offsets per conv geometry and input map size, cached on the device."""

    def __init__(self, height, width, wrap_pad):
        self.erp_size = (height, width)
        self.wrap_pad = wrap_pad
        self.cache = {}

    def __call__(self, conv, x):
        import torch

        key = (conv.kernel_size, conv.stride, conv.padding, conv.dilation, tuple(x.shape[2:]), str(x.device))
        if key not in self.cache:
            self.cache[key] = torch.from_numpy(
                tangent_offsets(conv, tuple(x.shape[2:]), self.erp_size, self.wrap_pad)
            ).to(x.device, x.dtype)
        return self.cache[key]


def _tangent_conv_forward(conv, offsets, x):
    from torchvision.ops import deform_conv2d

    offset = offsets(conv, x).expand(x.shape[0], -1, -1, -1)
    return deform_conv2d(
        x, offset, conv.weight, conv.bias, stride=conv.stride, padding=conv.padding, dilation=conv.dilation
    )


def patch_tangent_convs(network, offsets):
    """Swap each encoder conv wider than 1x1 for its tangent-plane version; returns their names."""
    import torch.nn as nn

    names = []
    for name, module in network.backbone.named_modules():
        if isinstance(module, nn.Conv2d) and max(module.kernel_size) > 1:
            if module.padding_mode != "zeros":
                raise ValueError("{} pads with {}".format(name, module.padding_mode))
            module.forward = functools.partial(_tangent_conv_forward, module, offsets)
            names.append(name)
    return names


def geometry_blind(range_m, window, tolerance, rows_columns):
    """At the given pixels: whether depth fills the surrounding window, and whether one plane
    fits the window's camera-centred points to RMS <= ``tolerance`` times their mean range."""
    valid = range_m > 0
    points = cp.erp_directions(*range_m.shape) * np.where(valid, range_m, 0)[..., None]
    half = window // 2
    channels = [valid.astype(np.float64), np.where(valid, range_m, 0).astype(np.float64)]
    channels += [points[..., i] for i in range(3)]
    channels += [points[..., i] * points[..., j] for i in range(3) for j in range(i, 3)]
    rows, columns = rows_columns
    sums = []
    for channel in channels:  # wrap across the seam; windows past a pole count as missing depth
        padded = np.pad(np.pad(channel, ((0, 0), (half, half)), mode="wrap"), ((half, half), (0, 0)))
        box = cv2.boxFilter(padded, cv2.CV_64F, (window, window), normalize=False, borderType=cv2.BORDER_CONSTANT)
        sums.append(box[half:-half, half:-half][rows, columns])
    count, range_sum, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz = sums
    full = count == window * window
    n = np.maximum(count, 1)
    mean = np.stack([sx, sy, sz], axis=-1) / n[:, None]
    second = np.stack([sxx, sxy, sxz, sxy, syy, syz, sxz, syz, szz], axis=-1).reshape(-1, 3, 3) / n[:, None, None]
    covariance = second - mean[:, :, None] * mean[:, None, :]
    residual = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance)[:, 0], 0))
    return full, full & (residual <= tolerance * range_sum / n)


def coplanar_counts(prediction, label, range_m, window, tolerance):
    """Per door/window: pixels with depth around them, and how many are geometry-blind, overall,
    among those predicted as wall and among those predicted correctly."""
    names = ev.EXPECTED_CLASSES[1:]  # model label order (label = class id - 1)
    wall = names.index("wall")
    counts = {}
    for name in COPLANAR_CLASSES:
        rows, columns = np.nonzero(label == names.index(name))
        full, blind = geometry_blind(range_m, window, tolerance, (rows, columns))
        predicted = prediction[rows, columns]
        groups = {"all": np.ones_like(full), "predicted_wall": predicted == wall,
                  "correct": predicted == names.index(name)}
        counts[name] = {
            group: {"with_depth": int((full & where).sum()), "blind": int((blind & where).sum())}
            for group, where in groups.items()
        }
    return counts


def add_counts(total, counts):
    for name, groups in counts.items():
        for group, values in groups.items():
            for field, value in values.items():
                total[name][group][field] += value


def blind_shares(total):
    return {
        name: {group: values["blind"] / values["with_depth"] if values["with_depth"] else None
               for group, values in groups.items()}
        for name, groups in total.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        epilog="Every other argument goes to eval_pano_transfer.py.",
    )
    parser.add_argument(
        "--attention-cap-deg", type=float, nargs="?", const=TRAINING_CORNER_DEG, default=None,
        help="geodesic key radius in degrees (alone: {:.1f}, the training frame's centre-to-corner angle)".format(
            TRAINING_CORNER_DEG),
    )
    parser.add_argument("--tangent-offsets", action="store_true", help="tangent-plane taps for the encoder convs")
    parser.add_argument("--coplanar-share", action="store_true", help="count geometry-blind door/window pixels")
    parser.add_argument("--coplanar-window", type=int, default=15, help="plane-fit window in eval pixels (odd)")
    parser.add_argument(
        "--coplanar-tolerance", type=float, default=0.005,
        help="plane-fit RMS bound as a fraction of the window's mean range",
    )
    args, passthrough = parser.parse_known_args(argv)
    if args.attention_cap_deg is None and not args.tangent_offsets and not args.coplanar_share:
        parser.error("set at least one of --attention-cap-deg, --tangent-offsets, --coplanar-share")
    if args.attention_cap_deg is not None and not 0 < args.attention_cap_deg <= 180:
        parser.error("--attention-cap-deg must be in (0, 180]")
    if args.coplanar_window < 3 or args.coplanar_window % 2 == 0:
        parser.error("--coplanar-window must be odd and at least 3")
    if not args.coplanar_tolerance > 0:
        parser.error("--coplanar-tolerance must be positive")
    output = argparse.ArgumentParser(add_help=False)
    output.add_argument("--output", type=Path)
    output = output.parse_known_args(passthrough)[0].output

    state = {"geometry": None, "convs": None, "sample": None}
    total = {name: {group: {"with_depth": 0, "blind": 0} for group in ("all", "predicted_wall", "correct")}
             for name in COPLANAR_CLASSES}
    plain = {name: getattr(ev, name) for name in ("predict_with_wrap", "load_sample", "predict_classes")}

    def predict_with_wrap(network, rgb, modal_x, wrap_pad, device):
        geometry = (int(rgb.shape[2]), int(rgb.shape[3]), int(wrap_pad))
        if state["geometry"] is None:
            if args.attention_cap_deg is not None:
                patch_attention(network, outside=AttentionCap(*geometry, args.attention_cap_deg))
            if args.tangent_offsets:
                state["convs"] = patch_tangent_convs(network, TangentOffsets(*geometry))
            state["geometry"] = geometry
        elif state["geometry"] != geometry:
            raise ValueError("the ERP size changed between panoramas")
        return plain["predict_with_wrap"](network, rgb, modal_x, wrap_pad, device)

    def load_sample(sample, size, *rest, **keywords):
        loaded = plain["load_sample"](sample, size, *rest, **keywords)
        state["sample"] = (sample, loaded[2])
        return loaded

    def predict_classes(logits, channels=None):
        prediction = plain["predict_classes"](logits, channels)
        if args.coplanar_share:
            sample, label = state["sample"]
            raw = cv2.imread(str(sample["depth"]), cv2.IMREAD_UNCHANGED)
            raw = cv2.resize(raw, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST)
            add_counts(total, coplanar_counts(
                prediction[0].cpu().numpy(), label, cp.decode_erp_depth(raw),
                args.coplanar_window, args.coplanar_tolerance,
            ))
        return prediction

    argv_before = sys.argv
    sys.argv = [argv_before[0]] + passthrough
    for name, wrapped in (("predict_with_wrap", predict_with_wrap), ("load_sample", load_sample),
                          ("predict_classes", predict_classes)):
        setattr(ev, name, wrapped)
    try:
        code = ev.main()
    finally:
        sys.argv = argv_before
        for name, function in plain.items():
            setattr(ev, name, function)
    if code != 0:
        return code

    probes = []
    if args.attention_cap_deg is not None:
        probes.append("attention cap {:g} deg".format(args.attention_cap_deg))
    if args.tangent_offsets:
        probes.append("tangent-plane conv offsets")
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    report["setting"] += ", eval-only probes: " + (", ".join(probes) or "none (predictions unchanged)")
    report["scope_probes"] = {
        "tool": "eval_pano_scope.py",
        "erp_size": list(state["geometry"][:2]),
        "wrap_pad": state["geometry"][2],
        "attention_cap_deg": args.attention_cap_deg,
        "attention_cap_scope": CAP_SCOPE if args.attention_cap_deg is not None else None,
        "tangent_offsets": args.tangent_offsets,
        "tangent_offset_convs": state["convs"],
        "coplanar_share": {
            "window_px": args.coplanar_window,
            "tolerance_of_range": args.coplanar_tolerance,
            "counts": total,
            "blind_share": blind_shares(total),
        } if args.coplanar_share else None,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
