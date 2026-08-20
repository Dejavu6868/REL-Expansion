#!/usr/bin/env python3
"""Run exact REL alignment, channel, data and one-step CMX smoke checks."""

import argparse
import ast
import csv
import gc
import importlib
import importlib.util
import json
import os
import random
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from third_party.rel_original import getImage, getREL


CHANNELS = ("EGVIA", "LOA", "ReD")


def function_body(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(
        body[0].value, (ast.Str, ast.Constant)
    ):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def assert_source_alignment(reference_root, compatibility_hha):
    comparisons = (
        (
            reference_root / "getREL.py",
            REPO_ROOT / "third_party" / "rel_original" / "rel.py",
            ("getImage", "getREL"),
        ),
        (
            reference_root / "utils" / "rgbd_util.py",
            REPO_ROOT / "third_party" / "rel_original" / "rgbd_util.py",
            (
                "processDepthImage_ERP",
                "getPointCloud_ERP",
                "computeNormalsSquareSupport_ERP",
            ),
        ),
        (
            compatibility_hha,
            REPO_ROOT / "third_party" / "rel_original" / "hha_util.py",
            (
                "filterItChopOff", "mutiplyIt", "invertIt", "getRMatrix",
                "rotatePC", "getGDir", "getGDirHelper",
            ),
        ),
    )
    checked = []
    for source, extracted, functions in comparisons:
        for name in functions:
            assert function_body(source, name) == function_body(extracted, name), name
            checked.append(name)
    return checked


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reference(reference_root, compatibility_hha):
    package = types.ModuleType("utils")
    package.__path__ = [str(reference_root / "utils")]
    sys.modules["utils"] = package
    load_module("utils.hha_util", compatibility_hha)
    load_module("utils.rgbd_util", reference_root / "utils" / "rgbd_util.py")
    return load_module("reference_getREL", reference_root / "getREL.py")


def unload_reference():
    for name in ("reference_getREL", "utils.rgbd_util", "utils.hha_util", "utils"):
        sys.modules.pop(name, None)


def channel_stats(array):
    records = []
    for index, semantic in enumerate(CHANNELS):
        values = array[:, :, index]
        records.append(
            {
                "index": index,
                "semantic": semantic,
                "dtype": str(values.dtype),
                "shape": list(values.shape),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "nan_count": int(np.isnan(values).sum()),
                "inf_count": int(np.isinf(values).sum()),
                "constant": bool(values.min() == values.max()),
            }
        )
    return records


def save_visualization(depth_path, rel, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    valid = raw_depth != np.iinfo(raw_depth.dtype).max
    depth_display = np.zeros(raw_depth.shape, dtype=np.uint8)
    if np.any(valid):
        values = raw_depth[valid].astype(np.float32)
        low, high = float(values.min()), float(values.max())
        if high > low:
            depth_display[valid] = np.clip(
                (values - low) * 255.0 / (high - low), 0, 255
            ).astype(np.uint8)
    cv2.imwrite(str(output_dir / "depth.png"), depth_display)
    for index, semantic in enumerate(CHANNELS):
        cv2.imwrite(str(output_dir / "channel_{}_{}.png".format(index, semantic)), rel[:, :, index])
    cv2.imwrite(str(output_dir / "rel_color.png"), rel)


def exact_rel_checks(reference, manifest, artifact_root, limit):
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["status"] in ("OK", "SKIPPED")
        ][:limit]
    assert len(rows) == limit

    results = []
    for index, row in enumerate(rows, start=1):
        depth_path = Path(row["depth_path"])
        disk_rel = cv2.imread(row["rel_path"], cv2.IMREAD_UNCHANGED)
        reference_depth = reference.getImage(str(depth_path), "Stanford2D3DPano")
        integrated_depth = getImage(str(depth_path), "Stanford2D3DPano")
        assert np.array_equal(reference_depth, integrated_depth)
        reference_rel = reference.getREL(reference_depth, alpha=45, lam=0.5)
        integrated_rel = getREL(integrated_depth, alpha=45, lam=0.5)
        assert np.array_equal(reference_rel, integrated_rel)
        assert np.array_equal(integrated_rel, disk_rel)

        reloaded = cv2.imread(row["rel_path"], cv2.IMREAD_UNCHANGED)
        assert np.array_equal(integrated_rel, reloaded)
        sample_id = "s3d_{:02d}".format(index)
        save_visualization(
            depth_path,
            integrated_rel,
            artifact_root / "channel_visualization" / sample_id,
        )
        results.append(
            {
                "sample_id": sample_id,
                "depth_path": str(depth_path),
                "shape": list(integrated_rel.shape),
                "dtype": str(integrated_rel.dtype),
                "reference_equal": True,
                "saved_reloaded_equal": True,
                "channels": channel_stats(integrated_rel),
            }
        )
        del reference_depth, integrated_depth, reference_rel, integrated_rel, disk_rel
        gc.collect()
    return results


def group_stats(model, prefixes):
    parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    ]
    count = sum(parameter.numel() for parameter in parameters)
    has_gradient = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all().item()
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in parameters
    )
    return {"parameters": count, "finite_nonzero_gradient": has_gradient}


def run_data_and_model_smoke(smoke_root, artifact_root, device_name):
    os.environ["CMX_REL_SMOKE_DATASET_ROOT"] = str(smoke_root)
    os.environ["CMX_REL_SMOKE_RUN_DIR"] = str(artifact_root / "runtime")
    config_module = importlib.import_module(
        "configs.stanford2d3dpano.cmx_mit_b2_rel_smoke"
    )
    sys.modules["config"] = config_module
    config = config_module.config

    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.dataloader import TrainPre, random_scale
    from models.builder import EncoderDecoder

    raw_dataset = RGBXDataset(config.data_setting, "val")
    raw_item = raw_dataset[0]
    disk_rel = cv2.imread(
        str(smoke_root / "REL" / (raw_item["fn"] + ".png")),
        cv2.IMREAD_UNCHANGED,
    )
    assert np.array_equal(raw_item["modal_x"], disk_rel)
    assert raw_item["modal_x"].shape[2] == 3

    old_size = (config.image_height, config.image_width)
    old_scales = config.train_scale_array
    try:
        config.image_height = 8
        config.image_width = 8
        config.train_scale_array = None
        grid = np.arange(256, dtype=np.uint8).reshape(16, 16)
        image = np.repeat(grid[:, :, None], 3, axis=2)
        random.seed(19)
        aligned_rgb, aligned_label, aligned_rel = TrainPre(
            np.zeros(3), np.ones(3)
        )(image, grid, image)
        assert np.array_equal(aligned_rgb, aligned_rel)
        assert np.array_equal(
            np.rint(aligned_rgb[0] * 255).astype(np.uint8), aligned_label
        )
        scaled_rgb, scaled_label, scaled_rel, scale = random_scale(
            image, grid, image, [0.5]
        )
        assert scale == 0.5
        assert np.array_equal(scaled_rgb, scaled_rel)
        assert scaled_rgb.shape[:2] == scaled_label.shape
    finally:
        config.image_height, config.image_width = old_size
        config.train_scale_array = old_scales

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    dataset = RGBXDataset(
        config.data_setting,
        "train",
        TrainPre(config.norm_mean, config.norm_std),
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    valid_labels = set(int(value) for value in torch.unique(batch["label"]).tolist())
    assert valid_labels.issubset(set(range(13)) | {255})
    assert valid_labels - {255}
    assert torch.isfinite(batch["data"]).all().item()
    assert torch.isfinite(batch["modal_x"]).all().item()

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(ignore_index=config.background)
    model = EncoderDecoder(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    rgb = batch["data"].to(device)
    rel = batch["modal_x"].to(device)
    label = batch["label"].to(device)

    print("Experiment: CMX-REL integration smoke test")
    print("Architecture: Original CMX")
    print("Backbone: MiT-B2")
    print("X modality: 3-channel REL")
    print("Gate: disabled")
    print("SMMF: disabled")
    print("DyMM: disabled")
    print("REL+: disabled")

    optimizer.zero_grad()
    logits = model(rgb, rel)
    loss = criterion(logits, label.long())
    assert torch.isfinite(logits).all().item()
    assert torch.isfinite(loss).item()
    assert list(logits.shape) == [1, 13, config.image_height, config.image_width]
    loss.backward()

    groups = {
        "rgb_encoder": group_stats(
            model,
            ("backbone.patch_embed", "backbone.block", "backbone.norm"),
        ),
        "rel_encoder": group_stats(model, ("backbone.extra_",)),
        "fusion": group_stats(model, ("backbone.FRMs", "backbone.FFMs")),
        "decoder": group_stats(model, ("decode_head",)),
    }
    assert all(group["finite_nonzero_gradient"] for group in groups.values())
    tracked = model.backbone.patch_embed1.proj.weight.detach().clone()
    optimizer.step()
    optimizer_updated = not torch.equal(
        tracked, model.backbone.patch_embed1.proj.weight.detach()
    )
    assert optimizer_updated

    rel_stats = []
    for channel, semantic in enumerate(CHANNELS):
        values = rel[:, channel]
        rel_stats.append(
            {
                "semantic": semantic,
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "mean": float(values.mean().item()),
                "std": float(values.std().item()),
            }
        )
    return {
        "sample_ids": list(batch["fn"]),
        "rgb_shape": list(rgb.shape),
        "rel_shape": list(rel.shape),
        "label_shape": list(label.shape),
        "valid_label_values": sorted(valid_labels),
        "rel_normalized_stats": rel_stats,
        "logits_shape": list(logits.shape),
        "loss": float(loss.item()),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "groups": groups,
        "optimizer_step_updated_parameter": optimizer_updated,
        "checkpoint_saved": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--compatibility-hha", required=True, type=Path)
    parser.add_argument("--generation-manifest", required=True, type=Path)
    parser.add_argument("--smoke-dataset-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    aligned_functions = assert_source_alignment(
        args.reference_root, args.compatibility_hha
    )
    reference = load_reference(args.reference_root, args.compatibility_hha)
    exact_results = exact_rel_checks(
        reference, args.generation_manifest, args.artifact_root, args.limit
    )
    unload_reference()
    model_result = run_data_and_model_smoke(
        args.smoke_dataset_root, args.artifact_root, args.device
    )

    results = {
        "status": "PASS",
        "source_aligned_functions": aligned_functions,
        "real_sample_count": len(exact_results),
        "exact_rel_results": exact_results,
        "channel_trace": {
            "core_ndarray": ["EGVIA", "LOA", "ReD"],
            "png_rgb_components": ["ReD", "LOA", "EGVIA"],
            "cv2_imread_unchanged": ["EGVIA", "LOA", "ReD"],
            "bgr_to_rgb_applied": False,
            "tensor": ["EGVIA", "LOA", "ReD"],
        },
        "model_smoke": model_result,
    }
    output = args.artifact_root / "smoke_results.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("status=PASS")
    print("results={}".format(output))


if __name__ == "__main__":
    main()
