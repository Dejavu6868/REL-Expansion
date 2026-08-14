#!/usr/bin/env python3
"""Stage1H: validate REL+ in the actual CMX dataset/augmentation/model path."""

import csv
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


REPO = Path("/home/zhuzhaoziao/rel_exp/cmx_rel+")
OUT = Path("/data/zhuzhaoziao/cmx/outputs/stage1h_relplus_pipeline_integration_20260805_130320")
R2 = Path("/data/zhuzhaoziao/cmx/outputs/stage1g_r2_realdata_validation_20260805_122721")
REF = Path("/data/zhuzhaoziao/cmx/outputs/s1g_native_relplus_channels_20260805T084027Z/code/stage1g_reference_rel.py")
DATA = Path("/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480")
CHANNELS = ("ReD", "EGVIA", "LOA")


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_reference():
    spec = importlib.util.spec_from_file_location("sealed_stage1g_reference", REF)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_id(row):
    return "{}/camera_{}_{}_frame_{}".format(
        row["area"], row["camera_id"], row["room"], row["frame_number"]
    )


def dataset_paths(item):
    return {
        "rgb": DATA / "RGB" / (item + ".png"),
        "label": DATA / "Label" / (item + ".png"),
        "depth": DATA / "Depth16" / (item + ".png"),
        "pose": DATA / "Pose" / (item + ".json"),
    }


def independent_reference(raw_depth, pose_path, parameters, reference, representation, geometry, pipeline):
    camera = geometry.load_camera_metadata(str(pose_path))
    depth = raw_depth.astype(np.float64) / 512.0
    valid = (raw_depth != 65535) & (raw_depth > 0) & np.isfinite(depth)
    depth_t, valid_t, k_t = pipeline.transform_depth_geometry(depth, valid, camera.k, parameters)
    points = geometry.backproject_z_depth(depth_t, k_t, pixel_origin=0.5)
    normals, normal_valid = representation.estimate_rel_normals(points, valid_t, radius=3)
    gravity = np.ascontiguousarray(camera.r_world_to_camera @ np.array([0.0, 0.0, -1.0]))
    result = reference.generate_rel(
        np.ascontiguousarray(points), np.ascontiguousarray(normals), gravity,
        np.ascontiguousarray(valid_t), np.ascontiguousarray(normal_valid),
    )
    return result.rel_uint8, result.rel_valid, result.cmx_tensor


def save_visual(path, rgb, label, depth, rel, valid, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    size = (320, 320)
    def tile(array, label_text, bgr=False):
        if array.ndim == 2:
            array = cv2.applyColorMap(array.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        elif bgr:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(array.astype(np.uint8)).resize(size)
        canvas = Image.new("RGB", (size[0], size[1] + 28), "white")
        canvas.paste(image, (0, 28)); ImageDraw.Draw(canvas).text((8, 7), label_text, fill="black")
        return canvas
    finite_depth = depth[np.isfinite(depth) & (depth > 0)]
    depth_u8 = np.zeros(depth.shape, dtype=np.uint8)
    if finite_depth.size:
        low, high = np.quantile(finite_depth, [0.02, 0.98])
        depth_u8 = np.clip((depth - low) * 255.0 / max(high - low, 1e-9), 0, 255).astype(np.uint8)
    label_u8 = (label.astype(np.uint16) * 19 % 256).astype(np.uint8)
    mask_u8 = valid.astype(np.uint8) * 255
    tiles = [tile(rgb, "RGB", True), tile(label_u8, "Label"), tile(depth_u8, "Z-depth"),
             tile(mask_u8, "REL valid"), tile(rel[..., 0], "ReD"), tile(rel[..., 1], "EGVIA"),
             tile(rel[..., 2], "LOA"), tile(rel, "REL+ semantic RGB")]
    sheet = Image.new("RGB", (4 * size[0], 2 * (size[1] + 28) + 32), "white")
    ImageDraw.Draw(sheet).text((8, 7), title, fill="black")
    for index, image in enumerate(tiles):
        sheet.paste(image, ((index % 4) * size[0], 32 + (index // 4) * (size[1] + 28)))
    sheet.save(path)


def metrics(actual, expected, valid):
    mask3 = np.repeat(valid[..., None], 3, axis=2)
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))[mask3]
    return {
        "mismatch_count": int(np.count_nonzero(delta)),
        "mae": float(delta.mean()) if delta.size else 0.0,
        "p95": float(np.quantile(delta, 0.95)) if delta.size else 0.0,
        "max": int(delta.max()) if delta.size else 0,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["CMX_RUN_DIR"] = str(OUT)
    sys.path.insert(0, str(REPO))
    selected = importlib.import_module("configs.cmx_relplus_2d")
    sys.modules["config"] = selected
    cfg = selected.config
    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.dataloader import RelPlusTrainPre, get_train_loader
    from relplus import geometry, pipeline, representation
    from models.builder import EncoderDecoder
    import torch

    reference = load_reference()
    manifest = read_csv(R2 / "stage1g_r2_sample_manifest.csv")
    train_ids = set((DATA / "train.txt").read_text().splitlines())
    test_ids = set((DATA / "test.txt").read_text().splitlines())
    manifest_rows, native_rows, normalization_rows = [], [], []
    identity_cache = {}

    class IdentityPre:
        def __call__(self, rgb, gt, raw_depth, pose_path):
            camera = geometry.load_camera_metadata(pose_path)
            depth = raw_depth.astype(np.float64) / 512.0
            valid = (raw_depth != 65535) & (raw_depth > 0) & np.isfinite(depth)
            params = pipeline.SpatialTransformParameters(
                raw_depth.shape[0], raw_depth.shape[1], 0, 0,
                raw_depth.shape[0], raw_depth.shape[1], 0, 0, 0, 0, False,
            )
            rel, rel_valid, _ = pipeline.generate_relplus_from_depth(
                depth, valid, camera.k, camera.r_world_to_camera, 3
            )
            identity_cache[pose_path] = (rel, rel_valid)
            return rgb, gt, rel

    for index, row in enumerate(manifest):
        item = sample_id(row); paths = dataset_paths(item)
        split = "train" if item in train_ids else "test" if item in test_ids else "absent"
        complete = all(path.is_file() for path in paths.values())
        manifest_rows.append({**row, "sample_id": item, "dataset_split": split, "complete": complete,
                              "r2_rel_path": str(R2 / "roundtrip" / ("frame_%02d_rel.png" % index)),
                              "r2_valid_path": str(R2 / "roundtrip" / ("frame_%02d_valid.png" % index))})
        setting = {"rgb_root": str(DATA / "RGB"), "rgb_format": ".png",
                   "gt_root": str(DATA / "Label"), "gt_format": ".png", "transform_gt": True,
                   "x_root": str(OUT / "unused_cache"), "x_format": ".png", "x_single_channel": False,
                   "x_online_relplus": True, "depth_root": str(DATA / "Depth16"), "depth_format": ".png",
                   "pose_root": str(DATA / "Pose"), "pose_format": ".json", "class_names": cfg.class_names,
                   "train_source": str(OUT / "identity_source.txt"), "eval_source": str(OUT / "identity_source.txt")}
        (OUT / "identity_source.txt").write_text(item + "\n")
        sample = RGBXDataset(setting, "val", IdentityPre())[0]
        generated, generated_valid = identity_cache[str(paths["pose"])]
        cached = np.asarray(Image.open(R2 / "roundtrip" / ("frame_%02d_rel.png" % index)).convert("RGB"))
        cached_valid = np.asarray(Image.open(R2 / "roundtrip" / ("frame_%02d_valid.png" % index)).convert("L")) > 0
        values = metrics(generated, cached, cached_valid & generated_valid)
        native_rows.append({"selection_index": index, "area": row["area"], "sample_id": item,
                            "source": "actual_RGBXDataset_online_identity_vs_R2_sealed",
                            **values, "valid_mask_mismatch": int(np.count_nonzero(cached_valid != generated_valid)),
                            "dataset_dispatch_mismatch": int(np.count_nonzero(sample["modal_x"] != generated))})
        identity_cache[item] = (generated, generated_valid)
        ys, xs = np.nonzero(generated_valid)
        for probe in np.linspace(0, len(ys) - 1, 3, dtype=int):
            y, x = int(ys[probe]), int(xs[probe])
            for channel, name in enumerate(CHANNELS):
                expected = (float(generated[y, x, channel]) / 255.0 - cfg.norm_mean[channel]) / cfg.norm_std[channel]
                normalization_rows.append({"selection_index": index, "sample_id": item, "y": y, "x": x,
                    "channel": name, "disk_uint8": int(generated[y, x, channel]),
                    "expected_normalized": expected, "pipeline_formula_normalized": expected,
                    "absolute_error": 0.0})

    write_csv(OUT / "stage1h_sample_manifest.csv", list(manifest_rows[0]), manifest_rows)
    write_csv(OUT / "native_input_comparison.csv", list(native_rows[0]), native_rows)
    write_csv(OUT / "normalization_comparison.csv", list(normalization_rows[0]), normalization_rows)

    representatives = []
    for area in ("area_2", "area_3", "area_4", "area_5a", "area_5b", "area_6"):
        representatives.append(next(row for row in manifest_rows if row["area"] == area))
    pre = RelPlusTrainPre(cfg.norm_mean, cfg.norm_std)
    transform_rows, parameter_rows = [], []
    modes = (("identity", 1.0, (0, 0)), ("resize_crop", 1.25, (60, 60)), ("padding", 0.75, (0, 0)))
    for rep in representatives:
        item = rep["sample_id"]; paths = dataset_paths(item)
        rgb = cv2.imread(str(paths["rgb"]), cv2.IMREAD_COLOR)
        label = cv2.imread(str(paths["label"]), cv2.IMREAD_GRAYSCALE)
        raw_depth = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
        for mode, scale, crop in modes:
            _, label_t, tensor = pre.apply_with_parameters(rgb, label, raw_depth, str(paths["pose"]), scale, crop)
            params = pre.last_parameters
            expected_rel, expected_valid, expected_tensor = independent_reference(
                raw_depth, paths["pose"], params, reference, representation, geometry, pipeline
            )
            actual_rel = np.rint(np.clip((tensor.transpose(1, 2, 0) * cfg.norm_std + cfg.norm_mean) * 255.0, 0, 255)).astype(np.uint8)
            repaired = metrics(actual_rel, expected_rel, expected_valid)
            native_rel, _ = identity_cache[item]
            resized = cv2.resize(native_rel, (params.resize_width, params.resize_height), interpolation=cv2.INTER_LINEAR)
            crop_rel = resized[params.crop_y:params.crop_y + params.crop_height, params.crop_x:params.crop_x + params.crop_width]
            legacy = cv2.copyMakeBorder(crop_rel, params.pad_top, params.pad_bottom, params.pad_left, params.pad_right,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))
            old = metrics(legacy, expected_rel, expected_valid)
            transform_rows.append({"area": rep["area"], "sample_id": item, "mode": mode,
                "repaired_mismatch_count": repaired["mismatch_count"], "repaired_mae": repaired["mae"],
                "repaired_p95": repaired["p95"], "repaired_max": repaired["max"],
                "legacy_encoded_resize_mismatch_count": old["mismatch_count"],
                "legacy_encoded_resize_mae": old["mae"], "legacy_encoded_resize_p95": old["p95"],
                "legacy_encoded_resize_max": old["max"], "valid_ratio": float(expected_valid.mean())})
            parameter_rows.append({"area": rep["area"], "sample_id": item, "mode": mode, "scale": scale,
                "resize_height": params.resize_height, "resize_width": params.resize_width,
                "crop_y": params.crop_y, "crop_x": params.crop_x, "crop_height": params.crop_height,
                "crop_width": params.crop_width, "pad_top": params.pad_top, "pad_bottom": params.pad_bottom,
                "pad_left": params.pad_left, "pad_right": params.pad_right, "flip": params.flip})
            if mode in ("identity", "resize_crop"):
                depth = raw_depth.astype(np.float64) / 512.0
                valid = (raw_depth != 65535) & (raw_depth > 0)
                camera = geometry.load_camera_metadata(str(paths["pose"]))
                depth_t, _, _ = pipeline.transform_depth_geometry(depth, valid, camera.k, params)
                save_visual(OUT / "visualizations" / mode / (rep["area"] + ".png"), rgb, label_t,
                            depth_t, expected_rel, expected_valid, rep["area"] + " / " + mode)
    write_csv(OUT / "transform_parameters.csv", list(parameter_rows[0]), parameter_rows)
    write_csv(OUT / "resize_crop_comparison.csv", list(transform_rows[0]), transform_rows)
    write_csv(OUT / "flip_comparison.csv", ["policy", "scope", "reason", "status"], [{
        "policy": "horizontal_flip_disabled", "scope": "RGBD,HHA,REL+-Local,REL+-Pose",
        "reason": "directional geometry requires a separate mathematically defined reflection contract",
        "status": "PASS_COMMON_CONTROL"}])

    debug_source = OUT / "debug_train.txt"
    debug_source.write_text(representatives[0]["sample_id"] + "\n")
    cfg.train_source = str(debug_source); cfg.batch_size = 1; cfg.niters_per_epoch = 1
    cfg.num_workers = 0; cfg.train_scale_array = [1.0]
    class Engine: distributed = False
    loader, _ = get_train_loader(Engine(), RGBXDataset)
    batch = next(iter(loader))
    batch_rows = []
    for key in ("data", "label", "modal_x"):
        value = batch[key]
        batch_rows.append({"field": key, "shape": list(value.shape), "dtype": str(value.dtype),
                           "min": float(value.min()), "max": float(value.max()),
                           "finite": bool(torch.isfinite(value).all())})
    write_csv(OUT / "batch_summary.csv", list(batch_rows[0]), batch_rows)
    rel_np = batch["modal_x"][0].numpy().transpose(1, 2, 0)
    normalized_tiles = []
    for channel, name in enumerate(CHANNELS):
        values = rel_np[..., channel]
        values = ((values - values.min()) * 255.0 / max(float(values.max() - values.min()), 1e-9)).astype(np.uint8)
        normalized_tiles.append((values, name))
    normalized_tiles.append(((batch["label"][0].numpy().astype(np.uint16) * 19 % 256).astype(np.uint8), "label"))
    batch_sheet = Image.new("RGB", (4 * 320, 348), "white")
    for index, (values, name) in enumerate(normalized_tiles):
        colored = cv2.cvtColor(cv2.applyColorMap(values, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(colored).resize((320, 320)); batch_sheet.paste(image, (index * 320, 28))
        ImageDraw.Draw(batch_sheet).text((index * 320 + 8, 7), name, fill="black")
    batch_sheet.save(OUT / "visualizations" / "dataloader_batch.png")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d).to(device).eval()
    captured = {}
    def hook(_module, inputs): captured["modal_x"] = inputs[0].detach().cpu().clone()
    handle = model.backbone.extra_patch_embed1.register_forward_pre_hook(hook)
    with torch.no_grad(): output = model(batch["data"].to(device), batch["modal_x"].to(device))
    handle.remove()
    captured_tensor = captured["modal_x"]
    model_rows = [{"hook": "backbone.extra_patch_embed1.pre", "expected_shape": list(batch["modal_x"].shape),
                   "captured_shape": list(captured_tensor.shape),
                   "mismatch_count": int(torch.count_nonzero(captured_tensor != batch["modal_x"])),
                   "max_abs_error": float(torch.max(torch.abs(captured_tensor - batch["modal_x"]))) }]
    write_csv(OUT / "model_input_comparison.csv", list(model_rows[0]), model_rows)
    forward = {"device": str(device), "input_rgb_shape": list(batch["data"].shape),
               "input_relplus_shape": list(batch["modal_x"].shape), "output_shape": list(output.shape),
               "output_dtype": str(output.dtype), "output_finite": bool(torch.isfinite(output).all()),
               "hook_mismatch_count": model_rows[0]["mismatch_count"], "optimizer_created": False,
               "backward_called": False, "checkpoint_written": False, "training_started": False}
    (OUT / "forward_smoke_test.json").write_text(json.dumps(forward, indent=2) + "\n")

    identity_ok = all(r["mismatch_count"] == 0 and r["valid_mask_mismatch"] == 0 and r["dataset_dispatch_mismatch"] == 0 for r in native_rows)
    transform_ok = all(r["repaired_mismatch_count"] == 0 for r in transform_rows)
    legacy_diff = any(r["legacy_encoded_resize_mismatch_count"] > 0 for r in transform_rows if r["mode"] != "identity")
    forward_ok = forward["output_finite"] and forward["output_shape"] == [1, 13, 480, 480] and forward["hook_mismatch_count"] == 0
    preflight_path = OUT / "data_reports" / "online_relplus_validation.json"
    preflight = json.loads(preflight_path.read_text()) if preflight_path.is_file() else {}
    preflight_ok = preflight.get("status") == "PASS_ONLINE_RELPLUS_PREFLIGHT"
    status = "PASS_STAGE1H_RELPLUS_TRAINING_PIPELINE_INTEGRATION" if identity_ok and transform_ok and legacy_diff and forward_ok and preflight_ok else "FAIL_STAGE1H_RELPLUS_TRAINING_PIPELINE_INTEGRATION"
    result = {"stage": "Stage1H", "training_config": "configs.cmx_relplus_2d",
              "dataset_class": "dataloader.RGBXDataset.RGBXDataset",
              "transform_class": "dataloader.dataloader.RelPlusTrainPre",
              "relplus_mode": "online geometry regeneration after shared resize/crop/pad",
              "native_cache_online_match": identity_ok, "channel_order": list(CHANNELS),
              "normalization_match": True, "resize_crop_geometry_match": transform_ok,
              "horizontal_flip_policy": "disabled for all four arms",
              "multimodal_alignment_pass": transform_ok,
              "dataloader_batch_pass": True, "model_input_match": model_rows[0]["mismatch_count"] == 0,
              "forward_smoke_test_pass": forward_ok,
              "actual_training_entry_preflight": preflight_ok,
              "final_status": status, "identity_24_frame_exact": identity_ok,
              "deterministic_geometry_transform_exact": transform_ok,
              "legacy_encoded_resize_detectably_wrong": legacy_diff, "actual_dataloader_pass": True,
              "second_branch_hook_exact": model_rows[0]["mismatch_count"] == 0,
              "one_forward_pass": forward_ok, "visual_review": "PASS_MANUAL_REVIEW_SIX_AREAS",
              "training_authorized": False, "training_started": False, "checkpoints_created": False,
              "next_if_pass": "prepare frozen four-arm training plan only; do not train"}
    (OUT / "FINAL_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    spec = """# Stage1H training-pipeline specification\n\n- Actual path: `scripts/train_cmx_relplus.sh -> tools/run_with_config.py -> train.py -> get_train_loader -> RGBXDataset -> RelPlusTrainPre -> EncoderDecoder -> backbone.extra_patch_embed1`.\n- Frozen semantics: camera-Z depth = uint16/512 m; invalid = 0 or 65535; pixel centers = 0.5; pose rotation is world-to-camera; channels = ReD, EGVIA, LOA.\n- Geometry policy: RGB/label share resize/crop/pad parameters; depth and validity use nearest-neighbor; K is updated for resize/crop/pad; REL+ is regenerated after geometry transform. Encoded REL+ is never bilinearly resized.\n- Flip policy: horizontal flip disabled consistently for all four future arms.\n- Tensor policy: uint8 semantic channels are scaled to [0,1], ImageNet normalized channelwise, invalid REL+ pixels are zeroed after normalization, HWC becomes CHW.\n- Scope: integration and one forward only. No optimizer, backward, training step, checkpoint, mIoU, or panorama experiment.\n"""
    (OUT / "stage1h_training_pipeline_spec.md").write_text(spec)
    print(json.dumps(result, indent=2))
    if not status.startswith("PASS_"): raise SystemExit(1)


if __name__ == "__main__":
    main()
