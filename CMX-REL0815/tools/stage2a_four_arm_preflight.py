#!/usr/bin/env python3
import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO = Path(__file__).resolve().parents[1]
ARMS = ("rawdepth", "hha", "relplus_local", "relplus_pose")


def run_arm(root, arm):
    arm_dir = root / "preflight" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CMX_RUN_DIR"] = str(arm_dir)
    os.environ["STAGE2A_COMMON_INITIAL_MODEL"] = str(root / "initialization/common_initial_model.pth")
    os.environ["CMX_INITIALIZATION_REPORT"] = str(arm_dir / "pretrained_load_report.json")
    sys.path.insert(0, str(REPO))
    selected = importlib.import_module("configs.stage2a_{}".format(arm))
    sys.modules["config"] = selected
    cfg = selected.config
    cfg.batch_size = 2
    cfg.niters_per_epoch = 1
    cfg.num_workers = 0
    cfg.train_scale_array = [1.0]

    import torch
    import torch.nn as nn
    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.dataloader import get_train_loader
    from models.builder import EncoderDecoder
    from stage2a.runtime import load_common_initial_model, seed_everything
    from utils.init_func import group_weight

    class Engine:
        distributed = False

    seed_everything(cfg.seed, deterministic=True)
    loader, _ = get_train_loader(Engine(), RGBXDataset)
    batch = next(iter(loader))
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.background)
    model = EncoderDecoder(cfg=cfg, criterion=criterion, norm_layer=nn.BatchNorm2d)
    initial_report = load_common_initial_model(model, cfg.common_initial_model)
    params = group_weight([], model, nn.BatchNorm2d, cfg.lr)
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device).train()
    rgb = batch["data"].to(device)
    label = batch["label"].to(device)
    modal = batch["modal_x"].to(device)
    optimizer.zero_grad()
    loss = model(rgb, modal, label)
    if not torch.isfinite(loss):
        raise ValueError("{} preflight loss is non-finite".format(arm))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradient_finite = bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)
    if not gradient_finite:
        raise ValueError("{} preflight gradient is missing or non-finite".format(arm))
    optimizer.step()
    preprocess = loader.dataset.preprocess
    observed_gravity = getattr(preprocess, "last_gravity_source", "not_applicable")
    expected_gravity = "local" if arm == "relplus_local" else "pose" if arm == "relplus_pose" else "not_applicable"
    if observed_gravity != expected_gravity:
        raise ValueError("{} gravity dispatch mismatch: {}".format(arm, observed_gravity))
    record = {
        "arm": arm, "second_modality_identity": cfg.second_modality_identity,
        "gravity_source": cfg.gravity_source, "observed_gravity_dispatch": observed_gravity,
        "input_shape": list(batch["modal_x"].shape), "input_dtype": str(batch["modal_x"].dtype),
        "input_min": float(batch["modal_x"].min()), "input_max": float(batch["modal_x"].max()),
        "loss": float(loss.detach().cpu()), "gradient_finite": gradient_finite,
        "optimizer_step_success": True, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "common_initial_tensor_count": initial_report["tensor_count"], "status": "PASS",
    }
    (arm_dir / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    values = batch["modal_x"][0].numpy()
    sheet = Image.new("RGB", (3 * 320, 348), "white")
    draw = ImageDraw.Draw(sheet)
    for channel in range(3):
        plane = values[channel]
        scaled = ((plane - plane.min()) * 255.0 / max(float(plane.max() - plane.min()), 1e-12)).astype(np.uint8)
        colored = np.stack([scaled, scaled, scaled], axis=-1)
        sheet.paste(Image.fromarray(colored).resize((320, 320)), (channel * 320, 28))
        draw.text((channel * 320 + 8, 7), "{} channel {}".format(arm, channel), fill="black")
    sheet.save(arm_dir / "batch.png")
    print(json.dumps(record, indent=2, sort_keys=True))


def aggregate(root):
    rows = []
    for arm in ARMS:
        command = [sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--arm", arm]
        completed = subprocess.run(command, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (root / "preflight" / (arm + ".log")).write_text(completed.stdout)
        if completed.returncode != 0:
            print(completed.stdout)
            raise SystemExit(completed.returncode)
        rows.append(json.loads((root / "preflight" / arm / "result.json").read_text()))
    fields = list(rows[0])
    with (root / "preflight/four_arm_preflight.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    images = [Image.open(root / "preflight" / arm / "batch.png").convert("RGB") for arm in ARMS]
    contact = Image.new("RGB", (images[0].width, sum(image.height for image in images)), "white")
    offset = 0
    for image in images:
        contact.paste(image, (0, offset)); offset += image.height
    contact.save(root / "preflight/four_arm_batch_visualization.png")
    print(json.dumps({"status": "PASS_FOUR_ARM_PREFLIGHT", "arms": list(ARMS)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.arm:
        run_arm(args.root.resolve(), args.arm)
    else:
        aggregate(args.root.resolve())


if __name__ == "__main__":
    main()
