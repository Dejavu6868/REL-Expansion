#!/usr/bin/env python3
import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ARMS = ("rawdepth", "hha", "relplus_local", "relplus_pose")


def run_arm(root, seed, arm):
    seed_dir = root / ("seed_{}".format(seed))
    arm_dir = seed_dir / "preflight" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    os.environ["STAGE2B_SEED"] = str(seed)
    os.environ["CMX_RUN_DIR"] = str(arm_dir)
    os.environ["STAGE2B_COMMON_INITIAL_MODEL"] = str(seed_dir / "common_initial_model.pth")
    os.environ["CMX_INITIALIZATION_REPORT"] = str(arm_dir / "pretrained_load_report.json")
    sys.path.insert(0, str(REPO))
    selected = importlib.import_module("configs.stage2b_{}".format(arm))
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
    model.to(device).train()
    optimizer.zero_grad()
    loss = model(batch["data"].to(device), batch["modal_x"].to(device), batch["label"].to(device))
    if not torch.isfinite(loss):
        raise ValueError("{} preflight loss is non-finite".format(arm))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradient_finite = bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)
    if not gradient_finite:
        raise ValueError("{} preflight gradient is missing or non-finite".format(arm))
    optimizer.step()
    observed = getattr(loader.dataset.preprocess, "last_gravity_source", "not_applicable")
    expected = "local" if arm == "relplus_local" else "pose" if arm == "relplus_pose" else "not_applicable"
    if observed != expected:
        raise ValueError("{} gravity dispatch mismatch: {}".format(arm, observed))
    record = {
        "seed": seed,
        "arm": arm,
        "gravity_source": cfg.gravity_source,
        "observed_gravity_dispatch": observed,
        "input_shape": list(batch["modal_x"].shape),
        "input_dtype": str(batch["modal_x"].dtype),
        "input_min": float(batch["modal_x"].min()),
        "input_max": float(batch["modal_x"].max()),
        "loss": float(loss.detach().cpu()),
        "gradient_finite": gradient_finite,
        "optimizer_step_success": True,
        "common_initial_tensor_count": initial_report["tensor_count"],
        "status": "PASS",
    }
    (arm_dir / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


def aggregate(root, seed):
    rows = []
    preflight = root / ("seed_{}".format(seed)) / "preflight"
    for arm in ARMS:
        command = [sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--seed", str(seed), "--arm", arm]
        completed = subprocess.run(command, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        preflight.mkdir(parents=True, exist_ok=True)
        (preflight / (arm + ".log")).write_text(completed.stdout)
        if completed.returncode != 0:
            print(completed.stdout)
            raise SystemExit(completed.returncode)
        rows.append(json.loads((preflight / arm / "result.json").read_text()))
    with (preflight / "four_arm_preflight.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"status": "PASS_FOUR_ARM_PREFLIGHT", "seed": seed, "arms": list(ARMS)}
    (preflight / "status.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(23456, 34567))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.arm:
        run_arm(args.root.resolve(), args.seed, args.arm)
    else:
        aggregate(args.root.resolve(), args.seed)


if __name__ == "__main__":
    main()
