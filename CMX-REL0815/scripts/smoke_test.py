#!/usr/bin/env python3
"""One-batch GPU smoke test covering train, validation, checkpoint, and resume."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_gradients(model):
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return bool(gradients) and all(torch.isfinite(gradient).all().item() for gradient in gradients)


def optimizer_for(model, config, group_weight):
    groups = group_weight([], model, nn.BatchNorm2d, config.lr)
    return torch.optim.AdamW(
        groups, lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(str(repo_root))
    sys.path.insert(0, str(repo_root))

    selected_config = importlib.import_module("configs.cmx_relplus_2d")
    config = selected_config.config
    config.batch_size = 1
    config.niters_per_epoch = 1
    config.num_workers = 0
    config.nepochs = 1
    smoke_split = run_dir / "data_reports" / "smoke_split.txt"
    if not smoke_split.is_file():
        raise FileNotFoundError(smoke_split)
    config.train_source = str(smoke_split)
    config.eval_source = str(smoke_split)
    sys.modules["config"] = selected_config

    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.dataloader import ValPre, get_train_loader
    from engine.engine import Engine
    from models.builder import EncoderDecoder
    from utils.init_func import group_weight
    from utils.metric import compute_score, hist_info
    from utils.transforms import normalize

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    device = torch.device("cuda:0")
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)

    original_argv = sys.argv[:]
    sys.argv = [str(Path(__file__).resolve()), "-d", "0"]
    with Engine(custom_parser=argparse.ArgumentParser()) as engine:
        train_loader, _ = get_train_loader(engine, RGBXDataset)
        batch = next(iter(train_loader))
        model = EncoderDecoder(config, criterion=criterion, norm_layer=nn.BatchNorm2d).to(device)
        optimizer = optimizer_for(model, config, group_weight)
        images = batch["data"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        relplus = batch["modal_x"].to(device, non_blocking=True)
        model.train()
        optimizer.zero_grad()
        loss = model(images, relplus, labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError("smoke loss is not finite")
        loss.backward()
        gradients_finite = finite_gradients(model)
        if not gradients_finite:
            raise RuntimeError("smoke gradients are not finite")
        optimizer.step()

        setting = {
            "rgb_root": config.rgb_root_folder,
            "rgb_format": config.rgb_format,
            "gt_root": config.gt_root_folder,
            "gt_format": config.gt_format,
            "transform_gt": config.gt_transform,
            "x_root": config.x_root_folder,
            "x_format": config.x_format,
            "x_single_channel": config.x_is_single_channel,
            "train_source": config.train_source,
            "eval_source": config.eval_source,
            "class_names": config.class_names,
        }
        val_dataset = RGBXDataset(setting, "val", ValPre())
        validation = val_dataset[0]
        val_rgb = normalize(validation["data"], config.norm_mean, config.norm_std)
        val_rel = normalize(validation["modal_x"], config.norm_mean, config.norm_std)
        val_rgb = torch.from_numpy(np.ascontiguousarray(val_rgb.transpose(2, 0, 1))[None]).float().to(device)
        val_rel = torch.from_numpy(np.ascontiguousarray(val_rel.transpose(2, 0, 1))[None]).float().to(device)
        model.eval()
        with torch.no_grad():
            logits = model(val_rgb, val_rel)
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("validation logits are not finite")
        prediction = logits.argmax(1)[0].cpu().numpy()
        hist, labeled, correct = hist_info(config.num_classes, prediction, validation["label"])
        iou, miou, _, _, mean_accuracy, pixel_accuracy = compute_score(hist, correct, labeled)
        metrics_finite = np.isfinite([miou, mean_accuracy, pixel_accuracy]).all()
        if not metrics_finite:
            raise RuntimeError("validation metrics are not finite")

        checkpoint = run_dir / "smoke" / "checkpoint.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
        engine.update_iteration(epoch=1, iteration=0)
        engine.save_checkpoint(str(checkpoint))
        first_loss = float(loss.detach().cpu())

    del optimizer, model, loss, logits
    torch.cuda.empty_cache()

    resumed_model = EncoderDecoder(config, criterion=criterion, norm_layer=nn.BatchNorm2d).to(device)
    resumed_optimizer = optimizer_for(resumed_model, config, group_weight)
    sys.argv = [str(Path(__file__).resolve()), "-d", "0", "-c", str(checkpoint)]
    with Engine(custom_parser=argparse.ArgumentParser()) as resumed_engine:
        resumed_engine.register_state(
            dataloader=train_loader, model=resumed_model, optimizer=resumed_optimizer
        )
        resumed_engine.restore_checkpoint()
        if resumed_engine.state.epoch != 2:
            raise RuntimeError("resume did not advance to epoch 2")
        resumed_model.train()
        resumed_optimizer.zero_grad()
        resumed_loss = resumed_model(images, relplus, labels)
        if not torch.isfinite(resumed_loss).item():
            raise RuntimeError("resumed loss is not finite")
        resumed_loss.backward()
        if not finite_gradients(resumed_model):
            raise RuntimeError("resumed gradients are not finite")
        resumed_optimizer.step()

    sys.argv = original_argv
    report = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "dataloader_batch_shape": list(images.shape),
        "relplus_batch_shape": list(relplus.shape),
        "forward_loss": first_loss,
        "gradients_finite": gradients_finite,
        "validation_logits_finite": True,
        "validation_miou": float(miou),
        "validation_pixel_accuracy": float(pixel_accuracy),
        "validation_mean_accuracy": float(mean_accuracy),
        "per_class_iou": [float(value) if np.isfinite(value) else None for value in iou],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_load": "PASS",
        "resume_epoch": 2,
        "resumed_optimizer_step": "PASS",
        "resumed_loss": float(resumed_loss.detach().cpu()),
    }
    output = run_dir / "smoke" / "smoke_report.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
