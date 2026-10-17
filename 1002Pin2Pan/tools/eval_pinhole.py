#!/usr/bin/env python3
"""Evaluate any CMX checkpoint on the area-5 pinhole test set with the frozen evaluator.

The frozen launcher (``evaluation/run_evaluation.py``) only accepts the 2026-09-20
epoch-200 checkpoints. This tool runs the same frozen full-test evaluator
(``tools/eval_rel_plus_v2_3_full.py``, with the frozen HHA input adapter for the HHA
arm) on a resolved arm config JSON and any checkpoint: seed runs, REL+ 3.x runs and
adapted checkpoints from ``train_pin2pan.py``. An RGB-only checkpoint
(``train_rgb_only.py``) is built with ``rgb_only.model_class()`` and evaluated on the
HHA arm's inputs, which its network ignores.

``--eight-class-predictions`` restricts the per-pixel argmax to the 8 SPan8 classes, as
``eval_pano_transfer.py --eight-class-predictions`` does on panoramas. Without it the
computation is the frozen one, so the confusion matrix should equal the frozen
evaluation's for the same checkpoint.

The frozen evaluator writes its usual files (metrics.json, confusion_matrix.csv, ...);
those are 13-class metrics. ``eval_pinhole.json`` adds the run's provenance and the
exact 8-class score (ground-truth rows of the other 5 classes ignored, all prediction
columns kept). Single process by default; for 8 GPUs launch it with
``python -m torch.distributed.launch --nproc_per_node=8 --use_env``.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

import relplus_variant as rv
import rgb_only as ro
from eval_pano_transfer import (
    DEFAULT_SOURCE_ROOT,
    EIGHT_CLASS_NAMES,
    eight_class_channels,
    load_config,
    predict_classes,
)

# Modules the frozen evaluator imports inside main(); imported before the HHA adapter,
# which puts its own (hard-coded) source path first on sys.path.
FROZEN_MODULES = (
    "tools.eval_rel_plus_v2_3_full",
    "engine.relplus_evaluator",
    "dataloader.RGBXDataset",
    "dataloader.data_setting",
    "dataloader.dataloader",
    "dataloader.samplers",
    "models.builder",
    "utils.training_protocol",
)


def import_frozen(source_root):
    source_root = Path(source_root).resolve()
    sys.dont_write_bytecode = True  # keep the frozen archive's file set unchanged
    sys.path.insert(0, str(source_root))
    modules = {name: importlib.import_module(name) for name in FROZEN_MODULES}
    for name, module in modules.items():
        if source_root not in Path(module.__file__).resolve().parents:
            raise RuntimeError("{} escaped the frozen source: {}".format(name, module.__file__))
    return modules


def use_hha_adapter(source_root, frozen, core):
    """Install the frozen evaluation/input_adapter.py, as evaluation/eval_rank.py does."""
    path = Path(source_root).resolve().parent / "evaluation" / "input_adapter.py"
    saved_path = list(sys.path)
    spec = importlib.util.spec_from_file_location("input_adapter", str(path))
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    sys.path[:] = saved_path
    if adapter.frozen_evaluator is not frozen or adapter.evaluator_core is not core:
        raise RuntimeError("input adapter imported a different evaluator")
    core.prepare_eval_sample = adapter.prepare_three_arm_eval_sample
    frozen._unbatch = adapter.unbatch_three_arm
    return path


def eight_class_evaluate(core, channels):
    """``core.evaluate_prepared_sample`` with the argmax restricted to ``channels``."""
    import torch

    def evaluate(network, prepared, *, class_num, ignore_index, device):
        del ignore_index  # hist_info already excludes labels outside [0, class_num).
        network.eval()
        with torch.no_grad():
            logits = network(prepared.rgb.to(device), prepared.modal_x.to(device))
        if logits.ndim != 4 or logits.shape[0] != 1:
            raise ValueError("CMX evaluator expected BCHW logits with batch size one")
        if tuple(logits.shape[2:]) != prepared.label.shape:
            raise ValueError("prediction/logit shape does not match label")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("CMX evaluator produced non-finite logits")
        prediction = predict_classes(logits, channels)[0].detach().cpu().numpy().astype(np.uint8)
        hist, labeled, correct = core.hist_info(class_num, prediction, prepared.label)
        return {
            "prediction": prediction,
            "hist": hist.astype(np.int64),
            "labeled": int(labeled),
            "correct": int(correct),
            "logits_finite": True,
            "diagnostic_mask_passed_to_model": False,
        }

    return evaluate


def eight_class_metrics(confusion, channels):
    """Exact 8-class score: the other ground-truth rows are ignored, every prediction column kept."""
    rows = np.asarray(confusion, dtype=np.float64)[channels]
    true_pixels = rows[np.arange(len(channels)), channels]
    ground_truth = rows.sum(axis=1)
    predicted = rows[:, channels].sum(axis=0)
    union = ground_truth + predicted - true_pixels
    iou = np.divide(true_pixels, union, out=np.full(len(channels), np.nan), where=union > 0)
    accuracy = np.divide(
        true_pixels, ground_truth, out=np.full(len(channels), np.nan), where=ground_truth > 0
    )
    return {
        "classes": list(EIGHT_CLASS_NAMES),
        "mIoU_percent": float(np.nanmean(iou) * 100),
        "pixel_accuracy_percent": float(true_pixels.sum() / ground_truth.sum() * 100),
        "mean_accuracy_percent": float(np.nanmean(accuracy) * 100),
        "per_class_iou_percent": dict(zip(EIGHT_CLASS_NAMES, (iou * 100).tolist())),
        "predicted_as_other_classes_percent": float(
            (ground_truth.sum() - rows[:, channels].sum()) / ground_truth.sum() * 100
        ),
        "evaluated_pixels": int(ground_truth.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="resolved arm config JSON")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-epoch", type=int, default=200,
                        help="payload epoch; adapted checkpoints keep their source epoch")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--eight-class-predictions", action="store_true",
        help="argmax over the 8 SPan8 classes only ({}); the network is unchanged".format(
            " ".join(EIGHT_CLASS_NAMES)
        ),
    )
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    # Results are written only at the end, so every rank can check without racing the others.
    finished = list((args.output / "evaluation").glob("rank_*_evaluation.json"))
    if finished or (args.output / "eval_pinhole.json").exists():
        raise FileExistsError("{} already holds an evaluation; refusing to overwrite".format(args.output))

    started = time.time()
    modules = import_frozen(args.source_root)
    frozen = modules["tools.eval_rel_plus_v2_3_full"]
    core = modules["engine.relplus_evaluator"]
    config = load_config(args.config)
    channels = eight_class_channels(config.class_names)
    adapter = None
    if config.x_mode == "hha_frozen_cache":
        adapter = use_hha_adapter(args.source_root, frozen, core)
    if args.eight_class_predictions:
        core.evaluate_prepared_sample = eight_class_evaluate(core, channels)
    if ro.is_rgb_only(config):
        # The frozen evaluator imports EncoderDecoder from models.builder. Patching the real
        # module would break the frozen __init__, whose super() call looks the name up there.
        view = types.ModuleType("models.builder")
        view.EncoderDecoder = ro.model_class()
        sys.modules["models.builder"] = view
    marker = rv.read_marker(Path(config.x_root_folder).parent) if config.x_mode == "rel_plus_v2_1" else None

    module = types.ModuleType("eval_pinhole_config")
    module.config = config
    sys.modules[module.__name__] = module
    sys.argv = [
        str(frozen.__file__), "--config-module", module.__name__,
        "--checkpoint", str(args.checkpoint), "--expected-epoch", str(args.expected_epoch),
        "--output", str(args.output / "evaluation"), "--device", args.device,
    ]
    code = frozen.main()
    if code != 0 or rank != 0:
        return code

    confusion = np.loadtxt(
        args.output / "evaluation" / "confusion_matrix.csv", delimiter=",", skiprows=1, dtype=np.int64
    )[:, 1:]
    report = {
        "status": "COMPLETED",
        "split": "area 5 pinhole test set ({} images)".format(config.num_eval_imgs),
        "checkpoint": str(args.checkpoint),
        "expected_epoch": args.expected_epoch,
        "config": str(args.config),
        "x_mode": config.x_mode,
        "architecture": config.get("architecture"),
        "relplus_cache_variant": None if config.x_mode != "rel_plus_v2_1" else (
            "v2_1" if marker is None else marker["variant"]
        ),
        "hha_input_adapter": None if adapter is None else str(adapter),
        "source_root": str(Path(args.source_root).resolve()),
        "prediction_classes": list(
            EIGHT_CLASS_NAMES if args.eight_class_predictions else config.class_names
        ),
        "seconds": time.time() - started,
        "classes_8": eight_class_metrics(confusion, channels),
        "warning": "evaluation/metrics.json scores all 13 classes"
        + ("; with 8-class predictions the other 5 score 0 by construction, so use classes_8"
           if args.eight_class_predictions else ""),
    }
    (args.output / "eval_pinhole.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("8-class mIoU {:.4f}%".format(report["classes_8"]["mIoU_percent"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
