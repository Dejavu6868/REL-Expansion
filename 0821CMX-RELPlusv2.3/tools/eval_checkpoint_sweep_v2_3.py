#!/usr/bin/env python3
"""Sequentially launch one multi-GPU V2.3 evaluation per checkpoint."""

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_EPOCHS = tuple(range(100, 201, 5))


def parse_checkpoint_epoch(path):
    matches = re.findall(r"(?:epoch[-_]?)(\d+)", Path(path).stem, flags=re.I)
    if len(matches) != 1:
        raise ValueError("cannot parse one checkpoint epoch from {}".format(path))
    return int(matches[0])


def discover_checkpoints(directory, epochs=DEFAULT_EPOCHS):
    expected = {int(epoch) for epoch in epochs}
    found = {}
    for path in sorted(Path(directory).glob("*.pth")):
        try:
            epoch = parse_checkpoint_epoch(path)
        except ValueError:
            continue
        if epoch in found:
            raise ValueError("duplicate checkpoint for epoch {}".format(epoch))
        found[epoch] = path
    missing = sorted(expected - set(found))
    if missing:
        raise FileNotFoundError("missing checkpoints for epochs {}".format(missing))
    return [found[epoch] for epoch in sorted(expected)]


def select_endpoints(rows):
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["epoch"]))
    primary = next((dict(row) for row in ordered if int(row["epoch"]) == 200), None)
    if primary is None:
        raise ValueError("epoch 200 is required as the primary endpoint")
    primary["endpoint"] = "primary_epoch_200"
    secondary = max(
        (dict(row) for row in ordered),
        key=lambda row: (float(row["mIoU"]), -int(row["epoch"])),
    )
    secondary["endpoint"] = "test_selected_best"
    secondary["selection_bias"] = (
        "uses the test split; descriptive paper-compatible endpoint only"
    )
    return ordered, primary, secondary


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_precomputed(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["epoch"] = int(row["epoch"])
        for field in ("mIoU", "pixel_accuracy", "mean_accuracy"):
            if row.get(field) not in (None, ""):
                row[field] = float(row[field])
                row[field + "_percent"] = row[field] * 100.0
        row["metric_unit"] = "fraction_0_to_1"
    return rows


def _copy_endpoint_artifacts(source, output, suffix):
    mapping = {
        "per_class_iou.csv": "per_class_iou_{}.csv".format(suffix),
        "confusion_matrix.csv": "confusion_matrix_{}.csv".format(suffix),
    }
    for source_name, target_name in mapping.items():
        source_path = source / source_name
        if source_path.is_file():
            shutil.copyfile(str(source_path), str(output / target_name))


def write_sweep_outputs(output, rows, evaluation_dirs=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ordered, primary, secondary = select_endpoints(rows)
    fields = sorted({field for row in ordered for field in row})
    _write_csv(output / "metrics_all_checkpoints.csv", ordered, fields)
    (output / "metrics_epoch200.json").write_text(
        json.dumps(primary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "metrics_test_selected_best.json").write_text(
        json.dumps(secondary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if evaluation_dirs:
        _copy_endpoint_artifacts(
            Path(evaluation_dirs[int(primary["epoch"])]), output, "epoch200"
        )
        _copy_endpoint_artifacts(
            Path(evaluation_dirs[int(secondary["epoch"])]),
            output,
            "test_selected_best",
        )
    return ordered, primary, secondary


def resolve_distributed_launcher(requested):
    if requested != "auto":
        return requested
    try:
        spec = importlib.util.find_spec("torch.distributed.run")
    except (ImportError, ModuleNotFoundError):
        spec = None
    if spec is not None:
        return "torch.distributed.run"
    return "torch.distributed.launch"


def build_evaluator_command(
    *,
    python,
    evaluator,
    config_module,
    checkpoint,
    output,
    epoch,
    nproc_per_node,
    launcher,
):
    if int(nproc_per_node) < 1:
        raise ValueError("nproc_per_node must be positive")
    return [
        str(python),
        "-m",
        str(launcher),
        "--nproc_per_node",
        str(int(nproc_per_node)),
        str(evaluator),
        "--config-module",
        str(config_module),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--expected-epoch",
        str(int(epoch)),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--config-module",
        default=(
            "configs.stanford2d3d_s2d."
            "cmx_mit_b2_rel_plus_v2_3_formal"
        ),
    )
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument(
        "--launcher",
        choices=("auto", "torch.distributed.run", "torch.distributed.launch"),
        default="auto",
    )
    parser.add_argument(
        "--epochs", default=",".join(str(value) for value in DEFAULT_EPOCHS)
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--precomputed-metrics",
        type=Path,
        help="plumbing-only input; never a scientific evaluator result",
    )
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("run the sweep parent once; do not launch it under DDP")

    epochs = tuple(int(value) for value in args.epochs.split(",") if value)
    checkpoints = discover_checkpoints(args.checkpoint_root, epochs)
    checkpoint_by_epoch = {
        parse_checkpoint_epoch(path): path for path in checkpoints
    }
    launcher = resolve_distributed_launcher(args.launcher)
    evaluation_dirs = {}
    commands = []
    if args.precomputed_metrics is not None:
        rows = _read_precomputed(args.precomputed_metrics)
        if {int(row["epoch"]) for row in rows} != set(epochs):
            raise ValueError("precomputed metric epochs do not match checkpoints")
        for row in rows:
            row["checkpoint"] = str(checkpoint_by_epoch[int(row["epoch"])])
            row["scientific_metric_reported"] = False
            row["claim"] = "synthetic checkpoint sweep plumbing only"
    else:
        rows = []
        evaluator = Path(__file__).with_name("eval_rel_plus_v2_3_full.py")
        for epoch in sorted(checkpoint_by_epoch):
            evaluation_dir = (
                args.output_root
                / "checkpoint_evaluations"
                / "epoch_{:03d}".format(epoch)
            )
            command = build_evaluator_command(
                python=sys.executable,
                evaluator=evaluator,
                config_module=args.config_module,
                checkpoint=checkpoint_by_epoch[epoch],
                output=evaluation_dir,
                epoch=epoch,
                nproc_per_node=args.nproc_per_node,
                launcher=launcher,
            )
            if args.save_predictions:
                command.append("--save-predictions")
            commands.append(command)
            subprocess.run(command, check=True)
            metrics = json.loads(
                (evaluation_dir / "metrics.json").read_text(encoding="utf-8")
            )
            metrics["epoch"] = epoch
            rows.append(metrics)
            evaluation_dirs[epoch] = evaluation_dir

    ordered, primary, secondary = write_sweep_outputs(
        args.output_root, rows, evaluation_dirs=evaluation_dirs or None
    )
    report = {
        "status": "PASS",
        "checkpoint_count": len(ordered),
        "primary_epoch": int(primary["epoch"]),
        "secondary_epoch": int(secondary["epoch"]),
        "secondary_endpoint": "test_selected_best",
        "selection_bias_disclosed": True,
        "nproc_per_node": args.nproc_per_node,
        "launcher": launcher,
        "sequential_checkpoint_launch": True,
        "commands": commands,
        "precomputed_plumbing_smoke": args.precomputed_metrics is not None,
        "scientific_metric_reported": args.precomputed_metrics is None,
        "file_hash_written": False,
    }
    (args.output_root / "sweep_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
