#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


METRIC_RE = re.compile(
    r"mean_IoU\s+([0-9.]+)%.*?mean_pixel_acc\s+([0-9.]+)%.*?pixel_acc\s+([0-9.]+)%"
)
EPOCH_RE = re.compile(r"epoch-([0-9]+|last)\.pth$")


def find_validation_log(run_dir):
    preferred = run_dir / "val_last.log"
    if preferred.exists():
        return preferred
    candidates = sorted(run_dir.glob("val_*.log"))
    return candidates[-1] if candidates else None


def parse_log(path):
    rows = []
    checkpoint = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("Model: "):
            checkpoint = line[len("Model: "):].strip()
            continue
        match = METRIC_RE.search(line)
        if match:
            epoch_match = EPOCH_RE.search(checkpoint or "")
            epoch = epoch_match.group(1) if epoch_match else "MISSING"
            if epoch == "last" and checkpoint:
                resolved_match = EPOCH_RE.search(str(Path(checkpoint).resolve()))
                if resolved_match:
                    epoch = resolved_match.group(1)
            rows.append({
                "checkpoint": checkpoint or "MISSING",
                "epoch": epoch,
                "miou": match.group(1),
                "mean_pixel_acc": match.group(2),
                "pixel_acc": match.group(3),
            })
    return rows


def load_metadata(run_dir):
    path = run_dir / "metadata.json"
    if not path.is_file():
        return {}, "MISSING"
    return json.loads(path.read_text()), str(path)


def load_eval_metadata(run_dir, epoch):
    preferred = run_dir / "eval_metadata_{}.json".format(epoch)
    if preferred.is_file():
        return json.loads(preferred.read_text()), str(preferred)
    candidates = list(run_dir.glob("eval_metadata_*.json"))
    if not candidates:
        return {}, "MISSING"
    path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    return json.loads(path.read_text()), str(path)


def result_rows(run_dir):
    metadata, metadata_path = load_metadata(run_dir)
    config = metadata.get("resolved_config", {})
    source_diff = run_dir / "source.diff"
    source_diff_path = str(source_diff) if source_diff.is_file() else "MISSING"
    log_path = find_validation_log(run_dir)
    parsed = parse_log(log_path) if log_path else []
    if not parsed:
        parsed = [{
            "checkpoint": "MISSING",
            "epoch": "MISSING",
            "miou": "MISSING",
            "mean_pixel_acc": "MISSING",
            "pixel_acc": "MISSING",
        }]

    rows = []
    for metrics in parsed:
        eval_metadata, eval_metadata_path = load_eval_metadata(run_dir, metrics["epoch"])
        modality = config.get("modality", "MISSING")
        checkpoint = metrics["checkpoint"]
        checkpoint_exists = checkpoint != "MISSING" and Path(checkpoint).exists()
        train_environment = metadata.get("environment", {})
        eval_environment = eval_metadata.get("environment", {})
        required_provenance = (
            metadata.get("git_commit"),
            metadata.get("config_path"),
            config.get("dataset_path"),
            config.get("seed"),
            config.get("effective_distributed_seeds"),
            train_environment.get("cuda_visible_devices"),
            eval_environment.get("cuda_visible_devices"),
            train_environment.get("pytorch"),
            train_environment.get("torch_cuda"),
            metadata.get("command"),
            eval_metadata.get("config_path"),
            eval_metadata.get("command"),
        )
        provenance_complete = all(value not in (None, "", "MISSING") for value in required_provenance)
        rows.append({
            "run_id": run_dir.name,
            "modality": modality,
            "backbone": config.get("backbone", "MISSING"),
            "decoder": config.get("decoder", "MISSING"),
            "eval_mode": "SS",
            "miou": metrics["miou"],
            "pixel_acc": metrics["pixel_acc"],
            "mean_pixel_acc": metrics["mean_pixel_acc"],
            "target_miou": "61.2" if modality == "hha" else "",
            "target_pixel_acc": "82.3" if modality == "hha" else "",
            "epoch": metrics["epoch"],
            "checkpoint": checkpoint,
            "checkpoint_exists": checkpoint_exists,
            "log_path": str(log_path) if log_path else "MISSING",
            "metadata_path": metadata_path,
            "eval_metadata_path": eval_metadata_path,
            "source_diff_path": source_diff_path,
            "config_module": metadata.get("config_module", "MISSING"),
            "config_path": metadata.get("config_path", "MISSING"),
            "dataset_root": config.get("dataset_path", "MISSING"),
            "seed": config.get("seed", "MISSING"),
            "effective_distributed_seeds": json.dumps(config.get("effective_distributed_seeds", "MISSING")),
            "train_gpu_ids": train_environment.get("cuda_visible_devices", "MISSING"),
            "eval_gpu_ids": eval_environment.get("cuda_visible_devices", "MISSING"),
            "pytorch": train_environment.get("pytorch", "MISSING"),
            "cuda": train_environment.get("torch_cuda", "MISSING"),
            "git_commit": metadata.get("git_commit", "MISSING"),
            "train_command": metadata.get("command", "MISSING"),
            "command": eval_metadata.get("command", "MISSING"),
            "provenance_complete": provenance_complete,
            "status": "OK" if (
                metrics["miou"] != "MISSING"
                and checkpoint_exists
                and metadata_path != "MISSING"
                and eval_metadata_path != "MISSING"
                and source_diff_path != "MISSING"
                and provenance_complete
            ) else "MISSING",
        })
    return rows


def write_markdown(path, rows):
    columns = ["run_id", "modality", "miou", "pixel_acc", "epoch", "checkpoint", "log_path", "status"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]).replace("|", "\\|") for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path, help="Repeat for each run")
    parser.add_argument(
        "--output", type=Path,
        default=Path("/data/zhuzhaoziao/cmx/outputs/results/cmx_stanford2d3d_reproduction.csv"),
    )
    args = parser.parse_args()

    rows = [row for run_dir in args.run_dir for row in result_rows(run_dir)]
    if not rows:
        raise ValueError("no run directories with metadata were found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(args.output.with_suffix(".md"), rows)
    print(args.output)


if __name__ == "__main__":
    main()
