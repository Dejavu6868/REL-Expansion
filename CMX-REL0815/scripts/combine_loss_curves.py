#!/usr/bin/env python3
"""Combine inherited train progress and offline diagnostic validation CE."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re


EXPECTED_EPOCHS = tuple(range(4, 33, 4))
TRAIN_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+Iter\s+(\d+)/(\d+):.*?total_loss="
    r"([-+]?(?:nan|inf(?:inity)?|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?))",
    re.IGNORECASE,
)
TRAIN_COLUMN = "train_logged_epoch_mean_ddp_batch_ce"
VALIDATION_COLUMN = "validation_pixel_weighted_mean_ce"
TRAIN_PROTOCOL = (
    "CMX train.log total_loss at the final iteration: arithmetic mean across "
    "iterations of each DDP rank-averaged minibatch CE"
)


def require_finite_number(value, description):
    if isinstance(value, bool):
        raise ValueError("{} must be numeric, got boolean".format(description))
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be numeric, got {!r}".format(description, value))
    if not math.isfinite(number):
        raise ValueError("{} must be finite, got {!r}".format(description, value))
    return number


def load_validation(run, epoch):
    validation_path = run / "metrics" / "validation_loss_epoch_{}.json".format(epoch)
    preflight_path = run / "metrics" / "validation_loss_preflight_epoch_{}.json".format(
        epoch
    )
    validation = json.loads(validation_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    for payload, name in ((validation, "validation"), (preflight, "preflight")):
        if payload.get("checkpoint_epoch") != epoch:
            raise ValueError("{} checkpoint epoch mismatch at epoch {}".format(name, epoch))
        if payload.get("expected_epoch") != epoch:
            raise ValueError("{} expected epoch mismatch at epoch {}".format(name, epoch))
        checksum = payload.get("checkpoint_sha256")
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("{} checkpoint SHA-256 is invalid at epoch {}".format(name, epoch))
    if validation.get("epoch") != epoch:
        raise ValueError("validation report epoch mismatch at epoch {}".format(epoch))
    if validation["checkpoint_sha256"] != preflight["checkpoint_sha256"]:
        raise ValueError("checkpoint changed after preflight at epoch {}".format(epoch))
    mean_ce = require_finite_number(
        validation.get("mean_cross_entropy"), "validation mean CE at epoch {}".format(epoch)
    )
    loss_sum = require_finite_number(
        validation.get("cross_entropy_sum"), "validation CE sum at epoch {}".format(epoch)
    )
    valid_pixels = validation.get("valid_pixels")
    if not isinstance(valid_pixels, int) or isinstance(valid_pixels, bool) or valid_pixels <= 0:
        raise ValueError("valid pixel count must be a positive integer at epoch {}".format(epoch))
    recomputed = loss_sum / valid_pixels
    if not math.isclose(mean_ce, recomputed, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("validation mean and sum/count disagree at epoch {}".format(epoch))
    protocol = validation.get("protocol")
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("validation protocol is absent at epoch {}".format(epoch))
    return validation, mean_ce


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    text = (run / "logs" / "train.log").read_text(errors="replace").replace("\r", "\n")
    train = {}
    for match in TRAIN_RE.finditer(text):
        epoch = int(match.group(1))
        total_epochs = int(match.group(2))
        iteration = int(match.group(3))
        total_iterations = int(match.group(4))
        if total_epochs != 32 or total_iterations != 4409:
            continue
        if iteration == total_iterations and epoch in EXPECTED_EPOCHS:
            train[epoch] = require_finite_number(
                match.group(5), "logged train loss at epoch {}".format(epoch)
            )
    missing_train = sorted(set(EXPECTED_EPOCHS) - set(train))
    if missing_train:
        raise RuntimeError("missing final-iteration train loss for epochs {}".format(missing_train))

    rows = []
    validation_protocols = set()
    checkpoint_sha256 = {}
    for epoch in EXPECTED_EPOCHS:
        validation, validation_mean = load_validation(run, epoch)
        validation_protocols.add(validation["protocol"])
        checkpoint_sha256[str(epoch)] = validation["checkpoint_sha256"]
        rows.append(
            {
                "epoch": epoch,
                TRAIN_COLUMN: train[epoch],
                VALIDATION_COLUMN: validation_mean,
                "train_protocol": TRAIN_PROTOCOL,
                "validation_protocol": validation["protocol"],
            }
        )
    if len(rows) != 8 or len(validation_protocols) != 1:
        raise RuntimeError("expected exactly eight points and one validation protocol")

    metrics_dir = run / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output = metrics_dir / "loss_curves.csv"
    csv_temporary = output.with_name("{}.{}.tmp".format(output.name, os.getpid()))
    with csv_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_output = metrics_dir / "loss_curves.png"
    png_temporary = png_output.with_name("{}.{}.tmp".format(png_output.name, os.getpid()))
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(EXPECTED_EPOCHS, [row[TRAIN_COLUMN] for row in rows], marker="o", label="train logged CE")
    axis.plot(
        EXPECTED_EPOCHS,
        [row[VALIDATION_COLUMN] for row in rows],
        marker="o",
        label="Area-5 diagnostic CE",
    )
    axis.set_xlabel("epoch")
    axis.set_ylabel("cross entropy")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(png_temporary), format="png", dpi=180)
    plt.close(figure)
    if not png_temporary.is_file() or png_temporary.stat().st_size == 0:
        raise RuntimeError("matplotlib did not produce a non-empty PNG")
    with png_temporary.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError("matplotlib output is not a PNG file")

    protocol_output = metrics_dir / "loss_curves_protocol.json"
    protocol = {
        "checkpoint_epochs": list(EXPECTED_EPOCHS),
        "checkpoint_sha256": checkpoint_sha256,
        "purpose": "post-training diagnostic only; not used for checkpoint selection",
        "train_column": TRAIN_COLUMN,
        "train_protocol": TRAIN_PROTOCOL,
        "validation_column": VALIDATION_COLUMN,
        "validation_protocol": next(iter(validation_protocols)),
    }
    protocol_temporary = protocol_output.with_name(
        "{}.{}.tmp".format(protocol_output.name, os.getpid())
    )
    protocol_temporary.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    if csv_temporary.stat().st_size == 0 or protocol_temporary.stat().st_size == 0:
        raise RuntimeError("diagnostic CSV or protocol report is empty")
    os.replace(str(csv_temporary), str(output))
    os.replace(str(png_temporary), str(png_output))
    os.replace(str(protocol_temporary), str(protocol_output))
    print(output)


if __name__ == "__main__":
    main()
