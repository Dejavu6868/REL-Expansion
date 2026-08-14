#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUN_DIR=${1:?Usage: train_cmx_relplus.sh RUN_DIR [RESUME_CHECKPOINT] [GPU_IDS]}
RESUME=${2:-}
GPU_IDS=${3:-0,1,2,3,4,5,6,7}
PYTHON=/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
PORT=${PORT:-16041}
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/status" "$RUN_DIR/checkpoints" \
  "$RUN_DIR/configs" "$RUN_DIR/environment"

TRAIN_EXIT="$RUN_DIR/status/train.exitcode"
train_postconditions_passed=0
rm -f "$TRAIN_EXIT"
record_train_exit() {
  local code=$?
  if [[ "$train_postconditions_passed" -ne 1 && "$code" -eq 0 ]]; then
    code=1
  fi
  local temporary="$TRAIN_EXIT.$$.tmp"
  printf '%s\n' "$code" > "$temporary"
  mv -f "$temporary" "$TRAIN_EXIT"
}
trap record_train_exit EXIT

printf '%s\n' 'online_relplus_validation' > "$RUN_DIR/status/current_stage"
set +e
"$PYTHON" "$SCRIPT_DIR/validate_relplus_online.py" --run-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/logs/online_relplus_validation.log"
gate_codes=("${PIPESTATUS[@]}")
set -e
if [[ "${gate_codes[0]}" -ne 0 ]]; then
  exit "${gate_codes[0]}"
fi
if [[ "${gate_codes[1]}" -ne 0 ]]; then
  exit "${gate_codes[1]}"
fi
cp "$RUN_DIR/data_reports/online_relplus_validation.json" \
  "$RUN_DIR/data_reports/online_relplus_validation_initial.json"
printf '%s\n' '0' > "$RUN_DIR/status/online_relplus_validation_initial.exitcode"

nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader \
  > "$RUN_DIR/environment/gpu_processes_before_train.txt"
if [[ -s "$RUN_DIR/environment/gpu_processes_before_train.txt" ]]; then
  printf 'Selected host has active GPU compute processes; refusing to preempt them:\n' >&2
  cat "$RUN_DIR/environment/gpu_processes_before_train.txt" >&2
  exit 75
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NPROC=${#GPU_ARRAY[@]}
if [[ "$NPROC" -ne 8 ]]; then
  printf 'Eight-physical/four-logical protocol requires exactly eight GPUs, got %s\n' "$GPU_IDS" >&2
  exit 2
fi
if [[ "$GPU_IDS" != "0,1,2,3,4,5,6,7" ]]; then
  printf 'Formal topology requires physical GPUs 0,1,2,3,4,5,6,7 in order.\n' >&2
  exit 2
fi
GPU_INVENTORY="$RUN_DIR/environment/gpu_inventory.csv"
if [[ ! -s "$GPU_INVENTORY" ]]; then
  printf 'Audited GPU inventory is absent: %s\n' "$GPU_INVENTORY" >&2
  exit 1
fi
if [[ -e "$RUN_DIR/checkpoints/epoch-32.pth" && -z "$RESUME" ]]; then
  printf 'Formal epoch-32 checkpoint already exists; refusing a duplicate train.\n' >&2
  exit 1
fi
if [[ -n "$RESUME" && ! -s "$RESUME" ]]; then
  printf 'Resume checkpoint is absent or empty: %s\n' "$RESUME" >&2
  exit 1
fi

if [[ -n "$RESUME" ]]; then
  "$PYTHON" - "$RESUME" \
    "$RUN_DIR/status/resume_checkpoint_validation.json" \
    "$RUN_DIR/status/resume_checkpoint.sha256" \
    "$RUN_DIR" <<'PY'
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import torch


checkpoint_path = Path(sys.argv[1]).resolve()
report_path = Path(sys.argv[2])
sha_path = Path(sys.argv[3])
run_dir = Path(sys.argv[4]).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


report = {
    "checkpoint": str(checkpoint_path),
    "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "failed",
    "exit_code": 1,
}
try:
    if checkpoint_path.parent != (run_dir / "checkpoints").resolve():
        raise ValueError("resume checkpoint must belong to this run's checkpoints directory")
    digest = sha256_file(checkpoint_path)
    report["sha256"] = digest
    checkpoint = torch.load(str(checkpoint_path), map_location=torch.device("cpu"))
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    required = {"model", "optimizer", "epoch", "iteration"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise KeyError("missing required keys: {}".format(missing))
    if not isinstance(checkpoint["model"], Mapping) or not checkpoint["model"]:
        raise TypeError("model must be a non-empty mapping")
    if not isinstance(checkpoint["optimizer"], Mapping):
        raise TypeError("optimizer must be a mapping")
    optimizer_missing = sorted({"state", "param_groups"} - set(checkpoint["optimizer"]))
    if optimizer_missing:
        raise KeyError("optimizer missing keys: {}".format(optimizer_missing))
    epoch = checkpoint["epoch"]
    iteration = checkpoint["iteration"]
    match = re.fullmatch(r"epoch-(\d+)\.pth", checkpoint_path.name)
    if match is None:
        raise ValueError(
            "resolved resume filename must be epoch-{{4,8,...,28}}.pth, got {!r}".format(
                checkpoint_path.name
            )
        )
    filename_epoch = int(match.group(1))
    allowed_epochs = set(range(4, 32, 4))
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch not in allowed_epochs:
        raise ValueError(
            "resume epoch must be one of {}, got {!r}".format(sorted(allowed_epochs), epoch)
        )
    if filename_epoch != epoch:
        raise ValueError(
            "checkpoint filename epoch {} disagrees with internal epoch {}".format(
                filename_epoch, epoch
            )
        )
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration != 4408:
        raise ValueError("resume iteration must equal 4408, got {!r}".format(iteration))
    report.update(
        {
            "status": "passed",
            "exit_code": 0,
            "keys": sorted(checkpoint.keys()),
            "required_keys": sorted(required),
            "epoch": epoch,
            "iteration": iteration,
            "model_entry_count": len(checkpoint["model"]),
            "optimizer_keys": sorted(checkpoint["optimizer"].keys()),
        }
    )
except Exception as error:
    report["error"] = "{}: {}".format(type(error).__name__, error)

atomic_write(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
if report["exit_code"] == 0:
    atomic_write(sha_path, "{}  {}\n".format(report["sha256"], checkpoint_path))
print(json.dumps(report, indent=2, sort_keys=True))
sys.exit(report["exit_code"])
PY
  resume_sha=$(awk 'NR == 1 {print $1}' "$RUN_DIR/status/resume_checkpoint.sha256")
  printf 'Validated resume checkpoint: path=%s sha256=%s\n' "$RESUME" "$resume_sha" \
    >> "$RUN_DIR/logs/train.log"
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export CMX_RUN_DIR="$RUN_DIR"
export CMX_INITIALIZATION_REPORT="$RUN_DIR/initialization_report.json"
export CMX_GPU_INVENTORY="$GPU_INVENTORY"
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
cd "$REPO"

set +e
"$PYTHON" "$SCRIPT_DIR/validate_training_topology.py" --run-dir "$RUN_DIR" --steps 4409 \
  --gpu-inventory "$GPU_INVENTORY" \
  --smoke-topology "$RUN_DIR/smoke/ddp/configs/training_topology.json" \
  2>&1 | tee "$RUN_DIR/logs/training_topology_preflight.log"
topology_preflight_codes=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${topology_preflight_codes[0]}" > "$RUN_DIR/status/topology_preflight.exitcode"
if [[ "${topology_preflight_codes[0]}" -ne 0 ]]; then exit "${topology_preflight_codes[0]}"; fi
if [[ "${topology_preflight_codes[1]}" -ne 0 ]]; then exit "${topology_preflight_codes[1]}"; fi

nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader \
  > "$RUN_DIR/environment/gpu_processes_immediately_before_train.txt"
if [[ -s "$RUN_DIR/environment/gpu_processes_immediately_before_train.txt" ]]; then
  printf 'Selected host acquired active GPU compute processes; refusing to contend:\n' >&2
  cat "$RUN_DIR/environment/gpu_processes_immediately_before_train.txt" >&2
  exit 75
fi

cmd=("$PYTHON" -m torch.distributed.launch --nproc_per_node="$NPROC" --master_port="$PORT" \
  tools/run_with_config.py --config configs.cmx_relplus_2d train.py -p "$PORT")
if [[ -n "$RESUME" ]]; then
  cmd+=(-c "$RESUME")
fi
command_history="$RUN_DIR/configs/command_history.txt"
printf '# %s\n' "$(date --iso-8601=seconds)" >> "$command_history"
printf '%q ' "${cmd[@]}" >> "$command_history"
printf '\n' >> "$command_history"
command_temporary="$RUN_DIR/configs/command.txt.$$.tmp"
printf '# %s\n' "$(date --iso-8601=seconds)" > "$command_temporary"
printf '%q ' "${cmd[@]}" >> "$command_temporary"
printf '\n' >> "$command_temporary"
mv -f "$command_temporary" "$RUN_DIR/configs/command.txt"
printf '%s\n' "$$" > "$RUN_DIR/status/train.pid"
printf '%s\n' 'formal_training' > "$RUN_DIR/status/current_stage"

set +e
"${cmd[@]}" 2>&1 | tee -a "$RUN_DIR/logs/train.log"
train_codes=("${PIPESTATUS[@]}")
set -e
if [[ "${train_codes[0]}" -ne 0 ]]; then
  exit "${train_codes[0]}"
fi
if [[ "${train_codes[1]}" -ne 0 ]]; then
  exit "${train_codes[1]}"
fi
set +e
"$PYTHON" "$SCRIPT_DIR/validate_training_topology.py" --run-dir "$RUN_DIR" --steps 4409 \
  --gpu-inventory "$GPU_INVENTORY" \
  --smoke-topology "$RUN_DIR/smoke/ddp/configs/training_topology.json" \
  --require-runtime 2>&1 | tee "$RUN_DIR/logs/training_topology_runtime.log"
topology_runtime_codes=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${topology_runtime_codes[0]}" > "$RUN_DIR/status/topology_runtime.exitcode"
if [[ "${topology_runtime_codes[0]}" -ne 0 ]]; then exit "${topology_runtime_codes[0]}"; fi
if [[ "${topology_runtime_codes[1]}" -ne 0 ]]; then exit "${topology_runtime_codes[1]}"; fi
test -s "$RUN_DIR/checkpoints/epoch-32.pth"
checkpoint_sha=$(sha256sum "$RUN_DIR/checkpoints/epoch-32.pth" | awk '{print $1}')
sha_temporary="$RUN_DIR/checkpoints/epoch-32.sha256.$$.tmp"
printf '%s  %s\n' "$checkpoint_sha" "$RUN_DIR/checkpoints/epoch-32.pth" > "$sha_temporary"
mv -f "$sha_temporary" "$RUN_DIR/checkpoints/epoch-32.sha256"
sha256sum -c "$RUN_DIR/checkpoints/epoch-32.sha256"
ln -sfn epoch-32.pth "$RUN_DIR/checkpoints/last.pth"
ln -sfn epoch-32.pth "$RUN_DIR/checkpoints/best.pth"
test -L "$RUN_DIR/checkpoints/last.pth"
test -L "$RUN_DIR/checkpoints/best.pth"
test "$(readlink "$RUN_DIR/checkpoints/last.pth")" = epoch-32.pth
test "$(readlink "$RUN_DIR/checkpoints/best.pth")" = epoch-32.pth
test "$(sha256sum "$RUN_DIR/checkpoints/last.pth" | awk '{print $1}')" = "$checkpoint_sha"
test "$(sha256sum "$RUN_DIR/checkpoints/best.pth" | awk '{print $1}')" = "$checkpoint_sha"
train_postconditions_passed=1
