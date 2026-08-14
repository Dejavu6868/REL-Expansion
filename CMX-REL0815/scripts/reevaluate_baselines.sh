#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUN_DIR=${1:?Usage: reevaluate_baselines.sh RUN_DIR [HHA_GPU] [RAWDEPTH_GPU]}
HHA_GPU=${2:-1}
RAW_GPU=${3:-2}
PYTHON=/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
HHA_RUN=/data/zhuzhaoziao/cmx/outputs/stanford2d3d_b2_hha_formal_seed12345_20260711_004244
RAW_RUN=/data/zhuzhaoziao/cmx/outputs/stanford2d3d_b2_rawdepth_formal_seed12345_20260711_163209
HHA_EXPECTED_SHA256=37df767cb312981e86e8266ff6e552263ebf7b5efc276a1d121d526c3bea0e3e
RAW_EXPECTED_SHA256=1f535608ec16b2d585cdd64f432d3dcd2a34cc20ddca4504019fc6acdb2b295b

mkdir -p \
  "$RUN_DIR/baseline_reeval/hha/configs" \
  "$RUN_DIR/baseline_reeval/rawdepth/configs" \
  "$RUN_DIR/environment" "$RUN_DIR/logs" "$RUN_DIR/status"
printf '%s\n' '1' > "$RUN_DIR/status/baseline_preflight.exitcode"
printf '%s\n' '1' > "$RUN_DIR/status/baseline_hha_eval.exitcode"
printf '%s\n' '1' > "$RUN_DIR/status/baseline_rawdepth_eval.exitcode"

if [[ "$HHA_GPU" == "$RAW_GPU" ]]; then
  printf 'HHA and RawDepth re-evaluation require distinct GPUs.\n' >&2
  exit 2
fi

assert_gpu_idle() {
  local mode=$1
  local gpu=$2
  local uuid
  local processes
  local evidence="$RUN_DIR/environment/gpu_processes_before_baseline_${mode}.txt"
  uuid=$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
  if [[ -z "$uuid" ]]; then
    printf 'Could not resolve physical GPU %s for %s.\n' "$gpu" "$mode" >&2
    exit 2
  fi
  processes=$(nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,name,used_memory --format=csv,noheader \
    | awk -F',' -v selected="$uuid" '
        {
          current=$1
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", current)
          if (current == selected) print $0
        }
      ')
  {
    printf 'mode=%s\nphysical_gpu=%s\ngpu_uuid=%s\n' "$mode" "$gpu" "$uuid"
    printf 'compute_processes:\n'
    if [[ -n "$processes" ]]; then
      printf '%s\n' "$processes"
    else
      printf '%s\n' '<none>'
    fi
  } > "$evidence"
  if [[ -n "$processes" ]]; then
    printf 'GPU %s selected for %s has external compute processes; refusing to launch:\n%s\n' \
      "$gpu" "$mode" "$processes" >&2
    exit 75
  fi
}

validate_checkpoint() {
  local mode=$1
  local source_run=$2
  local expected_sha256=$3
  local destination="$RUN_DIR/baseline_reeval/$mode"
  local checkpoint="$source_run/checkpoints/epoch-32.pth"
  local source_metadata="$source_run/metadata.json"
  test -s "$checkpoint"
  test -s "$source_metadata"
  "$PYTHON" - "$mode" "$checkpoint" "$expected_sha256" "$source_metadata" \
    "$destination/checkpoint_validation.json" <<'PY'
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

mode, checkpoint_arg, expected, metadata_arg, output_arg = sys.argv[1:]
checkpoint = Path(checkpoint_arg).resolve()
metadata = Path(metadata_arg).resolve()
output = Path(output_arg)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

actual = sha256(checkpoint)
metadata_sha = sha256(metadata)
error = None
epoch = None
iteration = None
model_key_count = 0
try:
    payload = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint root is not a mapping")
    epoch = payload.get("epoch")
    iteration = payload.get("iteration")
    model = payload.get("model")
    if type(epoch) is not int or epoch != 32:
        raise ValueError("checkpoint internal epoch is {!r}, expected 32".format(epoch))
    if type(iteration) is not int or iteration != 4408:
        raise ValueError(
            "checkpoint internal iteration is {!r}, expected 4408".format(iteration)
        )
    if not isinstance(model, Mapping) or not model:
        raise ValueError("checkpoint model mapping is absent or empty")
    model_key_count = len(model)
except Exception as exc:  # Preserve a machine-readable failed preflight record.
    error = "{}: {}".format(type(exc).__name__, exc)

report = {
    "mode": mode,
    "checkpoint": str(checkpoint),
    "expected_sha256": expected,
    "actual_sha256": actual,
    "sha256_match": actual == expected,
    "checkpoint_epoch": epoch,
    "checkpoint_iteration": iteration,
    "model_key_count": model_key_count,
    "source_metadata": str(metadata),
    "source_metadata_sha256": metadata_sha,
    "internal_validation_error": error,
    "verified": actual == expected and error is None,
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(str(temporary), str(output))
if actual != expected:
    raise SystemExit("{} SHA-256 mismatch: expected {}, got {}".format(mode, expected, actual))
if error is not None:
    raise SystemExit("{} checkpoint validation failed: {}".format(mode, error))
PY
}

write_evaluation_evidence() {
  local mode=$1
  local source_run=$2
  local destination="$RUN_DIR/baseline_reeval/$mode"
  local config="configs.stanford2d3d_b2_${mode}"
  local command_file="$destination/configs/command.txt"
  local metadata_file="$destination/configs/resolved_config.json"
  local -a cmd=(
    "$PYTHON" "$REPO/tools/run_with_config.py" --config "$config" eval.py -d 0
    -e "$source_run/checkpoints/epoch-32.pth"
  )
  printf '%q ' "${cmd[@]}" > "$command_file"
  printf '\n' >> "$command_file"
  (
    cd "$REPO"
    CUDA_VISIBLE_DEVICES='' \
    CMX_RUN_DIR="$destination" \
    PYTHONPATH="$REPO" \
    PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" "$REPO/tools/save_repro_metadata.py" \
      --config "$config" \
      --command "recorded in $command_file" \
      --output "$metadata_file"
  )
  test -s "$command_file"
  test -s "$metadata_file"
}

run_one() {
  local mode=$1
  local gpu=$2
  local source_run=$3
  local destination="$RUN_DIR/baseline_reeval/$mode"
  local config="configs.stanford2d3d_b2_${mode}"
  local -a cmd=(
    "$PYTHON" "$REPO/tools/run_with_config.py" --config "$config" eval.py -d 0
    -e "$source_run/checkpoints/epoch-32.pth"
  )
  (
    cd "$REPO"
    CUDA_VISIBLE_DEVICES="$gpu" \
    CMX_RUN_DIR="$destination" \
    CMX_METRICS_JSON="$destination/metrics.json" \
    PYTHONPATH="$REPO" \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 \
    "${cmd[@]}"
  ) > "$RUN_DIR/logs/baseline_${mode}_reeval.log" 2>&1
}

printf '%s\n' 'baseline_preflight' > "$RUN_DIR/status/current_stage"
validate_checkpoint hha "$HHA_RUN" "$HHA_EXPECTED_SHA256"
validate_checkpoint rawdepth "$RAW_RUN" "$RAW_EXPECTED_SHA256"
assert_gpu_idle hha "$HHA_GPU"
assert_gpu_idle rawdepth "$RAW_GPU"
write_evaluation_evidence hha "$HHA_RUN"
write_evaluation_evidence rawdepth "$RAW_RUN"
printf '%s\n' '0' > "$RUN_DIR/status/baseline_preflight.exitcode"

printf '%s\n' 'baseline_reevaluation' > "$RUN_DIR/status/current_stage"
rm -f "$RUN_DIR/baseline_reeval/hha/metrics.json" \
  "$RUN_DIR/baseline_reeval/hha/per_class_iou.csv" \
  "$RUN_DIR/baseline_reeval/rawdepth/metrics.json" \
  "$RUN_DIR/baseline_reeval/rawdepth/per_class_iou.csv"
printf '%s\n' '1' > "$RUN_DIR/status/baseline_hha_eval.exitcode"
printf '%s\n' '1' > "$RUN_DIR/status/baseline_rawdepth_eval.exitcode"
set +e
run_one hha "$HHA_GPU" "$HHA_RUN" &
hha_pid=$!
run_one rawdepth "$RAW_GPU" "$RAW_RUN" &
raw_pid=$!
wait "$hha_pid"; hha_code=$?
wait "$raw_pid"; raw_code=$?
set -e
if [[ "$hha_code" -eq 0 && ! -s "$RUN_DIR/baseline_reeval/hha/metrics.json" ]]; then
  hha_code=1
fi
if [[ "$raw_code" -eq 0 && ! -s "$RUN_DIR/baseline_reeval/rawdepth/metrics.json" ]]; then
  raw_code=1
fi
printf '%s\n' "$hha_code" > "$RUN_DIR/status/baseline_hha_eval.exitcode"
printf '%s\n' "$raw_code" > "$RUN_DIR/status/baseline_rawdepth_eval.exitcode"
if [[ "$hha_code" -ne 0 || "$raw_code" -ne 0 ]]; then
  exit 1
fi
