#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUN_DIR=${1:?Usage: validation_loss_curve.sh RUN_DIR [GPU_IDS]}
GPU_IDS=${2:-0,1,2,3,4,5,6,7}
PYTHON=/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
EPOCHS=(4 8 12 16 20 24 28 32)
STATUS_PATH="$RUN_DIR/status/diagnostics.exitcode"
pids=()
active=()
status_written=0

atomic_status() {
  local value=$1
  local temporary="$STATUS_PATH.$BASHPID.tmp"
  printf '%s\n' "$value" > "$temporary"
  mv -f -- "$temporary" "$STATUS_PATH"
}

terminate_children() {
  local pid
  local deadline=$((SECONDS + 10))
  for pid in "${pids[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  while (( SECONDS < deadline )); do
    local running=0
    for pid in "${pids[@]:-}"; do
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        running=1
      fi
    done
    (( running == 0 )) && break
    sleep 1
  done
  for pid in "${pids[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]:-}"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  active=()
}

on_exit() {
  local code=$?
  trap - EXIT INT TERM
  terminate_children
  if [[ "$status_written" -eq 0 && -d "$RUN_DIR/status" ]]; then
    if [[ "$code" -eq 0 ]]; then
      code=1
    fi
    atomic_status "$code"
  fi
  exit "$code"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${#GPUS[@]}" -ne "${#EPOCHS[@]}" ]]; then
  printf 'Eight fixed checkpoints require eight GPUs.\n' >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${GPUS[@]}"; do
  if [[ -z "$gpu" || "$gpu" =~ [[:space:]] || -n "${seen_gpus[$gpu]+present}" ]]; then
    printf 'GPU identifiers must be eight unique, non-empty values: %s\n' "$GPU_IDS" >&2
    exit 2
  fi
  seen_gpus[$gpu]=1
done

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/metrics" "$RUN_DIR/status" "$RUN_DIR/environment"
rm -f -- "$STATUS_PATH"
export PYTHONPATH="$REPO"
export CMX_RUN_DIR="$RUN_DIR"
export CMX_RELPLUS_ROOT="$RUN_DIR/relplus_cache"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1

# Preflight all eight checkpoint files (including their serialized epochs and SHA-256)
# before any GPU worker is launched.
for epoch in "${EPOCHS[@]}"; do
  "$PYTHON" "$REPO/scripts/validation_loss.py" \
    --run-dir "$RUN_DIR" --epoch "$epoch" --preflight-only \
    --output "$RUN_DIR/metrics/validation_loss_preflight_epoch_$epoch.json" \
    > "$RUN_DIR/logs/validation_loss_preflight_epoch_$epoch.log" 2>&1
done

# Refuse to share any selected target GPU with an external compute process.
gpu_gate_temporary="$RUN_DIR/environment/gpu_processes_before_diagnostics.$BASHPID.tmp"
: > "$gpu_gate_temporary"
for gpu in "${GPUS[@]}"; do
  if ! processes=$(nvidia-smi -i "$gpu" --query-compute-apps=pid,name,used_memory \
      --format=csv,noheader 2>> "$gpu_gate_temporary"); then
    printf 'Unable to inspect target GPU %s.\n' "$gpu" >&2
    exit 69
  fi
  printf 'GPU %s\n' "$gpu" >> "$gpu_gate_temporary"
  if [[ -n "$processes" ]]; then
    printf '%s\n' "$processes" >> "$gpu_gate_temporary"
    mv -f -- "$gpu_gate_temporary" "$RUN_DIR/environment/gpu_processes_before_diagnostics.txt"
    printf 'Target GPU %s has an external compute process; refusing to launch diagnostics.\n' "$gpu" >&2
    exit 75
  fi
  printf 'no compute processes\n' >> "$gpu_gate_temporary"
done
mv -f -- "$gpu_gate_temporary" "$RUN_DIR/environment/gpu_processes_before_diagnostics.txt"

for index in "${!EPOCHS[@]}"; do
  epoch=${EPOCHS[$index]}
  gpu=${GPUS[$index]}
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$REPO/scripts/validation_loss.py" \
    --run-dir "$RUN_DIR" --epoch "$epoch" --batch-size 4 --workers 4 \
    --output "$RUN_DIR/metrics/validation_loss_epoch_$epoch.json" \
    > "$RUN_DIR/logs/validation_loss_epoch_$epoch.log" 2>&1 &
  pids+=("$!")
  active+=("$!")
done

# Poll all workers so a failure in any position promptly terminates its peers.
while (( ${#active[@]} > 0 )); do
  next_active=()
  for pid in "${active[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      next_active+=("$pid")
      continue
    fi
    if wait "$pid"; then
      :
    else
      code=$?
      terminate_children
      exit "$code"
    fi
  done
  active=("${next_active[@]}")
  (( ${#active[@]} > 0 )) && sleep 1
done
pids=()

"$PYTHON" "$REPO/scripts/combine_loss_curves.py" --run-dir "$RUN_DIR"
for epoch in "${EPOCHS[@]}"; do
  test -s "$RUN_DIR/metrics/validation_loss_preflight_epoch_$epoch.json"
  test -s "$RUN_DIR/metrics/validation_loss_epoch_$epoch.json"
done
test -s "$RUN_DIR/metrics/loss_curves.csv"
test -s "$RUN_DIR/metrics/loss_curves.png"
test -s "$RUN_DIR/metrics/loss_curves_protocol.json"
atomic_status 0
status_written=1
