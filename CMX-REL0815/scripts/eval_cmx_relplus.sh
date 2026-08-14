#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUN_DIR=${1:?Usage: eval_cmx_relplus.sh RUN_DIR [GPU_ID]}
GPU_ID=${2:-0}
PYTHON=/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
CHECKPOINT="$RUN_DIR/checkpoints/epoch-32.pth"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/status" "$RUN_DIR/metrics" \
  "$RUN_DIR/visualizations/predictions" "$RUN_DIR/environment"

EVAL_EXIT="$RUN_DIR/status/eval.exitcode"
eval_postconditions_passed=0
rm -f "$EVAL_EXIT"
record_eval_exit() {
  local code=$?
  if [[ "$eval_postconditions_passed" -ne 1 && "$code" -eq 0 ]]; then
    code=1
  fi
  local temporary="$EVAL_EXIT.$$.tmp"
  printf '%s\n' "$code" > "$temporary"
  mv -f -- "$temporary" "$EVAL_EXIT"
}
trap record_eval_exit EXIT

test -s "$CHECKPOINT"
sha256sum -c "$RUN_DIR/checkpoints/epoch-32.sha256"
gpu_evidence="$RUN_DIR/environment/gpu_processes_before_eval.txt"
if ! processes=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,name,used_memory \
    --format=csv,noheader 2> "$gpu_evidence.error"); then
  printf 'Unable to inspect target GPU %s.\n' "$GPU_ID" >&2
  exit 69
fi
printf '%s\n' "$processes" > "$gpu_evidence"
rm -f "$gpu_evidence.error"
if [[ -n "$processes" ]]; then
  printf 'Target GPU %s has an external compute process; refusing evaluation.\n' "$GPU_ID" >&2
  exit 75
fi
rm -f "$RUN_DIR/metrics/metrics.json" "$RUN_DIR/metrics/per_class_iou.csv"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export CMX_RUN_DIR="$RUN_DIR"
export CMX_RELPLUS_ROOT="$RUN_DIR/relplus_cache"
export CMX_METRICS_JSON="$RUN_DIR/metrics/metrics.json"
export CMX_PREDICTION_LIMIT=16
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
cd "$REPO"
cmd=("$PYTHON" tools/run_with_config.py --config configs.cmx_relplus_2d eval.py \
  -d 0 -e 32 -p "$RUN_DIR/visualizations/predictions")
printf '%q ' "${cmd[@]}" > "$RUN_DIR/configs/eval_command.txt"
printf '\n' >> "$RUN_DIR/configs/eval_command.txt"
printf '%s\n' 'relplus_evaluation' > "$RUN_DIR/status/current_stage"

set +e
"${cmd[@]}" 2>&1 | tee "$RUN_DIR/logs/eval.log"
eval_codes=("${PIPESTATUS[@]}")
set -e
if [[ "${eval_codes[0]}" -ne 0 ]]; then
  exit "${eval_codes[0]}"
fi
if [[ "${eval_codes[1]}" -ne 0 ]]; then
  exit "${eval_codes[1]}"
fi
test -s "$RUN_DIR/metrics/metrics.json"
test -s "$RUN_DIR/metrics/per_class_iou.csv"
eval_postconditions_passed=1
