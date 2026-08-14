#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?Usage: resume_stage2a_after_eval_fix.sh STAGE2A_ROOT}
REPO=/home/zhuzhaoziao/rel_exp/cmx_rel+
PYTHON=/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
ARM=rawdepth
RUN_DIR="$ROOT/$ARM"

test -s "$RUN_DIR/checkpoints/epoch-32.pth"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  printf '%s\n' 'GPU compute processes are already active; refusing recovery run.' >&2
  exit 75
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CMX_RUN_DIR="$RUN_DIR"
export STAGE2A_COMMON_INITIAL_MODEL="$ROOT/initialization/common_initial_model.pth"
export CMX_INITIALIZATION_REPORT="$RUN_DIR/initialization_from_pretrain.json"
export CMX_GPU_INVENTORY="$ROOT/configs/gpu_inventory.csv"
export CMX_METRICS_JSON="$RUN_DIR/metrics/final_metrics.json"
export CMX_PREDICTION_LIMIT=16
export PYTHONPATH="$REPO"
export PYTHONHASHSEED=12345
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1

cd "$REPO"
date --iso-8601=seconds > "$RUN_DIR/status/eval.retry1.started_at"
set +e
"$PYTHON" tools/run_with_config.py --config configs.stage2a_rawdepth eval.py \
  -d 0,1,2,3,4,5,6,7 -e 32 -p "$RUN_DIR/visualizations/predictions" \
  > "$RUN_DIR/logs/eval.retry1.log" 2> "$RUN_DIR/logs/eval.retry1.stderr.log"
eval_code=$?
set -e
printf '%s\n' "$eval_code" > "$RUN_DIR/status/eval.exitcode"
date --iso-8601=seconds > "$RUN_DIR/status/eval.retry1.ended_at"
if [[ "$eval_code" -ne 0 ]]; then
  exit "$eval_code"
fi
test -s "$RUN_DIR/metrics/final_metrics.json"
printf '%s\n' 'COMPLETE' > "$RUN_DIR/status/arm.status"

exec "$REPO/scripts/run_stage2a_all.sh" "$ROOT"
