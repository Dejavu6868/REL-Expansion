#!/usr/bin/env bash
set -euo pipefail
STAGE2A_ROOT=${1:?Usage: run_stage2a_per_image_reeval.sh STAGE2A_ROOT STAGE2B_ROOT}
STAGE2B_ROOT=${2:?Usage: run_stage2a_per_image_reeval.sh STAGE2A_ROOT STAGE2B_ROOT}
REPO=/home/zhuzhaoziao/rel_exp/cmx_rel+
PYTHON=/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
mkdir -p "$STAGE2B_ROOT/seed_12345/status"
printf '%s\n' "$$" > "$STAGE2B_ROOT/seed_12345/status/reeval.pid"
date --iso-8601=seconds > "$STAGE2B_ROOT/seed_12345/status/reeval.started_at"
for arm in rawdepth hha relplus_local relplus_pose; do
  RUN_DIR="$STAGE2A_ROOT/$arm"
  test -s "$RUN_DIR/checkpoints/epoch-32.pth"
  mkdir -p "$RUN_DIR/visualizations/stage2b_reeval_predictions"
  if [[ -s "$RUN_DIR/metrics/per_image_metrics.csv" ]]; then
    continue
  fi
  active_processes=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader)
  if [[ -n "$active_processes" ]]; then
    printf 'GPU compute processes are already active; refusing to contend:\n%s\n' "$active_processes" >&2
    exit 75
  fi
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export CMX_RUN_DIR="$RUN_DIR"
  export STAGE2A_COMMON_INITIAL_MODEL="$STAGE2A_ROOT/initialization/common_initial_model.pth"
  export CMX_METRICS_JSON="$RUN_DIR/metrics/final_metrics_stage2b_reeval.json"
  export CMX_PER_IMAGE_METRICS_CSV="$RUN_DIR/metrics/per_image_metrics.csv"
  export CMX_PREDICTION_LIMIT=0
  export PYTHONPATH="$REPO"
  export PYTHONHASHSEED=12345
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
  cd "$REPO"
  "$PYTHON" tools/run_with_config.py --config "configs.stage2a_$arm" eval.py \
    -d 0,1,2,3,4,5,6,7 -e 32 -p "$RUN_DIR/visualizations/stage2b_reeval_predictions" \
    > "$RUN_DIR/logs/stage2b_per_image_reeval.log" \
    2> "$RUN_DIR/logs/stage2b_per_image_reeval.stderr.log"
  test -s "$RUN_DIR/metrics/per_image_metrics.csv"
done
printf '%s\n' 'COMPLETE_STAGE2A_PER_IMAGE_REEVALUATION' > "$STAGE2B_ROOT/seed_12345/status/reeval.status"
date --iso-8601=seconds > "$STAGE2B_ROOT/seed_12345/status/reeval.ended_at"
