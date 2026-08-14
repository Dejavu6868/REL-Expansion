#!/usr/bin/env bash
set -euo pipefail

ARM=${1:?Usage: train_stage2a_arm.sh ARM STAGE2A_ROOT [PORT]}
STAGE2A_ROOT=${2:?Usage: train_stage2a_arm.sh ARM STAGE2A_ROOT [PORT]}
PORT=${3:-16120}
case "$ARM" in
  rawdepth|hha|relplus_local|relplus_pose) ;;
  *) printf 'Unknown Stage2A arm: %s\n' "$ARM" >&2; exit 2 ;;
esac

REPO=/home/zhuzhaoziao/rel_exp/cmx_rel+
PYTHON=/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
RUN_DIR="$STAGE2A_ROOT/$ARM"
CONFIG="configs.stage2a_${ARM}"
INITIAL="$STAGE2A_ROOT/initialization/common_initial_model.pth"
GPU_IDS=0,1,2,3,4,5,6,7
mkdir -p "$RUN_DIR"/{checkpoints,status,logs,metrics,configs,environment,visualizations/predictions}
test -s "$INITIAL"
active_processes=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader)
if [[ -n "$active_processes" ]]; then
  printf 'GPU compute processes are already active; refusing to contend:\n%s\n' "$active_processes" >&2
  exit 75
fi
if [[ -s "$RUN_DIR/checkpoints/epoch-32.pth" ]]; then
  printf 'Formal Stage2A checkpoint already exists for %s; refusing overwrite.\n' "$ARM" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export CMX_RUN_DIR="$RUN_DIR"
export STAGE2A_COMMON_INITIAL_MODEL="$INITIAL"
export CMX_INITIALIZATION_REPORT="$RUN_DIR/initialization_from_pretrain.json"
export CMX_GPU_INVENTORY="$STAGE2A_ROOT/configs/gpu_inventory.csv"
export PYTHONPATH="$REPO"
export PYTHONHASHSEED=12345
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1

cd "$REPO"
cp "${CONFIG//.//}.py" "$RUN_DIR/configs/resolved_config.py"
cp configs/stage2a_common.py "$RUN_DIR/configs/stage2a_common.py"
git status --short --branch > "$RUN_DIR/configs/git_status.txt"
git diff --binary > "$RUN_DIR/configs/source.diff"
cmd=("$PYTHON" -m torch.distributed.launch --nproc_per_node=8 --master_port="$PORT" \
  tools/run_with_config.py --config "$CONFIG" train.py -p "$PORT")
printf '%q ' "${cmd[@]}" > "$RUN_DIR/configs/train_command.txt"; printf '\n' >> "$RUN_DIR/configs/train_command.txt"
printf '%s\n' "$$" > "$RUN_DIR/status/launcher.pid"
date --iso-8601=seconds > "$RUN_DIR/status/train.started_at"
set +e
"${cmd[@]}" > "$RUN_DIR/logs/train.log" 2> "$RUN_DIR/logs/train.stderr.log"
code=$?
set -e
printf '%s\n' "$code" > "$RUN_DIR/status/train.exitcode"
date --iso-8601=seconds > "$RUN_DIR/status/train.ended_at"
if [[ "$code" -ne 0 ]]; then exit "$code"; fi
test -s "$RUN_DIR/checkpoints/epoch-32.pth"
ln -sfn epoch-32.pth "$RUN_DIR/checkpoints/final.pth"
ln -sfn epoch-32.pth "$RUN_DIR/checkpoints/selected.pth"

export CMX_METRICS_JSON="$RUN_DIR/metrics/final_metrics.json"
export CMX_PREDICTION_LIMIT=16
eval_cmd=("$PYTHON" tools/run_with_config.py --config "$CONFIG" eval.py \
  -d 0,1,2,3,4,5,6,7 -e 32 -p "$RUN_DIR/visualizations/predictions")
printf '%q ' "${eval_cmd[@]}" > "$RUN_DIR/configs/eval_command.txt"; printf '\n' >> "$RUN_DIR/configs/eval_command.txt"
date --iso-8601=seconds > "$RUN_DIR/status/eval.started_at"
set +e
"${eval_cmd[@]}" > "$RUN_DIR/logs/eval.log" 2> "$RUN_DIR/logs/eval.stderr.log"
eval_code=$?
set -e
printf '%s\n' "$eval_code" > "$RUN_DIR/status/eval.exitcode"
date --iso-8601=seconds > "$RUN_DIR/status/eval.ended_at"
if [[ "$eval_code" -ne 0 ]]; then exit "$eval_code"; fi
test -s "$RUN_DIR/metrics/final_metrics.json"
printf '%s\n' 'COMPLETE' > "$RUN_DIR/status/arm.status"
