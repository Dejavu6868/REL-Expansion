#!/usr/bin/env bash
set -u

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: run_fold1_arm.sh ARM_NAME PAPER_TARGET PORT [RESUME_CHECKPOINT]" >&2
  exit 64
fi

ARM_NAME=$1
PAPER_TARGET=$2
PORT=$3
RESUME_CHECKPOINT=${4:-}
CODE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_ROOT=/data/zhuzhaoziao/RELPlus/outputs/CMX_S3D_Fold1_reproduction
RUN_DIR="$OUTPUT_ROOT/$ARM_NAME"
PYTHON=/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
CONFIG_MODULE=configs.stanford2d3dpano.fold1
CONFIG_FILE="$CODE_ROOT/configs/stanford2d3dpano/fold1.py"
EVAL_PORT=$((PORT + 100))

mkdir -p "$RUN_DIR"/{checkpoints,logs,status,metrics,tensorboard,predictions_best,visualizations}

stage() {
  printf '%s\n' "$1" > "$RUN_DIR/status/current_stage"
}

fail() {
  printf '%s\n' "$1" > "$RUN_DIR/status/run.exitcode"
  stage failed
  exit "$1"
}

printf '%s\n' "$$" > "$RUN_DIR/status/orchestrator.pid"
cp "$CONFIG_FILE" "$RUN_DIR/config_snapshot.py"
cd "$CODE_ROOT"
"$PYTHON" tools/export_resolved_config.py --config-module "$CONFIG_MODULE" --output-dir "$RUN_DIR" \
  > "$RUN_DIR/logs/config_export.log" 2>&1 || fail 65

{
  echo "code_path=$CODE_ROOT"
  echo "branch=$(git branch --show-current)"
  git log -1 --pretty='latest_commit_date=%ad%nlatest_commit_title=%s' --date=iso
  "$PYTHON" -c 'import platform, torch; print("python=" + platform.python_version()); print("pytorch=" + torch.__version__); print("torch_cuda=" + str(torch.version.cuda))'
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  echo "gpu_count=8"
  echo "global_batch=8"
  echo "per_gpu_batch=1"
  echo "sync_batchnorm=true"
  echo "amp=false"
  echo "dataset_path=$OUTPUT_ROOT/common/Stanford2D3DPano"
  echo "pretrained=/data/zhuzhaoziao/cmx/raw/pretrained/segformer/mit_b2.pth"
} > "$RUN_DIR/environment.txt"

GPU_COUNT=$(nvidia-smi -L | wc -l)
if [[ "$GPU_COUNT" -ne 8 ]]; then
  echo "expected 8 GPUs, observed $GPU_COUNT" >&2
  fail 66
fi
GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "${GPU_PROCESSES//[[:space:]]/}" ]]; then
  echo "GPU compute processes already exist" >&2
  fail 67
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TRAIN_COMMAND=(
  "$PYTHON" -m torch.distributed.launch
  --nproc_per_node=8
  --master_port="$PORT"
  tools/run_with_config.py
  --config-module "$CONFIG_MODULE"
  train.py
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  TRAIN_COMMAND+=(--continue "$RESUME_CHECKPOINT")
fi
printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train_command.txt"
printf '\n' >> "$RUN_DIR/train_command.txt"
printf 'cd %q && nohup bash tools/run_fold1_arm.sh %q %q %q %q > %q 2>&1 &\n' \
  "$CODE_ROOT" "$ARM_NAME" "$PAPER_TARGET" "$PORT" "$RUN_DIR/checkpoints/epoch-last.pth" \
  "$RUN_DIR/logs/orchestrator.resume.log" > "$RUN_DIR/resume_command.txt"

stage training
set +e
"${TRAIN_COMMAND[@]}" > "$RUN_DIR/train.log" 2>&1
TRAIN_RC=$?
set -e
printf '%s\n' "$TRAIN_RC" > "$RUN_DIR/status/train.exitcode"
if [[ "$TRAIN_RC" -ne 0 ]]; then
  fail "$TRAIN_RC"
fi

for epoch in $(seq 100 5 200); do
  if [[ ! -f "$RUN_DIR/checkpoints/epoch-${epoch}.pth" ]]; then
    echo "missing checkpoint epoch $epoch" >&2
    fail 68
  fi
done

stage evaluation
: > "$RUN_DIR/eval.log"
cat > "$RUN_DIR/eval_command.txt" <<EOF
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=8 --master_port=$EVAL_PORT tools/eval_fold1.py --config-module $CONFIG_MODULE --checkpoint CHECKPOINT --metrics METRICS --per-class PER_CLASS --confusion CONFUSION
EOF

for epoch in $(seq 100 5 200); do
  EPOCH_LOG="$RUN_DIR/logs/eval_epoch${epoch}.log"
  set +e
  "$PYTHON" -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port="$EVAL_PORT" \
    tools/eval_fold1.py \
    --config-module "$CONFIG_MODULE" \
    --checkpoint "$RUN_DIR/checkpoints/epoch-${epoch}.pth" \
    --metrics "$RUN_DIR/metrics/metrics_epoch${epoch}.json" \
    --per-class "$RUN_DIR/metrics/per_class_epoch${epoch}.csv" \
    --confusion "$RUN_DIR/metrics/confusion_epoch${epoch}.csv" \
    > "$EPOCH_LOG" 2>&1
  EVAL_RC=$?
  set -e
  printf '%s\n' "$EVAL_RC" > "$RUN_DIR/status/eval_epoch${epoch}.exitcode"
  cat "$EPOCH_LOG" >> "$RUN_DIR/eval.log"
  if [[ "$EVAL_RC" -ne 0 ]]; then
    fail "$EVAL_RC"
  fi
done

"$PYTHON" tools/summarize_checkpoints.py --run-dir "$RUN_DIR" --target "$PAPER_TARGET" \
  > "$RUN_DIR/logs/summarize.log" 2>&1 || fail 69
BEST_EPOCH=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_epoch"])' "$RUN_DIR/metrics_best.json")

stage best_prediction_export
set +e
"$PYTHON" -m torch.distributed.launch \
  --nproc_per_node=8 \
  --master_port="$EVAL_PORT" \
  tools/eval_fold1.py \
  --config-module "$CONFIG_MODULE" \
  --checkpoint "$RUN_DIR/checkpoints/epoch-${BEST_EPOCH}.pth" \
  --metrics "$RUN_DIR/metrics/metrics_best_replay.json" \
  --per-class "$RUN_DIR/metrics/per_class_best_replay.csv" \
  --confusion "$RUN_DIR/metrics/confusion_best_replay.csv" \
  --predictions "$RUN_DIR/predictions_best" \
  > "$RUN_DIR/logs/eval_best_predictions.log" 2>&1
PRED_RC=$?
set -e
printf '%s\n' "$PRED_RC" > "$RUN_DIR/status/best_predictions.exitcode"
if [[ "$PRED_RC" -ne 0 ]]; then
  fail "$PRED_RC"
fi

printf '0\n' > "$RUN_DIR/status/run.exitcode"
stage complete
