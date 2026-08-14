#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-}
RUN_DIR=${2:-}
EPOCH=${3:-last}
GPU_ID=${4:-0}
if [[ "$MODE" != "hha" && "$MODE" != "rawdepth" ]]; then
  echo "Usage: $0 <hha|rawdepth> <run_dir> [epoch|last] [physical_gpu_id]" >&2
  exit 2
fi
if [[ ! -d "$RUN_DIR/checkpoints" ]]; then
  echo "Checkpoint directory not found: $RUN_DIR/checkpoints" >&2
  exit 1
fi

REPO=/home/zhuzhaoziao/rel_exp/cmx
PYTHON=/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
CACHE_ROOT=/data/zhuzhaoziao/cmx/cache
CONFIG_MODULE="configs.stanford2d3d_b2_${MODE}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export CMX_RUN_DIR="$RUN_DIR"
export PYTHONPATH="$REPO"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_HOME="$CACHE_ROOT/torch"
export TMPDIR="$CACHE_ROOT/tmp"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR"

cd "$REPO"
CMD=("$PYTHON" tools/run_with_config.py --config "$CONFIG_MODULE" eval.py -d 0 -e "$EPOCH")
printf -v COMMAND_STRING '%q ' "${CMD[@]}"
COMMAND_STRING="CUDA_VISIBLE_DEVICES=$GPU_ID CMX_RUN_DIR=$RUN_DIR ${COMMAND_STRING% }"
SAFE_EPOCH=${EPOCH//\//_}
printf '%s\n' "$COMMAND_STRING" > "$RUN_DIR/eval_command_${SAFE_EPOCH}.txt"
"$PYTHON" tools/save_repro_metadata.py \
  --config "$CONFIG_MODULE" \
  --command "$COMMAND_STRING" \
  --output "$RUN_DIR/eval_metadata_${SAFE_EPOCH}.json"

"${CMD[@]}" 2>&1 | tee "$RUN_DIR/eval_${SAFE_EPOCH}.console.log"
