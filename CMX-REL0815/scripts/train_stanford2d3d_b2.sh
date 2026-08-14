#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-}
RUN_KIND=${2:-formal}
if [[ "$MODE" != "hha" && "$MODE" != "rawdepth" ]]; then
  echo "Usage: $0 <hha|rawdepth> [formal|smoke|ddp_smoke]" >&2
  exit 2
fi
if [[ "$RUN_KIND" != "formal" && "$RUN_KIND" != "smoke" && "$RUN_KIND" != "ddp_smoke" ]]; then
  echo "RUN_KIND must be formal, smoke, or ddp_smoke" >&2
  exit 2
fi
if [[ "$RUN_KIND" == "ddp_smoke" && "$MODE" != "rawdepth" ]]; then
  echo "ddp_smoke uses RawDepth to verify the shared distributed training path" >&2
  exit 2
fi

REPO=/home/zhuzhaoziao/rel_exp/cmx
PYTHON=/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
OUTPUT_ROOT=/data/zhuzhaoziao/cmx/outputs
CACHE_ROOT=/data/zhuzhaoziao/cmx/cache
PORT=${PORT:-16005}

if [[ "$RUN_KIND" == "formal" ]]; then
  GPU_IDS=${GPU_IDS:-0,1,2,3}
  CONFIG_MODULE="configs.stanford2d3d_b2_${MODE}"
elif [[ "$RUN_KIND" == "smoke" ]]; then
  GPU_IDS=${GPU_IDS:-0}
  CONFIG_MODULE="configs.stanford2d3d_b2_${MODE}_smoke"
else
  GPU_IDS=${GPU_IDS:-0,1,2,3}
  CONFIG_MODULE="configs.stanford2d3d_b2_rawdepth_ddp_smoke"
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NPROC=${#GPU_ARRAY[@]}
if [[ "$RUN_KIND" != "smoke" && "$NPROC" -ne 4 ]]; then
  echo "Formal and DDP smoke runs require exactly four visible GPUs; got $GPU_IDS" >&2
  exit 2
fi
if [[ "$RUN_KIND" == "smoke" && "$NPROC" -ne 1 ]]; then
  echo "Smoke tests require exactly one visible GPU; got $GPU_IDS" >&2
  exit 2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID=${RUN_ID:-stanford2d3d_b2_${MODE}_${RUN_KIND}_seed12345_${STAMP}}
RUN_DIR=${CMX_RUN_DIR:-$OUTPUT_ROOT/$RUN_ID}
if [[ -d "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty run directory: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
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
if [[ "$RUN_KIND" != "smoke" ]]; then
  CMD=("$PYTHON" -m torch.distributed.launch --nproc_per_node="$NPROC" tools/run_with_config.py --config "$CONFIG_MODULE" train.py -p "$PORT")
else
  CMD=("$PYTHON" tools/run_with_config.py --config "$CONFIG_MODULE" train.py -d 0)
fi
printf -v COMMAND_STRING '%q ' "${CMD[@]}"
COMMAND_STRING="CUDA_VISIBLE_DEVICES=$GPU_IDS CMX_RUN_DIR=$RUN_DIR ${COMMAND_STRING% }"
printf '%s\n' "$COMMAND_STRING" > "$RUN_DIR/command.txt"

"$PYTHON" tools/save_repro_metadata.py \
  --config "$CONFIG_MODULE" \
  --command "$COMMAND_STRING" \
  --output "$RUN_DIR/metadata.json"
NEW_FILES=(
  configs/__init__.py
  configs/stanford2d3d_b2_common.py
  configs/stanford2d3d_b2_hha.py
  configs/stanford2d3d_b2_hha_smoke.py
  configs/stanford2d3d_b2_rawdepth.py
  configs/stanford2d3d_b2_rawdepth_ddp_smoke.py
  configs/stanford2d3d_b2_rawdepth_smoke.py
  tools/run_with_config.py
  tools/save_repro_metadata.py
  tools/prepare_stanford2d3d.py
  tools/verify_stanford2d3d.py
  tools/parse_cmx_results.py
  scripts/train_stanford2d3d_b2.sh
  scripts/eval_stanford2d3d_b2.sh
)
git diff --binary > "$RUN_DIR/source.diff"
for file in "${NEW_FILES[@]}"; do
  git diff --no-index --binary /dev/null "$file" >> "$RUN_DIR/source.diff" || true
done
git status --short --branch > "$RUN_DIR/git_status.txt"
df -h / /data > "$RUN_DIR/disk_before.txt"
du -sh /data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480 > "$RUN_DIR/dataset_size.txt"
mkdir -p "$RUN_DIR/config_snapshot"
cp configs/stanford2d3d_b2_common.py "$RUN_DIR/config_snapshot/"
cp "${CONFIG_MODULE//.//}.py" "$RUN_DIR/config_snapshot/"

"${CMD[@]}" 2>&1 | tee "$RUN_DIR/train.console.log"
