#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUN_DIR=${1:?Usage: audit_environment.sh RUN_DIR}
PYTHON=/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
DATASET=/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480

mkdir -p "$RUN_DIR"/{environment,configs,logs,metrics,visualizations/predictions,status}
nvidia-smi > "$RUN_DIR/environment/nvidia-smi.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits \
  > "$RUN_DIR/environment/gpu_inventory.csv"
"$PYTHON" -m pip freeze > "$RUN_DIR/environment/pip-freeze.txt"
if command -v conda >/dev/null 2>&1; then
  conda env export > "$RUN_DIR/environment/conda-env.yaml"
else
  printf '%s\n' 'not a conda-managed environment; see pip-freeze.txt' > "$RUN_DIR/environment/conda-env.yaml"
fi
{
  printf 'repo=%s\n' "$REPO"
  git -C "$REPO" rev-parse HEAD
  git -C "$REPO" status --short --branch
  printf 'source_cmx=/home/zhuzhaoziao/rel_exp/cmx\n'
  git -C /home/zhuzhaoziao/rel_exp/cmx rev-parse HEAD
  git -C /home/zhuzhaoziao/rel_exp/cmx status --short --branch
  printf 'source_rel=/data/bxh_copy/Pano_MA_Seg\nsource_rel_commit=unknown-no-git-ref\n'
} > "$RUN_DIR/environment/git-info.txt"
{
  uname -a
  "$PYTHON" --version
  "$PYTHON" - <<'PY'
import cv2, numpy, torch
print('numpy={}'.format(numpy.__version__))
print('opencv={}'.format(cv2.__version__))
print('torch={}'.format(torch.__version__))
print('torch_cuda={}'.format(torch.version.cuda))
print('cudnn={}'.format(torch.backends.cudnn.version()))
print('cuda_available={}'.format(torch.cuda.is_available()))
print('gpu_count={}'.format(torch.cuda.device_count()))
PY
  lscpu
  free -h
  df -h / /data
} > "$RUN_DIR/environment/system.txt"
sha256sum "$DATASET/train.txt" "$DATASET/test.txt" \
  /data/zhuzhaoziao/cmx/raw/pretrained/segformer/mit_b2.pth \
  > "$RUN_DIR/environment/input-sha256.txt"

git -C "$REPO" diff --binary HEAD > "$RUN_DIR/code_diff.patch"
while IFS= read -r file; do
  git -C "$REPO" diff --no-index --binary /dev/null "$REPO/$file" >> "$RUN_DIR/code_diff.patch" || true
done < <(git -C "$REPO" ls-files --others --exclude-standard | sort)
(
  cd "$REPO"
  find . \( -path './.git' -o -name '__pycache__' \) -prune -o \
    -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum
) > "$RUN_DIR/code_manifest.sha256"

export CMX_RUN_DIR="$RUN_DIR"
export CMX_RELPLUS_ROOT="$RUN_DIR/relplus_cache"
export PYTHONPATH="$REPO"
cd "$REPO"
"$PYTHON" tools/save_repro_metadata.py \
  --config configs.cmx_relplus_2d \
  --command 'recorded separately in configs/command.txt' \
  --output "$RUN_DIR/configs/resolved_config.json"
cp configs/cmx_relplus_2d.py "$RUN_DIR/configs/"
cp configs/stanford2d3d_b2_common.py "$RUN_DIR/configs/"
cp AUDIT.md "$RUN_DIR/AUDIT.md"
cp DEVIATIONS.md "$RUN_DIR/DEVIATIONS.md"

repo_symlink=$(find "$REPO" -path "$REPO/.git" -prune -o -type l -print -quit)
if [[ -n "$repo_symlink" ]]; then
  printf 'Repository contains a symlink and cannot be immutably audited: %s\n' \
    "$repo_symlink" >&2
  exit 74
fi

printf '%s\n' '0' > "$RUN_DIR/status/audit.exitcode"
