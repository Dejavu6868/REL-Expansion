#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUN_DIR=${1:?Usage: smoke_test.sh RUN_DIR [GPU_IDS]}
GPU_IDS=${2:-0,1,2,3,4,5,6,7}
PYTHON=/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
SMOKE_DIR="$RUN_DIR/smoke/ddp"
mkdir -p "$RUN_DIR/smoke" "$SMOKE_DIR" "$RUN_DIR/logs" "$RUN_DIR/status"

SMOKE_EXIT="$RUN_DIR/status/smoke.exitcode"
smoke_postconditions_passed=0
rm -f "$SMOKE_EXIT"
record_smoke_exit() {
  local code=$?
  if [[ "$smoke_postconditions_passed" -ne 1 && "$code" -eq 0 ]]; then
    code=1
  fi
  local temporary="$SMOKE_EXIT.$$.tmp"
  printf '%s\n' "$code" > "$temporary"
  mv -f "$temporary" "$SMOKE_EXIT"
}
trap record_smoke_exit EXIT

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NPROC=${#GPU_ARRAY[@]}
if [[ "$NPROC" -ne 8 ]]; then
  printf 'DDP smoke requires eight physical GPUs, got %s\n' "$GPU_IDS" >&2
  exit 2
fi
if [[ "$GPU_IDS" != "0,1,2,3,4,5,6,7" ]]; then
  printf 'DDP smoke requires physical GPUs 0,1,2,3,4,5,6,7 in order.\n' >&2
  exit 2
fi
GPU_INVENTORY="$RUN_DIR/environment/gpu_inventory.csv"
if [[ ! -s "$GPU_INVENTORY" ]]; then
  printf 'Audited GPU inventory is absent: %s\n' "$GPU_INVENTORY" >&2
  exit 1
fi
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader \
  > "$RUN_DIR/environment/gpu_processes_before_smoke.txt"
if [[ -s "$RUN_DIR/environment/gpu_processes_before_smoke.txt" ]]; then
  printf 'Selected host has active GPU compute processes; refusing smoke contention:\n' >&2
  cat "$RUN_DIR/environment/gpu_processes_before_smoke.txt" >&2
  exit 75
fi

export PYTHONPATH="$REPO"
export CMX_RUN_DIR="$RUN_DIR"
export CMX_RELPLUS_ROOT="$RUN_DIR/relplus_cache"
export CMX_SMOKE_SOURCE="$RUN_DIR/data_reports/smoke_split.txt"
export CMX_INITIALIZATION_REPORT="$RUN_DIR/smoke/initialization_report.json"
export CUDA_VISIBLE_DEVICES=${GPU_IDS%%,*}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
cd "$REPO"

set +e
"$PYTHON" scripts/smoke_test.py --run-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/logs/smoke_single_gpu.log"
single_codes=("${PIPESTATUS[@]}")
set -e
if [[ "${single_codes[0]}" -ne 0 ]]; then
  exit "${single_codes[0]}"
fi
if [[ "${single_codes[1]}" -ne 0 ]]; then
  exit "${single_codes[1]}"
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export CMX_RUN_DIR="$SMOKE_DIR"
export CMX_RELPLUS_ROOT="$RUN_DIR/relplus_cache"
export CMX_SMOKE_SOURCE="$RUN_DIR/data_reports/smoke_split.txt"
export CMX_INITIALIZATION_REPORT="$SMOKE_DIR/initialization_report.json"
export CMX_GPU_INVENTORY="$GPU_INVENTORY"
mkdir -p "$SMOKE_DIR/configs" "$SMOKE_DIR/status"

"$PYTHON" "$SCRIPT_DIR/validate_training_topology.py" --run-dir "$SMOKE_DIR" --steps 1 \
  --gpu-inventory "$GPU_INVENTORY" \
  2>&1 | tee "$RUN_DIR/logs/smoke_topology_preflight.log"

run_phase() {
  local epochs=$1
  local port=$2
  local resume=${3:-}
  local log="$RUN_DIR/logs/smoke_ddp_epoch${epochs}.log"
  local cmd=("$PYTHON" -m torch.distributed.launch --nproc_per_node="$NPROC" --master_port="$port" \
    tools/run_with_config.py --config configs.cmx_relplus_2d_ddp_smoke train.py -p "$port")
  if [[ -n "$resume" ]]; then
    cmd+=(-c "$resume")
  fi
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader \
    > "$RUN_DIR/environment/gpu_processes_before_smoke_epoch${epochs}.txt"
  if [[ -s "$RUN_DIR/environment/gpu_processes_before_smoke_epoch${epochs}.txt" ]]; then
    printf 'Selected host has active GPU compute processes; refusing smoke contention:\n' >&2
    cat "$RUN_DIR/environment/gpu_processes_before_smoke_epoch${epochs}.txt" >&2
    return 75
  fi
  CMX_SMOKE_EPOCHS="$epochs" "${cmd[@]}" 2>&1 | tee "$log"
}

run_phase 1 16031
test -s "$SMOKE_DIR/checkpoints/epoch-1.pth"
run_phase 2 16032 "$SMOKE_DIR/checkpoints/epoch-1.pth"
test -s "$SMOKE_DIR/checkpoints/epoch-2.pth"
grep -q 'Epoch 2/2' "$RUN_DIR/logs/smoke_ddp_epoch2.log"

"$PYTHON" "$SCRIPT_DIR/validate_training_topology.py" --run-dir "$SMOKE_DIR" --steps 1 \
  --gpu-inventory "$GPU_INVENTORY" --require-runtime \
  2>&1 | tee "$RUN_DIR/logs/smoke_topology_runtime.log"

"$PYTHON" - <<PY
import json, torch
checkpoints = [
    ('$SMOKE_DIR/checkpoints/epoch-1.pth', 1),
    ('$SMOKE_DIR/checkpoints/epoch-2.pth', 2),
]
report = {}
for path, expected_epoch in checkpoints:
    checkpoint = torch.load(path, map_location='cpu')
    assert set(('model', 'optimizer', 'epoch', 'iteration')).issubset(checkpoint)
    assert checkpoint['epoch'] == expected_epoch, (path, checkpoint['epoch'])
    assert checkpoint['iteration'] == 0, (path, checkpoint['iteration'])
    report[path] = {'epoch': checkpoint['epoch'], 'iteration': checkpoint['iteration']}
with open('$RUN_DIR/smoke/ddp_checkpoint_report.json', 'w') as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
PY
smoke_postconditions_passed=1
