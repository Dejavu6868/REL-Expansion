#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
DATASET=/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480
OUTPUT_ROOT=/data/zhuzhaoziao/cmx/outputs
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
PREPARE_WORKERS=${PREPARE_WORKERS:-32}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR=${1:-$OUTPUT_ROOT/cmx_relplus_2d_stanford2d3d_mitb2_seed12345_$STAMP}
if [[ -e "$RUN_DIR" || -L "$RUN_DIR" ]]; then
  printf 'Run directory already exists; refusing to overwrite or reuse it: %s\n' "$RUN_DIR" >&2
  exit 73
fi
if ! mkdir "$RUN_DIR"; then
  printf 'Could not atomically reserve run directory: %s\n' "$RUN_DIR" >&2
  exit 73
fi
mkdir -p "$RUN_DIR/status" "$RUN_DIR/logs"
printf '%s\n' "$BASHPID" > "$RUN_DIR/status/run.pid"
printf '%s\n' "$RUN_DIR" > "$RUN_DIR/status/run_dir"

finish() {
  local code=$?
  printf '%s\n' "$code" > "$RUN_DIR/status/run.exitcode"
  if [[ "$code" -ne 0 ]]; then
    printf '# FAILED OR INTERRUPTED\n\nSee `status/current_stage` and stage exit files. Run exit code: %s.\n' "$code" > "$RUN_DIR/STATUS.md"
  fi
}
trap finish EXIT

stage() {
  printf '%s\n' "$1" > "$RUN_DIR/status/current_stage"
  printf '## %s %s\n' "$(date --iso-8601=seconds)" "$1" >> "$RUN_DIR/logs/timeline.log"
}

run_semantics_gate() {
  local log_path=$1
  set +e
  PYTHONPATH="$REPO" "$PYTHON" "$REPO/scripts/validate_relplus_semantics.py" \
    --run-dir "$RUN_DIR" 2>&1 | tee "$log_path"
  local gate_codes=("${PIPESTATUS[@]}")
  set -e
  if [[ "${gate_codes[0]}" -ne 0 ]]; then return "${gate_codes[0]}"; fi
  if [[ "${gate_codes[1]}" -ne 0 ]]; then return "${gate_codes[1]}"; fi
}

stage audit
"$REPO/scripts/audit_environment.sh" "$RUN_DIR" 2>&1 | tee "$RUN_DIR/logs/audit.log"

stage tests
set +e
PYTHONPATH="$REPO" "$PYTHON" -m unittest discover -s "$REPO/tests" -v 2>&1 | tee "$RUN_DIR/logs/tests.log"
tests_codes=("${PIPESTATUS[@]}")
set -e
tests_code=${tests_codes[0]}
if [[ "$tests_code" -eq 0 && "${tests_codes[1]}" -ne 0 ]]; then tests_code=${tests_codes[1]}; fi
printf '%s\n' "$tests_code" > "$RUN_DIR/status/tests.exitcode"
if [[ "$tests_code" -ne 0 ]]; then exit "$tests_code"; fi

stage prepare
set +e
PYTHONPATH="$REPO" "$PYTHON" "$REPO/scripts/prepare_relplus.py" \
  --dataset-root "$DATASET" --run-dir "$RUN_DIR" --workers "$PREPARE_WORKERS" --visual-count 16 \
  --overwrite \
  2>&1 | tee "$RUN_DIR/logs/prepare.log"
prepare_codes=("${PIPESTATUS[@]}")
set -e
prepare_code=${prepare_codes[0]}
if [[ "$prepare_code" -eq 0 && "${prepare_codes[1]}" -ne 0 ]]; then prepare_code=${prepare_codes[1]}; fi
printf '%s\n' "$prepare_code" > "$RUN_DIR/status/prepare.exitcode"
if [[ "$prepare_code" -ne 0 ]]; then exit "$prepare_code"; fi

stage semantics_validation
run_semantics_gate "$RUN_DIR/logs/semantics_validation.log"
cp "$RUN_DIR/data_reports/semantics_validation.json" \
  "$RUN_DIR/data_reports/semantics_validation_initial.json"
printf '%s\n' '0' > "$RUN_DIR/status/semantics_validation_initial.exitcode"

stage smoke
if [[ -s "$RUN_DIR/status/smoke.exitcode" ]] \
  && [[ "$(cat "$RUN_DIR/status/smoke.exitcode")" == "0" ]] \
  && [[ -s "$RUN_DIR/smoke/smoke_report.json" ]] \
  && [[ -s "$RUN_DIR/smoke/ddp/checkpoints/epoch-1.pth" ]] \
  && [[ -s "$RUN_DIR/smoke/ddp/checkpoints/epoch-2.pth" ]] \
  && PYTHONPATH="$REPO" "$PYTHON" "$REPO/scripts/validate_training_topology.py" \
    --run-dir "$RUN_DIR/smoke/ddp" --steps 1 \
    --gpu-inventory "$RUN_DIR/environment/gpu_inventory.csv" --require-runtime \
    > "$RUN_DIR/logs/smoke_topology_reuse.log" 2>&1; then
  printf 'Reusing previously passed smoke evidence in %s\n' "$RUN_DIR/smoke" \
    | tee -a "$RUN_DIR/logs/smoke_test.log"
else
  "$REPO/scripts/smoke_test.sh" "$RUN_DIR" "$GPU_IDS" 2>&1 | tee "$RUN_DIR/logs/smoke_test.log"
fi

stage train
"$REPO/scripts/train_cmx_relplus.sh" "$RUN_DIR" "" "$GPU_IDS"

stage eval
"$REPO/scripts/eval_cmx_relplus.sh" "$RUN_DIR" 0

stage baseline_reeval
"$REPO/scripts/reevaluate_baselines.sh" "$RUN_DIR" 1 2

stage diagnostics
"$REPO/scripts/validation_loss_curve.sh" "$RUN_DIR" "$GPU_IDS"

stage cache_revalidation
set +e
PYTHONPATH="$REPO" "$PYTHON" "$REPO/scripts/validate_relplus_cache.py" --run-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/logs/cache_revalidation.log"
cache_revalidation_codes=("${PIPESTATUS[@]}")
set -e
if [[ "${cache_revalidation_codes[0]}" -ne 0 ]]; then exit "${cache_revalidation_codes[0]}"; fi
if [[ "${cache_revalidation_codes[1]}" -ne 0 ]]; then exit "${cache_revalidation_codes[1]}"; fi
cp "$RUN_DIR/data_reports/cache_validation.json" \
  "$RUN_DIR/data_reports/cache_revalidation.json"
printf '%s\n' '0' > "$RUN_DIR/status/cache_revalidation.exitcode"

stage semantics_revalidation
run_semantics_gate "$RUN_DIR/logs/semantics_revalidation.log"
cp "$RUN_DIR/data_reports/semantics_validation.json" \
  "$RUN_DIR/data_reports/semantics_revalidation.json"
printf '%s\n' '0' > "$RUN_DIR/status/semantics_revalidation.exitcode"

stage compare
PYTHONPATH="$REPO" "$PYTHON" "$REPO/scripts/compare_results.py" --run-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/logs/compare.log"

stage finalize
PYTHONPATH="$REPO" "$PYTHON" "$REPO/scripts/finalize_run.py" --run-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/logs/finalize.log"
printf '%s\n' '0' > "$RUN_DIR/status/run.exitcode"
ln -sfn "$RUN_DIR" "$OUTPUT_ROOT/latest_cmx_relplus_2d"
