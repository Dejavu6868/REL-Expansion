#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT=/data/zhuzhaoziao/RELPlus/outputs/CMX_S3D_Fold1_reproduction
DATA_ROOT="$OUTPUT_ROOT/common/Stanford2D3DPano"
PYTHON=/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
STATUS_DIR="$OUTPUT_ROOT/status"
mkdir -p "$STATUS_DIR" "$OUTPUT_ROOT/logs"

printf '%s\n' "$$" > "$STATUS_DIR/preparation_pipeline.pid"
printf 'waiting_for_rel_generation\n' > "$STATUS_DIR/preparation_pipeline.stage"
REL_PID=$(cat "$STATUS_DIR/rel_generation.pid")
while kill -0 "$REL_PID" 2>/dev/null; do
  sleep 30
done

if [[ "$(cat "$STATUS_DIR/rel_generation.exitcode")" != "0" ]]; then
  echo "REL generation failed" >&2
  exit 71
fi
REL_COUNT=$(find "$DATA_ROOT/rel" -type f -name '*_rel.png' | wc -l)
if [[ "$REL_COUNT" -ne 1413 ]]; then
  echo "REL coverage is $REL_COUNT, expected 1413" >&2
  exit 72
fi

printf 'full_data_audit\n' > "$STATUS_DIR/preparation_pipeline.stage"
cd /home/zhuzhaoziao/RELPlus/CMX-REL
"$PYTHON" tools/audit_s3d_data.py \
  --dataset-root "$DATA_ROOT" \
  --audit-dir "$OUTPUT_ROOT/audit" \
  --workers 7 \
  > "$OUTPUT_ROOT/logs/s3d_data_audit.log" 2>&1
printf '0\n' > "$STATUS_DIR/s3d_data_audit.exitcode"

printf 'pretrain_gate\n' > "$STATUS_DIR/preparation_pipeline.stage"
"$PYTHON" tools/finalize_pretrain_audit.py --output-root "$OUTPUT_ROOT" \
  > "$OUTPUT_ROOT/logs/finalize_pretrain_audit.log" 2>&1
printf '0\n' > "$STATUS_DIR/pretrain_gate.exitcode"

printf 'formal_suite\n' > "$STATUS_DIR/preparation_pipeline.stage"
tools/run_fold1_suite.sh
printf 'complete\n' > "$STATUS_DIR/preparation_pipeline.stage"
