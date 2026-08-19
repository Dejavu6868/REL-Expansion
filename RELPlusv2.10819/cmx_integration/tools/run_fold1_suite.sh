#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT=/data/zhuzhaoziao/RELPlus/outputs/CMX_S3D_Fold1_reproduction
STATUS_DIR="$OUTPUT_ROOT/status"
mkdir -p "$STATUS_DIR"

if ! grep -q 'Decision: \*\*PRETRAIN_GO\*\*' "$OUTPUT_ROOT/audit/PRETRAIN_GO_NO_GO.md"; then
  echo "PRETRAIN_GO is not present" >&2
  exit 70
fi

printf '%s\n' "$$" > "$STATUS_DIR/suite.pid"
printf 'cmx_rgbd_fold1_seed12345\n' > "$STATUS_DIR/suite_stage"
/home/zhuzhaoziao/RELPlus/CMX-RGBD/tools/run_fold1_arm.sh \
  cmx_rgbd_fold1_seed12345 59.03 29600

printf 'cmx_hha_fold1_seed12345\n' > "$STATUS_DIR/suite_stage"
/home/zhuzhaoziao/RELPlus/CMX-HHA/tools/run_fold1_arm.sh \
  cmx_hha_fold1_seed12345 63.98 29610

printf 'cmx_rel_fold1_seed12345\n' > "$STATUS_DIR/suite_stage"
/home/zhuzhaoziao/RELPlus/CMX-REL/tools/run_fold1_arm.sh \
  cmx_rel_fold1_seed12345 64.47 29620

printf 'training_and_evaluation_complete\n' > "$STATUS_DIR/suite_stage"
/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python \
  /home/zhuzhaoziao/RELPlus/CMX-REL/tools/build_fold1_visualizations.py \
  > "$OUTPUT_ROOT/logs/build_visualizations.log" 2>&1
/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python \
  /home/zhuzhaoziao/RELPlus/CMX-REL/tools/finalize_fold1_results.py \
  > "$OUTPUT_ROOT/logs/finalize_results.log" 2>&1
printf 'complete\n' > "$STATUS_DIR/suite_stage"
printf '0\n' > "$STATUS_DIR/suite.exitcode"
