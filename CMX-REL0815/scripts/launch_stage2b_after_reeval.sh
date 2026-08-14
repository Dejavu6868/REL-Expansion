#!/usr/bin/env bash
set -euo pipefail
STAGE2B_ROOT=${1:?Usage: launch_stage2b_after_reeval.sh STAGE2B_ROOT}
STAGE2A_ROOT=/data/zhuzhaoziao/cmx/outputs/stage2a_four_arm_single_seed_20260805_230826
REPO=/home/zhuzhaoziao/rel_exp/cmx_rel+
mkdir -p "$STAGE2B_ROOT/status"
printf '%s\n' "$$" > "$STAGE2B_ROOT/status/master.pid"
date --iso-8601=seconds > "$STAGE2B_ROOT/status/master.started_at"
"$REPO/scripts/run_stage2a_per_image_reeval.sh" "$STAGE2A_ROOT" "$STAGE2B_ROOT"
"$REPO/scripts/run_stage2b_all.sh" "$STAGE2B_ROOT"
printf '%s\n' 'COMPLETE_STAGE2B_TRAINING_AND_SEED12345_REEVALUATION' > "$STAGE2B_ROOT/status/master.status"
date --iso-8601=seconds > "$STAGE2B_ROOT/status/master.ended_at"
