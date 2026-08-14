#!/usr/bin/env bash
set -euo pipefail
STAGE2B_ROOT=${1:?Usage: resume_stage2b_after_inventory_repair.sh STAGE2B_ROOT}
REPO=/home/zhuzhaoziao/rel_exp/cmx_rel+
mkdir -p "$STAGE2B_ROOT/status"
printf '%s\n' "$$" > "$STAGE2B_ROOT/status/master.pid"
date --iso-8601=seconds > "$STAGE2B_ROOT/status/master.resumed_at"
"$REPO/scripts/run_stage2b_all.sh" "$STAGE2B_ROOT"
printf '%s\n' 'COMPLETE_STAGE2B_TRAINING_AND_SEED12345_REEVALUATION' > "$STAGE2B_ROOT/status/master.status"
date --iso-8601=seconds > "$STAGE2B_ROOT/status/master.ended_at"
