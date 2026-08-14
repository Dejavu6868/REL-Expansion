#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?Usage: run_stage2a_all.sh STAGE2A_ROOT}
REPO=/home/zhuzhaoziao/rel_exp/cmx_rel+
mkdir -p "$ROOT/status"
controller_complete=0
record_controller_exit() {
  code=$?
  printf '%s\n' "$code" > "$ROOT/status/controller.exitcode"
  if [[ "$controller_complete" -ne 1 ]]; then
    printf '%s\n' 'PARTIAL_STAGE2A_RUN_INCOMPLETE' > "$ROOT/status/controller.status"
  fi
}
trap record_controller_exit EXIT
printf '%s\n' "$$" > "$ROOT/status/controller.pid"
if [[ -s "$ROOT/status/controller.started_at" ]]; then
  date --iso-8601=seconds > "$ROOT/status/controller.resumed_at"
else
  date --iso-8601=seconds > "$ROOT/status/controller.started_at"
fi
arms=(rawdepth hha relplus_local relplus_pose)
ports=(16120 16121 16122 16123)
for index in 0 1 2 3; do
  arm=${arms[$index]}
  printf '%s\n' "$arm" > "$ROOT/status/current_arm"
  if [[ -s "$ROOT/$arm/status/arm.status" ]] && grep -qx 'COMPLETE' "$ROOT/$arm/status/arm.status"; then
    continue
  fi
  "$REPO/scripts/train_stage2a_arm.sh" "$arm" "$ROOT" "${ports[$index]}"
done
printf '%s\n' 'COMPLETE_STAGE2A_FOUR_ARM_SINGLE_SEED' > "$ROOT/status/controller.status"
date --iso-8601=seconds > "$ROOT/status/controller.ended_at"
controller_complete=1
