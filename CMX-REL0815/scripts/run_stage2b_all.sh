#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?Usage: run_stage2b_all.sh STAGE2B_ROOT}
REPO=/home/zhuzhaoziao/rel_exp/cmx_rel+
mkdir -p "$ROOT/status"
controller_complete=0
record_controller_exit() {
  code=$?
  printf '%s\n' "$code" > "$ROOT/status/controller.exitcode"
  if [[ "$controller_complete" -ne 1 ]]; then
    printf '%s\n' 'PARTIAL_STAGE2B_RUN_INCOMPLETE' > "$ROOT/status/controller.status"
  fi
}
trap record_controller_exit EXIT
grep -q 'PASS_AREA1_WORLD_UP_SANITY' "$ROOT/geometry_gate/area1_world_up_sanity.json"
for seed in 23456 34567; do
  grep -q 'PASS_FOUR_ARM_PREFLIGHT' "$ROOT/seed_$seed/preflight/status.json"
done
printf '%s\n' "$$" > "$ROOT/status/controller.pid"
date --iso-8601=seconds > "$ROOT/status/controller.started_at"
arms=(rawdepth hha relplus_local relplus_pose)
for seed_index in 0 1; do
  if [[ "$seed_index" -eq 0 ]]; then seed=23456; else seed=34567; fi
  for arm_index in 0 1 2 3; do
    arm=${arms[$arm_index]}
    printf '%s/%s\n' "$seed" "$arm" > "$ROOT/status/current_arm"
    if [[ -s "$ROOT/seed_$seed/$arm/status/arm.status" ]] && grep -qx 'COMPLETE' "$ROOT/seed_$seed/$arm/status/arm.status"; then
      continue
    fi
    port=$((16220 + seed_index * 10 + arm_index))
    "$REPO/scripts/train_stage2b_arm.sh" "$arm" "$ROOT" "$seed" "$port"
  done
done
printf '%s\n' 'COMPLETE_STAGE2B_NEW_SEEDS_TRAINING' > "$ROOT/status/controller.status"
date --iso-8601=seconds > "$ROOT/status/controller.ended_at"
controller_complete=1
