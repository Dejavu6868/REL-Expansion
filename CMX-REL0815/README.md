# CMX-REL+ for calibrated 2D Stanford2D3D

This target repository runs the controlled model `cmx_rel+` with `RGB + [ReD, EGVIA, LOA]` without changing CMX. The REL-default representation uses K for perspective backprojection and W2C rotation for gravity-aligned, camera-centred points; translation is validated but excluded from the three channels. REL+ is generated at 1080x1080 and resized into a fresh run-local 480x480 cache.

The exact mathematics, coordinate decisions, source paths, baseline hashes, and conflicts are frozen in `AUDIT.md`. `DEVIATIONS.md` lists every known disclosure limit.

## Full run

```bash
REPO=/home/zhuzhaoziao/rel_exp/cmx_rel+
OUTPUT=/data/zhuzhaoziao/cmx/outputs
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT/cmx_relplus_2d_stanford2d3d_mitb2_seed12345_8gpu_rel_default_v2_$STAMP"
mkdir -p "$OUTPUT/.launchers"
nohup setsid /usr/bin/flock -n -E 75 /tmp/cmx_relplus_8gpu_formal.lock \
  /usr/bin/env GPU_IDS=0,1,2,3,4,5,6,7 PREPARE_WORKERS=32 PORT=16051 \
  "$REPO/run_experiment.sh" "$RUN_DIR" \
  </dev/null >"$OUTPUT/.launchers/$(basename "$RUN_DIR").log" 2>&1 &
```

The explicit run directory must not exist. The orchestrator refuses every pre-existing path before writing anything, and the outer lock prevents two formal runners from sharing the host. It performs environment audit, unit tests, fresh all-sample REL+ preparation, single-GPU and eight-GPU/resume smoke gates, formal eight-GPU training, epoch-32 evaluation, matched HHA/RawDepth re-evaluation, diagnostics, fairness comparison, and evidence finalization.

All shell entry points resolve the repository from their own location and use `set -euo pipefail`.

## Individual stages

```bash
scripts/audit_environment.sh RUN_DIR
PYTHONPATH=$PWD /data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python \
  scripts/prepare_relplus.py \
  --dataset-root /data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480 \
  --run-dir RUN_DIR --workers 32 --visual-count 16 --overwrite
scripts/smoke_test.sh RUN_DIR 0,1,2,3,4,5,6,7
scripts/train_cmx_relplus.sh RUN_DIR '' 0,1,2,3,4,5,6,7
scripts/eval_cmx_relplus.sh RUN_DIR 0
scripts/reevaluate_baselines.sh RUN_DIR 1 2
```

Resume from a complete epoch K is explicit:

```bash
scripts/train_cmx_relplus.sh RUN_DIR RUN_DIR/checkpoints/epoch-K.pth 0,1,2,3,4,5,6,7
```

Formal data preparation refuses a symlink cache and regenerates every PNG. Evaluation does not delete prior artifacts. Formal training refuses to start a duplicate run if epoch 32 already exists.

## Primary protocol

- Train Areas 1/2/3/4/6; evaluate Area 5a/5b.
- MiT-B2 + MLPDecoder, CE ignore 255.
- Eight physical GPUs implementing four logical data shards, global batch 12, 32 epochs, AdamW 6e-5.
- Single-scale/no-flip epoch-32 evaluation.
- Epoch 32 is the sole eligible checkpoint, so `best.pth == last.pth` by preregistration, not retrospective selection.

`STATUS.md` may say `COMPLETE` only after all stage exit codes are zero, the epoch-32 checkpoint hashes and loads, exact metrics and all 13 class IoUs are finite, and the baseline fairness gate passes.
