# Formal CMX-REL+ V2.3 training launch report

## Status

`FORMAL_TRAINING_STARTED`

The only authorized formal arm was launched at 2026-08-21 00:59:09 +08 after
all hard gates passed. This is retraining with backpropagation and optimizer
updates. It is not yet a completed 200-epoch run.

## Frozen run identity

- Arm: CMX-REL+ V2.3 only
- Representation: `RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`
- Invalid baseline: `SOURCE_COMPAT_STORAGE_255`
- Architecture/backbone/decoder: Original CMX / MiT-B2 / MLPDecoder
- Seed: 12345
- GPUs/ranks: 8/8
- Epochs: 200
- Global batch: 8
- AMP/SyncBN: false/true
- Loss: Focal, gamma 2, `none_then_mean`
- Optimizer: AdamW, LR `6e-5`, weight decay `0.01`
- Scheduler: iteration-wise WarmUpPolyLR, 10 warm-up epochs
- Primary/secondary endpoints: epoch 200 / `test_selected_best`

## Passed launch gates

- Frozen generator byte invariant: PASS, zero changed pixels
- Full cache generation: PASS, 70,496/70,496, zero failures
- Cache audit: PASS, 70 risk-regeneration samples, zero differences
- Full CMX training-data preflight: PASS, 70,496/70,496, zero failures
- Real DDP optimizer/checkpoint smoke: PASS, 8 ranks, 50+3 updates
- MiT-B2 pretrained model: present and loaded
- Eight visible GPUs and required ports: available before launch
- Launcher validate-only: exit code 0

## Durable launch

The server command was launched with `nohup`; the essential command is:

```text
python tools/run_and_record_exitcode.py \
  --exitcode /data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v2_3/logs/formal_training_launcher_attempt1.exitcode \
  -- python tools/launch_formal_training_v2_3.py \
  --config-module configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_3_formal \
  --cache-audit /data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v2_3/formal_cache/audit/cache_audit_summary.json \
  --training-data-preflight /data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v2_3/formal_cache/preflight/cmx_training_data_preflight_summary.json \
  --ddp-smoke /data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v2_3/ddp_smoke/ddp_optimizer_smoke_summary.json \
  --authorize-formal-training --accept-source-compatible-invalid \
  --nproc-per-node 8 --launcher torch.distributed.launch
```

- Durable wrapper PID: 3126627
- Distributed launcher PID: 3126628
- Rank PIDs at startup: 3126656, 3126657, 3126658, 3126659, 3126660,
  3126663, 3126665 and 3126667
- Launch ID: `CMX_RELPlus_v2_3_seed12345_20260821_005914`
- Launch report: `formal_training/formal_training_launch_report.json`
- PID record: `formal_training/formal_training.pid`
- Resolved config: `resolved_configs/resolved_formal_config.json`
- Outer log: `logs/formal_training_launcher_attempt1.log`
- Runtime log:
  `formal_training/CMX_RELPlus_v2_3_seed12345/logs/train.log`
- Runtime status:
  `formal_training/CMX_RELPlus_v2_3_seed12345/runtime_status.json`
- Output directory:
  `formal_training/CMX_RELPlus_v2_3_seed12345`

## Startup observation

At 2026-08-21 01:00:35 +08:

- status: `RUNNING`;
- epoch / iteration: 1 / 150;
- global iteration: 150;
- loss: 1.765695571899414;
- learning rate: 1.351882655375775e-07;
- optimizer step executed: true;
- author NaN replacement count: 0;
- world size: 8;
- all eight rank processes were alive and all eight GPUs were active;
- no final launcher exit code existed, as expected for a running process;
- no formal checkpoint existed, as expected before epoch 100.

The run has therefore satisfied the required first-batch and first-100-update
startup check. Completion, checkpoint count, final exit code and scientific
test metrics remain pending. No checkpoint was replaced and no file hash was
generated or written.
