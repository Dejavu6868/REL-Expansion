# Real 8-GPU DDP optimizer smoke report

## Final status

`PASS` (attempt1, exit code 0)

The smoke ran from approximately 00:56:16 to 00:57:05 on eight NVIDIA
GeForce RTX 3090 GPUs. It used the formal CMX-REL+ V2.3 data/configuration
path, real `DistributedDataParallel`, SyncBatchNorm, backpropagation and
AdamW optimizer updates.

## Result

- GPU/rank count: 8/8
- Updates before checkpoint: 50 per rank
- Updates after restore: 3 per rank
- Optimizer step executed: true
- Pretrained MiT-B2 loaded: true
- Loss, logits and gradients finite on all ranks: true
- Author NaN replacement count: 0 on every rank
- Parameter changes before checkpoint:
  `rgb_encoder=true`, `x_encoder=true`, `fusion=true`, `decoder=true`
- Checkpoint saved and restored: true
- Model snapshots match exactly after restore: true
- Optimizer LR before/after restore:
  `4.4457885982156354e-08` / `4.4457885982156354e-08`
- LR after resumed updates: `4.717979736881899e-08`
- LR continuous across restore and updated after resume: true/true
- Parameter changes after resume:
  `rgb_encoder=true`, `x_encoder=true`, `fusion=true`, `decoder=true`
- Residual smoke/training GPU processes after completion: 0

The saved file is explicitly disposable:

`ddp_smoke/DISPOSABLE_DDP_SMOKE_epoch-0_iter-50.pth`

It is outside the formal-training run directory and is not a formal endpoint,
evaluation candidate or replacement for any existing checkpoint.

## Evidence

- Aggregate report: `ddp_smoke/ddp_optimizer_smoke_summary.json`
- Per-rank reports: `ddp_smoke/rank_00.json` through `rank_07.json`
- Log: `logs/ddp_optimizer_smoke_attempt1.log`
- Exit code: `logs/ddp_optimizer_smoke_attempt1.exitcode`
- Wrapper PID record: `logs/ddp_optimizer_smoke_attempt1.wrapper.pid`

No file hash was generated or written.
