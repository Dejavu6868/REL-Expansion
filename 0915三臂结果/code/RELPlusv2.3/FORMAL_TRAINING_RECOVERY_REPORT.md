# Formal training recovery report

## Current status

`FORMAL_TRAINING_STARTED` (attempt2 resumed from epoch 100)

Attempt1 stopped with exit code 1 at 2026-08-24 03:40:07 +08. The training
loop had completed epoch 105, but rank 0 failed before writing the epoch-105
checkpoint. The last durable recovery point was therefore epoch 100.

## Root cause

The first checkpoint call created the intended `latest_CMX_RELPlus_v2_3_logs`
symlink. On the next checkpoint call, `validate_log_link_paths()` compared
`realpath(log_dir)` with `realpath(log_dir_link)`. Because the existing valid
symlink resolves to `log_dir`, the guard incorrectly raised:

```text
ValueError: refusing a self-referential log directory link
```

The distributed launcher then terminated all eight ranks. This was a
checkpoint-link validation defect, not a data, GPU-memory, NaN, model or
optimizer failure.

## Recovery boundary

`checkpoints/epoch-100.pth` was loaded read-only before recovery:

- payload epoch/iteration: 100/6612;
- model state keys: 837;
- optimizer parameter groups/state entries: 2/810;
- checkpoint bytes: 799,645,865.

Epochs 101 through 105 from attempt1 were not checkpointed and are rerun.
No epoch-105 state was guessed or reconstructed.

## Fix and verification

The path guard now compares normalized absolute path names without following
the destination symlink. It still rejects a literal self-link while allowing
the intended existing alias. The formal launcher also gained a fail-closed
`--resume-checkpoint` path that verifies directory, filename epoch, payload
epoch/iteration, model state and optimizer state before appending
`--continue` to the distributed training command.

- TDD RED: 2 failed, 16 deselected, exit code 1.
- Focused GREEN: 2 passed, 16 deselected, exit code 0.
- Final isolated live-source regression: 126 passed, exit code 0.
- Resume validate-only with the real epoch-100 checkpoint: exit code 0.

The invalid attempt9 test log is retained: pointing `REL_AUTHORITATIVE_ROOT`
at the full V2.2 project shadowed V2.3 tools on `sys.path`, so its seven
failures are not accepted as regression evidence. Attempt10 used the frozen
REL copy in the current audit tree and the independent Perspective reference.

## Attempt2 startup observation

Attempt2 launched at 2026-08-25 16:41:17 +08 via `nohup`.

- launch ID: `CMX_RELPlus_v2_3_seed12345_20260825_164123`;
- durable wrapper PID: 612798;
- distributed launcher PID: 612799;
- rank PIDs: 612828, 612829, 612830, 612831, 612832, 612835, 612837,
  612839;
- all eight ranks logged successful restore from `epoch-100.pth`;
- observation at 16:42:51: epoch 101, iteration 180, global iteration
  661480, loss 0.0090136, LR 3.2145371e-05;
- new optimizer updates after the recovery point: 180;
- world size: 8; author NaN replacement count: 0;
- attempt2 final exit code: not present because the process is running.

## Evidence

- Attempt1 log/exit: `logs/formal_training_launcher_attempt1.log` and
  `.exitcode`
- Attempt1 archived controls: `formal_training_launch_report_attempt1.json`,
  `formal_training_attempt1.pid`, `resolved_formal_config_attempt1.json`, and
  `runtime_status_attempt1_failure_snapshot.json`
- RED/GREEN: `tests/tdd_checkpoint_resume_red.*` and
  `tests/tdd_checkpoint_resume_green.*`
- Full regression: `tests/pytest_full_live_resume_fix_attempt10.*`
- Resume validation: `logs/formal_training_resume_validate_attempt1.*`
- Attempt2 log/exit: `logs/formal_training_launcher_attempt2.log` and
  `.exitcode`
- Current launch report/config/status: `formal_training_launch_report.json`,
  `resolved_formal_config.json`, and `runtime_status.json`

No file hash was generated or written.
