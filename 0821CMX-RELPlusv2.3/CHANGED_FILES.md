# V2.2 to V2.3 changed files

Files not listed here are inherited from V2.2. The complete `rel_plus/` tree
and Original CMX `models/` tree remain unchanged.

## Modified implementation

- `configs/stanford2d3d_s2d/common.py`
- `engine/engine.py`
- `engine/relplus_evaluator.py`
- `tools/audit_full_relplus_cache.py`
- `tools/generate_full_relplus_cache.py`
- `tools/run_with_config.py`
- `train.py`
- `utils/training_protocol.py`

## New implementation

- `configs/stanford2d3d_s2d/cmx_mit_b2_rel_plus_v2_3_formal.py`
- `tools/audit_three_arm_x_modalities_v2_3.py`
- `tools/benchmark_cache_throughput_v2_3.py`
- `tools/eval_checkpoint_sweep_v2_3.py`
- `tools/eval_rel_plus_v2_3_full.py`
- `tools/launch_formal_training_v2_3.py`
- `tools/preflight_cmx_training_data_v2_3.py`
- `tools/run_and_record_exitcode.py`
- `tools/run_ddp_optimizer_smoke_v2_3.py`
- `tools/validate_eval_rank_consistency_v2_3.py`
- `utils/resolved_config.py`

## Tests

- `tests/test_v2_3_infrastructure.py`

## V2.3 documentation

- `CMX_RELPLUS_V2_3_SPEC.md`
- `V2_2_TO_V2_3_DIFF.md`
- `FULL_CACHE_GENERATION_REPORT.md`
- `FULL_CACHE_AUDIT_REPORT.md`
- `FULL_TRAINING_DATA_PREFLIGHT_REPORT.md`
- `DDP_OPTIMIZER_SMOKE_REPORT.md`
- `FORMAL_TRAINING_PROTOCOL.md`
- `FORMAL_TRAINING_LAUNCH_REPORT.md`
- `FORMAL_EVALUATION_PROTOCOL.md`
- `THREE_ARM_DATA_CONTRACT.md`
- `CHANGED_FILES.md`
- `IMPLEMENTATION_REPORT_CN.md`
- `README.md`
