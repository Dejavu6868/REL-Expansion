# V2.1 → V2.2 变更文件

以下均直接对应 V2.2 prompt；未列出的 REL+ generator/core 文件保持不变。

## 新增配置与 sampler

- `configs/stanford2d3d_s2d/cmx_mit_b2_rel_plus_v2_2_formal.py`
- `configs/stanford2d3d_s2d/cmx_mit_b2_rel_plus_v2_2_pilot.py`
- `configs/stanford2d3d_s2d/cmx_mit_b2_rgbd_v2_2_formal.py`
- `configs/stanford2d3d_s2d/cmx_mit_b2_hha_v2_2_formal.py`
- `configs/stanford2d3d_s2d/comparison_v2_2.py`
- `dataloader/samplers.py`
- `utils/training_runtime.py`

## 修改训练、数据与 evaluator

- `configs/stanford2d3d_s2d/common.py`
- `dataloader/RGBXDataset.py`
- `dataloader/dataloader.py`
- `dataloader/profiles.py`
- `engine/engine.py`
- `engine/logger.py`
- `engine/relplus_evaluator.py`
- `rel_plus/integration/cmx_preprocess.py`
- `train.py`
- `utils/training_protocol.py`
- `utils/transforms.py`

## 新增或修订工具

- `tools/generate_full_relplus_cache.py`
- `tools/audit_full_relplus_cache.py`
- `tools/preflight_cmx_training_data_v2_2.py`
- `tools/eval_rel_plus_v2_1_smoke.py`
- `tools/eval_rel_plus_v2_2_full.py`
- `tools/eval_checkpoint_sweep_v2_2.py`
- `tools/validate_formal_startup_no_step_v2_2.py`
- `tools/benchmark_dataloader_v2_2.py`
- `tools/validate_cache_tools_smoke_v2_2.py`
- `tools/validate_eval_rank_consistency_v2_2.py`
- `tools/collect_v2_2_evidence.py`
- `tools/trace_three_arm_dataloaders_v2_2.py`
- `tools/verify_generator_byte_invariant.py`
- `tools/analyze_pilot_invalid.py`
- `tools/eval_rel_plus_v2_1.py`
- `tools/trace_comparison_profile.py`（旧构造性 trace 入口退役并 fail loud）

## 测试与 fixture

- `tests/test_v2_2_infrastructure.py`
- `tests/test_formal_configs_and_guards.py`
- `tests/test_comparison_transform_profile.py`
- `tests/fixtures/synthetic_sweep_metrics.csv`

## 文档

- `CMX_RELPLUS_V2_2_SPEC.md`
- `V2_1_TO_V2_2_DIFF.md`
- `FORMAL_TRAINING_PROTOCOL.md`
- `FORMAL_EVALUATION_PROTOCOL.md`
- `FULL_CACHE_PROTOCOL.md`
- `CHECKPOINT_ENDPOINT_PROTOCOL.md`
- `THREE_ARM_COMPARISON_PROTOCOL.md`
- `CHANGED_FILES.md`
- `IMPLEMENTATION_REPORT_CN.md`
- `COMPLETION_CHECKLIST.md`
- `README.md`

冻结且未修改的核心至少包括 `rel_plus/generator.py`、`encoding.py`、`depth.py`、`camera.py`、`source_helpers.py`、normal helper、storage contract 以及 Original CMX `models/` tree。
