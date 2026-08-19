# Changed Files

## Added

- `dataloader/data_setting.py`: sole train/eval dataset-setting builder.
- `dataloader/profiles.py`: shared no-flip comparison transform profile.
- `utils/training_protocol.py`: authorization, seed, cuDNN, criterion and optimizer helpers.
- `engine/relplus_evaluator.py`: canonical full-image evaluator plumbing.
- `configs/stanford2d3d_s2d/`: common, 36-sample pilot and fail-closed formal configs.
- `tools/eval_rel_plus_v2_1.py`: canonical evaluation entry.
- `tools/verify_generator_byte_invariant.py`, `smoke_pilot_loaders.py`,
  `trace_comparison_profile.py`, `analyze_pilot_invalid.py`,
  `audit_mit_b2_initialization.py`: non-training audits and smokes.
- Seven CMX protocol regression test modules under `tests/`.
- The five required protocol/report documents and `SOURCE_NOTICE.md`.

## Modified

- `README.md`: points to this independent REL+ integration and its gates.
- `dataloader/RGBXDataset.py`: strict REL+/mask contract without channel conversion.
- `dataloader/dataloader.py`: shared profile, four-value ValPre and one setting builder.
- `rel_plus/integration/cmx_preprocess.py`: author-compatible sampling/reference path and label-joint diagnostics.
- `utils/loss_opr.py`: live/default gamma 2.
- `train.py`: pre-construction gates and author loss/seed/cuDNN/optimizer flow.
- `eval.py`, `tools/eval_fold1.py`: shared builder and explicit REL+ legacy rejection.
- `tools/validate_single_batch_v2_1.py`: formal Focal backward without optimizer.
- `tools/prepare_and_launch_fold1.sh`, `run_fold1_arm.sh`,
  `run_fold1_suite.sh`: legacy-copy guards; historical external scripts unchanged.
- `tests/test_augmentation_contract.py`: reference now imports actual CMX transforms.

## Explicitly unchanged

- All Original CMX files under `models/`.
- Frozen REL+ generator core: `camera.py`, `constants.py`, `depth.py`,
  `encoding.py`, `generator.py`, `geometry.py`, `normal_diagnostics.py`,
  `profiles.py`, `source_helpers.py`, `stanford_s2d.py`, `storage.py`.
- `third_party/rel_original/hha_util.py`.
- Existing external RGBD/HHA/REL experiment directories, logs and checkpoints.

No file-hash artifact was created.
