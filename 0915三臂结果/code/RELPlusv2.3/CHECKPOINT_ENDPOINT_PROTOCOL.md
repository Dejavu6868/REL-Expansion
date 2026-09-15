# Checkpoint endpoint protocol

## Primary

The primary endpoint is epoch 200. It is fixed before training and is always
reported; the test split is not used to select it.

## Secondary

The secondary descriptive endpoint is the largest test mIoU among epochs 100,
105, ..., 200. It must be named `test_selected_best` and disclose test-set
selection bias. It is not an unbiased validation-selected checkpoint.

## Evaluator and sweep

`tools/eval_rel_plus_v2_3_full.py` requires the expected epoch and rejects a
checkpoint whose payload epoch is missing or different. Metrics are emitted
as both fraction and percent.

`tools/eval_checkpoint_sweep_v2_3.py` runs once as a parent process and
sequentially launches one eight-rank full evaluator for each of the 21 frozen
epochs. It retains metrics for every checkpoint and writes both endpoint
artifacts. Running the sweep parent itself under DDP is an error.

Synthetic and 1/2/8-rank checks are evaluator-plumbing evidence only, never a
scientific segmentation result.
