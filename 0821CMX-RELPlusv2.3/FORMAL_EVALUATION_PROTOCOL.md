# Formal evaluation protocol

## One checkpoint

`tools/eval_rel_plus_v2_3_full.py` evaluates the complete ordered 17,593-image
test split. It requires `--expected-epoch`; the checkpoint payload must contain
the same epoch. A missing or mismatched payload epoch fails before evaluation.

Frozen settings are full-image 480 x 480, scale `[1]`, flip off,
`align_corners=False`, 13 classes and ignore ID 255. The model and checkpoint
are loaded once per rank. Output includes:

- `metrics.json`;
- `per_class_iou.csv`;
- `confusion_matrix.csv`;
- `evaluation_manifest.csv` with rank ownership;
- optional raw/palette predictions and visualizations.

Every headline metric is explicitly emitted both as a 0--1 fraction and as a
percent: mIoU, pixel accuracy, mean accuracy and per-class IoU.

## Distributed identity

`DistributedEvalSamplerNoPad` assigns `indices[rank::world_size]`, so it does
not pad or duplicate samples. Each rank records owned sample IDs and an int64
confusion matrix. The all-reduced matrix is checked again against the merged
rank files, and the total sample count must equal 17,593.

The pre-training plumbing test uses synthetic labels and predictions. Its
1-rank, 2-rank and 8-rank confusion matrices must be exactly identical. It is
not a scientific metric result. A full 1-vs-8-rank checkpoint comparison can
run after the first formal checkpoint exists.

## Checkpoint sweep

The sweep parent must run once outside DDP. It discovers exactly epochs 100,
105, ..., 200 and launches one eight-rank evaluator per checkpoint
sequentially. It never launches the sweep parent itself under DDP.

- Primary: epoch 200, always reported.
- Secondary: `test_selected_best`, always labelled as test-selected and
  potentially biased.
- All 21 checkpoint metrics remain in `metrics_all_checkpoints.csv`; reporting
  only the best checkpoint is prohibited.
