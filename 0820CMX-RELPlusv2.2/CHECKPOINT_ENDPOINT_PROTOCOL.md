# Checkpoint Endpoint 协议

## Primary

正式三臂主 endpoint 在训练前冻结为 epoch 200。RGBD、HHA 和 REL+ 使用同一规则，不借助 test split 选择 checkpoint。

## Secondary

允许同时报告 epoch 100–200、每 5 epochs 中 test mIoU 最高者，但名称必须为 `test_selected_best`。它使用 test 集进行选择，是描述性、paper-compatible oracle endpoint，存在 selection bias，不是无偏主结果。

任何正式报告都必须同时给出 epoch 200；禁止只报告 best。当前没有独立 validation split，因此不虚构 `best_validation_selected`。

## 工具与验证

`tools/eval_checkpoint_sweep_v2_2.py` 默认要求 epoch 100,105,...,200 全部存在，逐 checkpoint 调用 full evaluator，并输出：

- `metrics_all_checkpoints.csv`
- `metrics_epoch200.json`
- `metrics_test_selected_best.json`
- 两个 endpoint 的 per-class IoU 与 confusion matrix

合成 smoke 使用 epoch 100/105/200，正确选择 epoch 200 为 primary、105 为 `test_selected_best`；缺失 epoch 105 时 fail loud。合成数值被标为 plumbing only，不是科学结果。
