# 正式评价协议

## 单 checkpoint full evaluator

入口：`tools/eval_rel_plus_v2_2_full.py`。它读取完整 test list，断言汇总样本数等于 17,593，并在循环前只加载一次 checkpoint、只把模型移动到 device 一次。

冻结 profile：full-image 480×480、scale `[1]`、flip=False、align_corners=False、13 类、ignore=255。它不复用 S3D ERP 的 1080 crop/stride。输出包括 `metrics.json`、`per_class_iou.csv`、`confusion_matrix.csv`、`evaluation_manifest.csv`，以及可选 predictions/visualizations。

指标为 mIoU、Pixel Accuracy、Mean Accuracy、13 类 IoU、有效像素数和评价样本数。

## 多 GPU

`DistributedEvalSamplerNoPad` 使用 `indices[rank::world_size]`，不补齐也不重复。每 rank 写 owned sample IDs、sample count 和 confusion matrix；最终对 confusion matrix 和 sample count 做 all-reduce。rank 文件再通过共享聚合函数复算一次，并与 all-reduce 逐元素核对。

1/2/8-rank 合成验证的 confusion matrix 完全一致。prediction 保存只写本 rank 拥有的 sample，并核对最终文件计数。

## smoke 与科学结果边界

1–3 张入口已明确命名为 `tools/eval_rel_plus_v2_1_smoke.py`，只能声明 evaluator plumbing PASS。当前 3 张 pilot smoke 验证了 Val loader、forward、prediction、13×13 confusion 和 raw/palette prediction 保存。

本轮没有运行完整 17,593 张 test，也没有产生或报告科学 mIoU。
