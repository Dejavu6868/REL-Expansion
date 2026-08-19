# CMX-REL+ v2.1 接入实现报告

## 状态

CMX-REL+ v2.1 的数据、训练前协议与评价接口闭环已完成受限验证。Formal
配置完整但仍保持 `training_authorized=False`、`data_ready=False`；本报告不构成
全量 cache 或正式训练授权。

## 已静态审计

- 冻结 REL+ v2.1、Original CMX、作者 REL 与既有 RGBD/HHA/REL 参考副本。
- Original CMX `models/` 未改；Gate、SMMF、DyMM、SGA 均关闭。
- Focal、seed、cuDNN、AdamW 参数组、WarmUpPolyLR 与初始化执行路径已逐项核对。
- 历史 Fold 1 启动脚本只在本次独立副本中增加默认拒绝 guard，外部历史目录未改。

## 已数组/权重对拍

- 固定输入的 REL+ generator 前后：changed pixels 0、changed channels 0、max difference 0。
- Focal loss map、最终 scalar、logits gradient、第一层参数 gradient 与作者源码一致；gamma 1/2 可区分。
- WarmUpPolyLR 前 300 iteration 与作者源码逐值一致。
- MiT-B2 RGB/X encoder 各 332 个映射 tensor 的 key、shape、value 完全一致；无 missing/unexpected/shape/value mismatch。
- RGB 与 `[EGVIA, LOA, ReD]` sentinel 经磁盘、loader、增强、normalize、CHW/DataLoader 后未交换。

## 已 pytest

最终完整测试在显式提供冻结 `rel.py`、`hha_util.py` 与作者 `rgbd_util.py` 路径后为
93 passed、0 skipped、1 个来自未修改 Original CMX `collections.Iterable` 的弃用警告。

## 已 pilot loader

36 张 pilot 的首个真实样本已分别通过 Train DataLoader 与 Validation DataLoader。
Train 输出为 float32 BCHW RGB/X、int64 label、bool mask；Val 输出保留 uint8 HWC
RGB/X 与 bool mask，并由 canonical evaluator 统一 normalize。no-flip 在真实 TrainPre 生效。

## 已 forward/backward

真实 pilot 单 batch 使用 MiT-B2 双 encoder、MLPDecoder 与正式
`FocalLoss2d(gamma=2,reduction=none).mean()`。logits/loss 有限，RGB encoder、X
encoder、fusion、decoder 梯度均 finite/nonzero。执行了 backward，但没有构造
optimizer、没有 optimizer/scheduler step、没有 epoch loop、没有 checkpoint。

## 已 evaluator smoke

1 张真实 pilot 完成 Val DataLoader -> CMX forward -> 480x480 prediction -> 13x13
confusion accumulation，并成功保存 raw/color PNG。模型为随机初始化；该结果只说明
`evaluator plumbing smoke PASS`，未计算或报告科学 mIoU。

## 已 invalid 联合诊断

36 张 pilot 在冻结 `SOURCE_COMPAT_STORAGE_255` 输入上完成原始/变换后 invalid、
双线性影响比例、mean/max deviation、label-ignore/有效语义比例与分班计数。诊断未
修改任何正式 REL+ 输入，也未把 mask 传入模型。

## 尚未执行

- 未生成全量 REL+ cache。
- 未执行 optimizer.step 或 scheduler.step。
- 未产生 checkpoint。
- 未启动正式训练或多 seed 实验。
- 未在完整 test split 上计算或报告 mIoU。
