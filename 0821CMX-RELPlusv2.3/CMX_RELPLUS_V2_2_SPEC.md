# CMX-REL+ V2.2 规格

## 身份与边界

CMX-REL+ V2.2 是 REL+ v2.1 的训练、缓存和科学评价基础设施升级，不是新的几何表示。集成协议为 `CMX_RELPLUS_V2_2`，表示协议仍为 `RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`。

冻结内容包括 ReD、EGVIA、LOA、K、pose、normal、厘米编码、depth-only invalid、通道顺序 `[EGVIA, LOA, ReD]` 和 offline canonical 480。正式 invalid 策略仍为 `SOURCE_COMPAT_STORAGE_255`：cache invalid 为 `[255,255,255]`，REL+ 使用 `INTER_LINEAR`，归一化后不额外置零，valid mask 仅用于诊断且不进入模型。

## 正式数据合同

- Dataset：`Stanford2D3D_S2D`，split 为 `Stanford2D3D_S2D_official_train_test`，不是 S3D Fold 1。
- train/test：52,903 / 17,593，共 70,496。
- RGB、label、REL+ 和 valid mask 的 canonical shape 均为 480×480；REL+ 为 uint8 三通道，mask 为 uint8 二值单通道。
- label 合法 stored ID 从真实 `class_mapping.json` 读取；loader 用有符号整数显式执行 stored ID 减一，0 映射为 255。
- Dataset 长度只等于真实 list 长度。逻辑 epoch 为 52,904，由 `FixedLengthDistributedSampler` 每 epoch 一次 O(N) shuffle 后补 1 个样本实现。

## 正式模型与训练协议

Original CMX、MiT-B2、MLPDecoder，Gate/SMMF/DyMM/SGA 均关闭。全局 batch 8，200 epochs，AdamW，作者参数组，`WarmUpPolyLR`，warmup 10 epochs，Focal gamma=2、`reduction='none'` 后对完整 H×W 求 mean，ignore loss 为 0 但仍在空间均值分母中。

作者代码中的 NaN 行为被显式复刻：loss map 出现 NaN 时先 `torch.nan_to_num(..., nan=0.0)`，并累计替换数量；这不是静默改成另一种 loss。

## 授权状态

正式配置固定：

```python
training_authorized = False
full_cache_authorized = False
data_ready = False
```

训练入口只接受两个现实门禁：显式训练授权，以及结构/数值审计为 `PASS` 的 `cache_audit_summary.json`。配置 import 不访问服务器文件；真实 list、路径和审计只在 runtime 检查。

## 运行环境

正式最低 Python 版本为 3.8；基础测试也支持较新 Python。V2.2 已移除 `collections.Iterable` 和错误 package-root 注入，并限定 transform RNG 为 Python `random.Random`/`random` 模块或 NumPy `Generator`。
