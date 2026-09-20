# CMX S2D 新参数三臂：固定epoch200完整评估（2026-09-20）

**三臂全量评估及完整性复核均PASS。** 三臂mIoU均高于旧0915配方。

新训练配方为global batch56、lr1.2e-4；旧0915为batch8、lr6e-5。两者都使用seed12345、固定200轮、同一17593张测试图。

## 1. 新旧主终点对照

| 模型 | 旧0915 mIoU（%） | 新配方mIoU（%） | 差值（百分点） | 新Pixel Accuracy（%） | 新Mean Accuracy（%） |
|---|---:|---:|---:|---:|---:|
| RGBD | 54.8358 | 55.4112 | +0.5754 | 79.4078 | 63.4511 |
| HHA | 59.1558 | 60.9883 | +1.8325 | 82.4203 | 69.2016 |
| RELPlus | 59.5854 | 61.0169 | +1.4315 | 83.0179 | 69.1388 |

新配方内，REL+−HHA为 **+0.0286个百分点**，REL+−RGBD为 **+5.6057个百分点**。

## 2. 新配方逐类别结果

| 类别 | RGBD IoU（%） | HHA IoU（%） | REL+ IoU（%） | REL+−HHA（百分点） |
|---|---:|---:|---:|---:|
| beam | 0.4188 | 3.8183 | 3.0581 | -0.7601 |
| board | 72.5440 | 75.5906 | 73.5804 | -2.0103 |
| bookcase | 64.1659 | 66.5779 | 66.9225 | +0.3446 |
| ceiling | 90.2857 | 91.6349 | 92.4266 | +0.7917 |
| chair | 76.0579 | 83.1779 | 81.8664 | -1.3115 |
| clutter | 44.6521 | 51.1803 | 51.0241 | -0.1561 |
| column | 14.1599 | 21.2197 | 25.8120 | +4.5923 |
| door | 32.3838 | 43.0642 | 47.4402 | +4.3760 |
| floor | 94.9259 | 96.6188 | 96.4295 | -0.1893 |
| sofa | 23.8582 | 40.6369 | 36.0570 | -4.5799 |
| table | 71.6052 | 77.5124 | 76.1255 | -1.3869 |
| wall | 74.0841 | 77.3009 | 78.6035 | +1.3026 |
| window | 61.2042 | 64.5147 | 63.8740 | -0.6407 |

完整逐类新旧差值见CSV。

## 3. 执行和完整性证据

- 三臂均只读取新训练的epoch-200.pth；评估过程中没有训练、反向传播、optimizer更新或权重覆盖。
- 每臂exit0，8rank均完成；每臂17593张唯一样本，顺序分片严格等于test_ids[rank::8]，计数2200+7×2199。
- 每臂3973620198有效像素；新三臂和旧三臂的13类GT像素直方图完全相同。
- rank混淆矩阵求和与CSV一致，服务端和本地均独立重算mIoU、Pixel Accuracy、Mean Accuracy及逐类IoU。
- 三个checkpoint的路径、size、mtime和SHA前后不变；训练bundle文件、评估实现及数据证据指纹核验通过。
- 837个模型state keys完整匹配；浮点权重和推理logits均检查有限性。
- 输入处理复用冻结loader和evaluator；RGBD/HHA adapter仅改变源码定位路径。实际import路径均来自本次训练source快照。
- 推理：480×480整图、scale1/no-flip、align_corners=false、float32张量、未启用AMP、TF32保持原环境默认并记录、8卡×batch1、无补齐sampler。
- 总评估与串行审计耗时约8.54分钟。

## 4. 结论边界

这是单seed、固定测试协议的描述性比较。batch与学习率同时改变，优化器更新次数由1322600降至189000；不能把结果单独归因于某个参数，不能据此宣称跨seed稳定性或统计显著性。

保留与0915相同的criterion=None评估构造：decoder BN eps实际为1e-5，而训练为1e-3。此差异未在本轮修正，影响量未知；没有重新前向评估或改写旧0915结果。HHA通道物理来源、K/gravity完整provenance的历史缺项仍保留。

## 5. 权重与来源

| 臂 | epoch200 checkpoint SHA-256 |
|---|---|
| RGBD | `8a9fe5f6509dfd068f00c06968d1eae15f97d2fc13c0056d57b03744cf57210b` |
| HHA | `c5cf78b12d2458a48fe5505a4b7099ef84308dd4790f8283c02a605ba05ccd03` |
| RELPlus | `bf17b065bbbedf01f78c698b039c384c9b220a2332b24b930ca685db552e1232` |

远端评估证据根：`/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_20260915/attempt1/evaluation_epoch200_20260920_attempt1`。
远端训练源码：`/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-20260915/source`。

## 6. 归档证据入口

- [验证后的新旧对照](<results/comparison_verified.json>)
- [新旧指标CSV](<results/new_vs_0915_metrics.csv>)
- [逐类新旧CSV](<results/new_vs_0915_per_class.csv>)
- [服务端三臂原始比较](<evidence/evaluation/three_arm_comparison.json>)
- [全量评估状态](<evidence/evaluation/status.json>)
- [数据、配置与权重预检](<evidence/evaluation/preflight.json>)
- [本次评估协议](<code/evaluation/PROTOCOL.json>)
- [传输核验](<evidence/local_verification/TRANSFER_VERIFICATION.json>)
