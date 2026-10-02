# 0927调参结果：当前最高观测 REL+ 配方冻结归档

本目录按用户要求冻结 **gamma=1、lr=0.00012、weight_decay=0.01** 的现有已完成配方，并收录最近学习率实验作为对照。主配方的 REL+ mIoU 为 **61.40637240%**，HHA 为 **60.18289885%**。

归档创建于2026-10-02；`0927调参结果`是用户指定的目录名。所选配方实际于2026-09-20启动、2026-09-24 00:27:47完成。2026-09-27启动的是下表lr0.00010对照，它于2026-09-30完成。本次归档没有重新训练或修改实验参数。

| 已完成配方 | HHA mIoU (%) | REL+ mIoU (%) | 在本归档中的角色 |
|---|---:|---:|---|
| **gamma1 / lr0.00012 / wd0.01** | **60.18289885** | **61.40637240** | **所选冻结配方** |
| gamma1 / lr0.00010 / wd0.01 | 61.18139160 | 61.35387870 | 最近完成的学习率对照 |
| gamma1.5 / lr0.00012 / wd0.01 | 60.47034661 | 61.16096760 | 历史gamma对照 |
| gamma2 / lr0.00012 / wd0.01 | 60.98827182 | 61.01689760 | 原始配方对照 |

“最高”仅指这四组已完成实验中，单seed、固定epoch200的REL+ mIoU最高。它不是HHA最高的一组。lr0.00010的REL+低0.05249370个百分点，HHA高0.99849275个百分点；不能据此认定两者稳定优劣。此前的共同配方候选筛选仍要求REL+超过61.40637239796889%、HHA不低于60.182898850089714%；lr0.00010和gamma1.5均为执行PASS、候选FAIL。

| 冻结项 | 设置 |
|---|---|
| 模型与表示 | 双流MiT-B2 + MLPDecoder512；CMX_RELPLUS_V2_3；RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT；[EGVIA, LOA, ReD] |
| 主要优化参数 | AdamW；lr0.00012；focal_gamma1；weight_decay0.01；betas(0.9,0.999)；no_decay组0 |
| 日程与规模 | warmup10、Poly power0.9；batch56；8GPU×每卡7；200轮；每臂189000更新；seed12345 |
| 数据与增强 | S2D Area1/2/3/4/6训练52903图，Area5a/5b测试17593图；480×480；原6档尺度；no-flip |
| 评估 | 固定epoch200，全量17593图，scale1/no-flip；原BN/数值语义保持不变 |

[所选参数摘要](FROZEN_PARAMETERS.json) · [HHA完整配置](configs/hha.json) · [REL+完整配置](configs/relplus.json) · [四组比较CSV](results/completed_recipes.csv)

`selected_recipe/`逐字节复制已发布的[0920归档](https://github.com/Dejavu6868/REL-Expansion/tree/c8ff74502cf6717fc554adc47db4d14e9ddd42a9/0920%E8%B0%83%E5%8F%82%E7%BB%93%E6%9E%9C)，保留其459个文件、原SHA256清单、200文件实际执行bundle及174个source文件；可独立核验。原始文档保留0920身份、原日期和历史运行/未发布状态，它们不覆盖本目录的冻结说明。

| 路径 | 内容 |
|---|---|
| `selected_recipe/code/training/` | 所选配方实际执行的训练/评估代码、suite和bundle清单 |
| `selected_recipe/evidence/` | 所选配方完整准备、启动、完成和评估证据 |
| `configs/` | 所选HHA/REL+完整配置，等同于selected_recipe中的原件 |
| `comparisons/gamma1_lr10/` | lr0.00010评分矩阵、完整配置、评估前置记录和已有完成审计摘要 |
| `comparisons/gamma15_lr12/` | gamma1.5的同类对照证据摘要 |
| `results/` | 四组已完成配方的全精度JSON与CSV |
| `provenance/` | 复制来源/哈希、归档边界、最终权重身份 |

对照目录用于复算评分和解释选择，不声称包含其全部执行bundle及逐rank样本记录。未执行的调参建议和参数网格不纳入已完成结果。

在仓库根目录运行以下只读离线核验，无需GPU、PyTorch、数据或服务器连接：

```bash
python3 -B '0927调参结果/verify_archive.py'
```

核验器检查全目录文件集合与SHA、原459文件归档一致性、所选完整训练/评估证据、参数摘要与原配置、四组混淆矩阵指标和选择结果。不执行模型训练或推理。

权重与数据字节不上传GitHub；保留服务器路径、大小及SHA记录。所选权重为：

- HHA：`d4d210ea92a473cc700b1660dff7da644e8d1b9934be759b216c0d4931ae2b4d`。
- REL+：`6506a922b9d31919576b3f998864993a24ca7f07baca868d71a5ff3b6a581831`。

详见[权重身份](provenance/checkpoints.json)与[归档范围](provenance/archive_scope.json)。服务器权重哈希是既有完成审计记录，本次离线归档核验没有重新读取权重字节。
