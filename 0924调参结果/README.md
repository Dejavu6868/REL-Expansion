# 0924调参结果：focal_gamma=1.5 的 HHA / REL+ 双臂冻结归档

本目录冻结 **focal_gamma=1.5、lr=0.00012、batch56、seed12345** 的两臂完整参数、实际执行代码、200轮训练结果和全量评估证据。其余科学参数沿用 `0920调参结果`。目录名 `0924` 表示参数确定和训练启动日期；实验于 **2026-09-27 12:03:45（北京时间）**完成，2026-09-27 14:41:49完成服务器最终只读复核，并于当天冻结归档。

**执行与评估 PASS；参数候选晋级 FAIL。** 本次按用户要求保存这一版本，保留原筛选结论：相对gamma1，HHA提升，REL+下降，未满足“REL+高于gamma1且HHA不低于gamma1”的共同配方候选标准。gamma1与gamma2历史归档均保留。尚未执行的 `lr=0.00010` 建议不属于本次归档。

| 模型 | gamma1 mIoU（%） | 本轮 gamma1.5 mIoU（%） | 1.5−1（百分点） | gamma2 mIoU（%） | 1.5−2（百分点） |
|---|---:|---:|---:|---:|---:|
| HHA | 60.18289885 | **60.47034661** | +0.28744776 | 60.98827182 | −0.51792521 |
| REL+ | 61.40637240 | **61.16096760** | −0.24540479 | 61.01689760 | +0.14407001 |

本轮 REL+−HHA 为 **+0.69062100** 个百分点；gamma1为+1.22347355，gamma2为+0.02862578。差距不能代替同臂提升判断，本轮仅一个seed。

[完整报告](REPORT.md) · [总体指标与变化](results/gamma15_vs_baselines_metrics.csv) · [13类IoU与gamma1差值](results/gamma15_per_class.csv) · [原始比较JSON](results/dual_arm_comparison.json)

| 冻结项目 | 设置 |
|---|---|
| 模型 | CMX衍生双MiT-B2 + MLPDecoder512 |
| 数据划分 | S2D Area1/2/3/4/6训练52,903图；Area5a/5b测试17,593图；13类、ignore255 |
| 规模与终点 | 8GPU × 每卡batch7；200epochs；每epoch945步；每臂189,000次更新 |
| 优化 | AdamW，lr0.00012，betas(0.9,0.999)，weight decay0.01；Focal gamma1.5 |
| 学习率日程 | WarmUpPolyLR，warmup10epochs/9450updates，power0.9 |
| 数据处理 | 480×480；原尺度增强[0.5,0.75,1,1.25,1.5,1.75]；无翻转 |
| 初始化与数值 | 同一预训练权重和原初始化；seed12345；AMP关闭，保留原TF32设置 |
| 评估 | 固定epoch200，480整图，scale1/no-flip，align_corners=false |

完整参数以 [HHA配置](configs/hha.json)、[REL+配置](configs/relplus.json)、[共享suite](code/training/suite.json) 为准。逐项比较129/122个配置字段，唯一科学变化是 `focal_gamma: 1→1.5`；实验标识与独立路径相应改变。174个source Python文件保持原冻结字节，两臂16个训练rank记录的实际损失均为gamma1.5。详见 [配置差异](evidence/startup/REMOTE_RESOLVED_CONFIG.json)、[执行前协议](code/training/PROTOCOL.md) 与 [最终冻结协议](provenance/frozen_protocol.json)。

两臂训练均达到200轮、189000次更新，8 rank完成、exit0、NaN替换0；总训练约73.01小时。评估各覆盖17593个唯一样本，精确按 `test_ids[rank::8]` 有序分片，每臂有效像素3,973,620,198。逐rank混淆矩阵、样本顺序、汇总指标、逐类IoU、gamma1/gamma2差值均已复核，三组实验GT直方图相同。

[最终流水线状态](evidence/completed/pipeline_status.json) · [训练完成记录](evidence/completed/train_status.json) · [评估状态](evidence/completed/evaluation_epoch200/status.json) · [309项服务器复核](evidence/completed/LIVE_FINAL_RECHECK.json) · [1073项独立离线审计](evidence/completed/INDEPENDENT_OFFLINE_AUDIT.json)

```text
configs/                  两臂原始完整参数
code/training/            实际206文件bundle及清单，含174个source Python文件
code/reference_notices/   原许可证和依赖说明（执行bundle之外的补充文件）
results/                  总体指标CSV、逐类IoU CSV、完整比较JSON
baseline_gamma1/          0920冻结基线的原始矩阵、指标及来源记录
baseline_gamma2/          0915冻结基线的原始矩阵、指标及来源记录
evidence/completed/       121份远端完成输出/输入列表及传输、远端、独立审计
evidence/preparation/     79份原始准备与smoke证据
evidence/startup/         历史进度、配置核验、部署与准备传输清单
evidence/local_original/  发布前本地文档与状态快照
provenance/               协议、代码来源、权重身份、环境、原审计依赖映射
verify_archive.py         标准库只读离线核验器
SHA256SUMS                全目录SHA-256清单（不包含自身）
```

gamma1主基线来自提交 [`c8ff745`](https://github.com/Dejavu6868/REL-Expansion/tree/c8ff74502cf6717fc554adc47db4d14e9ddd42a9/0920%E8%B0%83%E5%8F%82%E7%BB%93%E6%9E%9C)，gamma2历史基线来自 [`791f11d`](https://github.com/Dejavu6868/REL-Expansion/tree/791f11db670239a19e78f3956c6f3d4466f17dc3/0915%E8%B0%83%E5%8F%82%E7%BB%93%E6%9E%9C)。用于复算的基线矩阵与指标已复制到本目录，无需读取旧归档或连接服务器。

从仓库根目录执行：

```bash
python3 '0924调参结果/verify_archive.py'
```

核验器只使用Python标准库，检查文件集合与哈希、原始206文件bundle、完整配置、训练终点、样本分片和矩阵求和，独立重算新旧指标与候选判定；不执行训练或推理。可将整个 `0924调参结果` 文件夹复制到其他位置运行。`evidence/completed/independent_offline_audit.py` 是未改动的历史审计脚本，保留原工作区路径；可移植只读入口是本目录根部的 `verify_archive.py`。

历史证据中的 `RUNNING`、访问失败和 `github_published:false` 表示当时快照；最终执行状态以 `evidence/completed/pipeline_status.json` 的PASS为准。本轮正式训练和评估均为attempt1。准备阶段曾修正一处Python3.8测试兼容问题，原失败记录与修正后的部署证据一并保留。

数据图像/缓存、预训练权重和训练权重字节不包含在Git归档中。两臂最终权重仍位于服务器：

```text
/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_FG15_20260924/attempt1/runs/<arm>/checkpoints/epoch-200.pth
```

| arm | 权重SHA-256 |
|---|---|
| hha | `cb5d1d9a456a32f98952f9b1eb429e1393493988f17218511436042e889ac405` |
| relplus | `1812ce2f4c1109b1ce5308a51e5a3aa06376a9f6abc8ffc43194ac6658358be6` |

实际服务器权重已在最终只读审计中重新计算SHA；离线脚本核验归档中的身份记录。重训或推理仍需原数据、权重、运行环境，并调整历史绝对路径。详见 [权重身份](provenance/checkpoints.json)、[来源](provenance/source_provenance.json) 与 [归档范围](provenance/archive_completeness.json)。

为与冻结基线一致，保留原 `criterion=None` 评估构造：decoder BN eps训练1e-3、评估1e-5；AMP关闭，原TF32设置保留。历史HHA物理通道及K/gravity来源缺项仍在。该归档记录CMX主体与项目适配器的实际执行版本，不宣称完整官方CMX协议复现。结果属于单seed开发性比较；Area5参与了多次选参，不能据此认定跨seed稳定优劣或独立于选参的泛化表现。
