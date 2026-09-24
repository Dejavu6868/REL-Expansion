# 0920调参结果：focal_gamma=1 的 HHA / REL+ 双臂冻结归档

本目录冻结 **focal_gamma=1** 的两臂参数、实际执行代码、200轮训练结果和全量评估证据。其余科学参数沿用 `0915调参结果`：**batch56、lr=0.00012、seed12345**。目录名 `0920` 表示参数确定和训练启动日期；实验于 **2026-09-24 00:27:47（北京时间）**完成，并于当天核验、归档。

| 模型 | 原 gamma=2 mIoU（%） | 本轮 gamma=1 mIoU（%） | 变化（百分点） |
|---|---:|---:|---:|
| HHA | 60.98827182 | **60.18289885** | −0.80537297 |
| REL+ | 61.01689760 | **61.40637240** | +0.38947480 |

本轮 REL+−HHA 为 **+1.22347355** 个百分点，原差距为 +0.02862578。差距扩大更多来自 HHA 下降；本轮仅一个 seed，不能据此宣称稳定优势。

[完整报告](REPORT.md) · [总体指标与变化](results/gamma1_vs_gamma2_metrics.csv) · [13类 IoU](results/gamma1_per_class.csv) · [原始比较 JSON](results/dual_arm_comparison.json)

| 冻结项目 | 设置 |
|---|---|
| 模型 | CMX衍生双MiT-B2 + MLPDecoder512 |
| 数据划分 | S2D Area1/2/3/4/6训练52,903图；Area5a/5b测试17,593图；13类、ignore255 |
| 规模与终点 | 8GPU × 每卡batch7；200epochs；每epoch945步；每臂189,000次更新 |
| 优化 | AdamW，lr0.00012，betas(0.9,0.999)，weight decay0.01；Focal gamma1 |
| 学习率日程 | WarmUpPolyLR，warmup10epochs/9450updates，power0.9 |
| 数据处理 | 480×480；原尺度增强[0.5,0.75,1,1.25,1.5,1.75]；无翻转 |
| 初始化与数值 | 同一预训练权重和原初始化；seed12345；AMP关闭，保留原TF32设置 |
| 评估 | 固定epoch200，480整图，scale1/no-flip，align_corners=false |

完整参数以 [HHA配置](configs/hha.json)、[REL+配置](configs/relplus.json)、[共享suite](code/training/suite.json) 为准。已逐项比较129/122个配置字段，唯一科学变化是 `focal_gamma: 2→1`；实验标识与独立路径相应改变。两臂共16个训练rank记录的实际损失均为gamma1，配置与保存文件完全一致。详见 [配置差异](evidence/startup/resolved_config_difference.json)、[原实验协议](code/training/PROTOCOL.md) 和 [冻结协议](provenance/frozen_protocol.json)。

两臂训练均达到200轮、189000步，8 rank完成、exit0、NaN替换0。评估各覆盖17593个唯一样本，精确按 `test_ids[rank::8]` 分片，每臂有效像素3,973,620,198。逐rank混淆矩阵、样本顺序、汇总指标、逐类IoU及gamma2差值已复核。

[最终流水线状态](evidence/completed/pipeline_status.json) · [训练完成记录](evidence/completed/train_status.json) · [评估状态](evidence/completed/evaluation_epoch200/status.json) · [服务器只读复核](evidence/completed/LIVE_FINAL_RECHECK.json) · [独立离线审计](evidence/completed/INDEPENDENT_OFFLINE_AUDIT.json)

目录内容：

```text
configs/                  两臂原始完整参数
code/training/            实际200文件bundle及清单，含训练、评估入口与174个source Python文件
code/reference_notices/   原许可证和依赖说明（执行bundle之外的补充文件）
results/                  指标CSV、逐类IoU、比较JSON
baseline_gamma2/          0915冻结基线的原始矩阵/指标及来源记录
evidence/completed/       114份原始完成证据及传输、远端、独立审计
evidence/preparation/     80份原始准备与smoke证据及清单
evidence/startup/         历史进度、配置核验与部署记录
evidence/local_original/  原本地文档和状态快照，保留原路径及当时发布状态
provenance/               协议、代码来源、权重身份、环境、复制路径映射
verify_archive.py         标准库只读离线核验器
SHA256SUMS                全目录SHA-256清单（不包含自身）
```

gamma2基线来自提交 [`791f11db670239a19e78f3956c6f3d4466f17dc3`](https://github.com/Dejavu6868/REL-Expansion/tree/791f11db670239a19e78f3956c6f3d4466f17dc3/0915%E8%B0%83%E5%8F%82%E7%BB%93%E6%9E%9C) 的 `0915调参结果`。本目录内已有用于复算的两臂矩阵和指标副本，无需读取上一目录或连接服务器。

从仓库根目录执行：

```bash
python3 '0920调参结果/verify_archive.py'
```

核验器仅需Python标准库，检查文件集合/哈希、完整参数、训练终点、样本分片、矩阵求和，并重算新旧指标及差值；不执行训练或推理。历史证据中的 `RUNNING`、`QUEUED` 和失败预检代表当时状态，最终状态以 `evidence/completed/pipeline_status.json` 的 PASS 为准。attempt1仅预检查失败且0训练步，本轮正式训练和评估使用attempt2。

权重与数据字节不包含在Git归档中。两臂最终权重仍位于服务器：

```text
/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_FG1_20260920/attempt2/runs/<arm>/checkpoints/epoch-200.pth
```

| arm | 权重SHA-256 |
|---|---|
| hha | `d4d210ea92a473cc700b1660dff7da644e8d1b9934be759b216c0d4931ae2b4d` |
| relplus | `6506a922b9d31919576b3f998864993a24ca7f07baca868d71a5ff3b6a581831` |

实际服务器权重已在最终只读审计中核对；离线脚本仅核验归档中的身份记录。重训或推理仍需原数据、权重、运行环境，并调整历史绝对路径。详见 [权重身份](provenance/checkpoints.json)、[来源](provenance/source_provenance.json) 与 [归档范围](provenance/archive_completeness.json)。

为与冻结基线一致，decoder BN eps仍为训练1e-3、评估1e-5；AMP关闭不代表TF32关闭，实际两个TF32开关均为true。历史HHA物理通道及K/gravity来源缺项仍保留。该归档记录CMX主体与项目适配器的实际执行版本，不代表完整官方CMX实验协议复现。
