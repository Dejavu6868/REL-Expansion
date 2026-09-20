# 0915调参结果：CMX 三臂固定 epoch200 归档

本目录冻结 **global batch56、lr1.2e-4、seed12345** 的 RGBD / HHA / REL+ 三臂正式训练及评估结果。训练、全测试集评估和完整性复核均已完成。

目录名中的 `0915` 指本轮配方与训练启动日期；评估和归档完成日期为 **2026-09-20**。沿用原训练/测试划分，固定报告第200轮，没有另留验证集或进行测试集 checkpoint 扫选。

## 结果

| 臂 | 旧0915 mIoU（%） | 本轮 mIoU（%） | 增量（百分点） | 本轮 Pixel Accuracy（%） | 本轮 Mean Accuracy（%） |
|---|---:|---:|---:|---:|---:|
| RGBD | 54.8358 | **55.4112** | +0.5754 | 79.4078 | 63.4511 |
| HHA | 59.1558 | **60.9883** | +1.8325 | 82.4203 | 69.2016 |
| REL+ | 59.5854 | **61.0169** | +1.4315 | 83.0179 | 69.1388 |

三臂在本次单seed比较中均提高。新配方内 REL+−HHA 为 **+0.0286个百分点**，两者分数非常接近；REL+−RGBD 为 **+5.6057个百分点**。本轮同时改变 batch 和学习率，且每臂更新次数由1,322,600降至189,000，不能单独归因，也不能据此宣称跨seed稳定优势或统计显著性。

[完整评估报告及逐类结果](REPORT.md) · [精确新旧指标CSV](results/new_vs_0915_metrics.csv) · [逐类新旧CSV](results/new_vs_0915_per_class.csv) · [验证后的比较JSON](results/comparison_verified.json)

旧基线来自本仓库提交 [`da823a691f8666f9877df017743387d6b6f9074e`](https://github.com/Dejavu6868/REL-Expansion/tree/da823a691f8666f9877df017743387d6b6f9074e/0915%E4%B8%89%E8%87%82%E7%BB%93%E6%9E%9C) 的 `0915三臂结果`；用于离线复算的原始指标副本位于 [baseline_0915](baseline_0915)。本轮没有重跑旧基线。

## 训练与评估协议

| 项目 | 本轮固定设置 |
|---|---|
| 数据 | Stanford2D3D S2D；训练52,903，测试17,593；13类，ignore255 |
| 划分 | train areas 1/2/3/4/6；test areas 5a/5b |
| 模型 | CMX衍生双MiT-B2 + MLPDecoder512；三臂改变X输入表示 |
| 初始化 | 同一MiT-B2预训练；三臂同一初始化权重指纹；seed12345 |
| 训练规模 | 8GPU × 每卡batch7 = global56；200epochs |
| 更新数 | 每轮945；每臂189,000；每轮逻辑样本52,920，补齐17 |
| 优化 | AdamW，lr1.2e-4，betas(0.9,0.999)，weight decay0.01；Focal gamma2 |
| 日程 | WarmUpPolyLR；warmup10epochs = 9,450updates；power0.9 |
| 增强 | 480×480；scale[0.5,0.75,1,1.25,1.5,1.75]；NO_FLIP |
| 数值与线程 | float32张量，AMP关闭；每rank16workers；OpenCV/PyTorch/OMP线程数1 |
| 评估 | 固定epoch200；480×480整图，scale1/no-flip，align_corners=false |
| 评估分片 | 8GPU × batch1；无补齐；每臂17,593唯一样本 |

精确配置原文件：[RGBD](configs/rgbd.json)、[HHA](configs/hha.json)、[REL+](configs/relplus.json)。这三个文件逐字节复制自训练启动时保存的配置，并与最终评估预检中读取的训练配置逐字段核对一致。

完整约定见 [原训练协议](code/training/PROTOCOL.md)、[原suite](code/training/suite.json)、[评估协议](code/evaluation/PROTOCOL.json) 和 [冻结协议索引](provenance/frozen_protocol.json)。线程设置及前100轮恢复checkpoint策略等执行差异也记录在原训练协议中。

## 执行证据

三臂训练均达到epoch200 / global_iteration189000，各8rank完成，退出码0，NaN替换数0；总训练约109.3小时。最终状态见 [training/completion.json](evidence/training/completion.json)。

三臂评估各退出0，24rank全部完成，每臂17,593唯一样本、3,973,620,198有效像素。rank混淆矩阵之和与总矩阵相同；指标和逐类IoU由混淆矩阵独立复算。新旧六组结果的GT类别像素计数相同。评估及串行审计约8.54分钟。

- [评估最终状态](evidence/evaluation/status.json)、[三臂完整性比较](evidence/evaluation/three_arm_comparison.json)、[数据/配置/代码/权重预检](evidence/evaluation/preflight.json)。
- 原始结果：[RGBD](evidence/evaluation/rgbd/evaluation)、[HHA](evidence/evaluation/hha/evaluation)、[REL+](evidence/evaluation/relplus/evaluation)。各含metrics、逐类IoU、混淆矩阵、样本manifest和8rank记录。
- [远端证据原始SHA清单](evidence/evaluation/EVIDENCE.sha256)覆盖102个文件；[本地传输核验](evidence/local_verification/TRANSFER_VERIFICATION.json)。

`evidence/training/startup_20260915/` 保留启动、smoke和首次预检失败的历史证据，其中的 `RUNNING` 只代表当时状态；**最终训练状态以 `evidence/training/completion.json` 为准**。

## 归档内容与来源

```text
README.md / REPORT.md       中文摘要、完整结果与局限
configs/                    三臂原始完整resolved配置
code/training/              执行bundle：174个Python源码 + 10个协议/入口/测试文件
code/evaluation/            本轮评估入口、输入适配器和审计实现
code/reference_notices/     原代码许可证和依赖说明（补充文件）
code/local_analysis_original/ 原本地汇总脚本（保留原路径语义）
results/                    本轮与旧0915对照JSON/CSV
baseline_0915/              固定旧提交的指标、协议和适配器副本
evidence/                   训练完成、启动/smoke、全量评估与传输证据
provenance/                 原文件来源、环境、协议和checkpoint指纹
verify_archive.py           标准库离线归档核验器
SHA256SUMS                  全归档文件哈希清单（不含自身）
```

源码保持实际执行时字节；184文件训练bundle与远端评估前记录逐项SHA一致。执行来源以 [source_provenance.json](provenance/source_provenance.json) 和 [copied_files.json](provenance/copied_files.json) 为准。代码含CMX模型主体及项目数据/协议适配，不等同于完整官方CMX实验协议。历史配置中的服务器绝对路径保留作追溯用途。

## 权重与复现范围

本目录保存代码、参数、结果和证据。**不含数据集、输入缓存、预训练权重或训练checkpoint文件**，也不含完整逐epoch训练日志与全程资源轨迹。预测图片未在本次评估中生成。详见 [archive_completeness.json](provenance/archive_completeness.json)。

三臂最终checkpoint仍位于服务器训练输出根：

```text
/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_20260915/attempt1/runs/<arm>/checkpoints/epoch-200.pth
```

| arm | SHA-256 |
|---|---|
| rgbd | `8a9fe5f6509dfd068f00c06968d1eae15f97d2fc13c0056d57b03744cf57210b` |
| hha | `c5cf78b12d2458a48fe5505a4b7099ef84308dd4790f8283c02a605ba05ccd03` |
| relplus | `bf17b065bbbedf01f78c698b039c384c9b220a2332b24b930ca685db552e1232` |

完整路径、大小、mtime和哈希见 [checkpoints.json](provenance/checkpoints.json)。原远端评估已核对权重前后未变；离线归档核验只验证这些记录相互一致，不重新读取未收录的权重。

从仓库根目录可执行：

```bash
python3 '0915调参结果/verify_archive.py'
```

仅需Python标准库；检查文件集合和SHA、原bundle及评估证据、三臂完成记录、逐rank样本覆盖与矩阵汇总，并重算新旧指标。不运行训练或模型推理。重新训练/推理还需要原始数据、权重和对应环境，以及调整服务器绝对路径；已观测环境见 [observed_environment.json](provenance/observed_environment.json)。

## 保留的限制

1. 单seed，batch和lr联合变化；本轮不是独立验证集调参搜索，不能宣称稳定或显著优势。
2. 为保持旧评估语义，decoder BN eps在训练为1e-3、评估为1e-5；此差异影响未知，本轮未修正。
3. AMP关闭、张量为float32；实测 `cudnn_allow_tf32=true`、`matmul_allow_tf32=true`，保留原环境默认，不能视作TF32关闭。
4. HHA物理通道、K/gravity完整来源仍有历史缺项；冻结输入与执行路径不意味着这些几何来源问题已解决。

冻结后如需修正或追加实验，应以新提交/新目录明确记录，保留本次提交中的原始结果。
