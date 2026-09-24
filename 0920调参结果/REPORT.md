# focal_gamma=1：HHA / REL+ 双臂训练与评估结果

已完成。两臂训练和固定 epoch 200 全量评估均为 PASS，队列于 **2026-09-24 00:27:47（北京时间）**结束；**2026-09-24 10:06:33**完成服务器只读最终复核。HHA 的 mIoU 下降，REL+ 的 mIoU 小幅上升。

| 模型 | 冻结 gamma=2 mIoU（%） | 本轮 gamma=1 mIoU（%） | 本轮减冻结值（百分点） |
|---|---:|---:|---:|
| HHA | 60.98827182 | 60.18289885 | -0.80537297 |
| REL+ | 61.01689760 | 61.40637240 | +0.38947480 |

本轮 REL+−HHA 为 **+1.22347355** 个百分点，原 gamma=2 为 +0.02862578 个百分点，差距扩大 +1.19484777 个百分点。其中 HHA 下降贡献 0.80537297，REL+ 上升贡献 0.38947480 个百分点。这是单 seed 的实际观测，不能据此认定稳定优势或两臂共同受益。

| 模型 | gamma=1 像素准确率（%） | gamma=1 平均类别准确率（%） |
|---|---:|---:|
| HHA | 82.49702495 | 68.10313214 |
| REL+ | 82.91289230 | 69.36485739 |

唯一科学参数变化为 focal_gamma 2→1；实验标识及独立输出路径相应改变。HHA 的 129 字段、REL+ 的 122 字段完整配置已逐项比对。其余沿用 0915调参结果：batch56（8×7）、lr=0.00012、AdamW、weight_decay=0.01、seed12345、200 epochs、每 epoch 945 步、warmup 10 epochs、Poly power 0.9、480×480、原预训练权重与初始化、原数据与增强协议。

S2D Fold1 沿用 Area1/2/3/4/6 训练（52,903 图），Area5a/5b 测试（17,593 图）。两臂各完成 189,000 次更新，8 rank 完成、exit 0、NaN 替换 0。评估固定 epoch 200，480 整图、scale=1、无翻转；每臂均覆盖 17,593 个唯一测试样本、3,973,620,198 个有效像素。原评估约定继续保留：criterion=None 时 decoder BN eps=1e-5，训练 eps=1e-3，与冻结基线相同。

完成阶段的全部 114 份远端证据（8,005,685 字节）已下载并逐文件核对 SHA-256。最终只读复核通过：200 个冻结 bundle 文件、19 个输入/前置证据文件、gamma=1 和 gamma=2 共四个终点权重、精确样本分片、8 rank 混淆矩阵求和、IoU/准确率复算及对照差值均一致。本地独立复核另用整数/Fraction 从原始混淆矩阵重新计算，共 189 项检查 PASS（浮点尾数差小于 1e-10 个百分点）。两臂训练总计约 72.82 小时，自动评估约 5.64 分钟。

[完整协议](<code/training/PROTOCOL.md>) · [全部总体指标与变化 CSV](<results/gamma1_vs_gamma2_metrics.csv>) · [13 类 IoU CSV](<results/gamma1_per_class.csv>)

[本地独立复核](<evidence/completed/INDEPENDENT_OFFLINE_AUDIT.json>) · [原始双臂比较](<evidence/completed/evaluation_epoch200/dual_arm_comparison.json>) · [服务器最终复核](<evidence/completed/LIVE_FINAL_RECHECK.json>) · [证据传输校验](<evidence/completed/TRANSFER_VERIFICATION.json>) · [完整配置差异](<evidence/startup/resolved_config_difference.json>)

实际代码位置：`/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-FG1-20260920`。入口 `pipeline.py --phase execute`，训练后接 `evaluation/run_evaluation.py` / `eval_rank.py`。174 个 source Python 文件与冻结源版本逐字节一致，完整 bundle 清单 SHA-256 为 `42df431764b0c0aad2f2a71214bda28dfc6a34d9d06eea2ab9bebad6b342892c`。

远端权重根目录：`/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_FG1_20260920/attempt2/runs`。HHA：`hha/checkpoints/epoch-200.pth`，SHA-256 `d4d210ea92a473cc700b1660dff7da644e8d1b9934be759b216c0d4931ae2b4d`；REL+：`relplus/checkpoints/epoch-200.pth`，SHA-256 `6506a922b9d31919576b3f998864993a24ca7f07baca868d71a5ff3b6a581831`。每个权重 799,760,393 字节；权重保留在服务器。

attempt1 仅发生预检查失败、0 训练步，失败证据保留；本报告只使用 attempt2。本目录按用户授权冻结参数、代码、结果与证据；GitHub 提交标识记录在发布回执中。
