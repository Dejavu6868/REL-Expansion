# HHA / REL+ focal_gamma 单因素双臂实验准备

状态：AWAITING_PARAMETER_SELECTION。当前未启动新训练或评估，未部署候选配置。用户已要求执行双臂实验；待明确 gamma 数值与冻结基线版本，这不是重复索取执行授权。

## 与冻结主线的关系、阶段与问题

本次为 S2D Fold1 单因素超参数对照，准备进入 stage 6 训练，再做 stage 7 固定终点评估。科学问题是：改变 focal_gamma 后，同一表示各自的 mIoU 如何变化，以及 REL+−HHA 的差值是否变化。原 0915调参结果与 0915三臂结果保持冻结。

## 可审查候选

建议选择最新 0915调参结果作为基线，仅把两臂 focal_gamma 从 2 改为 1。其他 shared 字段逐项相同，候选完整字段与差异见 PREPARATION.json。数值尚未由用户选择，不能把此文件当已批准运行配置。

两臂均从相同 MiT-B2 预训练重新起跑，使用原 S2D Fold1：Area1/2/3/4/6 训练 52,903 张，Area5a/5b 测试 17,593 张。batch56、lr1.2e-4、seed12345、200 epochs、FP32、480×480、NO_FLIP、16 workers/rank、AdamW/weight decay/warmup/Poly、缓存/通道/归一化和既有 BN 语义均沿用该基线。每臂945步/轮、189000更新、9450预热更新。按上一轮相同两臂约72.9小时作粗略预算；新gamma实际耗时未知。

## 已核接线

suite_common.py 的 build_config 将 shared 字段写入真实 config；source/utils/training_protocol.py 的 build_author_criterion 读取 focal_gamma；source/utils/loss_opr.py 的 FocalLoss2d 将 gamma 用于 (1-softmax)^gamma。source/tools/eval_rel_plus_v2_3_full.py 以 criterion=None 构造模型，故只改旧权重的评估配置不能检验 gamma 的训练效果。

参考：[Focal Loss 原论文](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)。gamma1 是用于检验减弱难样本加权的候选，不是论文对本项目的最优参数保证。

## 前提、门检与停止条件

- 科学前提：仅单seed描述性对照，不扩展到全景迁移或跨seed稳定收益。
- 数据前提：相同训练/测试有序列表、预训练、HHA/REL+缓存及数据审计文件SHA，保持原输入协议。
- 控制与工具前提：复制原174文件source快照到独立bundle，仅变外层实验身份、双臂范围、gamma和输出路径；用完整resolved配置差异证明只改变gamma。各臂8rank短测须证明实际criterion.gamma、有限loss/logits/gradients、NaN0、保存恢复和16rank相同初始化。新入口须拒绝suite.shared额外键；完整resolved差异白名单仅gamma、实验ID、输出路径及ddp_smoke_report。runtime另断言criterion.gamma、reduction=none和ignore_index=255，避免只记录配置而未证明实际损失。
- 保持原GPU allocator75%、至少20%余量、85°C温度门与主存/SHM保护；失败保留现场并停队列，不自动换参数或重试。

运行PASS要求每臂exit0、8rank完成、epoch200/189000更新、NaN0和最终权重SHA。此PASS只解锁已请求的固定epoch200全量评估。

评估PASS要求每臂17593唯一ID和精确无补齐分片、混淆矩阵/指标独立复算、GT直方图一致、checkpoint/源码/输入证据保持不变。输出新gamma双臂结果、各臂相对gamma2的差值、REL+−HHA及其变化。

科学方向标记：新REL+−新HHA>0为该seed的正方向，否则为非正方向；提升须另看新旧同臂差值。运行PASS不等于性能提升或统计显著性。证据缺失/资源不可用为BLOCKED，执行或完整性失败为FAIL；都不能以历史best checkpoint替代终点。

## 尚待完成

参数选择后生成最终独立bundle与端到端队列，核验部署/配置/数据，完成双臂短测，再启动HHA→REL+的200轮训练，全部成功后自动执行已请求的双臂epoch200评估。当前未创建Codex自动监控。
