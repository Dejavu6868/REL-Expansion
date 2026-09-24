# HHA / REL+ focal_gamma=1 双臂训练与固定终点评估协议

状态：AUTHORIZED_PREPARATION。用户明确指定 focal_gamma=1；其余按最新冻结 0915调参结果 执行。当前准备部署与短测，通过后执行已请求的200轮双臂训练及epoch200评估。实际运行状态以独立状态文件为准。

## 与冻结主线的关系、阶段与问题

本次为 S2D Fold1 单因素超参数对照，准备进入 stage 6 训练，再做 stage 7 固定终点评估。科学问题是：改变 focal_gamma 后，同一表示各自的 mIoU 如何变化，以及 REL+−HHA 的差值是否变化。原 0915调参结果与 0915三臂结果保持冻结。

## 已定实验参数

选择最新 0915调参结果作为基线，仅把两臂 focal_gamma 从 2 改为 1。其他 shared 字段逐项相同，完整运行配置见 suite.json。PREPARATION.json 与 READINESS.md 是选择前的历史准备材料。

两臂均从相同 MiT-B2 预训练重新起跑，使用原 S2D Fold1：Area1/2/3/4/6 训练 52,903 张，Area5a/5b 测试 17,593 张。batch56、lr1.2e-4、seed12345、200 epochs、FP32、480×480、NO_FLIP、16 workers/rank、AdamW/weight decay/warmup/Poly、缓存/通道/归一化和既有 BN 语义均沿用该基线。每臂945步/轮、189000更新、9450预热更新。按上一轮相同两臂约72.9小时作粗略预算；新gamma实际耗时未知。

## 已核接线

suite_common.py 的 build_config 将 shared 字段写入真实 config；source/utils/training_protocol.py 的 build_author_criterion 读取 focal_gamma；source/utils/loss_opr.py 的 FocalLoss2d 将 gamma 用于 (1-softmax)^gamma。source/tools/eval_rel_plus_v2_3_full.py 以 criterion=None 构造模型，故只改旧权重的评估配置不能检验 gamma 的训练效果。

参考：[Focal Loss 原论文](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)。gamma1 是用于检验减弱难样本加权的候选，不是论文对本项目的最优参数保证。

## 前提、门检与停止条件

- 科学前提：仅单seed描述性对照，不扩展到全景迁移或跨seed稳定收益。
- 数据前提：相同训练/测试有序列表、预训练、HHA/REL+缓存及数据审计文件SHA，保持原输入协议。
- 控制与工具前提：复制原174文件source快照到独立bundle，仅变外层实验身份、双臂范围、gamma和输出路径；用完整resolved配置差异证明只改变gamma。各臂8rank短测须证明实际criterion.gamma、有限loss/logits/gradients、NaN0、保存恢复和16rank相同初始化。新入口须拒绝suite.shared额外键；完整resolved差异白名单仅gamma、实验ID、输出路径、ddp_smoke_report及准确指向新冻结source的root_dir/abs_dir。runtime另断言criterion.gamma、reduction=none和ignore_index=255，避免只记录配置而未证明实际损失。
- 保持原GPU allocator75%、至少20%余量、85°C温度门与主存/SHM保护；失败保留现场并停队列，不自动换参数或重试。

运行PASS要求每臂exit0、8rank完成、epoch200/189000更新、NaN0和最终权重SHA。此PASS只解锁已请求的固定epoch200全量评估。

评估PASS要求每臂17593唯一ID和精确无补齐分片、混淆矩阵/指标独立复算、GT直方图一致、checkpoint/源码/输入证据保持不变。输出新gamma双臂结果、各臂相对gamma2的差值、REL+−HHA及其变化。

科学方向标记：新REL+−新HHA>0为该seed的正方向，否则为非正方向；提升须另看新旧同臂差值。运行PASS不等于性能提升或统计显著性。证据缺失/资源不可用为BLOCKED，执行或完整性失败为FAIL；都不能以历史best checkpoint替代终点。

## 执行顺序

独立bundle/source保留174文件冻结快照。pipeline.py --phase prepare 核验配置/数据并完成HHA、REL+各8rank的20+2步短测；pipeline.py --phase execute 核对prepare成功后启动HHA→REL+的200轮训练，全部成功后执行已请求的双臂epoch200评估。phase均持久独立会话且拒绝重复启动，不创建Codex自动监控。

## 精确路径与最终证据

- 远端bundle：`/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-FG1-20260920`。
- 远端输出：`/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_FG1_20260920/attempt2`。
- pipeline_status.json记录训练→评估总状态，train_status.json记录双臂训练，各臂runs/<arm>/runtime_status.json和exitcode记录进度/退出。
- evaluation_epoch200/status.json与双臂对照JSON记录固定终点评估和相对gamma2变化。
- 所有正式权重仍在runs/<arm>/checkpoints/epoch-200.pth；baseline/留存冻结gamma2配置、指标与初始模型哈希来源。
- 第一次执行前检查184份基线bundle内容hash均匹配，Git工作区初始无跟踪差异；本次不发布Git。

预检查attempt1在0训练更新时因root_dir/abs_dir随源码快照目录迁移而拒绝；已限定这两字段必须准确指向本新bundle/source，未改变任何数据/训练行为。失败attempt1与完整当时代码快照保留，修正后使用attempt2重新进行预检查。
