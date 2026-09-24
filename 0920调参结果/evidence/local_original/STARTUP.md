# focal_gamma=1 双臂正式训练启动记录

状态：FORMAL_TRAINING_RUNNING，评估已接入同一持久队列。实测时间：2026-09-20T23:34:14.589889+08:00。

## 已执行

用户指定 focal_gamma=1。沿用最新冻结 `0915调参结果`：global batch56（8×7）、lr1.2e-4、seed12345、200epochs、945步/轮、189000更新/臂、9450预热更新。保留原S2D Fold1训练52903/测试17593、480×480、FP32、NO_FLIP、AdamW、增强、缓存、通道、归一化、workers与BN语义。HHA和REL+从同一MiT-B2预训练重新初始化。

HHA现为epoch1/200，第70/945步；loss=1.912390，NaN替换0，8rank初始化通过并等于gamma2基线。REL+在队列中等待。尚无本轮epoch200权重或新mIoU。当前资源实查任卡最低空闲6.294GiB、最高60°C。

## 验证

- source174文件与冻结版逐字节一致；200项新bundle部署SHA通过。
- HHA129字段、REL+122字段全量差异检查：科学参数仅focal_gamma从2变1；实验ID、输出路径以及准确的新source目录属于路径身份迁移。
- 27项训练/隔离测试与9项评估测试本地、远端均PASS。
- 双臂各8rank，20步+保存恢复+2步短测PASS；16rank实际criterion为FocalLoss2d/gamma1/reduction none/ignore255，loss/logits/gradients有限，NaN0、恢复验证正常；16rank初始模型hash全部等于冻结gamma2。
- 原gamma2两臂epoch200 checkpoint与184文件训练bundle均通过本轮实时SHA复核，旧结果未改变。
- 80份远端准备证据已拉回并逐项SHA核验。

首次预检查attempt1在0训练更新时因source迁移带来的root_dir/abs_dir变化被保守检查拒绝；修正为仅允许准确的新source路径后，使用attempt2重做并通过。失败记录与当时完整代码保留在attempt1/failed_bundle_snapshot，无正式训练因此被覆盖或续训。

## 持久队列与后续评估

远端bundle：`/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-FG1-20260920`。
远端输出：`/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_FG1_20260920/attempt2`。

总supervisor PID=1563106、starttime=1058262511；已脱离SSH，进程身份在上面的实测时间核对一致。只有对应身份复核后才能操作该进程。

`pipeline_status.json`记录全流程；`train_status.json`记录HHA→REL+；`runs/<arm>/runtime_status.json`、`arm_status.json`、`exitcode`、`checkpoints/`为逐臂证据。

两臂训练都满足exit0、8rank完成、epoch200/189000更新、NaN0与权重hash后，队列自动执行已请求的固定epoch200全量评估。结果写入`evaluation_epoch200/status.json`、`dual_arm_comparison.json`以及逐臂混淆矩阵/逐类IoU；报告新双臂结果、同臂gamma1−gamma2、REL+−HHA及差值变化。保持原criterion=None评估及decoder eps1e-5；其与训练eps1e-3的既有差异保持原样。

按上一轮HHA+REL+训练用时粗估约72.9小时（约3天），不是本轮完成时间保证。原资源保护保留；失败即停并留证，不自动重试/改参数/选checkpoint。不创建Codex自动监控，未进行Git发布。一次seed的分数变化只作描述性结果。

## 证据入口

- `PROTOCOL.md` / `suite.json`：最终协议和配置。
- `evidence/startup_snapshot.json`：本次实际启动与GPU/进程观测。
- `evidence/remote_preparation/`：预检查、完整resolved配置、双臂短测和启动回执。
- `evidence/resolved_config_difference.json`：相对冻结gamma2的字段差异。
- `evidence/TRANSFER_VERIFICATION.json`：80文件传输核验。
- `evidence/deployment_attempt2.json`：部署及失败attempt保留回执。

本地快照不会自动刷新；后续查询须读取上述远端状态，不能重复启动队列。
