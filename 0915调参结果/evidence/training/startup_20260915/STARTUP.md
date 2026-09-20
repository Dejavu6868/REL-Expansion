# CMX 新三臂正式训练启动记录

**状态：FORMAL_TRAINING_RUNNING。** 观测时间：2026-09-15T23:55:47.141186+08:00。当前RGBD运行，HHA和REL+已排队；训练尚未完成，尚无新mIoU。

## 已执行与验证

- 用户已授权RGBD/HHA/REL+使用总batch56、lr1.2e-4、seed12345、200epochs，并明确沿用原52903/17593划分。
- 每臂945步/轮、189000次更新；共同AdamW/Focal gamma2/wd.01/warmup10/Poly.9/480×480/NO_FLIP/FP32。
- 174个源文件与冻结指纹一致；新bundle184文件传输指纹一致。原三臂源码、结果、checkpoint和数据未改；没有Git操作。
- 本地和远端23项测试PASS；新采样器epoch1/2完整覆盖52903张真实图片，三臂逐rank顺序一致。
- 三臂各8rank、20步+保存恢复+2步smoke全部PASS；24个rank初始化SHA相同；loss/logits/gradients有限、NaN替换0、参数及rank RNG恢复通过。
- RGBD正式8rank初始化与smoke一致；启动观测已完成epoch1的120/945更新，loss=1.331606，NaN替换0。当时lr=1.511111111e-06，属于10轮预热，配置base lr为1.2e-4。
- 启动资源检查PASS；观测任卡最低free 6.293945GiB、最高64°C，8个GPU进程全部属于当前作业。

## 运行与输出

服务器：zhuzhaoziao@172.20.50.33。独立执行bundle：`/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-20260915`，模型与训练源位于其`source/`。

服务器输出根：`/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_20260915/attempt1`。

队列顺序为RGBD→HHA→REL+；每臂占用8卡。服务器supervisor PID=817388（仅本次历史身份，后续操作必须复核starttime），已脱离SSH会话。每臂只有exit0、8rank完成、epoch200/189000、NaN0、最终checkpoint存在并记录SHA后才允许启动下一臂。任何失败会停止本次队列；不自动改参数、重试或恢复，不停止其他服务。

- 队列状态：`/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_20260915/attempt1/train_status.json`
- 当前训练状态：`/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_20260915/attempt1/runs/rgbd/runtime_status.json`
- 训练日志：`/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_20260915/attempt1/runs/rgbd/launcher.log`
- 最终权重：`/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_20260915/attempt1/runs/<arm>/checkpoints/epoch-200.pth`
- 每臂100轮前每5轮保存两个交替恢复槽；100轮起每5轮保存正式checkpoint。

按此前HHA工程吞吐估算纯训练合计约107小时（4.5天），不含保存/冷缓存/模态差异；不是实测完成时间。没有恢复Codex自动监控；服务器内的资源保护与串行队列持续执行已授权训练。

## 结论边界与保留事项

本轮同时改变batch和lr，三臂之间只变X表示；不额外运行lr6e-5对照，不能单独归因于学习率。保持单seed结论边界。本次队列仅训练，后续应按固定epoch200完整测试集统一评估，禁止test checkpoint扫选。

CPU构造核验已确认旧evaluator decoder BN eps=1e-5，训练配置eps=1e-3；真实训练BN也已在rank_init记录。分数影响UNKNOWN，本次不修改原训练/评估实现或覆盖0915分数。

首轮部署预检查因传输规则遗漏source/tests而失败；补齐后重新核验PASS。失败日志与preflight_attempt1_FAIL.json均保留，无训练因此启动或覆盖。

## 本地证据

- [训练协议](</Users/maxzhu/Documents/Codex/old-projects/windows/REL Expansion/work/CMX_S2D_B56_LR12_20260915/PROTOCOL.md>)
- [实际启动快照](</Users/maxzhu/Documents/Codex/old-projects/windows/REL Expansion/work/CMX_S2D_B56_LR12_20260915/evidence/remote/startup_snapshot.json>)
- [队列启动回执](</Users/maxzhu/Documents/Codex/old-projects/windows/REL Expansion/work/CMX_S2D_B56_LR12_20260915/evidence/remote/train_supervisor_launch.json>)
- [三臂smoke结果](</Users/maxzhu/Documents/Codex/old-projects/windows/REL Expansion/work/CMX_S2D_B56_LR12_20260915/evidence/remote/smoke_status.json>)
- [预检查与数据/BN审计](</Users/maxzhu/Documents/Codex/old-projects/windows/REL Expansion/work/CMX_S2D_B56_LR12_20260915/evidence/remote/preflight.json>)
- [部署校验](</Users/maxzhu/Documents/Codex/old-projects/windows/REL Expansion/work/CMX_S2D_B56_LR12_20260915/evidence/remote/deployment.json>)
- [本地检查日志](</Users/maxzhu/Documents/Codex/old-projects/windows/REL Expansion/work/CMX_S2D_B56_LR12_20260915/evidence/all_tests.log>)

本地快照不会自动更新；后续询问状态时须重新读取服务器文件。
