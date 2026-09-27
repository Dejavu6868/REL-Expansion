# focal_gamma=1.5 双臂训练与评估启动记录

截至 **2026-09-24T10:58:18.965715+08:00**，HHA已进入正式训练第1/200轮、第40/945步，REL+排队。两臂训练完成后，服务器会自动评估各自固定epoch200权重。当前没有本轮mIoU结果。

## 配方与边界

基于最新冻结 `0920调参结果`（提交c8ff74502cf6717fc554adc47db4d14e9ddd42a9），两臂仅改变 `focal_gamma: 1→1.5`。保持batch56、lr0.00012、seed12345、200epochs、945步/轮、原52903训练/17593测试Area5留出划分及其他全部冻结训练/数据/评估参数。两臂从相同原MiT-B2预训练和初始化重新训练；不会使用短测恢复权重作为正式起点。

## 已验证

- 原174文件source字节不变，原FocalLoss2d公式不变；CUDA FP32七个随机/饱和输入用例loss及梯度有限。
- 服务器实际HHA129字段、REL+122字段完整配置相对gamma1仅gamma科学参数变化；模型/输入/原BN语义保持冻结。
- 27项训练合同、14项评估合同检查通过；206文件部署逐SHA校验通过，训练和评估独立复核通过。
- 两臂各8rank真实20+2步短测均PASS，loss/logits/gradients有限、NaN替换0，保存恢复、RNG/采样状态与参数恢复通过；全部16rank实际gamma1.5且初始化SHA与旧基线一致。
- 79份准备证据已传回本地并逐SHA核对。旧gamma1/gamma2两臂共四个epoch200权重SHA在启动前重新核对一致，旧实验目录保持只读。
- 正式HHA8rank初始化PASS，实际gamma1.5；已完成40次更新，loss=1.44893038，NaN替换0，没有rank错误。该观测仅证明启动与初始进度。

## 后台队列与结果

远端bundle：`/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-FG15-20260924`

远端输出：`/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_FG15_20260924/attempt1`

持久supervisor PID=2057311，starttime=1088289304，独立会话已脱离SSH；实际进程身份与命令行已复核。不要重复启动。

`pipeline_status.json`记录训练→评估全流程，`train_status.json`记录HHA→REL+队列，`runs/<arm>/runtime_status.json`记录实时进度。固定终点评估输出在`evaluation_epoch200/dual_arm_comparison.json`，将报告gamma1.5相对gamma1、gamma2的同臂差值和REL+−HHA差值。评估使用原17593张测试图、480整图、scale1/no-flip和原BN构造语义。

沿用原资源保护与失败停队列机制，不自动换参数、重试、追加seed或选test-best。训练执行PASS与性能晋级PASS分开：后者采用完整精度REL+高于gamma1且HHA不低于gamma1，仅筛选待复验候选，不能认定稳定优势。

双臂训练参考上一轮约72.82小时，粗略约3天；实际完成时间取决于本轮运行。没有新增Codex定时监控或发布Git。

[完整实验协议](PROTOCOL.md) · [状态快照](status.json) · [实际启动证据](evidence/startup_snapshot.json) · [准备证据校验](evidence/PREPARATION_ACCEPTANCE.json) · [传输校验](evidence/PREPARATION_TRANSFER_VERIFICATION.json)

服务器当前GPU快照：

```text
0, 17682 MiB, 53, 100 %
1, 17682 MiB, 58, 66 %
2, 17682 MiB, 55, 62 %
3, 17682 MiB, 55, 95 %
4, 17682 MiB, 54, 45 %
5, 17682 MiB, 49, 58 %
6, 17682 MiB, 53, 90 %
7, 17682 MiB, 53, 51 %
```
