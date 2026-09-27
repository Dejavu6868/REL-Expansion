# HHA / REL+ focal_gamma=1.5 双臂实验协议

用户已于2026-09-24明确要求调整focal_gamma=1.5，并进行HHA/REL+双臂评估，其余使用原先冻结参数。执行状态由本轮状态文件记录，本协议不代表训练已完成。

## 主线关系、阶段与科学问题

本次为冻结S2D源域协议的参数敏感性补充，进入训练及固定终点评估。以最新0920调参结果（提交c8ff74502cf6717fc554adc47db4d14e9ddd42a9）的gamma1为主基线，保留0915 gamma2历史对照。科学问题：gamma1.5能否提高REL+绝对mIoU，同时缓解gamma1下HHA的下降。该实验不替代几何验证或四臂表示消融。

## 唯一改变及冻结控制

两臂仅把focal_gamma从1改为1.5；另变独立实验标识及输出路径。原174文件source快照逐字节保持一致；原FocalLoss2d表达式保持不变。新suite_common.py对完整resolved配置逐字段校验，只放行gamma及明确的身份路径；每个实际rank记录criterion.gamma=1.5、reduction=none、ignore255及初始化哈希。

训练仍为Area1/2/3/4/6的52903图，测试Area5a/5b的17593图，13类。两个模型均从相同MiT-B2预训练与seed12345初始化重训；batch56、每卡7、lr0.00012、AdamW betas(0.9,0.999)、weight_decay0.01、200epochs、945更新/轮、189000更新、warmup10轮/9450更新、Poly0.9、16workers/rank、FP32、480×480、原scale增强与NO_FLIP、缓存/通道/归一化均冻结。BN保持训练eps1e-3、原criterion=None评估decoder eps1e-5。

## 前提与检查

科学前提：单seed开发性对照；反复使用Area5选参后，它不是独立于选参的泛化证据。关注REL+自身变化并单独报告HHA，不把HHA下降导致的gap扩大当作REL+改进。

数据/访问前提：原预训练、有序列表、缓存及数据审计证据可读，指纹不变。工具前提：新独立bundle与输出目录；完整配置差异符合约定；原式gamma1.5 CUDA数值检查有限；HHA和REL+各8rank的20+2更新保存恢复短测PASS，16rank实际gamma1.5、相同初始化、loss/logits/gradients有限、NaN替换0。

沿用原资源保护：allocator75%、至少20%显存余量、85°C温度门、主存启动64GiB/运行32GiB、SHM启动4GiB/运行2GiB。任何配置/输入漂移、非有限loss/gradient、NaN替换或资源越界立即停止本队列并保留现场；不自动重试、改参数、选test-best或恢复其他作业。

## 终点、PASS/FAIL/BLOCKED及解锁

执行顺序：独立准备检查 → 双臂短测 → HHA正式200轮 → REL+正式200轮 → 两臂固定epoch200全量评估。训练PASS须各臂exit0、8rank完成、epoch200/189000更新、NaN0及最终权重SHA；只解锁本次已授权评估。

评估PASS须两臂17593唯一ID与精确无补齐分片、相同GT直方图、混淆矩阵及指标独立复算，权重/源码/数据证据不变。保存mIoU、逐类IoU、pixel/mean accuracy及gamma1.5相对gamma1和gamma2各臂差值、各配方REL+−HHA与gap变化。

候选晋级标准沿用本次候选建议：REL+ mIoU高于gamma1的61.40637240%，且HHA不低于gamma1的60.18289885%（比较使用完整精度原始数值）。有效完成但未达到此目标，记录候选晋级FAIL；执行或完整性失败记录运行FAIL。必需输入、资源或证据不可用为BLOCKED，不用旧权重替代本轮终点。正常停止于固定200轮，不中途根据test分数改变方案。

候选PASS仅说明gamma1.5值得后续成对新seed复验；不自动启动额外实验，不宣称统计显著或稳定优势，不解锁全景迁移。主线总体仍为约20样本几何检查→四臂单seed二维矩阵→满足门槛后ERP-21/Cubemap-6/Tangent-18迁移；本次不声称这些历史门均已通过。

## 路径、证据与用时

- 新远端bundle：/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-FG15-20260924
- 新输出：/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_FG15_20260924/attempt1
- 队列：pipeline_status.json、train_status.json；逐臂runs/<arm>/runtime_status.json、rank_init、rank_done、exitcode、checkpoints/epoch-200.pth。
- 评估：evaluation_epoch200/status.json与dual_arm_comparison.json；完整配置、源码/输入SHA、采样序列、短测/实际初始化、逐rank样本/矩阵审计均保留。
- baseline_gamma1保留最新冻结配置/协议/结果；baseline保留gamma2历史结果/协议及主基线gamma1同臂配置（文件用途由代码与哈希约束）。历史0920/0915实验目录只读。
- 双臂训练粗略预算约72.82小时，依据上一轮实际用时，不保证本轮完成时刻。服务器持久队列脱离SSH后继续执行，失败停队列；不创建Codex定时监控。
