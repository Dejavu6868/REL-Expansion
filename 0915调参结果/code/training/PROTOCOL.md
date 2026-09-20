# CMX S2D 新三臂正式训练：batch56 / lr1.2e-4

状态：参数和范围已由本会话用户授权；运行状态以独立证据文件为准。

## 问题与冻结主线的关系

研究载体仍是普通RGB-D、Stanford2D3D S2D的CMX表示比较。本次为stage 6执行阶段：在同一个新训练配方下，训练RGBD、HHA和REL+，为固定epoch200三臂比较提供权重。0915三臂结果、源码、checkpoint、输入缓存、数据列表保持原样。

相对0915配方共同改变全局batch8→56、lr6e-5→1.2e-4，联动重算每轮更新/逻辑样本/补样本。三臂彼此只改变X输入表示。额外统一OpenCV/PyTorch/OMP线程数1及REL+前100轮的恢复checkpoint策略；这些是明确记录的执行差异。本次不能单独估计学习率或batch的贡献，不能凭单seed宣称稳定科研收益。

## 用户授权

用户明确请求：以这组参数把CMX三臂进行训练。

数据划分已确认：沿用原划分，做三臂正式训练。故本次不是上一轮提议的独立validation调参筛选；不建立4～6候选排名校准，也不新增lr6e-5对照臂。三臂都从相同mit_b2预训练重新开始；不使用旧科学checkpoint或smoke checkpoint续训。

## 实验矩阵

| 顺序 | 臂 | X输入 | 模态身份 |
|---|---|---|---|
| 1 | CMX-RGBD | uint8深度repeat3 | RGBD_UINT8_REPEAT3_SOURCECOMPAT |
| 2 | CMX-HHA | 冻结HHA数组顺序 | HHA_FROZEN_CACHE_EXECUTABLE_ORDER |
| 3 | CMX-REL+ | 冻结EGVIA/LOA/ReD | RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT |

共同条件：52903训练/17593测试、13类、ignore255、480×480、双MiT-B2+MLPDecoder512、seed12345、8GPU×每卡7、AdamW betas(.9,.999)、lr1.2e-4、wd.01、Focal gamma2、完整200轮WarmUpPolyLR日程、预热10轮、power.9、FP32、每rank16workers。保留现有多尺度和NO_FLIP，保持原正式TrainPre随机流；不使用测速探针的逐样本固定增强RNG。

每轮945更新、52920逻辑样本、17个padding；每rank6615样本=945×7。每臂189000更新，warmup9450更新。保留optimizer.step后设置lr的时序，首个更新使用base lr，随后迭代0调度为0。decoder SyncBN、encoder FFM普通BN，保持现有实现。

## 前提与检查

- 科学前提：这只是单seed新配方三臂训练；短训预测全训和跨seed稳定性没有获得支持。
- 数据前提：原列表/预训练指纹一致，旧严格数据校验继续执行；三臂原有模态身份保留，避免掉入legacy校验。新sampler逐rank验证epoch1/2的完整覆盖和跨臂序列一致。
- 控制与工具前提：新来源快照174Python文件与冻结指纹一致；独立新配置与入口、输出隔离、实际lr和初始化哈希记录；三臂各一次20步训练+保存恢复+2步更新smoke；三臂24rank初始化必须一致。
- BN语义：独立CPU枚举冻结evaluator的criterion=None构造，smoke记录真实训练BN。若发现构造差异，保留既有实现和差异证据，分数影响仍UNKNOWN；本次不改冻结evaluator或重算旧分数。正式训练与评估是不同动作，本次队列不进行checkpoint扫选。

## 终点、停止与解锁

运行PASS：每臂8rank完成、exit0、FORMAL_TRAINING_COMPLETED、epoch200/global_iteration189000、NaN替换0、epoch200权重存在且哈希已记录；随后队列才解锁下一臂。三臂完成只解锁固定epoch200的完整测试集评估与审计，不解锁额外seed、调参搜索或全景迁移。

科学比较终点：后续统一整图480×480、scale1/no-flip、17593测试图的epoch200 mIoU和逐类IoU。预先报告REL+−HHA及REL+−RGBD；单seed正差仅为描述性结果，负差不能被best checkpoint替换。当前训练状态不得冒充科学PASS。

运行FAIL/BLOCKED：非有限数值、任rank错误、非零退出、证据缺失、资源门不满足、来源变化、外来GPU作业或进度1800秒未前进时，停止本次所属进程组并停止队列，保留失败证据。不自动重试、恢复或改参数。

资源沿用已验证保护：单rank allocator75%；任卡空闲至少max(3GiB,20%标称容量)；温度低于85°C；启动主存64GiB/SHM4GiB、运行主存32GiB/SHM2GiB；仅核对PID/starttime/PGID后终止自建进程组。持续资源记录是本训练进程的内部保护，不恢复Codex自动监控。

## 保存、预算与证据

每臂正式checkpoint保存100/105/…/200轮；100轮前每5轮用两个交替恢复槽。默认从预训练起跑；恢复必须另核同臂同配置身份。本次没有提前淘汰或约67轮选点。

按历史HHA工程吞吐条件估计每臂纯训练35.77小时，三臂107.31小时；不含冷缓存/保存/各模态差异，不作为测量结果或硬性总超时。输出空间启动要求至少250GiB；实际资源、进度与退出状态持续留存。

证据包括suite.json、完整resolved配置、源码及输入SHA、环境、采样检查、真实初始化SHA、BN属性、smoke保存恢复结果、argv/env、stdout/stderr、资源轨迹、逐臂运行状态、8rank完成记录、checkpoint/退出码/失败原因。所有实验件位于新bundle或新输出attempt目录。
