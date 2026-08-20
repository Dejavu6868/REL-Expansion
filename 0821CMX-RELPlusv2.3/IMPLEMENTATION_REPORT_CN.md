# CMX-REL+ V2.3 实现、验证与正式训练启动报告

## 最终状态

`FORMAL_TRAINING_STARTED`

V2.3 代码、70,496 样本正式 cache、完整 cache audit、完整 CMX
training-data preflight 和真实 8-GPU optimizer/checkpoint smoke 均已完成并
通过。唯一获授权的 CMX-REL+ V2.3、seed 12345、8-GPU、200-epoch 正式
训练已启动；本报告不把“已启动”写成“已完成”。

## 主线、科学问题与冻结控制

- 主线阶段：冻结 REL+ v2.1 表示后的 CMX-REL+ 单臂正式训练。
- 后续科学问题：在严格冻结 CMX、数据、增强、loss、scheduler 和 endpoint
  时，REL+ 相对 RGBD/HHA 是否改善 S2D test 表现。
- 本轮改变因素：缓存、审计、训练门禁、真实 DDP、checkpoint 身份和评价
  基础设施；未改变 X 表示。
- 冻结数据/控制：S2D official train/test、Original CMX、MiT-B2、
  MLPDecoder、global batch 8、200 epochs、Focal gamma 2、AdamW、
  WarmUpPolyLR、no-flip、13 类、ignore 255、seed 12345。
- 主 endpoint：epoch 200；次 endpoint：`test_selected_best`，后者明确存在
  test-selection bias。

## 路径

- V2.2 基线：`/home/zhuzhaoziao/RELPlus/RELPlusv2.2`
- V2.3 代码：`/home/zhuzhaoziao/RELPlus/RELPlusv2.3`
- full manifest：
  `/data/zhuzhaoziao/RELPlus/outputs/REL_plus_v2_1_implementation/full_manifest.csv`
- 运行证据：`/data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v2_3`
- 正式训练：
  `/data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v2_3/formal_training/CMX_RELPlus_v2_3_seed12345`

V2.2 及历史目录均保留不动。

## REL+ 表示不变量

正式表示仍为
`RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`，通道顺序仍为
`[EGVIA, LOA, ReD]`，invalid policy 仍为
`SOURCE_COMPAT_STORAGE_255`。V2.3 的 `rel_plus/` 与 V2.2 source tree
一致，没有重新设计 K、pose、normal、ReD、EGVIA、LOA 或 invalid。

冻结生成器与 V2.3 调用路径的合成样本和 12 个真实 S2D 样本 byte
regression 均为：changed pixels 0、changed channels 0、maximum difference
0。

## V2.3 修复

1. auditor duplicate 检查由 O(N²) `list.count` 风险改为 `Counter` O(N)。
2. generation、audit、preflight 和 training gate 绑定归一化绝对 cache、
   manifest、resolved manifest、ordered train/test 列表及协议 ID；错误身份
   fail closed。
3. 正式训练同时要求 cache generation PASS、cache audit PASS 和完整 CMX
   preflight PASS，不接受旧 `data_ready` 回退。
4. 70 个重生成样本按七个 area、invalid、normal quality、gravity、
   room/camera 和固定随机风险分层；逐像素对拍。
5. `full_cache_generated` 只有在全 manifest 成功生成/严格 resume 且零失败时
   才为 true；PNG 原子写入，损坏/shape 错误文件重新生成。
6. checkpoint 文件名 epoch 与 payload epoch 对拍，不一致直接失败。
7. sweep 每个 checkpoint 顺序启动一次 multi-GPU evaluator；1/2/8-rank
   confusion 完全一致。
8. evaluator 内部统一 fraction 0..1，正式表格显式输出 percent；
   `test_selected_best` 保留偏差声明且不会替代 epoch 200 主 endpoint。
9. 新增真实 8-rank DDP step/save/restore smoke、显式 resolved config 和
   fail-closed 正式 launcher。

## 分阶段执行结果

### 代码与测试完成

- 最终 live-source regression：124 passed，exit code 0。
- V2.3 基础设施测试：16 passed。
- 初始 TDD RED：14 failed / 1 passed，随后实现至全绿；area_5a/5b 的真实
  `area`/`area_group` 差异也保留独立 RED/GREEN 证据。
- evaluator rank consistency：1/2/8 ranks 逐元素一致。
- frozen generator byte invariant：PASS，所有差异为 0。

### Full cache 完成

- 生成：70,496/70,496，train 52,903，test 17,593，failure 0。
- REL+ PNG：70,496；ValidMask PNG：70,496；临时 PNG：0。
- generation 状态：PASS，`full_cache_generated=true`。
- 16-worker 实测吞吐：47.0309 images/s，500 张零失败。

### Audit 完成

- attempt1 因 auditor 错误地把 area_5a/5b 折叠为 area_5 而失败；失败日志
  保留，cache 未改动。
- 一行分组修复和回归测试后完整重跑 attempt2：PASS，failure 0。
- 70 个风险分层样本（七个 area 各 10）全部重新生成一致，非零 changed
  pixel/channel/max-difference 行数均为 0。
- invalid 插值诊断只记录风险，不改变正式输入。

### Training-data preflight 完成

- 70,496/70,496 RGB、label、REL+、ValidMask 均在本次运行中实际解码。
- train/test 为 52,903/17,593；missing、decode、shape、dtype、range、
  class mapping、pairing 和 split identity failure 均为 0。
- 类别映射来自真实配置文件；未猜测 class ID。

### DDP smoke 完成

- 8 GPUs / 8 ranks，50 次 checkpoint 前更新和 3 次恢复后更新。
- 执行真实 backpropagation 与 AdamW `optimizer.step()`。
- rgb encoder、X encoder、fusion、decoder 四组参数在保存前后均确认更新。
- disposable checkpoint 保存/恢复成功，恢复后模型快照一致，optimizer LR
  连续且继续更新。
- loss、logits、gradients 全部有限；NaN replacement count 为 0。
- disposable checkpoint 位于 `ddp_smoke/`，未进入正式 sweep，也未替换任何
  既有 checkpoint。

### 正式训练已启动，尚未完成

- launch ID：`CMX_RELPlus_v2_3_seed12345_20260821_005914`
- launcher PID：3126628；durable wrapper PID：3126627。
- 8 ranks 均已启动并加载 MiT-B2。
- 2026-08-21 01:00:35 +08 观测：epoch 1、iteration 150、global
  iteration 150、loss 1.7656955719、LR 1.3518826554e-07、world size 8、
  NaN replacement count 0。
- 该运行执行 backpropagation 与 optimizer update，属于正式 retraining。
- 尚未到 epoch 100，因此没有正式 checkpoint，也没有 checkpoint replacement。
- 200 epochs 尚未完成，训练最终退出码和 17,593 张 test split 的科学
  mIoU 尚不存在。

## 三臂边界

RawDepth 和 HHA 的 70,496 样本只读数据合同均为 READY，未阻塞本轮 REL+
单臂。但本轮没有启动 RGBD、HHA、原始 REL、多 seed、Fold、调参或其他
实验，亦没有报告三臂科学比较结果。

本轮未生成或写入任何文件哈希。
