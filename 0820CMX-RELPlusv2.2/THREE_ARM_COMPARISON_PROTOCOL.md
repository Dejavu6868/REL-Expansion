# RGBD / HHA / REL+ 三臂比较协议

## 冻结控制

三个未授权 formal config 统一 Original CMX、MiT-B2、MLPDecoder、关闭 Gate/SMMF/DyMM/SGA、同一 RGB/label/split/seed/sampler/transform sampler/loss/scheduler/evaluator/endpoint，且 no flip。

唯一允许变化的是 X 模态及必要读取：

- RGBD：CMX canonical `RawDepth`，单通道复制为三通道；
- HHA：CMX HHA 三通道；
- REL+：offline canonical 480 `[EGVIA, LOA, ReD]` 与诊断 valid mask。

## 真实 trace

正式证据入口为 `tools/trace_three_arm_dataloaders_v2_2.py`。它创建三个真实 Dataset、三个真实 DataLoader、三个独立 RNG 实例，在相同 epoch/rank/base seed 下分别运行，并记录 sample ID、occurrence、rank、worker、scale、crop、scaled/output shape 和四边 pad。

本轮比较每臂前 50 项，mismatch=0。`constructed_trace_copy_used=False`。旧 `tools/trace_comparison_profile.py` 已退役并 fail loud；保留的 synthetic RNG helper 只用于单元测试，不能作为三臂证据。

## 历史边界

V2.2 使用 offline canonical 480 + no-flip。Stage1H/Stage2B 使用不同表示与增强协议；旧 Stage2B legacy relplus 数字不可直接作为控制组；旧 S3D mirror-on RGBD/HHA/REL 数字也不是严格 no-flip 对照。
