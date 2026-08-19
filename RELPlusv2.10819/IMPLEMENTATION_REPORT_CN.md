# REL+ v2.1 代码修订、CMX 接入闭环与训练前验证报告

## 状态

`REL+ v2.1 已完成，具备提交用户进行全量 cache 与训练审批的条件`。

该结论只覆盖协议实现、真实 CMX 接入、全数据训练前 preflight、固定 pilot
与一次 wiring backward，不授权全量 cache 或正式训练。

## 版本与路径

- 未修改的 v2 基线：`/home/zhuzhaoziao/RELPlus/RELPlusv2`
- 独立 v2.1 代码：`/home/zhuzhaoziao/RELPlus/RELPlusv2.1`
- 隔离的真实 CMX 副本：上述目录中的 `cmx_integration/`
- 运行证据：
  `/data/zhuzhaoziao/RELPlus/outputs/REL_plus_v2_1_implementation`
- 全量 preflight：运行证据下的 `preflight/`
- 36 张 pilot：运行证据下的 `pilot_cache/`

v2 源码与现有 `CMX-REL`、`CMX-RGBD`、`CMX-HHA` 均未被覆盖。独立
byte-regression 调用 v2 CLI 曾产生 10 个 Python bytecode cache；这些本轮
运行副作用已按明确文件清单清除。最终 v2 恢复为 59 个原文件且 cache 为 0，
没有源码或文档内容变更。

## 核心三通道不变量

- v2.1 的 generator 直接委托冻结的 v2 生成路径；ReD、EGVIA、LOA、
  alpha 45、lambda 0.5、radius 2、厘米编码、full-image statistics、
  depth-only production validity、uint8 截断和 `[EGVIA, LOA, ReD]` 顺序均未改。
- 独立部署的 v2 CLI 与 v2.1 对 1 个合成样本和 12 个真实样本逐数组比较：
  13/13 PASS，changed pixel 0，changed channel 0。
- depth-invalid 仍正式存为 `[255,255,255]`；diagnostic valid mask 不改写
  production bytes，也不进入模型。

## 已完成的接口与验证修订

### K 与 dataset profile

新增 `STANFORD_S2D_PROFILE`，独立冻结 native `(1080,1080)`、canonical
`(480,480)`、JSON half-pixel K 和 W2C 3x4 pose。生产
`load_canonical_frame()` 必须显式接收 profile；depth 不再自我声明 K 的
reference shape。CameraGeometry 额外拒绝 nonfinite、非正 focal、非零 skew、
越界/与 profile 不相容的 principal point、非正交 R、非 +1 determinant 和
metadata camera centre 不一致。

### Pose evidence

物理验证明确分为 `PASS_STRONG`、`PASS_WEAK`、`REVIEW_REQUIRED`、`FAIL`、
`NOT_APPLICABLE`；缺少证据不再算 PASS。semantic normal 使用
`depth_valid & finite & nonzero & normal_quality`。测试覆盖 global-XYZ 强证据、
semantic-only 正例、证据不足、转置自洽错误 pose 强反例和 semantic-only
反例。

12 张真实 review 样本为 10 `PASS_STRONG` + 2 Area 1 `PASS_WEAK`；Area 1
没有 global XYZ，报告保留这一限制，不升级为强验证。

### Invalid、transform 与 dtype

- 正式策略命名为 `SOURCE_COMPAT_STORAGE_255`：REL+ 按 CMX `INTER_LINEAR`
  scale、normalize、crop/pad、HWC→CHW，不额外置零。
- nearest valid mask 同步传播，但只用于统计、审计和可视化；模型输入仍是
  RGB + 3-channel REL+。
- `SpatialTransform` 记录 source/scaled/output shape；错误 source 或 scaled
  shape 立即失败。RGB、REL+、label、mask 共用同一 transform。
- no-flip 在真实 `TrainPre` 构造时实际调用 policy validator；REL+ 模式不会
  调用 legacy `random_mirror()`。
- RGB/REL+ model input 明确为 float32 CHW。photometric callback 只接收 RGB，
  并检查返回 shape 与 uint8 dtype。
- 新增独立 OpenCV reference test，对缩小补边与放大裁剪两条 legacy CMX
  数组链逐数组对拍。

## 真实 CMX 接入

`cmx_integration/` 是从服务器当前 REL fork 派生、去除 Git metadata/cache/
runtime products 的隔离副本。新增 `x_mode="rel_plus_v2_1"`：

- REL+ 用 `cv2.IMREAD_UNCHANGED` 读取，不执行 BGR/RGB 转换；
- RGB 保持现有三臂共同的 cv2 返回字节行为，没有顺便改变公共通道；
- `RGBXDataset -> TrainPre -> DataLoader` 调用共享 adapter；
- Original CMX MiT-B2、dual encoder、MLPDecoder 保持不变；Gate、DyMM、
  SMMF、SGA 均为 off。

真实 loader sentinel 通过：REL+ `[11,22,33]` 与 RGB `[7,13,29]` 从 PNG
到 model 前 tensor 均保持逻辑顺序；输出 `torch.float32 [1,3,4,5]`，
no-flip，mask 在 batch 中但提供给模型的 mask channel 数为 0。

## Full preflight

正式 manifest 为 70,496 张：train 52,903、test 17,593。完整逐样本扫描
70,496/70,496 PASS，FAIL 0；所有 Area 的 K/pose algebra/gravity/file/depth
结构失败均为 0。CSV 70,497 行（含表头），状态
`READY_FOR_PILOT_CACHE_WITH_REVIEW`。

normal 全量统计：

- nonfinite ratio：min/median/mean/p95/max 均为 0；
- zero-normal：mean 1.3572e-8，p95 0，max 1.3822e-5；
- low-support：mean 1.4938e-7，p95 0，max 4.6975e-5；
- normal quality：min 0.999953025，mean 0.999999851；
- depth-invalid：median 0.000703125，mean 0.004997451，p95 0.024743924，
  max 0.100247396。

详见 `FULL_PREFLIGHT_REPORT.md` 和证据目录的逐样本 CSV/summary。

## 36 张 pilot 与 invalid 风险

pilot 生成 36/36 PASS，Area 1、2、3、4、5、6 各 6 张；Area 5 同时覆盖
5a/5b。输出恰有 36 个 REL+ PNG、36 个 valid mask、36 个 debug summary、
36 个全通道 montage、36 个增强后 montage；train/test 列表各 36 行。
`full_cache_generated=false`。

stratified pilot 的 invalid interpolation diagnostic：

- source invalid ratio：median 0.00966146，p95 0.09668294，max 0.10024740；
- boundary-near ratio：median 0.00278646，p95 0.01782661，max 0.01821181；
- bilinear invalid-affected ratio：median 0.00031250，mean 0.00211432，
  p95 0.01000868，max 0.01223524；
- affected pixels 的平均 channel deviation：median 51.2199，max-sample
  mean 116.5909；单 channel deviation 最大 251。

transformed nearest-invalid ratio 包含 0.75 scale 后 centre padding，因此其
median 0.06868273、max 0.48926649 不能解释为原始 depth-invalid 比例。
这些量只暴露 frozen source-compatible 255 在 bilinear 边界上的影响；v2.1
没有静默切换到 mask-zero 或 masked interpolation。

## 测试与实跑证据

- TDD red：缺失 v2.1 profile/generator/diagnostic 时 collection 失败，退出码 2；
  红灯证据保留。
- 完整 pytest：70 passed，退出码 0。
- live-source：1 passed、69 deselected，退出码 0，实际调用审计源码。
- v2→v2.1 byte regression：13/13 PASS，0 changed pixel/channel。
- K：native/native、canonical/canonical PASS；两种交叉 resolution、错误
  profile、非零 skew、越界 principal point 均 FAIL。
- Pose：strong/weak/review/fail、错误 pose 反例、quality mask 排除均通过。
- Invalid/transform/dtype/photometric、固定 seed、storage channel sentinel、
  独立 CMX 数组链对拍均通过。
- full preflight：70,496/70,496 PASS。
- pilot：36/36 PASS。
- 真实 loader + RGB/channel sentinel：PASS。

### 一次真实 single-batch wiring

使用 pilot 的 Area 1 样本执行一次真实
`DataLoader -> Original CMX forward -> CrossEntropy loss -> backward`：

- RGB、modal_x：`[1,3,480,480]` float32；label `[1,480,480]`；
- loss 2.9312751293，ignore index 255，ignore pixels 16,709；
- RGB encoder 332、X encoder 332、fusion 132、decoder 14 个 gradient tensor；
  四组均 finite 且至少一个 nonzero；
- backward 只执行这一次；optimizer 未构造，optimizer/scheduler step 均未
  执行，epoch loop 未启动，checkpoint 未写入。

实跑后证据目录与代码 staging 中的 `.pth/.pt/.ckpt` 数量均为 0，GPU compute
process 数回到 0。

## 人工目检

实际打开 7 个代表样本的 14 张图，覆盖 Area 1 weak、最高/高 invalid、
ceiling、最高 gravity tilt、最低 normal quality、0.75 scale centre padding。
RGB/depth/label/REL+/mask 的 crop/pad 边界一致，未见通道交换、空间错位或
整图退化。高 invalid 边界可见窄 bilinear 过渡，与数值诊断相符。详情见
`VISUAL_REVIEW_CN.md`；通道结论以数值 sentinel/byte regression 为正式证据，
不依赖颜色观感。

## 范围确认

本轮仅有一次训练前 wiring backward；没有 optimizer update，也没有 checkpoint
替换。

本轮未生成全量 REL+ cache，未启动正式训练，未执行 optimizer.step，未产生 checkpoint，未计算 mIoU。
