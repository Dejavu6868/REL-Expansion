# REL+ v2 源码修订、验证与训练前接口实现报告

## 状态

`REL+ v2 已完成，具备进入 CMX 接入审查的条件`。

该状态只覆盖源码、训练前 adapter 和几何/数值验证，不授权或代表训练结果。

## 已由 source regression 确认

- `getRMatrix`、`rotatePC`、perspective radius-2 normal/offset 与 REL 三通道均有离线 golden 对拍。
- 服务器 live-source 测试实际调用审计源码，1/1 通过。
- ReD、EGVIA、LOA、通道顺序和 uint8 截断保持原始可执行行为。

## 已由数学证明与合同测试确认

- Depth16 `raw/512`、invalid `{0,65535}`、half-pixel K、显式 W2C 和 camera centre。
- K 绑定 `intrinsics_shape`；1080/480 匹配通过，交叉错配失败。
- aligned metres 到 encoding centimetres 的唯一单位转换。
- 正式 mask 只来自 depth；normal finite/nonzero/support 仅诊断。
- frozen alpha 45、lambda 0.5、radius 2 和 canonical 480。
- anti-parallel gravity 使用 typed exception；preflight 与 batch 均记录样本级失败。
- storage 与 CMX adapter 的 `[11,22,33]` sentinel 全链路不换通道。
- 一个随机 `SpatialTransform` 同步用于 RGB/REL+/label；no-flip policy 在 adapter 内实际调用。

## 已由合成几何确认

- 八组独立射线-plane 解析场景覆盖正视/倾斜地板、天花板、墙面以及 pitch+roll；均走完整 depth-to-generator 路径。
- 自洽错误 pose 使用 `R_wrong=R_true.T` 和相同 camera centre：正交性、det 和 centre 均通过，但独立物理验证正确失败。
- analytic native-to-canonical nearest 测试通过，canonical reprojection P95 是正式判据。

## 已由真实 global XYZ 确认

- 12 张 review 样本覆盖 Area 1–6、不同 room/camera、gravity 和 optical-axis；生成 12/12 成功。
- Area 2–6 共 10 张有 global XYZ：native 与 canonical geometry 10/10 PASS。
- 最大 native component P95：`0.0009794085 m`。
- 最大 canonical component P95：`0.0118121547 m`，低于按 observed depth 和 frozen subpixel mapping 预先定义的逐样本容差。
- 最大 canonical reprojection P95：`0.8643245363 px`；最大值 `0.8650060820 px`，均低于冻结 `1.0/1.5 px` 判据。
- pose physical status：12 PASS、0 FAIL。

## 只通过弱物理检查

Area 1 的 2 张样本没有可用 global XYZ，明确记录 `geometry_oracle=unavailable` 和 `validation_level=weak`；完成 camera centre、pose algebra、gravity、normal/label 和 montage 检查，但未伪造强 oracle 结论。

## v1 到 v2 数值影响

- dense-valid：0 changed pixel。
- depth-invalid：0 changed pixel。
- NaN-normal：1 pixel、2 channels 按 source-exact mask 预期改变。
- zero-normal：0 unexpected change。
- degenerate unit branch：1 pixel、2 channels 按 centimetre 修订预期改变。
- unexpected difference：0。

## 测试执行

- `python -m pytest -q`：50 passed、1 live-source skipped、退出码 0。
- 带审计路径的 `python -m pytest -q -m live_source`：1 passed、退出码 0。
- v1-v2 diff、单位回归、normal policy、K resolution、canonical geometry、pose counterexample、gravity singularity、八组合成平面、augmentation contract、storage/model channel sentinel 均通过。
- 12-row dataset preflight：12 PASS，0 core failure。
- 单图正式 CLI：480x480x3 uint8、`[EGVIA,LOA,ReD]`，退出码 0。

## 目检

12 张 contact sheet 已实际打开。RGB、深度、depth mask、camera/gravity normal、quality mask、ReD、EGVIA、LOA 和 display-only 图空间对齐；通道均有结构且未见整图退化、错位或交换。`rel_plus_display_only.png` 不作为存储或模型合同证据。

## 尚未验证

- 尚未改动生产 CMX，compatibility adapter 仍需下一轮接入审查。
- Area 1 强 global-XYZ oracle 仍不可用。
- 未生成全量 cache，未验证全数据集奇异样本是否为零；preflight 工具已具备失败闭合能力。

## 范围确认

本轮未修改生产 CMX，未生成全量 REL+，未训练，未执行 backpropagation 或 optimizer update，未产生 checkpoint，未计算 mIoU。
