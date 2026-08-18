# REL+ v1 源码实现与验证报告

## 状态

`REL+ v1 已完成`。

代码根目录：`/home/zhuzhaoziao/RELPlus/RELPlus`

验证产物目录：
`/data/zhuzhaoziao/RELPlus/outputs/REL_plus_v1_implementation`

## 已由原始源码确认

- 正式 S3D REL 生成器实际导入
  `/home/zhuzhaoziao/RELPlus/CMX-REL/third_party/rel_original`。
- `rel.py::getREL` 冻结了 ReD min/max、EGVIA P1/P99 与 `~is_horizontal` 融合、
  ERP LOA degree 截断、invalid 写入和 `[EGVIA,LOA,ReD]` stack 顺序。
- ERP REL 使用 radius=2 法向；重力对齐调用 `getRMatrix(g0.T,gDir)`，再将
  `R.T` 传给 `rotatePC`。
- perspective 法向来自公开
  `REL-SF4PASS-reference/utils/rgbd_util.py::computeNormalsSquareSupport`；本实现显式传
  canonical radius=2。
- live source encoder 三通道最终 uint8 逐像素回归通过；live perspective normal 和
  offset 逐值回归通过。

## 已由真实数据确认

- 生产接口为 `/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480`：Depth16/Pose 为
  native 1080×1080，RGB 为 canonical 480×480。
- 深度是 z-depth，单位为 `raw/512`；0 与 65535 无效。
- JSON K 直接采用 half-pixel convention；native 到 canonical 时 K 的前两行分别按
  `480/1080` 缩放。
- `camera_rt_matrix` 是 W2C `[R|t]`；10/10 样本均通过
  `-R.T@t == camera_location` 和 native global XYZ 对拍。
- global XYZ 对拍每张固定 1024 个 joint-valid pixels，三个分量 P95 阈值为
  `1/512 m = 0.001953125 m`；本次最坏分量 P95 为
  `0.0009692177195181895 m`，10/10 PASS。
- 最大 camera center 绝对误差为 `1.0009887034811982e-06 m`；最大重力对齐误差为
  `4.440892098500626e-16`。
- 从 60,169 个完整候选中按 6 个 area 覆盖和 pose 极值固定选择 10 个不同 camera；
  pitch 范围 `[-89.8752°, 89.8391°]`，roll 范围
  `[-179.9910°, 179.9663°]`，native invalid 比例范围
  `[0, 0.0101312]`。
- 10/10 输出均为 480×480×3 uint8，invalid triplet 正确，三个有效通道均非常量。

真实 manifest：
`/data/zhuzhaoziao/RELPlus/outputs/REL_plus_v1_implementation/real_validation/real_samples_manifest.csv`

总 montage：
`/data/zhuzhaoziao/RELPlus/outputs/REL_plus_v1_implementation/real_validation/review_samples_montage.png`

## 由数学推导得到

- JSON half-pixel K 的数组索引反投影是
  `X=(u+0.5-cx)Z/fx`、`Y=(v+0.5-cy)Z/fy`。
- 透视水平径向 `r=[x,y,0]/hypot(x,y)` 的右手切向量为
  `t=[r_y,-r_x,0]`；在 ERP 参数化下严格退化为
  `[cos(phi),-sin(phi),0]`。
- 对水平镜像后的真实几何，连续量满足 `hcos'=-hcos`、
  `LOA'=180°-LOA`；因此不能把普通 `np.fliplr` 当作合法增强。
- 当 `hypot(x,y)=0` 时，冻结切向量为零，LOA 为 90°。

## 测试结果

- pytest：19 passed，退出码 0。
- source encoder regression：PASS，最终三通道逐像素一致。
- perspective normal source regression：PASS，radius=2 normal/offset 逐值一致。
- K/half-pixel/resize：PASS。
- W2C pose/camera center/global XYZ：10/10 PASS。
- gravity/source rotation：PASS。
- LOA ERP reduction、mirror、axis singularity：PASS。
- horizontal flip policy：PASS（明确拒绝）。
- channel sentinel `[11,22,33]`：PASS。
- synthetic floor/wall/ceiling EGVIA：PASS。
- 单图 CLI 与 debug bundle：PASS，480×480×3 uint8。

第一次真实几何脚本尝试因隔离环境没有 `Imath` 退出 1；该次没有形成几何结论。
随后实测服务器 OpenCV 4.5.5 可读取同一 EXR，并显式把 OpenCV BGR 恢复为 EXR
`R/G/B = X/Y/Z`，同一批 10 张重跑后退出 0。失败与成功日志均保留。

## 可视化目检

Codex 已实际打开 10 张总拼图，并以全分辨率复查明显倾斜的 sample 06 和 sample 08。

- RGB 与 depth/normal 的物体边缘一致，样本配对正确。
- 未见转置、镜像或通道交换。
- invalid 在数值检查中均为 `[255,255,255]`，可视化位置与缺失深度一致。
- 所有样本三通道均非全黑/全白。
- 墙面、地面/桌面和天花区域的 EGVIA 响应有合理分层。
- LOA 的明显变化与几何边界或 invalid 对齐，未见无来源的全局断裂。
- 不同 pitch/roll 下 gravity-aligned normal 保持稳定。

目检状态：`PASS_CODEX_INSPECTION`。这不是语义 ground truth 指标，也不替代用户后续独立复核。

## 仍未验证或明确不在本轮范围

- 未接入 CMX，因本任务明确禁止；因此 CMX loader 端到端消费尚未验证。
- 未训练、未反向传播、未更新 optimizer、未替换 checkpoint。
- 未生成全量 REL+ cache，未计算 mIoU。
- 未实现 horizontal/vertical flip 或任意旋转增强；v1 对 horizontal flip 明确报错，其他
  几何增强不在支持范围。

这些是范围边界，不影响本轮独立 REL+ v1 源码与 10 张真实样本验证结论。

## 范围确认

本轮未接入 CMX、未训练、未生成全量 REL+、未产生 checkpoint。
