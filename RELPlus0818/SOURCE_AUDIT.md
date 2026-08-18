# SOURCE_AUDIT

## 审计范围

本文件只记录 2026-08-18 在服务器真实路径上的只读定位结果。历史 `cmx_rel+` 的表示实现仅用于追溯；本项目不以它为数学基线。

## 原始 REL 权威链

`AUTHORITATIVE_REL_ROOT = /home/zhuzhaoziao/RELPlus/CMX-REL/third_party/rel_original`

正式 S3D 生成入口是 `/home/zhuzhaoziao/RELPlus/CMX-REL/tools/generate_stanford2d3d_rel.py`，它在第 16 行导入上述包，并在 `generate_one()` 中调用 `getImage()` 与 `getREL()`。

- `rel.py::getREL`：ReD、EGVIA、ERP-LOA、missing 写入、clip、stack 与 uint8 截断；默认 `alpha=45`、`lam=0.5`；输出 `[EGVIA, LOA, ReD]`。
- `rgbd_util.py::getPointCloud_ERP`、`computeNormalsSquareSupport_ERP`、`processDepthImage_ERP`：ERP 点云、半径 2 法向、估计重力与对齐。
- `hha_util.py::getRMatrix`、`rotatePC`：正式重力旋转与数组旋转。
- Perspective helper：`/home/zhuzhaoziao/RELPlus/REL-SF4PASS-reference/utils/rgbd_util.py::getPointCloudFromZ` 与 `computeNormalsSquareSupport`。其 `processDepthImage()` 的 HHA 路径选择半径 3；REL+ v1 显式传半径 2，以保持 ERP REL 的局部支持半径。
- 缺失的公开 `utils/hha_util.py` 由 `/data/bxh_copy/Pano_MA_Seg/utils/hha_util.py` 提供；它与正式 vendored `hha_util.py` 逐字节相同。

`/home/zhuzhaoziao/RELPlus/REL/rel_original` 与权威包的 `rel.py`、`rgbd_util.py` 相同，但不是当前正式生成器的实际 import 目标，因此仅作对照。

## S2D 生产接口

- Native 数据：`/data/zhuzhaoziao/datasets/Stanford2D3D/with_xyz`，RGB/depth/pose/global_xyz 均为 1080×1080 对齐数据。
- 生产数据：`/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480`；`Depth16` 和 `Pose` 保留 native 1080×1080 深度及原始 K/pose，RGB/label 为 480×480。
- 生产 loader：`/home/zhuzhaoziao/rel_exp/cmx_rel+/dataloader/dataloader.py::ValPre`；深度=`raw/512`，无效=`raw==0 or raw==65535`。
- 空间转换：`/home/zhuzhaoziao/rel_exp/cmx_rel+/relplus/pipeline.py::transform_depth_geometry`；深度/valid 最近邻缩放，K 的前两行按 `sx/sy` 缩放。
- Pose parser：`/home/zhuzhaoziao/rel_exp/cmx_rel+/relplus/geometry.py::load_camera_metadata`；强制 JSON `camera_rt_matrix` 为 3×4 world-to-camera `[R|t]`，并用 `-R.T@t == camera_location` 验证。
- K：JSON half-pixel convention；反投影使用 `(u+0.5-cx)/fx`。
- 世界向下：`[0,0,-1]`；相机重力=`R_world_to_camera@[0,0,-1]`。

## 外部几何证据

`/data/zhuzhaoziao/cmx/outputs/stage1g_r2_realdata_validation_20260805_122721` 的 24 个跨区域真实样本使用 global XYZ 对拍：全部 `geometry_wiring_pass=True`，各轴 P95 小于 `1/512 m`；最终状态为 `PASS_STAGE1G_R2_REAL_DATA_RELPLUS_VALIDATION`。本项目仍会独立选择约 10 张样本重新验证，不能把旧 PASS 冒充本次运行结果。

## 冻结裁决

- REL+ v1 正式定义在 canonical 480×480 空间。
- native Depth16 先最近邻缩放到 480×480；K 同比例更新；不 crop、不 pad、不 flip。
- `normal_radius=2` 表示 canonical 空间中的 2 pixels。
- 历史 `(raw+1)/512` 来自旧 HHA 生成路径，不进入本 REL+ v1。

