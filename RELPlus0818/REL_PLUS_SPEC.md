# REL+ v1 冻结规格

- 数据域：Stanford2D3D S2D 透视图。
- 正式空间：canonical 480×480；native 1080×1080 depth 以 nearest-neighbor 缩放，K 前两行按 480/1080 缩放。
- 深度：camera z-depth，`float32(raw_uint16/512)`；0、65535 无效并写为 0。
- K：JSON half-pixel-center convention；数组索引 `(u,v)` 的中心是 `(u+0.5,v+0.5)`。
- Pose：`X_camera=R_world_to_camera@X_world+t_world_to_camera`；必须通过 camera center 验证。
- 世界重力：down=`[0,0,-1]`；相机重力=`R_world_to_camera@down`。
- 法向：原始 perspective square-support helper；canonical radius=2 pixels；保持 source orientation。
- 重力对齐：原始 `getRMatrix(target_down, gravity_camera)` 后用其转置旋转点与法向。
- ReD：对齐点的 `hypot(x,y)`，保留原始整图 min/max、clip 与截断。
- EGVIA：保留原始 `N_z=-normal_z`、P1/P99、`alpha=45`、`lambda=0.5` 和 `~is_horizontal` 融合。
- LOA：透视水平径向切向量；角度为 degree uint8 截断，不映射到 0–255；轴奇点为 90°。
- 输出：H×W×3 uint8，byte channel 0/1/2=`EGVIA/LOA/ReD`；无效像素 `[255,255,255]`。
- PNG：OpenCV `IMREAD_UNCHANGED`/`imwrite`，不做 BGR/RGB 转换。
- 增强：horizontal/vertical flip 与任意旋转均不在 v1 支持范围；明确拒绝 horizontal flip。
- 范围：只生成约 10 张审查样本；不接 CMX、不训练、不生成全量 cache、不评价 mIoU。

