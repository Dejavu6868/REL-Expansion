# REL 到 REL+ v1 的映射

| 环节 | 分类 | v1 处理 | 权威位置 |
|---|---|---|---|
| ReD | SOURCE | 保持 min/max、clip、uint8 截断与 missing 写入顺序 | `CMX-REL/third_party/rel_original/rel.py::getREL` |
| EGVIA | SOURCE | 保持 P1/P99、`~is_horizontal` 融合、clip 与量化 | 同上 |
| LOA 角度编码 | SOURCE | 保持 `arccos`、degree 和 `astype(uint8)` | 同上 |
| K 反投影 | ADAPTATION | JSON half-pixel K，使用 `u+0.5` | 生产 loader/global XYZ 对拍 |
| Pose gravity | ADAPTATION | `R_world_to_camera@[0,0,-1]` 替代 `getGDir` | 生产 pose parser |
| 重力对齐 | SOURCE | 调用 source-exact `getRMatrix` 与 `rotatePC(..., R.T)` | `hha_util.py` |
| Perspective normal | SOURCE + PARAMETER | 复用 `computeNormalsSquareSupport`，显式 R=2 | `REL-SF4PASS-reference/utils/rgbd_util.py` |
| Perspective tangent | ADAPTATION | `r=[x,y,0]/hypot(x,y)`，`t=[r_y,-r_x,0]` | ERP tangent 的解析推广 |
| 外参平移 | VALIDATION ONLY | 只验证 camera center/世界点，不进三通道 | pose/global XYZ 验证 |
| no-flip | REL+ v1 POLICY | 拒绝普通 horizontal flip | LOA 连续量镜像关系 |

