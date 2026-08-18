# REL+ v1（Stanford2D3D S2D）

这是原始可执行 REL 的 Stanford2D3D S2D 透视几何适配。正式输出为
`H×W×3 uint8`，byte channel 顺序固定为 `[EGVIA, LOA, ReD]`，无效像素固定为
`[255,255,255]`。

本项目只实现和验证 REL+ v1；没有接入或修改 CMX，没有训练，没有生成全量 cache，
也没有计算 mIoU 或产生 checkpoint。

## 冻结接口

- 深度：Stanford `Depth16`，`float32(raw_uint16 / 512)`；0 和 65535 无效。
- 正式空间：480×480；native depth 最近邻缩放，K 的前两行同步缩放。
- K：JSON half-pixel convention；反投影使用 `u+0.5`、`v+0.5`。
- Pose：`X_camera = R_world_to_camera @ X_world + t_world_to_camera`。
- 重力：`R_world_to_camera @ [0,0,-1]`，随后按原始 `getRMatrix` / `rotatePC(..., R.T)` 对齐。
- 法向：公开 perspective square-support helper 的数学路径，半径固定为 canonical 2 pixels。
- ReD/EGVIA：保持权威 `getREL()` 的统计、融合、clip 和 uint8 截断顺序。
- LOA：透视水平径向的切向量 `[r_y,-r_x,0]`；保存角度制 uint8，不映射到 0–255。
- 存储：OpenCV 原样写入/读取，不做 RGB/BGR 通道转换。
- 增强：普通 horizontal flip 被明确拒绝。

完整裁决见 `REL_PLUS_SPEC.md`；源码链见 `SOURCE_AUDIT.md`。

## 环境与测试

服务器验证环境为 Python 3.8.16、NumPy 1.21.6、OpenCV 4.5.5、pytest 7.4.4。
建议使用隔离环境：

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

两项 live-source 回归测试依赖本服务器审计过的只读源码路径：

- `/home/zhuzhaoziao/RELPlus/CMX-REL/third_party/rel_original`
- `/home/zhuzhaoziao/RELPlus/REL-SF4PASS-reference`
- `/data/bxh_copy/Pano_MA_Seg/utils/hha_util.py`

核心生成代码运行时不导入 CMX 或上述原始仓库。

## 单图生成

在项目根目录执行：

```bash
python tools/generate_rel_plus.py \
  --depth /path/to/native_depth16.png \
  --camera-json /path/to/camera_pose.json \
  --output /path/to/rel_plus.png \
  --debug-dir /path/to/debug
```

CLI 统一走 `rel_plus.stanford_s2d.load_canonical_frame()`，不会另写一套 K/pose 解释。
`debug_arrays.npz` 保存正式生成器返回的中间量；调试图中的 RGB 面板为黑色占位，因为该
CLI 不接收 RGB。

## 真实样本复现

以下命令只选择并生成 10 张审查样本，不会生成全量 cache：

```bash
python tools/generate_review_samples.py \
  --output-root /data/zhuzhaoziao/RELPlus/outputs/REL_plus_v1_implementation/real_validation

python tools/validate_real_geometry.py \
  --manifest /data/zhuzhaoziao/RELPlus/outputs/REL_plus_v1_implementation/real_validation/real_samples_manifest.csv \
  --output-root /data/zhuzhaoziao/RELPlus/outputs/REL_plus_v1_implementation/real_validation
```

本次已验证产物位于：
`/data/zhuzhaoziao/RELPlus/outputs/REL_plus_v1_implementation`。
