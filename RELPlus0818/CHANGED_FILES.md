# CHANGED_FILES

本目录是一次完整的新交付，不是在旧 `RELPlus` 上增量修改。目标目录原内容在最终替换时
整体移出，因此下面列出的均为本次 REL+ v1 文件。

## 核心代码

- `rel_plus/__init__.py`：公开 API。
- `rel_plus/camera.py`：K、显式 W2C pose、half-pixel 反投影、canonical K 更新和相机重力。
- `rel_plus/depth.py`：`raw/512` 解码、0/65535 invalid、最近邻缩放。
- `rel_plus/source_helpers.py`：最小 source-aligned perspective normal、`getRMatrix` 和 `rotatePC` 路径。
- `rel_plus/encoding.py`：source-exact ReD/EGVIA/角度量化与 perspective LOA。
- `rel_plus/generator.py`：正式端到端生成器和 debug 输出。
- `rel_plus/stanford_s2d.py`：生产 Depth16/pose 到 canonical 480×480 的唯一 adapter。
- `rel_plus/storage.py`：无颜色转换的 PNG 读写。
- `rel_plus/policy.py`：拒绝 horizontal flip。

## 工具

- `tools/generate_rel_plus.py`：单图 CLI。
- `tools/generate_review_samples.py`：从完整候选中按 area 与 pose 统计固定选择 10 张并生成审查图。
- `tools/validate_real_geometry.py`：用 native global XYZ 对拍 K、W2C pose、camera center 和 gravity。
- `tools/visualize_rel_plus.py`：调试视图、逐样本 montage 与总拼图。

## 测试

- `tests/conftest.py`：测试路径准备。
- `tests/test_depth_camera.py`：深度、half-pixel、K adapter、canonical resize。
- `tests/test_generator.py`：公共输出/debug contract 与合成平面 EGVIA。
- `tests/test_loa_erp_reduction.py`：ERP 退化、非零镜像关系与轴奇点 90°。
- `tests/test_pose_gravity.py`：W2C parser 与 source gravity rotation。
- `tests/test_source_encoder_regression.py`：与 live 权威 `getREL()` 最终三通道逐像素一致。
- `tests/test_source_perspective_normal_regression.py`：与 live perspective helper 的 radius=2 法向逐值一致。
- `tests/test_storage_policy.py`：通道 sentinel round-trip 与 no-flip。
- `pytest.ini`、`tests.log`：测试配置与最终日志。

## 规格与报告

- `SOURCE_AUDIT.md`：真实权威源码和生产数据调用链。
- `REL_TO_REL_PLUS_MAPPING.md`：SOURCE / ADAPTATION / VALIDATION ONLY / POLICY 边界。
- `REL_PLUS_SPEC.md`：冻结数学与数据规格。
- `IMPLEMENTATION_REPORT_CN.md`：实现、测试、真实数据和未验证项报告。
- `THIRD_PARTY_NOTICE.md`、`LICENSE`：适配来源、运行时边界和上游许可证文本。
- `README.md`、`requirements.txt`：使用说明和最小依赖。

## 未修改项

- 原始 REL：未修改。
- `CMX-REL` 与其他 CMX 目录：未修改。
- 数据集：未修改。
- 外层 `/home/zhuzhaoziao/RELPlus` 的其他项目：未修改。
