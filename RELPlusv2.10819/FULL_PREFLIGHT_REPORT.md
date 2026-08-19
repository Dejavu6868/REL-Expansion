# REL+ v2.1 全数据集 preflight 报告

协议：`RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`。

## 结论

`PASS`。正式 manifest 共 70,496 个唯一样本，逐样本扫描 70,496/70,496
通过，失败 0；最终状态为 `READY_FOR_PILOT_CACHE_WITH_REVIEW`。输出 CSV
共 70,497 行（含表头），扫描支持 partial CSV 断点续跑。

本扫描是只读的 K/pose algebra、gravity、depth 与 normal diagnostic
preflight；没有生成全量 REL+ cache。

## 覆盖

| 划分/Area | 样本数 | FAIL |
|---|---:|---:|
| train | 52,903 | 0 |
| test | 17,593 | 0 |
| Area 1 | 10,327 | 0 |
| Area 2 | 15,714 | 0 |
| Area 3 | 3,704 | 0 |
| Area 4 | 13,268 | 0 |
| Area 5a | 6,261 | 0 |
| Area 5b | 11,332 | 0 |
| Area 6 | 9,890 | 0 |

每行检查 RGB/label/Depth16/Pose 配对、Depth16 `uint16` 与 native
`1080x1080`、dataset profile、K finite/reference shape/principal point/zero
skew、3x4 W2C、R 正交、`det(R)=+1`、metadata camera centre、gravity 与
anti-parallel singularity，以及 canonical 480 depth 和 normal diagnostics。

结构性失败计数均为 0：文件读取/配对、K、pose、gravity singularity、
整图 depth-invalid、整图 normal-nonfinite 均未出现。

## 全量数值分布

| 指标 | min | median | mean | p95 | max |
|---|---:|---:|---:|---:|---:|
| depth invalid ratio | 0 | 0.000703125 | 0.004997451 | 0.024743924 | 0.100247396 |
| normal nonfinite ratio | 0 | 0 | 0 | 0 | 0 |
| zero-normal ratio | 0 | 0 | 1.3572e-8 | 0 | 1.3822e-5 |
| low-support ratio | 0 | 0 | 1.4938e-7 | 0 | 4.6975e-5 |
| normal quality ratio | 0.999953025 | 1 | 0.999999851 | 1 | 1 |
| gravity angle (degree) | 66.310204 | 90.206656 | 90.091591 | 108.512089 | 113.566988 |

normal quality 只用于诊断/semantic validator，不改变 source-exact production
mask。没有根据这些比率引入方法定义之外的 hard threshold。

## 代表性 outlier

- 最高 depth-invalid：
  `area_5b/camera_f3303e70e1124075bb4ffabb387c55b2_lobby_1_frame_0`，
  0.100247396。
- 最高 zero-normal：
  `area_6/camera_985d599271064881945d5a4057a601b8_office_14_frame_24`，
  1.3822e-5。
- 最高 low-support、最低 normal quality：
  `area_6/camera_9747cbdf06434a3293a88b62b75aff05_office_15_frame_44`，
  4.6975e-5 / 0.999953025。
- normal nonfinite 的全量最大值为 0。

这些样本已进入 pilot 的 stratified review 范围；outlier 没有触发核心几何
失败。

## Pose 与 Area 1

另用冻结的 12 张真实 review manifest 运行 v2.1 physical validator：12/12
通过。Area 2–6 的 10 张 global-XYZ 样本为 `PASS_STRONG`；Area 1 的 2 张因
没有 global XYZ，仅为 quality-masked semantic `PASS_WEAK`，不是强 oracle。
`REVIEW_REQUIRED=0`、`FAIL=0`、`NOT_APPLICABLE=0`。全量 preflight 对所有
70,496 张检查了 pose algebra 与 gravity，但不把不存在的 global XYZ
伪造成全量强物理证据。

## 证据路径

- manifest：
  `/data/zhuzhaoziao/RELPlus/outputs/REL_plus_v2_1_implementation/full_manifest.csv`
- 逐样本 CSV：
  `/data/zhuzhaoziao/RELPlus/outputs/REL_plus_v2_1_implementation/preflight/full_preflight.csv`
- 汇总：
  `/data/zhuzhaoziao/RELPlus/outputs/REL_plus_v2_1_implementation/preflight/full_preflight.summary.json`
- 运行日志/退出码：同目录 `full_preflight.log` 与
  `full_preflight.exitcode`（0）。
- 12 张 pose 验证：
  `/data/zhuzhaoziao/RELPlus/outputs/REL_plus_v2_1_implementation/real_pose_validation/`
