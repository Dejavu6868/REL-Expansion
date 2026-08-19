# REL+ v2.1 pilot 人工目检记录

本记录对应协议 `RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`。实际打开并检查了
7 个 pilot 样本的 `montage.png` 与 `pilot_montage.png`，共 14 张图，而不是
只确认文件存在。

## 已检查代表样本

- Area 1 最低 tilt：
  `camera_316d272b63464affb889d9eade4e2b5a_conferenceRoom_2_frame_20`。
- Area 1 高 invalid：
  `camera_5e167d912c03482dbd50607f9ca51a4f_office_29_frame_15`。
- Area 2 ceiling-visible：
  `camera_d69c3e2b9f8845bf872b93e60a0c2b96_auditorium_1_frame_45`。
- Area 3 高 invalid：
  `camera_6cf9b1a7e96d44aeb9a8a147328b6e76_hallway_4_frame_16`。
- Area 5b 全数据最高 invalid、pilot 最高边界污染：
  `camera_f3303e70e1124075bb4ffabb387c55b2_lobby_1_frame_0`。
- Area 5b 最高 gravity tilt：
  `camera_f7cb917ede0e4e70866bf4da9f53a632_storage_1_frame_35`。
- Area 6 pilot 最低 normal quality：
  `camera_9747cbdf06434a3293a88b62b75aff05_office_15_frame_44`。

## 实际观察

- RGB、z-depth、depth-valid、camera normal、normal quality、
  gravity-aligned normal、EGVIA、LOA、ReD 与组合 REL+ 的结构边界一致；
  未见整图退化、轴向翻转或明显空间错位。
- 原始 stored label 与变换后的 model label 在类别编号减一后保持相同几何
  边界；它们与 transformed REL+ 使用同一 crop/pad。高 invalid 样本中的
  空洞也与 nearest valid mask 对齐。
- Area 5b 最高 tilt 样本清楚覆盖了 0.75 scale 后的 centre padding：
  transformed REL+ 的 pad 是 CMX 在归一化空间中的零值，label pad 为 255，
  valid-mask pad 为 false，三者边界一致。该现象是冻结链行为，不是把
  invalid REL+ 静默置零。
- 高 invalid 样本可见 source-compatible 255 区域以及缩放后边界处的窄过渡；
  nearest mask 保持硬边界。该现象与独立 invalid contamination 数值诊断一致，
  正式 `modal_x` 未被诊断 mask 改写。
- 最高 tilt 样本的 gravity-aligned normal 在墙/地面等大平面上仍连续；
  Area 2 ceiling-visible 样本的 ceiling normal 与 REL 三通道具有一致结构。
- 最低 normal-quality 样本的质量剔除仅为稀疏局部，未出现大范围 normal
  崩塌。Area 1 两个样本只作为 weak physical review，目检不把它们升级为
  global-XYZ 强证据。

通道顺序结论不依赖颜色观感；正式依据仍是 `[11,22,33]` storage/loader
sentinel、RGB byte sentinel 和 v2→v2.1 逐字节回归。
