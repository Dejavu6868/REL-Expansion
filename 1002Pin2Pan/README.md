# 1002Pin2Pan：针孔→全景迁移的准备工具

本目录用于 Pin2Pan 实验的第2步和第3步。先检查透视与全景两条代码路径是否描述同一几何（REL+ 与 HHA 各自检查），再把已冻结的 S2D（针孔）权重直接用于 Stanford2D3D 全景测试（source-only，不训练）。两者都直接调用 `0927调参结果/selected_recipe/code/training/source` 中的冻结代码，没有复制或修改其中任何文件。

## 1. 跨投影一致性检查 `tools/cross_projection.py`

对每张全景深度图，按已知旋转渲染90°针孔裁剪（默认 yaw 0/90/180/270 × pitch −20/0/20，480×480），用冻结的 `generate_rel_plus_v2_1` 计算裁剪图的 REL+；再用原始 `rel.py` 在整张全景上计算 REL，并按相同的三维射线取样，然后比较。

报告分两组：

- **原始几何量**：高度（cm）、水平半径（cm）、EGVIA 混合前的法向夹角（°）、LOA（°）。两条路径描述的是同一场景，这些量应一致；中位数超过阈值（默认 2 cm / 2°）即 FAIL，说明存在几何或约定错误。
- **编码字节**：EGVIA、LOA、ReD。它们还包含逐图归一化（ReD 的 min/max、高度的 1/99 百分位）。90° 裁剪与 360° 全景的归一化范围本来就不同，所以这里的差异不判 PASS/FAIL，只用于量化归一化带来的域差。

```bash
python3 1002Pin2Pan/tools/cross_projection.py \
  /data/zhuzhaoziao/datasets/Stanford2D3D/area_1/pano/depth/<id>_depth.png [...] \
  --output <输出目录>/cross_projection.json
```

**合成场景结果**（`tests/test_cross_projection_synthetic.py`：长方体房间加一张桌子，全景 512×1024，裁剪 160×160）：

- 9 个裁剪的原始几何量全部一致。高度差中位数 ≤0.4 cm，水平半径差 ≤0.16 cm，EGVIA 夹角差 ≤0.63°，LOA 差 ≤0.82°。
- 全景路径估计的重力为竖直方向。
- 编码字节差异明显。ReD 中位差 7–44（共 255 级），EGVIA 中位差最高为 5，均来自逐图归一化。
- 负对照通过：给裁剪相机提供错误重力（忽略 20° pitch）时，检查判为 FAIL。

真实数据仍需在服务器上用 Stanford2D3D 全景运行。加 `--hha` 可同时检查全景 HHA（见第3节）。

## 2. 全景 source-only 评估 `tools/eval_pano_transfer.py`

加载冻结的 CMX 双流 MiT-B2 和 S2D epoch200 权重，在 area_5a/5b 全景上评估：

- **RGB**：用冻结 loader 的原调用读取，即 `cv2.imread(path, cv2.COLOR_BGR2RGB)`，通道顺序与训练一致。
- **REL+ 的 X 输入**：原始 ERP `getREL`，在最近邻缩放到评估尺寸的深度上计算。这对应透视缓存在 480×480 模型尺寸上生成的做法。
- **HHA 的 X 输入**：采用缓存报告确认的生成顺序（原尺寸生成再缩放，或先缩放深度再生成）和通道顺序。
- **标签**：通过 `semantic_labels.json` 映射到 13 类，`<UNK>` 记为 ignore255，缩放用最近邻。
- **推理**：整图一次前向，左右各做 128 列环形填充，保证 ±180° 接缝处两侧都有上下文。
- **评估尺寸**：默认 1024×2048（约 5.7 px/°），接近 S2D 480 图的像素角密度；如更改须在结果中注明。

```bash
python3 1002Pin2Pan/tools/eval_pano_transfer.py \
  --stanford-root /data/zhuzhaoziao/datasets/Stanford2D3D \
  --semantic-labels /data/zhuzhaoziao/cmx/raw/reference_repos/2D-3D-Semantics/assets/semantic_labels.json \
  --config 0927调参结果/configs/relplus.json \
  --checkpoint <REL+ epoch-200.pth> \
  --output <输出目录>/relplus_pano_eval
```

HHA 臂：把 `--config` 换成 `0927调参结果/configs/hha.json`，并加上 `--hha-cache-report <hha_cache_check.json>`（见第3节）。

输出包括 `metrics.json`、`per_class_iou.csv`、`confusion_matrix.csv` 和 `samples.txt`，另按 ERP 仰角分带评分：90–60°、60–25°、25–0°、0–−25°、−25–−60°、−60–−90°（每带含上边界），写入 `elevation_band_metrics.csv`、`elevation_band_confusion.npy` 和 `metrics.json` 的 `elevation_bands`。加 `--limit 3` 可先做冒烟测试，报告标记为 `SMOKE`。`--limit` 必须为正；缺少任一指定区域、重复指定区域或没有有效标签像素时拒绝完成评估。

加 `--eight-class-predictions` 时，argmax 只在 SPan8 的 8 类（ceiling、chair、door、floor、sofa、table、wall、window）中取，网络和权重不变，`metrics.json` 的 `prediction_classes` 记录所用类别。混淆矩阵仍为 13×13；8 类分数需忽略其余 5 类的真值行（与 Trans4PASS+ `stanford2d3d_pan8.py` 一致），此时 `metrics.json` 中的 13 类指标不再有意义。

### 针孔评估 `tools/eval_pinhole.py`

冻结启动器（`evaluation/run_evaluation.py`）只接受 2026-09-20 的两个第 200 轮检查点。本工具对任意配置和检查点（其他种子、REL+ 3.x、`train_pin2pan.py` 的适配检查点）运行同一个冻结全测试集评估器 `tools/eval_rel_plus_v2_3_full.py`（HHA 臂套用冻结的 `evaluation/input_adapter.py`），测试集仍是区域 5 的 17,593 张针孔图。不加 `--eight-class-predictions` 时计算与冻结评估相同，同一检查点的混淆矩阵应与冻结结果一致；加上后 argmax 只在 SPan8 的 8 类中取，与全景评估的同名参数相同。

```bash
python3 1002Pin2Pan/tools/eval_pinhole.py \
  --config <该检查点训练时的 resolved 配置.json> \
  --checkpoint <epoch-200.pth 或适配后的 checkpoint.pth> \
  --eight-class-predictions \
  --output <输出目录>/hha_pinhole_8pred
```

REL+ 臂读取配置中缓存路径指向的针孔缓存，变体名（v2_1、v3_0、v3_1）写入 `eval_pinhole.json`。适配检查点保存的是源检查点的轮数，`--expected-epoch` 保持默认 200。默认单卡；8 卡时用 `python -m torch.distributed.launch --nproc_per_node=8 --use_env 1002Pin2Pan/tools/eval_pinhole.py ...`。`evaluation/` 下是冻结评估器的原样输出（13 类 `metrics.json`、`confusion_matrix.csv` 等）；`eval_pinhole.json` 记录检查点、配置、预测类别，以及精确 8 类分数 `classes_8`（忽略其余 5 类真值行，保留全部预测列）。加 `--eight-class-predictions` 时 `evaluation/metrics.json` 的 13 类 mIoU 把其余 5 类记为 0，请看 `classes_8`。输出目录已有评估结果时拒绝运行。

## 3. 全景 HHA `tools/hha.py` 与缓存核对 `tools/check_hha_cache.py`

CMX README 指定用 [Depth2HHA-python](https://github.com/charlesCXK/Depth2HHA-python) 生成 HHA，已收录于 `vendor/depth2hha`（MIT；打包和数值类型兼容改动见其 `SOURCE_NOTICE.md`）。

全景 HHA 沿用 Depth2HHA 中与相机模型无关的全部定义：

- 角度通道：法向用 3 邻域窗口，重力由 10 邻域法向经 `getYDir` 估计（阈值 45°/15°，迭代 5+5 次），通道值为 `angle+128−90`。
- 高度通道：`h−yMin`，并保留 `yMin>−90` 时取 `−130` 的规则。
- 视差通道：`31000/max(depth_cm,100)`。

点云与法向来自原始 REL 的 ERP 函数。唯一的定义改动是视差：ERP 没有 z-depth，因此改用射线距离（range）。

**合成场景结果**：

- 角度差中位数 ≤0.37°，高度差 ≤0.77 cm。
- 角度、高度两个字节通道的中位差为 0–2。
- 视差字节中位差为 23–37，因为同一点的 z/range 中位数约为 0.78。这一差异来自 ERP 本身，无法消除，与 REL+ 的 ReD 归一化差异属于同一类域差。

**使用前必须先核对缓存。** 训练用 HHA 缓存（`Stanford2D3D_480/HHA`）的生成参数没有记录。`check_hha_cache.py` 从原始 Depth16 和位姿 K 重新计算 HHA，并与缓存逐字节比较；它会尝试两种分辨率顺序（先算后缩放时再分别尝试 OpenCV 最近邻、像素中心最近邻、双线性、area、双三次五种缩放核）和两种通道顺序：

```bash
python3 1002Pin2Pan/tools/check_hha_cache.py \
  --manifest /data/zhuzhaoziao/RELPlus/outputs/REL_plus_v2_1_implementation/full_manifest.csv \
  --hha-root /data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480/HHA \
  --output <输出目录>/hha_cache_check.json
```

`MATCH` 要求同一种生成顺序和通道顺序下，**每个抽样文件、每个通道的所有字节均一致**。中位数、P95 和最大差异作为诊断保留，不允许用中位数掩盖局部错误或失败样本。这只证明报告所列样本的缓存匹配，不证明整个缓存相同，也不消除 z-depth/range 的差异。

评估脚本的 HHA 臂必须提供该报告（`--hha-cache-report`），核对报告中的缓存目录与配置一致，重新检查逐样本证据，并采用报告确认的生成顺序和通道顺序。旧版仅按中位数判定的报告必须重新生成；`NO_MATCH`、空证据或报告与证据不一致时拒绝运行。

若最优配方只差取整级误差（每个样本各通道 P95 ≤1；最多一个通道最大差 >2，且该通道 ≥90% 像素完全一致），报告为 `NEAR_MATCH`，并优先选出满足该条件的配方。评估脚本默认拒绝 `NEAR_MATCH`，需显式加 `--accept-hha-near-match`，并在 `metrics.json` 的 `hha_recipe.cache_status` 中记录。

## 4. 裁剪拼接评估 `tools/eval_pano_crops.py`

第2节的整图评估给网络的是 ERP 输入，逐图归一化覆盖整个 360°。对冻结清单全部 52,903 张训练图的 K 矩阵核对发现，训练视场为 **45.00–75.00°，中位数 62.47653165897473°**，并非 90°；17,593 张测试图也在 45–75°。统计及清单哈希见 [`evidence/training_fov_20261003.json`](evidence/training_fov_20261003.json)。

本工具默认采用训练视场中位数，生成 **22 个针孔裁剪**：pitch 0 每隔 45° 一个（8 个），pitch ±45 各每隔 60° 一个（12 个），再各加一个 pitch ±80°、yaw 0 的极区裁剪。极区避开恰好 −90° 时冻结 REL+ 重力对齐的反平行奇点，同时覆盖两极。每个裁剪按训练缓存的做法生成输入并单独推理，再把 softmax 概率投回 ERP 标签网格，用与第2节相同的 1024×2048 标签评分。运行时逐像素检查覆盖，存在空洞即拒绝评测。

`--crop-fov-deg` 可更改视场，但必须满足完整覆盖。默认值只对齐训练视场的中位数，不代表复现了训练集的整个视场分布；原先 16 个 90° 裁剪的布局不能只缩小视场后继续使用。

**水平切片与训练视角核对。** 2026-10-03 的完整切片评测中，HHA 从 55.67 跌到 36.21：倾斜切片上 Depth2HHA 的重力估计失效（抽查一张全景，±80° 切片偏差约 90°，22 个切片中 8 个超过 80°），而仰角 |e|>约30° 的像素（占 ERP 网格的 66%）只由倾斜切片覆盖。为此：

- `--layout level` 只用 8 个 pitch 0 切片（水平切片上重力偏差 0.16–0.65°），未覆盖的像素不计分。默认视场下 ±25° 仰角带全部覆盖，可与第2节整图结果的同两带（25–0°、0–−25°）直接比较。
- 两种评测都输出仰角分带结果（见第2节），可看出各臂在哪个仰角带失分。
- `tools/audit_training_views.py` 用冻结位姿加载器读取清单中全部位姿，按 split 报告相机 pitch、roll 分位数、pitch 直方图、各切片 pitch ±10° 内的图像比例，以及训练图像素（每图 32×32 射线采样）落在各仰角带的比例，并与 ERP 网格各带像素比例并列：

```bash
python3 1002Pin2Pan/tools/audit_training_views.py \
  --manifest /data/zhuzhaoziao/RELPlus/outputs/REL_plus_v2_1_implementation/full_manifest.csv \
  --output <输出目录>/training_views.json
```

- `tools/audit_view_coverage.py`（只用 CPU，不训练）检查训练视角的冗余度，以及按位姿删掉 15% 训练图会去掉什么。按位姿文件名的 `camera_<uuid>_` 前缀把视角分到拍摄位置，报告每个位置的视角数、pitch/roll/视场分布，以及每个位置上球面各方向（`--directions` 个近似等立体角方向）被几个视角看到（按仰角带）。再比较三种删法：|pitch| 最大的先删、均匀随机（`--random-trials` 次）、贪心覆盖规则（每次删掉使最少方向降到 `--greedy-k` 个视角以下的那张）。“空洞”指某位置原来至少一个视角看得到、删后一个都看不到的方向。判定规则预先写在脚本中：随机删除的空洞占比低于 1% 判为 `REDUNDANT_RANDOM_SUFFICES`；贪心规则把空洞至少减半判为 `COVERAGE_RULE_HAS_ROOM`；否则 `COVERAGE_RULE_NO_BETTER`。这只是覆盖层面的筛查，mIoU 是否变化仍需训练。

```bash
python3 1002Pin2Pan/tools/audit_view_coverage.py \
  --manifest /data/zhuzhaoziao/RELPlus/outputs/REL_plus_v2_1_implementation/full_manifest.csv \
  --output <输出目录>/view_coverage.json
```

- **渲染**：裁剪先在 S2D 原生尺寸 1080×1080 上从原生 ERP 渲染。RGB 用双线性采样，再用 INTER_LINEAR 缩放到 480（与第2节相同；训练集 480 RGB 的缩放核没有记录）。z-depth 由最近 ERP 像素的射线距离换算。
- **REL+**：深度最近邻缩放到 480，K 同步缩放（即 `load_canonical_frame` 的做法），再以裁剪的已知旋转作为重力调用冻结的 `generate_rel_plus_v2_1`。ReD 与高度按每个裁剪归一化，与训练一致。
- **HHA**：在裁剪上运行 Depth2HHA，生成顺序、缩放核和通道顺序取自 `check_hha_cache.py` 报告。规则同第2节，`NEAR_MATCH` 同样需要 `--accept-hha-near-match`。
- **拼接**：每个 ERP 像素取所有覆盖它的裁剪的 softmax 之和，再取 argmax。

参数与第2节相同（没有 `--wrap-pad`），另有 `--workers N`，用 N 个 CPU 进程按全景并行生成裁剪输入，每个进程依次生成一张全景的 22 个裁剪。HHA 臂的 Depth2HHA 在原开发环境中约 15 s/个 1080 裁剪，服务器耗时需实测；可加 `--workers 32` 并设 `OMP_NUM_THREADS=1`：

```bash
OMP_NUM_THREADS=1 python3 1002Pin2Pan/tools/eval_pano_crops.py \
  --stanford-root /data/zhuzhaoziao/datasets/Stanford2D3D \
  --semantic-labels /data/zhuzhaoziao/cmx/raw/reference_repos/2D-3D-Semantics/assets/semantic_labels.json \
  --config 0927调参结果/configs/hha.json \
  --checkpoint <HHA epoch-200.pth> \
  --hha-cache-report <hha_cache_check.json> --accept-hha-near-match \
  --workers 32 \
  --output <输出目录>/hha_crop_eval
```

REL+ 臂换成 `relplus.json` 与 REL+ 权重，去掉两个 HHA 参数。

**解读**：分别计算两臂“裁剪评估 − 整图评估”的变化。本比较同时改变投影、视场与上下文、法向估计、REL+ 的重力来源与归一化范围、HHA 的 z-depth/range 定义，以及多视角概率融合。REL+ 若获得更大增益，可以支持进一步检验其输入表示的跨投影差异，但不能单独证明逐图归一化是主因；若增益接近，也不能据此排除归一化影响或认定场景本身是原因。归因需要固定其他因素、只改变归一化统计范围的对照实验。

**合成测试**：全景按方位角（每 45° 一区）和仰角（±30°、±70°）分区着色，用逐像素按颜色分类的网络推理并拼接。当前布局要求 ERP 标签复原率超过 99%，两极所在的首末行全部正确；把拼接的 yaw 或 pitch 偏移 5°，复原率必须下降超过 1 个百分点。另检查 60°、训练中位视场、75° 在 64×128 与 1024×2048 上的完整覆盖，以及旧 16 切片布局在训练中位视场下会留下空洞。

## 5. REL+ 3.0 `tools/relplus_variant.py`、`tools/relplus_variant_cache.py`

全景底部极区（−60° 到 −90°）里，REL+ 把地板大量判成桌子：整图评估 5.04M 像素，22 切片评估 16.26M 像素；HHA 只有 15k。切片按训练方式逐个归一化反而更差，所以原因不是全景 ReD 的归一化。冻结编码中，EGVIA 只对偏离水平超过 alpha = 45° 的表面混入高度（`rel_plus/encoding.py:95-102`，ERP `getREL` 规则相同）。地板和桌面都是水平面，EGVIA 相同，LOA 都约为 90，几何通道里只有 ReD（水平距离）能区分二者。

REL+ 3.0（变体名 `v3_0`）对所有表面混入高度。冻结流程只接受 v2.1 的协议字符串，所以本次训练的冻结报告仍写 `RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`，3.0 记录在缓存标记 `relplus_variant.json` 和评估的 metrics.json 中。两个冻结编码器都先把角度截到 [0, 255]，再判断 `angle <= t or angle >= 255 - t`，其中 `t = alpha * 255 / 180`；取 alpha = −1 时没有像素被判为水平。LOA、ReD、有效掩码和无效值 255 都不变。合成房间（相机离地 1.4 m，桌面高 0.75 m）上，地板仍约为 0，天花板约为 254，桌面从约 0 变为 32（ERP）或 45（pitch −20° 切片）。

**1. 生成缓存。** `generate` 原样调用冻结的 `tools/generate_full_relplus_cache.py`，其余参数照传。清单用现有 `formal_cache/cache_generation_summary.json` 里记录的 `manifest_path`。工具先在输出目录写入 `relplus_variant.json`，再开始生成图像；如果目录里已有另一种变体，或者有 REL+ 图像却没有标记，就拒绝运行（冻结的 `--resume` 只检查 PNG 能否解码）。缓存进程用 fork 启动，以继承变体设置：

```bash
python3 1002Pin2Pan/tools/relplus_variant_cache.py generate \
  --manifest <manifest_path> \
  --output-root /data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v3_0/formal_cache \
  --workers 32 --authorize-full-cache
```

**2. 审计缓存。** `audit` 原样调用冻结的 `tools/audit_full_relplus_cache.py`，变体从缓存标记读取，70 个样本按该变体逐字节重新生成并比对：

```bash
python3 1002Pin2Pan/tools/relplus_variant_cache.py audit \
  --manifest <manifest_path> \
  --cache-root /data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v3_0/formal_cache \
  --output-dir /data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v3_0/formal_cache/audit
```

**3. 训练。** 复制 `relplus.json`，把所有指向 `CMX_RELPlus_v2_3/formal_cache` 的路径（包括 `x_root_folder`、`x_valid_root_folder`、split 列表、缓存报告、审计报告、预检报告，以及嵌套块中的同名字段）改成新缓存。配方与种子不变（gamma 1、lr 1.2e-4、seed 12345）。之后按原流程跑预检和 DDP smoke：启动器会核对 smoke 报告里的缓存审计路径（`tools/launch_formal_training_v2_3.py:102-113`），所以 smoke 也要针对新缓存重跑。第 200 轮的针孔测试集评估读取新缓存的 `test.txt`，可以直接与 61.41 比较。

**4. 全景评估。** 第2、4节的两个评估都新增了 `--relplus-variant`，默认 `v2_1`。该值必须与检查点配置所用训练缓存的标记一致（`x_root_folder` 的上一级目录；没有标记即为 `v2_1`），否则拒绝运行；HHA 臂只接受 `v2_1`。所用变体和 alpha 会写入 metrics.json。

```bash
python3 1002Pin2Pan/tools/eval_pano_transfer.py \
  --stanford-root /data/zhuzhaoziao/datasets/Stanford2D3D \
  --semantic-labels /data/zhuzhaoziao/cmx/raw/reference_repos/2D-3D-Semantics/assets/semantic_labels.json \
  --config <变体配置.json> --checkpoint <变体 epoch-200.pth> \
  --relplus-variant v3_0 \
  --output <输出目录>/relplus_v3_0_whole
```

**对照指标**：针孔 mIoU（v2.1 为 61.41），整图全景 mIoU（53.47），底部极区地板→桌子像素数（5.04M）与桌子 IoU（4.96），以及各仰角带在两臂共有类别上与 HHA 的差值。

### REL+ 3.1（变体名 `v3_1`，`--variant v3_1`）

即离线对比（下一小节）中分离效果最好的 `banded_metric` 规则。和 3.0 一样，冻结流程里的协议字符串仍是 v2.1，3.1 只记录在缓存标记和评估 metrics.json 中。它不改 alpha，而是替换 EGVIA 公式：

EGVIA = 85 · [S((θ − 45°)/15°) + S((θ − 135°)/15°) + clip((z − z_lo)/(z_hi − z_lo), 0, 1)]，S(x) = ½ + ¾x − ¼x³（x 先截到 [−1, 1]）

θ 是未混合的表面角度（0° 朝上，90° 竖直，180° 朝下），z 是相对相机的高度（cm），[z_lo, z_hi] = [−150, 210]（训练集地板 p5 到天花板 p95）。朝上、竖直、朝下的平面分别落在 0–85、85–170、170–255，没有跳变；高度是米制的，同一张桌面在针孔帧和全景里取值相同。LOA、ReD、有效掩码和无效值 255 都不变。针孔侧把冻结生成器调用的 `encode_rel_channels` 换成只改写第 0 通道的包装；全景侧用 `get_rel_banded_metric`，LOA 和 ReD 逐句沿用 `getREL`。测试核对两者的 LOA/ReD 与冻结编码器逐字节一致。

缓存、训练和评估都按上面 1–4 步进行，只是把 `--variant`/`--relplus-variant` 换成 `v3_1`，输出目录换成新的（例如 `CMX_RELPlus_v3_1/formal_cache`），复制的配置里所有缓存路径都指向它。

### EGVIA 规则离线对比 `tools/check_egvia_rules.py`（只用 CPU，不训练）

用同一组角度 A 和高度 H 计算六种 EGVIA 规则，按类别比较：`v2_1`（REL 公开代码与 REL+ v2.1，在 45° 处跳变）、`paper_eq4`（REL 论文式 4，同样跳变，方向相反）、`v3_0`（处处 0.5A + 0.5H）、`banded`（85·o(A) + H/3，o 在 45° 与 135° 两侧 ±15° 内平滑地从 0 过渡到 1 再到 2；朝上、竖直、朝下的平面分别落在 0–85、85–170、170–255，高度仍逐图归一化）、`banded_metric`（同 `banded`，但 H 换成相对相机的米制高度，截断到 `--metric-range-cm`，默认 −150 到 210 cm，即训练集地板 p5 到天花板 p95）、`v3_0_metric`（3.0 的公式换成米制高度，用来把高度的改动和分带分开看）。每张图都会核对 `v2_1` 规则与冻结编码器的 EGVIA 逐字节一致。后两种只是候选，尚未接入缓存生成。

```bash
python3 1002Pin2Pan/tools/check_egvia_rules.py \
  --manifest /data/zhuzhaoziao/RELPlus/outputs/REL_plus_v2_1_implementation/full_manifest.csv \
  --config 0927调参结果/configs/relplus.json \
  --stanford-root /data/zhuzhaoziao/datasets/Stanford2D3D \
  --semantic-labels /data/zhuzhaoziao/cmx/raw/reference_repos/2D-3D-Semantics/assets/semantic_labels.json \
  --workers 32 --output <输出目录>/egvia_rules
```

样本与上面的地板/桌面检查相同：默认 3000 帧针孔训练图，以及训练区域（1、2、3、4、6，不含区域 5）的全部全景。`egvia_rules.json` 按子集给出：各规则下关键类别对的 EGVIA 直方图重叠（两类归一化直方图逐 bin 取最小值再求和，1 为相同，0 为完全分开）及全部类别对的平均重叠；各类别距 45°/135° 不到 5° 的像素比例（`v2_1` 与 `paper_eq4` 在这里跳变）；各类别相对相机高度的 p5/中位数/p95（用训练集确定米制范围）。完整重叠矩阵在 `egvia_overlap.csv`，直方图在 `egvia_histograms.csv`。EGVIA 重叠只是代理指标：RGB、LOA、ReD 仍能区分 EGVIA 分不开的类别。

### 输入缺陷检查 `tools/check_input_defects.py`（阶段0，只用 CPU，不训练）

在不改任何缓存、检查点和冻结源码的前提下，度量 REL+ 输入的四个已知问题有多大，用来决定是否值得重训：

- **ReD**：针孔训练帧中有无效像素的比例；无效像素（存为相机位置，水平距离 0）把逐图最小值拉到 0 后，有效像素的 ReD 字节相对"只用有效像素求 min/max"偏移多少；水平距离的 cm 分位数（用于定米制 ReD 的范围 R，取 p99）。另在训练区域全景上渲染针孔裁剪（yaw 0/90/180/270 × pitch −20/0/20，FOV 60°，同 `cross_projection.check_panorama`），比较同一射线上裁剪与全景的 ReD 字节差：现行逐图 ReD，以及由水平距离差换算的米制 ReD（距离/R）。
- **插值混入无效值**：对配置所指的 REL+ 缓存（变体取自缓存标记）和 `--hha-root` 的 HHA 缓存，按训练的缩放/裁剪采样（两臂同一变换），调用冻结的 `analyze_invalid_interpolation`，统计输入被双线性缩放混入无效像素的有标签像素比例。HHA 没有有效掩码，用 480 深度的有效像素代替。
- **水平翻转**：对前 `--flip-frames` 帧，把深度、内参和重力一起镜像后重新编码，与翻转后的原编码比较（v2.1 与 3.1）。EGVIA、ReD 应完全一致；LOA 分别按 180−LOA 和 179−LOA 计分（LOA 字节是截断的，所以合成场景上恰好成立的是 179−LOA）。
- **全景输入**：全景深度中原始值 0（ERP 解码会把它当成 2 mm 的有效点）和 65535 的比例；`getGDir` 估计重力相对 ERP 竖直方向的倾角（与 `cross_projection.py` 一样假设全景是水平存储的），以及该倾角对 3.1 的高度（cm）和 EGVIA 字节的影响。

全程不读全景标签。全景计数默认覆盖全部区域，裁剪只用训练区域。`input_defects.json` 的 `plan_gates` 按修复计划预先写好的阈值给出结论：阶段1（米制 ReD）要求多数裁剪现行 ReD 中位差 ≥10 级且米制 ReD ≤3 级；阶段2（缩放修复）要求任一臂受影响的有标签像素超过 1%。每帧、每张全景、每个裁剪的明细在 `pinhole_frames.csv`、`panoramas.csv`、`crops.csv`。

```bash
OMP_NUM_THREADS=1 nice -n 19 python3 1002Pin2Pan/tools/check_input_defects.py \
  --manifest /data/zhuzhaoziao/RELPlus/outputs/REL_plus_v2_1_implementation/full_manifest.csv \
  --config <REL+ 3.1 训练所用的配置.json> \
  --hha-root /data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480/HHA \
  --stanford-root /data/zhuzhaoziao/datasets/Stanford2D3D \
  --workers 8 --output <输出目录>/input_defects
```

`--skip-blend` 跳过缓存读取（不需要 torch）。

## 6. 领域自适应 `tools/build_target_cache.py`、`tools/gen_pseudo_labels.py`、`tools/train_pin2pan.py`

按 Trans4PASS（`adaptations/`，commit 758f2301）的流程把冻结的针孔检查点适配到全景：先做输出空间对抗预热（warm-up），再用预热后的模型生成伪标签，最后做 MPA（伪标签、原型对齐和对抗三项）。源域是冻结训练用的针孔缓存，配置、数据增强和 focal loss 都不变；目标域是区域 1、2、3、4、6 的全景，训练时不读取它们的真值。区域 5 只用于最终评估：工具拒绝把它作为目标训练区域，也不在区域 5 上挑选检查点，只保存最后一次迭代的结果。

**与 Trans4PASS 的差异**（Trans4PASS 从 ImageNet 权重开始训练，这里微调已经训练好的网络）：

- 优化器沿用冻结配方的 AdamW，学习率取冻结值的 1/10（1.2e-5）并按 poly 衰减；Trans4PASS 用 SGD。这个学习率和迭代次数（预热 2,000 步、MPA 10,000 步）都是估计值，没有调过。
- 两个域都用冻结的 focal loss（gamma 1，对全部像素取平均）。伪标签为 255 的像素按 0 计入，所以目标项的实际权重约等于伪标签覆盖率。各项权重沿用 Trans4PASS：对抗 0.001，原型每个域 0.001，伪标签 1。
- 伪标签阈值规则相同（每类置信度的中位数，上限 0.9），但只在评估尺寸上做单尺度推理。Trans4PASS 用 0.5 到 1.75 共六个尺度，而 1.75 倍的 1024×2048 全景放不进 24 GB 显存。
- 原型记忆的更新修正了 Trans4PASS 的问题。原代码 `np.mean(列表)` 没有指定轴，所有通道被同一个标量代替；缓冲区从不清空；没出现的类被拉向 0。这里按通道求均值，只更新出现过的类，每次更新后清空缓冲区，并在各 GPU 之间同步。
- 特征取解码器融合层 `decode_head.linear_fuse` 的 512 通道输出（1/4 分辨率）。
- 目标裁剪保持全高，两极都在其中；宽 1024，起始方位随机，可以跨过接缝。每卡每步 2 张源图、1 张目标裁剪。

**1. 目标缓存。** 缓存与评估完全相同的输入：RGB、该臂的 ERP X 和真值 Label，尺寸 1024×2048。Label 只在第 3 步的报告中使用。REL+ 变体参数和 HHA 报告参数与第 2、5 节相同：

```bash
python3 1002Pin2Pan/tools/build_target_cache.py \
  --stanford-root /data/zhuzhaoziao/datasets/Stanford2D3D \
  --semantic-labels /data/zhuzhaoziao/cmx/raw/reference_repos/2D-3D-Semantics/assets/semantic_labels.json \
  --config 0927调参结果/configs/relplus.json --workers 32 \
  --output <输出目录>/relplus_target
```

HHA 臂换成 `hha.json`，并加上 `--hha-cache-report <报告> --accept-hha-near-match`。

**2. 预热。** 与冻结启动器一样用 `torch.distributed.launch`，每张卡一个进程：

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29511 \
  1002Pin2Pan/tools/train_pin2pan.py --stage warmup \
  --config 0927调参结果/configs/relplus.json --checkpoint <REL+ epoch-200.pth> \
  --target-cache <输出目录>/relplus_target --output <输出目录>/relplus_warmup
```

**3. 伪标签。** 单卡运行：

```bash
python3 1002Pin2Pan/tools/gen_pseudo_labels.py \
  --target-cache <输出目录>/relplus_target \
  --config 0927调参结果/configs/relplus.json \
  --checkpoint <输出目录>/relplus_warmup/checkpoint.pth \
  --output <输出目录>/relplus_pseudo
```

`pseudo_labels.json` 给出伪标签覆盖率、每类阈值和保留像素的准确率，以及该检查点在目标训练全景上的 mIoU、各仰角带分数和底部极区地板→桌子像素数。这些数字用目标域真值计算，只用来观察，不能据此调整阈值或挑选模型。

**4. MPA。** 从预热检查点开始：

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29512 \
  1002Pin2Pan/tools/train_pin2pan.py --stage mpa \
  --config 0927调参结果/configs/relplus.json \
  --checkpoint <输出目录>/relplus_warmup/checkpoint.pth \
  --target-cache <输出目录>/relplus_target \
  --pseudo-labels <输出目录>/relplus_pseudo \
  --output <输出目录>/relplus_mpa
```

每个阶段输出 `checkpoint.pth`、`adaptation_state.pth`（判别器与原型记忆）、`pin2pan.json`（本次设置）和 `train_log.jsonl`（每 20 步各项损失）。检查点保存为 `{"epoch": 200, "model": ...}`，epoch 沿用源检查点，所以第 2、4 节的评估可以直接使用，`--expected-epoch 200` 不变。如果 1024 宽的目标裁剪超出显存，可改用 `--target-crop-width 512`（仍保持全高）。`--lambda-adv` 设置对抗权重（默认 0.001），设为 0 即关闭对抗项；判别器仍会训练并记录损失，但不再影响分割网络。显存和速度还没有在 GPU 上测过。

**5. 评估。** 用第 2 节的整图评估在区域 5 上分别评估预热和 MPA 检查点。

**对照指标**：整图全景 mIoU（source-only：REL+ 53.47，HHA 55.67）、各仰角带分数，以及底部极区地板→桌子像素数（REL+ source-only 为 5.04M）。

## 7. RGB-only 基线 `tools/rgb_only.py`、`tools/train_rgb_only.py`

用来回答深度是否帮助针孔→全景迁移。网络是冻结 CMX 的 RGB 流：先按 CMX 原样构建双流 MiT-B2 + MLPDecoder512（同一种子下随机数消耗相同，RGB 编码器和解码器的初始权重与 HHA 臂一致），再删去 X 流（`extra_*`）和融合模块（FRM、FFM），四个阶段的 RGB 特征直接进入解码器。解码器保留 CMX 的 512 通道（SegFormer-B2 原版为 768），所以与 HHA 臂只差 X 流和融合。参数量 25.78M（编码器 24.20M），CMX 为 66.57M。`forward(rgb, modal_x)` 保持 CMX 接口并忽略 `modal_x`。

**训练。** 输入某个种子的 HHA 臂 resolved 配置，种子、数据、配方和数据检查都取自它；工具只改架构、臂名和输出路径，然后用冻结的 `source/train.py` 训练 RGB-only 网络。数据仍按 HHA 臂读取（HHA 输入被网络忽略），所以同一种子下与 HHA 臂的批次和增强相同（CPU 测试中用两个 loader worker 与 CMX 网络逐批比对过，服务器上没有追踪）。与套件的 `run_arm.py` 一样，每个进程用 1 个 OpenCV 和 1 个 torch 线程，遇到 NaN 损失即停止，world size 必须为 8。

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29513 \
  1002Pin2Pan/tools/train_rgb_only.py \
  --config 0927调参结果/configs/hha.json \
  --output <输出目录>/rgb_only_s12345
```

输出目录必须为空。先加 `--smoke-iterations 20`（换一个输出目录）只训练一轮 20 步并保存 `checkpoints/epoch-1.pth`，用来检查显存和整条路径。中断后换新的输出目录，加 `--continue <旧目录>/checkpoints/epoch-N.pth` 续训（参数原样交给冻结 Engine）。输出目录中有 `resolved_config.json`（评估时作 `--config`）、`rgb_only.json`（架构、来源配置、种子、参数量）以及冻结训练的日志和检查点；冻结的 train.log 仍打印 "Architecture: Original CMX"。

**评估。** 第 2 节的全景评估和针孔评估按配置中的 `architecture` 构建 RGB-only 网络，并把它写入输出报告。全景评估把 X 置零，不需要 `--hha-cache-report`；针孔评估仍经过冻结的 HHA 输入适配器，输入同样被忽略。

```bash
python3 1002Pin2Pan/tools/eval_pano_transfer.py --eight-class-predictions \
  --stanford-root /data/zhuzhaoziao/datasets/Stanford2D3D \
  --semantic-labels /data/zhuzhaoziao/cmx/raw/reference_repos/2D-3D-Semantics/assets/semantic_labels.json \
  --config <输出目录>/rgb_only_s12345/resolved_config.json \
  --checkpoint <输出目录>/rgb_only_s12345/checkpoints/epoch-200.pth \
  --output <评估目录>/rgb_only_s12345_pano_8pred
```

## 8. 已知限制

- **原始深度（RGBD）臂不能在全景上评估。** 针孔 z-depth 字节在 ERP 中没有对应定义。
- **HHA 的视差通道在两种投影之间存在系统差异**（见第3节）。
- **全景 REL 的重力来自 `getGDir` 估计；透视 REL+ 使用位姿真值重力。** 一致性报告中给出了每张全景的重力估计角，可用来判断这一差异的影响。
- **逐图归一化造成的编码差异仍保留在评估输入中。** 如果第1节真实数据也显示 ReD/EGVIA 字节差异很大，应考虑在 Pin2Pan 研究中增加固定物理单位归一化的 REL+ 臂。
- **REL+ 3.0 的高度仍按每张图 1–99 分位数归一化。** 几乎全是地板的切片会把很小的高度起伏拉伸到整个范围；这对切片评估的影响大于整图评估。

## 测试

```bash
cd 1002Pin2Pan && python3 -m pytest -q tests/
```

几何/HHA 测试需要 numpy、scipy、opencv、pytest；环形推理测试另需 torch。实际 CMX 评估还需要冻结源码依赖的 timm、easydict、Pillow 和 PyYAML。测试包含针孔评估（冻结评估器在 CPU 上端到端跑通，含 8 类预测）、错误重力、非有限几何量、缓存局部错误/失败样本、HHA 生成顺序、裁剪覆盖与拼接几何、仰角分带、水平切片覆盖、训练视角核对、训练视角覆盖与子采样核对、REL+ 3.0 变体（编码、评估门控、缓存工具的多进程生成）、EGVIA 规则离线对比、输入缺陷检查（ReD 偏移、镜像编码与 LOA 重映射、全景倾角与原始 0 值、端到端运行；插值一项需 torch）、领域自适应（特征 KL 损失与伪标签阈值逐项对照 Trans4PASS 原代码，原型记忆更新，跨接缝裁剪，以及在小网络上用 CPU 跑通目标缓存、伪标签、预热和双进程 MPA）、RGB-only 基线（与 CMX 的初始权重和预训练加载一致、输出与 X 无关，以及在小网络上用 CPU 跑通训练、针孔评估和全景评估，并与 CMX 网络逐批比对训练批次）、测试区域缺失及空评估参数等回归场景。以上测试在 PyTorch 2.x 和 1.8.1（Python 3.9）上都通过；输入缺陷检查的测试只在无 torch 的 Python 3.13 上跑过，插值一项待在服务器上运行。测试通过不代表已经在真实服务器数据上验证迁移效果。
