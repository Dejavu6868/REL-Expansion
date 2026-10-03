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

输出包括 `metrics.json`、`per_class_iou.csv`、`confusion_matrix.csv` 和 `samples.txt`。加 `--limit 3` 可先做冒烟测试，报告标记为 `SMOKE`。`--limit` 必须为正；缺少任一指定区域、重复指定区域或没有有效标签像素时拒绝完成评估。

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

第2节的整图评估给网络的是 ERP 输入，逐图归一化覆盖整个 360°，与训练时的 90° 针孔图不同。本工具把每张全景切成 16 个 90° 针孔裁剪：pitch 0 每隔 45° 一个，pitch ±45 各每隔 90° 一个，合起来覆盖全部 ERP 像素。每个裁剪按训练缓存的做法生成输入并单独推理，再把 softmax 概率投回 ERP 标签网格，用与第2节相同的 1024×2048 标签评分。

- **渲染**：裁剪先在 S2D 原生尺寸 1080×1080 上从原生 ERP 渲染。RGB 用双线性采样，再用 INTER_LINEAR 缩放到 480（与第2节相同；训练集 480 RGB 的缩放核没有记录）。z-depth 由最近 ERP 像素的射线距离换算。
- **REL+**：深度最近邻缩放到 480，K 同步缩放（即 `load_canonical_frame` 的做法），再以裁剪的已知旋转作为重力调用冻结的 `generate_rel_plus_v2_1`。ReD 与高度按每个裁剪归一化，与训练一致。
- **HHA**：在裁剪上运行 Depth2HHA，生成顺序、缩放核和通道顺序取自 `check_hha_cache.py` 报告。规则同第2节，`NEAR_MATCH` 同样需要 `--accept-hha-near-match`。
- **拼接**：每个 ERP 像素取所有覆盖它的裁剪的 softmax 之和，再取 argmax。

参数与第2节相同（没有 `--wrap-pad`），另有 `--workers N`，用 N 个 CPU 进程并行生成裁剪输入。HHA 臂的 Depth2HHA 在 1080 裁剪上本地约 15 s/个，每张全景 16 个，建议加 `--workers 32` 并设 `OMP_NUM_THREADS=1`：

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

**解读**：分别计算两臂“裁剪评估 − 整图评估”的提升。若 REL+ 的提升明显大于 HHA，整图上多出的降幅主要来自 ERP 输入（逐图归一化），固定归一化的 REL+ 臂值得训练；若两者提升相近，原因在全景场景本身，改归一化不一定有效。裁剪视场固定为 90°，若训练针孔图的视场与此相差较大，结论需要打折扣。

**合成测试**：全景按方位角（每 45° 一区）和仰角（±30°、±70°）分区着色，用逐像素按颜色分类的网络推理并拼接，ERP 标签复原率 99.7%，两极所在的首末行全部正确；把裁剪 yaw 偏 5° 时降到 97.5%，pitch 偏 5° 时降到 91.2%。

## 5. 已知限制

- **原始深度（RGBD）臂不能在全景上评估。** 针孔 z-depth 字节在 ERP 中没有对应定义。
- **HHA 的视差通道在两种投影之间存在系统差异**（见第3节）。
- **全景 REL 的重力来自 `getGDir` 估计；透视 REL+ 使用位姿真值重力。** 一致性报告中给出了每张全景的重力估计角，可用来判断这一差异的影响。
- **逐图归一化造成的编码差异仍保留在评估输入中。** 如果第1节真实数据也显示 ReD/EGVIA 字节差异很大，应考虑在 Pin2Pan 研究中增加固定物理单位归一化的 REL+ 臂。

## 测试

```bash
cd 1002Pin2Pan && python3 -m pytest -q tests/
```

几何/HHA 测试需要 numpy、scipy、opencv、pytest；环形推理测试另需 torch。实际 CMX 评估还需要冻结源码依赖的 timm、easydict、Pillow 和 PyYAML。测试包含错误重力、非有限几何量、缓存局部错误/失败样本、HHA 生成顺序、裁剪覆盖与拼接几何、测试区域缺失及空评估参数等回归场景。测试通过不代表已经在真实服务器数据上验证迁移效果。
