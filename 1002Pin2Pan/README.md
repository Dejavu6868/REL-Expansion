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

每个阶段输出 `checkpoint.pth`、`adaptation_state.pth`（判别器与原型记忆）、`pin2pan.json`（本次设置）和 `train_log.jsonl`（每 20 步各项损失）。检查点保存为 `{"epoch": 200, "model": ...}`，epoch 沿用源检查点，所以第 2、4 节的评估可以直接使用，`--expected-epoch 200` 不变。如果 1024 宽的目标裁剪超出显存，可改用 `--target-crop-width 512`（仍保持全高）。显存和速度还没有在 GPU 上测过。

**5. 评估。** 用第 2 节的整图评估在区域 5 上分别评估预热和 MPA 检查点。

**对照指标**：整图全景 mIoU（source-only：REL+ 53.47，HHA 55.67）、各仰角带分数，以及底部极区地板→桌子像素数（REL+ source-only 为 5.04M）。

## 7. 已知限制

- **原始深度（RGBD）臂不能在全景上评估。** 针孔 z-depth 字节在 ERP 中没有对应定义。
- **HHA 的视差通道在两种投影之间存在系统差异**（见第3节）。
- **全景 REL 的重力来自 `getGDir` 估计；透视 REL+ 使用位姿真值重力。** 一致性报告中给出了每张全景的重力估计角，可用来判断这一差异的影响。
- **逐图归一化造成的编码差异仍保留在评估输入中。** 如果第1节真实数据也显示 ReD/EGVIA 字节差异很大，应考虑在 Pin2Pan 研究中增加固定物理单位归一化的 REL+ 臂。
- **REL+ 3.0 的高度仍按每张图 1–99 分位数归一化。** 几乎全是地板的切片会把很小的高度起伏拉伸到整个范围；这对切片评估的影响大于整图评估。

## 测试

```bash
cd 1002Pin2Pan && python3 -m pytest -q tests/
```

几何/HHA 测试需要 numpy、scipy、opencv、pytest；环形推理测试另需 torch。实际 CMX 评估还需要冻结源码依赖的 timm、easydict、Pillow 和 PyYAML。测试包含错误重力、非有限几何量、缓存局部错误/失败样本、HHA 生成顺序、裁剪覆盖与拼接几何、仰角分带、水平切片覆盖、训练视角核对、REL+ 3.0 变体（编码、评估门控、缓存工具的多进程生成）、领域自适应（特征 KL 损失与伪标签阈值逐项对照 Trans4PASS 原代码，原型记忆更新，跨接缝裁剪，以及在小网络上用 CPU 跑通目标缓存、伪标签、预热和双进程 MPA）、测试区域缺失及空评估参数等回归场景。以上测试在 PyTorch 2.x 和 1.8.1（Python 3.9）上都通过。测试通过不代表已经在真实服务器数据上验证迁移效果。
