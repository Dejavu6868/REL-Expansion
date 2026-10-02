# 1002Pin2Pan：针孔→全景迁移的准备工具

本目录用于 Pin2Pan 实验的第2步和第3步：先检查透视 REL+ 与全景 REL 两条代码路径是否描述同一几何，再把已冻结的 S2D（针孔）权重直接用于 Stanford2D3D 全景测试（source-only，不训练）。两者都直接调用 `0927调参结果/selected_recipe/code/training/source` 中的冻结代码，没有复制或修改其中任何文件。

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

真实数据仍需在服务器上用 Stanford2D3D 全景运行。

## 2. 全景 source-only 评估 `tools/eval_pano_transfer.py`

加载冻结的 CMX 双流 MiT-B2 和 S2D epoch200 权重，在 area_5a/5b 全景上评估：

- **RGB**：用冻结 loader 的原调用读取，即 `cv2.imread(path, cv2.COLOR_BGR2RGB)`，通道顺序与训练一致。
- **X 输入**：原始 ERP `getREL`，在最近邻缩放到评估尺寸的深度上计算。这对应透视缓存在 480×480 模型尺寸上生成的做法。
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

输出包括 `metrics.json`、`per_class_iou.csv`、`confusion_matrix.csv` 和 `samples.txt`。加 `--limit 3` 可先做冒烟测试。

## 3. 已知限制

- **HHA 和原始深度两臂暂不能在全景上评估。** 它们的冻结输入只针对针孔相机定义，仓库中没有 ERP 版本，脚本会直接拒绝。要比较 HHA 与 REL+ 的迁移差距，需要先确定全景 HHA 的定义。
- **全景 REL 的重力来自 `getGDir` 估计；透视 REL+ 使用位姿真值重力。** 一致性报告中给出了每张全景的重力估计角，可用来判断这一差异的影响。
- **逐图归一化造成的编码差异仍保留在评估输入中。** 如果第1节真实数据也显示 ReD/EGVIA 字节差异很大，应考虑在 Pin2Pan 研究中增加固定物理单位归一化的 REL+ 臂。

## 测试

```bash
cd 1002Pin2Pan && python3 -m pytest -q tests/
```

需要 numpy、opencv、pytest；评估测试另需 torch。
