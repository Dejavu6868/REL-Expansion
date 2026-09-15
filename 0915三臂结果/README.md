# 0915三臂结果：CMX-RGBD / CMX-HHA / CMX-REL+

**冻结日期：2026-09-15。结果状态：三臂固定 epoch200 全量评估完成。** 本目录保存本次 Stanford2D3D Perspective S2D 三臂实验的源码快照、参数、结果和来源证据。后续调 batch、精度、模型或表示的实验应使用新目录和新协议，不回写本目录。

固定主终点：RGBD **54.835780%**、HHA **59.155782%**、REL+ **59.585366% mIoU**。REL+ − HHA = **+0.429584 个百分点**，REL+ − RGBD = **+4.749585 个百分点**。

这是 **seed12345 下的描述性结果**，不能据此声称跨 seed 稳定收益、统计显著或普遍优越。REL+在13类中6类高于HHA、7类低于HHA。

## 1. 固定 epoch200 结果

全部使用相同的17,593张测试图片、13类、ignore255。以下指标单位为百分比；JSON保留原始精度。

| 模型 | mIoU (%) | Pixel Accuracy (%) | Mean Accuracy (%) |
|---|---:|---:|---:|
| CMX-RGBD | 54.835780 | 79.355698 | 62.950568 |
| CMX-HHA | 59.155782 | 81.908146 | 67.286482 |
| CMX-REL+ | 59.585366 | 81.853932 | 67.642209 |

主表只用epoch200。REL+历史扫描中epoch115的较高结果属于test-selected描述，不作为本次三臂主终点。

| 类别 | RGBD IoU (%) | HHA IoU (%) | REL+ IoU (%) |
|---|---:|---:|---:|
| beam | 0.9312 | 3.6095 | 1.8447 |
| board | 71.2721 | 74.4841 | 71.1179 |
| bookcase | 63.4661 | 65.2762 | 65.7749 |
| ceiling | 90.3828 | 91.1234 | 92.3529 |
| chair | 75.1929 | 82.3530 | 82.1280 |
| clutter | 47.0387 | 50.0496 | 49.8385 |
| column | 12.5681 | 17.2009 | 24.3497 |
| door | 31.5612 | 42.5419 | 39.1347 |
| floor | 94.7718 | 96.6183 | 96.3556 |
| sofa | 20.6300 | 29.4410 | 34.0877 |
| table | 71.0032 | 76.9232 | 76.2333 |
| wall | 73.1574 | 76.6981 | 77.2104 |
| window | 60.8897 | 62.7059 | 64.1815 |

原始文件：[三臂汇总JSON](results/three_arm_comparison.json)、[总指标CSV](results/three_arm_metrics.csv)、[分类别CSV](results/three_arm_per_class_iou.csv)。每臂的metrics、分类IoU和混淆矩阵分别位于[RGBD](results/RGBD/)、[HHA](results/HHA/)、[REL+](results/RELPlus/)。三臂有效像素均为 **3,973,620,198**，真实标签的逐类像素总数一致。

## 2. 数据和输入表示

| 参数 | 冻结值 |
|---|---|
| 数据集 | Stanford2D3D Perspective S2D，480×480 |
| 训练区域 | Areas 1、2、3、4、6 |
| 测试区域 | Area 5a、5b |
| Train / Test | 52,903 / 17,593，交集0 |
| 类别 | 13；顺序见结果表 |
| 标签转换 | 存储1–13映射到模型0–12；0和255作为ignore255 |
| RGB归一化与X预处理 | 使用封存loader与cmx_preprocess的实际实现；mean=[0.485,0.456,0.406]，std=[0.229,0.224,0.225] |
| 缩放 | [0.5,0.75,1.0,1.25,1.5,1.75] |
| 裁剪 / padding | RGB、X、label共享变换，输出480×480 |
| 几何增强 | 水平/垂直翻转、任意旋转、透视warp均关闭 |

| Arm | X输入 | 冻结行为 |
|---|---|---|
| RGBD | RawDepth uint8单通道缓存 | IMREAD_UNCHANGED后重复为3通道；`RGBD_UINT8_REPEAT3_SOURCECOMPAT` |
| HHA | 已有uint8三通道HHA缓存 | IMREAD_UNCHANGED，保留可执行数组顺序；`HHA_FROZEN_CACHE_EXECUTABLE_ORDER` |
| REL+ | Perspective REL+ v2.1离线uint8三通道缓存 | `[EGVIA, LOA, ReD]`；`RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT` / `SOURCE_COMPAT_STORAGE_255` |

REL+还读取诊断valid mask；它不是额外的模型输入通道。HHA的物理通道命名与原生成K/gravity来源仍有未闭合项，本次冻结没有重新生成HHA或重新定义其通道。

训练list SHA-256：`96788184f2a1b318a05395a2c6b3867759526e0adb612a90eb6af59b1491b011`。测试list SHA-256：`b9de196c6c1aa8f9ac37926910af0806ce59b91eb068998711ddbb78eb24423a`。本目录不分发原始图片、模态缓存或模型权重。

## 3. 模型与训练参数

| 参数 | 冻结值 |
|---|---|
| 模型 | Original CMX派生/对齐的dual MiT-B2 + MLPDecoder |
| Decoder embedding | 512 |
| Gate / SMMF / DyMM / SGA | 全部关闭 |
| 初始化 | 共享MiT-B2预训练权重，随机初始化政策一致 |
| Seed | 12345，一个实验seed；不是八个独立seed |
| epoch/rank seed | 12345 + epoch + local_rank × 1000 |
| GPU / DDP | 8×RTX3090，8 ranks |
| 每卡 / 总batch | 1 / 8 |
| Epochs | 200 |
| 每轮 / 总optimizer updates | 6,613 / 1,322,600 |
| 每轮逻辑样本 | 52,904，补齐1个；FixedLengthDistributedSampler |
| Loss | FocalLoss2d，gamma=2，ignore255，reduction=none后求mean |
| Optimizer | AdamW，lr=6e-5，betas=(0.9,0.999)，weight decay=0.01 |
| 不衰减参数组 | bias/norm等按作者group_weight策略，weight decay=0 |
| Scheduler | iteration-wise WarmUpPolyLR，power=0.9 |
| Warmup | 10epochs，即66,130updates |
| 精度 / SyncBN | AMP=false，SyncBN=true |
| DataLoader workers | 每rank16，合计128；无显式worker_init_fn |
| cuDNN | benchmark=false，deterministic=false |
| 科学checkpoint | epoch100、105、…、200；固定主终点epoch200 |

预训练文件原路径：`/data/zhuzhaoziao/cmx/raw/pretrained/segformer/mit_b2.pth`。SHA-256：`ced22617efb7bae3c34ad0a80f20a9b8afb4d27368cb0835a23456baa9d0e092`。

机器可读的完整参数摘要：[frozen_protocol.json](provenance/frozen_protocol.json)。9月15日性能诊断中的workers/OpenCV候选和增大batch的讨论，均未应用到这次三臂正式结果。

## 4. 评估协议

| 参数 | 冻结值 |
|---|---|
| Checkpoint | 三臂均epoch200 |
| 数据 | 完整17,593张测试图片，无重复padding |
| Scale / flip | [1] / false |
| Crop / stride rate | [480,480] / 1.0 |
| align_corners | false |
| 每rank batch / ranks | 1 / 8 |
| 各rank样本数 | [2200,2199,2199,2199,2199,2199,2199,2199] |
| 指标 | argmax预测后的全局混淆矩阵汇总，mIoU对13类等权 |

RGBD/HHA实际使用的[评估输入适配器](evaluation/eval_three_arm_v1_full_adapter.py)也一并保存；其SHA-256为`374b244a4c2ed3d8e77384dae5721616920e757feb7979b08c89bfac36e09445`。它是项目适配代码，不能称为未经改动的官方整仓evaluator。REL+沿用其封存的V2.3 evaluator。

## 5. 训练代码的真实来源

三臂协议一致，历史上并非在同一个目录执行。归档保留两份源码树，不改写源码来制造“同一工作树”的历史。

| Arm | 本目录代码 | 实际服务器代码根 | 配置模块 |
|---|---|---|---|
| REL+ | [code/RELPlusv2.3](code/RELPlusv2.3/) | `/home/zhuzhaoziao/RELPlus/RELPlusv2.3` | `configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_3_formal` |
| RGBD | [code/CMX-S2D-ThreeArm-v1](code/CMX-S2D-ThreeArm-v1/) | `/home/zhuzhaoziao/RELPlus/CMX-S2D-ThreeArm-v1` | `configs.stanford2d3d_s2d.cmx_mit_b2_rgbd_three_arm_v1` |
| HHA | 同上 | 同上 | `configs.stanford2d3d_s2d.cmx_mit_b2_hha_three_arm_v1` |

- 两份树都保留`train.py`、`models/`、`dataloader/`、`utils/`、`engine/`、`configs/`、REL+表示代码、工具、测试夹具和许可证。
- REL+封存的是包含训练恢复及评估启动器修复的代码，来源为2026-09-01保存的210文件源码包，与本地工作副本逐文件一致。旧GitHub目录`0821CMX-RELPlusv2.3`不含这些后续修复，不能替代本次归档。
- RGBD/HHA树的174个Python源码指纹与2026-09-15服务器诊断记录一致。源码、来源与逐文件指纹见[provenance](provenance/)。
- 共享树通过原有训练代码、初始化、采样器、增强轨迹及评估等价门禁；相关历史审计的路径与指纹见[prerequisite_report_references.json](provenance/prerequisite_report_references.json)。
- 模型主体对齐公开Original CMX；S2D模态读取、训练基础设施、评估adapter及Perspective REL+适配属于项目代码。

源码树内README、协议文档中的“future/未授权/未完成”是原版本历史文字，为保持源码快照未改写。**当前冻结状态以本README、2026-09-14结果JSON及本目录审计为准。**

## 6. Checkpoint身份与环境

| Arm | epoch200 checkpoint SHA-256 |
|---|---|
| CMX-RGBD | `508bd8ab921050675dae4d316f68438a48f2cd2aa710fd8a0126b878e11fd7d2` |
| CMX-HHA | `3d1f85fb78b36ee06c65e586fd314cbd6df2762486639408d530dd429ca367db` |
| CMX-REL+ | `ed249f4365fa1f1bd31fcc9e50803b348f4e18c732e1bb5735cffa6de7ecdaa2` |

三个checkpoint的原路径、字节数和已记录时间见[checkpoints.json](provenance/checkpoints.json)。权重未上传GitHub；文件hash用于将结果与服务器上的具体模型对应。

已记录运行环境：Python3.8.16、PyTorch1.8.2、CUDA11.1、cuDNN8005、NumPy1.21.6、OpenCV4.5.5、SciPy1.7.3。[环境证据](provenance/observed_environment.json)。源码中的`requirements.txt`是历史依赖列表，含下限约束，不是完整可重建的环境锁文件。

## 7. 查看与核验

```text
0915三臂结果/
├── README.md
├── code/
│   ├── RELPlusv2.3/                 # REL+实际历史源码快照
│   └── CMX-S2D-ThreeArm-v1/        # RGBD/HHA共享源码快照
├── evaluation/                     # RGBD/HHA实际评估adapter
├── results/                        # 总表及三臂原始指标、混淆矩阵
├── provenance/                     # 参数、来源、配置与审计记录
├── SHA256SUMS                      # 本次冻结文件清单
└── verify_archive.py               # 无需GPU的归档核验
```

进入本目录后运行：

```bash
python3 verify_archive.py
```

核验内容：全部文件SHA-256、从三臂混淆矩阵重算mIoU/PA/MA、有效像素数及真实标签直方图一致性。它只读本目录，不运行模型，不启动训练。

训练实际通过各树的`tools/run_with_config.py`加载指定配置后进入`train.py`。HHA实际启动命令和resolved配置见[provenance/training/HHA](provenance/training/HHA/)。记录中的绝对路径是历史服务器位置；在其他机器复现需准备相同数据、权重、审计依赖和独立输出目录，不能直接把默认`config.py`当作S2D三臂入口。

本归档冻结已完成结果，不是包含数据、权重、完整环境镜像的即开即跑分发包。正式三臂模型和服务未因本次归档而修改。

原始材料范围：三臂均包含配置源码、精确参数摘要和原始结果文件；HHA另附当时的resolved运行配置与启动报告。RGBD/REL+的resolved运行配置原件及若干等价审计全文目前仅保留服务器路径/指纹引用，未混充为本地已封存文件。来源详情见[archive_completeness.json](provenance/archive_completeness.json)。这不影响本目录对现有代码、参数与指标的固定，但限制了完整运行环境的复现。
