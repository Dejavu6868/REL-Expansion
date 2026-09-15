#!/usr/bin/env python3
"""Create the final evidence-bound CMX S3D Fold 1 reproduction conclusion."""

import csv
import json
import re
from pathlib import Path

import numpy as np


OUTPUT_ROOT = Path("/data/zhuzhaoziao/RELPlus/outputs/CMX_S3D_Fold1_reproduction")
ARMS = {
    "CMX-RGBD": (OUTPUT_ROOT / "cmx_rgbd_fold1_seed12345", "Raw depth", 59.03),
    "CMX-HHA": (OUTPUT_ROOT / "cmx_hha_fold1_seed12345", "HHA", 63.98),
    "CMX-REL": (OUTPUT_ROOT / "cmx_rel_fold1_seed12345", "REL", 64.47),
}


def diagnosis(value, target):
    difference = abs(value - target)
    if difference <= 0.5:
        return "高度接近"
    if difference <= 1.0:
        return "基本接近"
    if difference <= 2.0:
        return "部分复现，需要解释"
    return "未复现，需要定位"


def loss_summary(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Epoch (\d+)/200 Iter \d+/131:.*?total_loss=([0-9.eE+-]+)", text)
    by_epoch = {}
    for epoch, loss in matches:
        by_epoch[int(epoch)] = float(loss)
    if not by_epoch:
        return {"first": None, "last": None, "minimum": None, "epochs": 0}
    ordered = sorted(by_epoch.items())
    return {
        "first": ordered[0][1],
        "last": ordered[-1][1],
        "minimum": min(by_epoch.values()),
        "epochs": len(by_epoch),
    }


def main():
    data = json.loads((OUTPUT_ROOT / "audit" / "s3d_data_audit.json").read_text(encoding="utf-8"))
    rows = {}
    checkpoint_rows = {}
    losses = {}
    for name, (run_dir, modality, target) in ARMS.items():
        best = json.loads((run_dir / "metrics_best.json").read_text(encoding="utf-8"))
        epoch200 = json.loads((run_dir / "metrics_epoch200.json").read_text(encoding="utf-8"))
        with (run_dir / "metrics_all_checkpoints.csv").open(encoding="utf-8") as handle:
            checkpoint_rows[name] = list(csv.DictReader(handle))
        losses[name] = loss_summary(run_dir / "train.log")
        rows[name] = {
            "run_dir": run_dir,
            "modality": modality,
            "target": target,
            "best": best,
            "epoch200": epoch200,
        }

    best_values = {name: item["best"]["mIoU_percent"] for name, item in rows.items()}
    ranking_correct = best_values["CMX-REL"] > best_values["CMX-HHA"] > best_values["CMX-RGBD"]
    absolute_differences = {name: abs(item["best"]["mIoU_percent"] - item["target"]) for name, item in rows.items()}
    if ranking_correct and all(value <= 1.0 for value in absolute_differences.values()):
        next_stage = "A. 三组 Fold 1 均已冻结，可以进入三折或多 seed"
    elif ranking_correct:
        next_stage = "B. 相对排序正确，但至少一组绝对精度仍需定位"
    elif absolute_differences["CMX-RGBD"] > 2 and max(absolute_differences["CMX-HHA"], absolute_differences["CMX-REL"]) <= 2:
        next_stage = "C. RGBD 输入协议需要修正"
    elif absolute_differences["CMX-HHA"] > 2 and max(absolute_differences["CMX-RGBD"], absolute_differences["CMX-REL"]) <= 2:
        next_stage = "D. HHA 输入协议需要修正"
    elif absolute_differences["CMX-REL"] > 2 and max(absolute_differences["CMX-RGBD"], absolute_differences["CMX-HHA"]) <= 2:
        next_stage = "E. REL 训练或评价链路需要修正"
    else:
        next_stage = "F. 公共训练/评价协议未完全复现"

    result_lines = [
        "# CMX S3D Fold 1 reproduction conclusion", "",
        "## 1. 实验身份", "",
        "Stanford2D3D S3D；Official Fold 1；Original CMX；MiT-B2；MLPDecoder；No Gate；No SMMF；No DyMM；No SGA；single seed 12345。", "",
        "本轮是完整重新训练：执行了反向传播、optimizer 更新并生成了新的 checkpoint；训练前单步更新 checkpoint 不替代正式 checkpoint。", "",
        "## 2. 代码", "",
        "- CMX-RGBD: `/home/zhuzhaoziao/RELPlus/CMX-RGBD`",
        "- CMX-HHA: `/home/zhuzhaoziao/RELPlus/CMX-HHA`",
        "- CMX-REL: `/home/zhuzhaoziao/RELPlus/CMX-REL`",
        "- 三个 `models/` 目录一致；参数量均为 66,567,573；本轮没有修改模型数学定义。",
        "- 统一修改仅包括配置真实生效、Focal Loss、seed、输入读取、训练前门检和分布式评价工具。", "",
        "## 3. 数据", "",
        "- 正式 raw 数据链：`/data/zhuzhaoziao/datasets/Stanford2D3D/no_xyz` archives。",
        "- 正式整理目录：`{}`。".format(data["dataset_root"]),
        "- ERP 总数 1413；Fold 1 train/test 为 1040/373；train/test 交集为 0。",
        "- RGB、Label、Depth、HHA、REL 完整可读交集：{}。".format(data["all_modalities_readable_and_aligned"]),
        "- 原始标签 0 为 ignore、1–13 为类别；loader 的 uint8 减一后为 255 ignore、0–12 类。", "",
        "类别顺序：beam, board, bookcase, ceiling, chair, clutter, column, door, floor, sofa, table, wall, window。", "",
        "## 4. 三种 X 模态", "",
        "- RGBD：uint16 depth 显式取高 8 位并复制三通道；不做逐图 min-max。invalid 65535 编为 255；Original CMX 没有独立 invalid mask。复制后通道相同，但固定 ImageNet channel-wise normalization 后数值不同。",
        "- HHA：使用完整冻结 `nhha` 文件；其 1413 个对应 depth 与正式 no_xyz-derived depth 全部一致。CMX 以 BGR→RGB 读取为 horizontal disparity、height、angle。当前优化生成器与冻结文件不一致，确切历史生成 revision 为 UNKNOWN，候选输出未用于训练。",
        "- REL：当前已验证可执行代码，alpha=45°、lambda=0.5、P1/P99、现有 `~is_horizontal` 与 `[EGVIA, LOA, ReD]` 顺序；使用 unchanged 读取，不执行 BGR2RGB。论文—代码差异已记录，但本项目采用已验证代码行为，差异不阻塞主复现。", "",
        "## 5. 训练与评价协议", "",
        "MiT-B2 ImageNet 初始化同时加载 RGB/X 分支；Focal Loss；AdamW；lr 6e-5；power 0.9；weight decay 0.01；warm-up 10；200 epochs；131 iterations/epoch；逻辑长度 1048；8×RTX 3090、per-GPU batch 1、global batch 8、SyncBN、AMP disabled；crop 1080；六尺度训练增强与同步 mirror；checkpoint 100–200 每 5 epoch；评价 scale 1、no flip、crop 1080、stride 720。", "",
        "## 6. 主结果", "",
        "| 模型 | X 模态 | 论文目标 | 实测 best | Best epoch | Epoch 200 | 与目标差值 | 诊断 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in rows.items():
        best = item["best"]["mIoU_percent"]
        epoch200 = item["epoch200"]["mIoU_percent"]
        result_lines.append(
            "| {} | {} | {:.2f} | {:.3f} | {} | {:.3f} | {:+.3f} | {} |".format(
                name, item["modality"], item["target"], best, item["best"]["checkpoint_epoch"],
                epoch200, best - item["target"], diagnosis(best, item["target"])
            )
        )
    result_lines.extend(
        [
            "", "两两 best mIoU 差值：",
            "- HHA − RGBD = {:+.3f} pp（论文 +4.95）".format(best_values["CMX-HHA"] - best_values["CMX-RGBD"]),
            "- REL − RGBD = {:+.3f} pp（论文 +5.44）".format(best_values["CMX-REL"] - best_values["CMX-RGBD"]),
            "- REL − HHA = {:+.3f} pp（论文 +0.49）".format(best_values["CMX-REL"] - best_values["CMX-HHA"]),
            "- 排序 `REL > HHA > RGBD`：{}。".format("OBSERVED PASS" if ranking_correct else "OBSERVED FAIL"),
            "", "### PAcc、Mean Accuracy 与 checkpoint 波动", "",
            "| 模型 | Best PAcc | Best Mean Accuracy | checkpoint mIoU min | max | std | loss first→last (min) |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name, item in rows.items():
        values = np.array([float(row["mIoU_percent"]) for row in checkpoint_rows[name]])
        loss = losses[name]
        loss_text = "UNKNOWN" if loss["first"] is None else "{:.4f}→{:.4f} ({:.4f})".format(loss["first"], loss["last"], loss["minimum"])
        result_lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
                name, item["best"]["pixel_accuracy_percent"], item["best"]["mean_class_accuracy_percent"],
                values.min(), values.max(), values.std(), loss_text
            )
        )

    result_lines.extend(["", "### Best checkpoint per-class IoU (%)", ""])
    result_lines.append("| Class | RGBD | HHA | REL | REL−HHA |")
    result_lines.append("|---|---:|---:|---:|---:|")
    class_names = rows["CMX-RGBD"]["best"]["class_names"]
    for index, class_name in enumerate(class_names):
        rgbd = rows["CMX-RGBD"]["best"]["iou"][index] * 100
        hha = rows["CMX-HHA"]["best"]["iou"][index] * 100
        rel = rows["CMX-REL"]["best"]["iou"][index] * 100
        result_lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:+.3f} |".format(class_name, rgbd, hha, rel, rel - hha))

    result_lines.extend(
        [
            "", "## 7. 证据分级结论", "",
            "- OBSERVED：上述配置、数据数量、训练日志、checkpoint、confusion matrix 和指标均由本轮产物直接给出。",
            "- INFERENCE：与目标的偏差可能来自输入表示 provenance、公开环境差异或优化随机性；只有与已观察差异相连的推断才保留。",
            "- UNKNOWN：冻结 HHA 文件的确切历史生成 revision/相机参数 provenance 未留存在当前代码目录。",
            "- PROPOSAL：本轮不自动扩展实验；以下判断提交审阅。", "",
            "## 8. 下一阶段判断", "",
            "**{}**".format(next_stage), "",
            "## 9. 停止条件", "",
            "Fold 1 已完成。本任务未启动 Fold 2、Fold 3、多 seed、SMMF、DyMM、REL+、SGA 或额外调参。",
        ]
    )
    (OUTPUT_ROOT / "CMX_S3D_FOLD1_REPRODUCTION_CONCLUSION.md").write_text(
        "\n".join(result_lines) + "\n", encoding="utf-8"
    )

    with (OUTPUT_ROOT / "main_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "x_modality", "paper_target", "best_mIoU", "best_epoch", "epoch200_mIoU", "difference", "diagnosis"])
        for name, item in rows.items():
            best = item["best"]["mIoU_percent"]
            writer.writerow([name, item["modality"], item["target"], best, item["best"]["checkpoint_epoch"], item["epoch200"]["mIoU_percent"], best - item["target"], diagnosis(best, item["target"])])
    print("final_report={}".format(OUTPUT_ROOT / "CMX_S3D_FOLD1_REPRODUCTION_CONCLUSION.md"))


if __name__ == "__main__":
    main()
