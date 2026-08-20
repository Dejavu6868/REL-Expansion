# CMX-REL+ V2.2 实现与验证报告

## 状态

CMX-REL+ V2.2 代码修复已完成，具备提交用户审批 full cache 与正式训练的条件。这里的“完成”仅指代码和被允许的 pilot/no-step 验证闭环，不表示 full cache、正式训练或科学评价已经完成。

## 主线与问题

- 主线阶段：V2.2 训练前基础设施与授权门禁。
- 未来科学问题：在冻结 Original CMX、数据、增强、loss、scheduler 和 endpoint 时，仅改变 X 模态，REL+ 相对 RGBD/HHA 的表现如何。
- 本轮改变因素：启动、Dataset/sampler、cache generator/auditor、full evaluator、checkpoint endpoint、真实三臂 trace 和兼容性。
- 冻结控制/数据：REL+ v2.1 三通道表示、S2D official train/test、Original CMX MiT-B2、no-flip、13 类与 ignore=255。
- 主 endpoint：epoch 200；次 endpoint：`test_selected_best`（有 test-selection bias）。
- 本轮 endpoint：基础设施 PASS/FAIL，不产生科学 mIoU。

## 路径

- v2.1 基线：`/home/zhuzhaoziao/RELPlus/RELPlusv2.1`
- V2.2 代码：`/home/zhuzhaoziao/RELPlus/RELPlusv2.2`
- 验证输出：`/data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v2_2_integration`
- V2.2 formal config：`configs/stanford2d3d_s2d/cmx_mit_b2_rel_plus_v2_2_formal.py`

## 核心不变量

`rel_plus/generator.py` 及冻结几何/编码核心未修改。固定合成输入和 12 个真实 S2D 样例的 byte regression 均为：changed pixels=0、changed channels=0、max difference=0。

## 已修复

1. S2D 使用 `dataset_split`，兼容字段 `dataset_fold=None`，不再误称 Fold 1。
2. Engine 无条件提供 non-DDP `local_rank=0, world_size=1`。
3. 日志 loss 累加转换为 Python float，不保留历史计算图。
4. logger 使用真实 `log_dir/log_file`；修复 FileHandler formatter 实例错误；拒绝自指 symlink。
5. Dataset 只保留 52,903 个真实条目，删除 per-item 全量排列；logical 52,904 由 sampler 管理。
6. full-cache generator、结构/数值 auditor、runtime cache readiness 闭环完成。
7. full evaluator、no-pad rank 划分、all-reduce、prediction 保存与 checkpoint sweep 完成。
8. 三臂 formal config 和三个真实 DataLoader trace 完成；旧构造性入口退役。
9. RNG 类型、Python 3.8+ Iterable、package root、无用 import 和 label uint8 回绕问题修复。
10. invalid=255 仍是 frozen baseline，诊断继续报告但不改变模型输入。

## 实际验证结果

- pytest：108 passed（包含 live-source）；其中新增 V2.2 基础设施测试 15 项。
- byte regression：合成 + 12 个真实样例 PASS，三项差异均为 0。
- pilot cache：36 张 generate PASS；resume 36/36 PASS；30 train / 6 test。
- pilot audit：failure=0；36/36 原始数据重生成逐像素一致。
- cache fault injection：缺失、损坏、shape、dtype、channels、extra 全部检出。
- CMX 数据 preflight：36/36 PASS；resume 合同 36/36 复用。
- formal startup：single-GPU PASS；DDP mock PASS。两者均完成真实 MiT-B2 forward/backward，梯度 finite/nonzero。
- DataLoader：1,000 样本用时 17.87 秒（55.97 samples/s），Dataset per-item randperm=0，sampler 每 epoch randperm=1；formal padding=1。
- evaluator smoke：3 张 pilot、13×13 confusion、raw+palette 共 6 个 PNG，plumbing PASS。
- evaluator rank consistency：1/2/8-rank confusion 逐元素完全一致。
- checkpoint sweep：合成 100/105/200 plumbing PASS；primary=200，secondary=105；缺失 checkpoint 正确报错。
- 三臂真实 trace：每臂 50 项，mismatch=0，未使用构造性复制。

最终汇总应以服务器输出目录中的最新日志和 JSON 为准；本报告不使用文件哈希。

## 本轮明确未执行

本轮未生成 70,496 张全量 REL+ cache，未执行 optimizer.step，未产生正式 checkpoint，未启动 200-epoch 训练，未在 17,593 张完整 test split 上计算或报告科学 mIoU。

本轮的 single-batch 检查执行了 backpropagation，但没有 optimizer update、scheduler-driven training 或 checkpoint replacement；它是 training-startup plumbing 验证，不是 retraining。

## 下一步解锁条件

只有用户明确批准 full cache，且全量 cache audit 为 PASS，才解锁“是否批准正式训练”的下一次独立决策。本轮没有自动扩大授权。
