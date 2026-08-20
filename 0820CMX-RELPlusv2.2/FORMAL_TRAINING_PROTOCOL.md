# 正式训练协议

## 启动顺序

`train.py` 在 Engine、logger、TensorBoard、model 和 optimizer 之前执行：

1. `training_authorized == True`；
2. cache audit report 存在且 `status == PASS`；
3. integration/representation protocol、总数、train/test 数和 failure count 全部一致；
4. runtime 再核对 train/test list 的真实非空行数。

正式配置仍未授权，因此当前直接启动会 fail closed。

## 共享 runtime

`utils/training_runtime.py::build_training_runtime()` 同时供正式入口和 no-step smoke 使用，构造 Engine 配套 logger、作者 seed、DataLoader、Original CMX MiT-B2、预训练初始化、Focal、AdamW 和 WarmUpPolyLR。single-GPU 默认 `local_rank=0, world_size=1`；正式 DDP 才覆盖并启用 SyncBatchNorm/DDP。

## 样本与 seed

- actual train=52,903；logical=52,904；padding=1；global batch=8；iterations/epoch=6,613。
- sampler 每 epoch 仅做一次 O(N) shuffle，支持 `set_epoch(epoch)`，然后以 stride 按 rank 分配。
- seed 公式保持作者源码：`base_seed + epoch + rank * 1000`（仅 distributed 时加入 rank 项）。
- cuDNN：`benchmark=False`、`deterministic=False`，与已核对作者行为一致。

## loss 与更新

FocalLoss2d 使用 gamma=2、ignore=255、`reduction='none'`，训练循环再 `mean()`。日志累计使用 `float(loss.detach().item())`，不会持有历史计算图。作者源码确实在 NaN 时执行 `torch.nan_to_num(..., nan=0.0)`；V2.2 复刻并记录替换数量。

本轮的 single-GPU 与 DDP mock 仅完成一个真实 batch 的 forward/loss/backward。执行了 backpropagation，但没有执行 optimizer update、scheduler step、epoch loop 或 checkpoint replacement；这不是 retraining，也没有产生训练结果。

## 正式训练尚未执行

未运行 `optimizer.step()`，未启动 200 epochs，未生成正式 checkpoint。下一阶段只有在用户单独批准 full cache、正式 cache 审计 PASS，再单独批准训练后才可进入。
