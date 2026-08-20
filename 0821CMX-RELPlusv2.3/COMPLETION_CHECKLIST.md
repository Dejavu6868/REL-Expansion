# CMX-REL+ V2.3 完成检查

- [x] V2.2 保留不动，V2.3 位于独立指定目录。
- [x] REL+ v2.1 generator/core 未修改；合成 + 12 个真实样本 byte
  regression 的 changed pixel/channel/max difference 均为 0。
- [x] auditor duplicate 检查为 Counter O(N)，并通过 70,496 synthetic ID
  性能测试。
- [x] cache/audit/preflight/training gate 绑定实际绝对路径、协议、manifest
  和 ordered train/test identity；负向身份测试 fail closed。
- [x] full-cache 原子写入、严格 resume、失败状态和 damaged/shape recovery
  测试通过。
- [x] 70,496 cache 全部生成；train/test 52,903/17,593；failure 0。
- [x] 完整 cache audit PASS；70 个风险分层样本重新生成差异全零。
- [x] 70,496 样本完整 CMX RGB/label/REL+/ValidMask preflight PASS；
  failure 0。
- [x] checkpoint filename/payload epoch mismatch 会 FAIL。
- [x] evaluator 1/2/8-rank confusion 一致，sweep 为逐 checkpoint
  multi-GPU，fraction/percent 单位明确。
- [x] RawDepth/HHA 全量只读合同均 READY；没有据此启动额外训练。
- [x] 最终 live-source pytest：124 passed，exit code 0。
- [x] 真实 8-GPU DDP smoke：50+3 optimizer updates，四组参数更新，
  disposable checkpoint save/restore 和 LR continuity 全部 PASS。
- [x] 正式 launcher validate-only exit code 0。
- [x] 唯一 seed 12345、8-GPU、200-epoch Original CMX REL+ 正式训练已
  启动，并确认前 150 次更新稳定。
- [ ] 200 epochs 尚未完成；因此训练退出码、正式 checkpoints 和 full-test
  科学评价尚未产生。

最终状态：`FORMAL_TRAINING_STARTED`。

本轮没有启动 RGBD、HHA、原始 REL、多 seed 或其他新实验，也没有生成或
写入文件哈希。
