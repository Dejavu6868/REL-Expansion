# CMX-REL+ V2.2 完成检查

- [x] REL+ v2.1 generator/core 未修改；合成 + 12 个真实样例 byte regression 全零。
- [x] V2.2 独立目录与 import-safe formal config。
- [x] dataset split、single-GPU rank、loss float、logger、自指 symlink 修复。
- [x] Dataset actual length + fixed-length 1/2/8-rank sampler。
- [x] full-cache generator/auditor 代码和 36 张 generate/resume/audit/fault smoke。
- [x] cache audit report 驱动 data readiness。
- [x] full evaluator、no-pad rank 聚合、prediction 保存、checkpoint sweep。
- [x] primary epoch 200；secondary `test_selected_best` 及 bias 声明。
- [x] shared runtime single-GPU 与 DDP mock no-step backward。
- [x] Focal/seed/scheduler/invalid=255 协议未回退。
- [x] 三个真实 Dataset/DataLoader 的 50 项 trace，mismatch=0。
- [x] Python 3.8、RNG、Iterable、package import 与 label signed mapping。
- [x] `training_authorized=False`、`full_cache_authorized=False`。
- [x] 未执行 full cache、optimizer step、正式 checkpoint、200 epochs 或完整 test mIoU。

结论：代码与允许范围内的验证闭环已完成；full cache 与正式训练仍等待用户分别授权。
