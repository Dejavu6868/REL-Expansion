# REL+ v2.1 → CMX-REL+ V2.2 差异

| 项 | V2.1 | V2.2 | 是否改变 REL+ 三通道 |
|---|---|---|---|
| generator | frozen | frozen | 否 |
| Dataset indexing | per-item expansion 风险 | sampler 管理 | 否 |
| formal evaluator | smoke only | full evaluator | 否 |
| cache | pilot only | full generator/auditor code | 否 |
| endpoint | 未冻结 | epoch 200 + test-selected best | 否 |
| single GPU | `local_rank` 启动风险 | 完整默认值与 no-step 验证 | 否 |
| logging | formatter/路径不完整 | `train.log` 真实落盘，拒绝自指链接 | 否 |
| three-arm trace | 构造性复制 | 三个真实 DataLoader 独立 trace | 否 |

V2.2 修改的是启动、索引、审计、评价与可追溯性。`rel_plus/generator.py`、`encoding.py`、`depth.py`、`camera.py`、`source_helpers.py`、normal helper 和 storage contract 均未修改。

字节回归覆盖固定合成输入和 12 个真实 S2D 样例；总计 changed pixel=0、changed channel=0、max difference=0。任何未来改变三通道字节的方案必须使用新方法名，不能并入 V2.2。
