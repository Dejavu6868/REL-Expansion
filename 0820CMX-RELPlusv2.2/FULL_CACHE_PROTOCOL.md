# Full REL+ Cache 协议

## 生成

`tools/generate_full_relplus_cache.py` 读取冻结 manifest，并且只调用冻结的 `load_canonical_frame()` 和 `generate_rel_plus_v2_1()`。它支持 workers、limit、dry-run、resume 和逐样本失败记录；不重新实现 depth decode、K resize、pose、normal 或 REL encoding。

PNG 写入流程为同目录临时文件 → 解码/shape/dtype 验证 → `os.replace()`。resume 同时检查文件存在、可解码、shape、dtype、三/单通道、mask 二值和 manifest protocol；不会只看 sample ID。

`full_cache_authorized=False` 时，只允许 dry-run 或显式 `--limit <= 36`。没有授权的全 manifest 实写会在创建 cache 前拒绝。

## 审计

`tools/audit_full_relplus_cache.py` 检查 manifest/train/test 数、ID 唯一、两类 PNG 文件数、缺失/额外文件、decode、480×480 shape、dtype、通道、mask 二值、invalid→`[255,255,255]`、train/test sample/path 无交叉。

抽样审计从原始 depth/K/pose 重新生成，并逐像素比较。正式输出固定为：

- `cache_audit_summary.json`
- `cache_audit_failures.csv`
- `cache_audit_sample_regeneration.csv`
- `cache_manifest_resolved.csv`

最终状态只有 PASS/FAIL；人工建议放在 `review_notes`，不增加审批状态。

## 本轮 pilot 证据

36 张 pilot 已完成 generate、resume 和 audit；30 train / 6 test，failure=0；36/36 原始数据重生成与 cache 逐像素一致。故障注入也验证了缺文件、损坏 PNG、错误 shape、错误 dtype、错误通道和额外文件均会被发现。

本轮没有生成 70,496 张 full cache。未使用文件哈希；data readiness 由结构和数值审计决定。formal data readiness 仍为 False。
