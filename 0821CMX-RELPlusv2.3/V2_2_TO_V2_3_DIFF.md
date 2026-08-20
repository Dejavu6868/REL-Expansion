# V2.2 to V2.3 difference

| Item | V2.2 | V2.3 | Changes REL+? |
|---|---|---|---|
| Auditor duplicate check | O(N^2) risk | `Counter`, O(N) | No |
| Audit identity | counts and protocol | absolute cache/manifest/split identity plus ordered IDs | No |
| Training gate | cache audit | cache audit plus full CMX preflight | No |
| Regeneration | biased toward low risk | 70 risk-stratified rows, ten per area | No |
| Cache completion | ambiguous on partial failure | exact all-rows/zero-failure predicate | No |
| Cache workers | possible nested OpenCV threads | one OpenCV thread per Python worker | No |
| Writes and resume | basic presence checks | verified atomic PNG and corrupt-file regeneration | No |
| Checkpoint epoch | filename-led | filename expectation must equal payload epoch | No |
| Metric units | implicit | fraction and percent both named | No |
| Evaluation sweep | default single process | parent launches one 8-rank evaluator per checkpoint sequentially | No |
| DDP training smoke | mock or no optimizer step | real NCCL/DDP/SyncBN step, save, restore and resume | No |
| Formal training | not authorized | one run only after every hard gate passes | No |

The model-facing X tensor remains `[EGVIA, LOA, ReD]` under
`RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT` and
`SOURCE_COMPAT_STORAGE_255`.
