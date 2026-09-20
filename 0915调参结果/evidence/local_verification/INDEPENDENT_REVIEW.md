# 本次评估实现独立复核

复核对象：eval_rank.py、run_evaluation.py、input_adapter.py、audit_eval.py。

## 来源与实现复核

- RGBD/HHA adapter与0915版本仅SOURCE_ROOT一行不同。
- 实际import路径必须位于新训练source快照；模型/输入处理与冻结evaluator一致。
- 旧launcher需要--use_env；本次已保留。
- 重建配置必须与训练resolved完全相等，且训练resolved SHA保持不变。
- 新输出目录不可复用，失败保留；资源保护只处理本次已验证PID/starttime/PGID。
- 每臂退出0、8rank完成后，独立复算样本覆盖、矩阵、全部指标并校验checkpoint完整性。
- 四个部署Python文件的语法检查通过，远端CPU prelaunch check通过，部署SHA与本地一致。

## 审计器合成数据检查

独立审计Agent用17593样本合成fixture验证：合法输入PASS；错误count、浮点矩阵、JSON字符串矩阵、rank与CSV矩阵不一致、shard顺序反转、错误mIoU、checkpoint改变均被拒绝。临时fixture已清理。此项只证明这些工程检查行为，不代表评估模型性能或跨环境复现。

## 已解决问题

启动前检查发现data_evidence字段名应为data_evidence_files；训练前不存在、训练后产生的smoke报告没有历史SHA，比较条件应为prior.exists=true。两项均在正式启动前修复并经远端预检验证，未产生失败GPU评估。

## 旧基线本地复核

独立Agent从0915三臂结果/results/{RGBD,HHA,RELPlus}的原始混淆矩阵重算指标，与metrics及per_class一致；文件SHA与旧封存三臂比较JSON一致。三臂GT直方图相同，有效像素3973620198。本轮没有重跑或更改旧基线。
