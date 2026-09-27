from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,shutil
P=Path(__file__).resolve().parents[1]
E=P/'evidence/completed_20260927'
J=lambda p:json.loads(Path(p).read_text())
C=J(E/'evaluation_epoch200/dual_arm_comparison.json');L=J(E/'LIVE_FINAL_RECHECK.json');A=J(E/'INDEPENDENT_OFFLINE_AUDIT.json');T=J(E/'TRANSFER_VERIFICATION.json');S=J(P/'suite.json')
assert L['status']=='PASS' and A['status'].startswith('PASS') and T['status']=='PASS'
assert C['status']=='PASS_FIXED_EPOCH200_DESCRIPTIVE_COMPARISON'
TZ=timezone(timedelta(hours=8))
stamp=lambda t:datetime.fromtimestamp(t,TZ).strftime('%Y-%m-%d %H:%M:%S')
rows=[]
for arm,label in [('hha','HHA'),('relplus','REL+')]:
 v1=C['baseline_gamma1']['metrics'][arm]['mIoU_percent'];v15=C['arms'][arm]['metrics']['mIoU_percent'];v2=C['baseline_gamma2']['metrics'][arm]['mIoU_percent']
 rows.append(f'| {label} | {v1:.8f} | {v15:.8f} | {v15-v1:+.8f} | {v2:.8f} | {v15-v2:+.8f} |')
gap=C['differences_percentage_points']['relplus-hha']['mIoU_percent'];oldgap=C['baseline_gamma1']['relplus_minus_hha_pp']['mIoU_percent'];hours=sum(v['hours'] for v in L['arm_times'].values())
weights={r['arm']:r for r in L['weights'] if r['gamma']==1.5}
report=f"""# focal_gamma=1.5：HHA / REL+ 双臂最终结果

**训练与评估完成并复核PASS；本轮参数候选晋级FAIL。** 两臂于北京时间{stamp(L['pipeline_finished_at'])}完成全部流程，{stamp(L['checked_at'])}完成服务器最终只读复核，本地独立复算PASS。

以最新冻结0920调参结果的gamma1为主基线，只调整focal_gamma=1.5。HHA mIoU回升0.28744776个百分点，但REL+下降0.24540479个百分点，未满足“REL+高于gamma1且HHA不低于gamma1”的共同配方候选标准。原gamma1/gamma2冻结结果不变，不自动替换冻结配方或启动其他实验。

| 模型 | gamma1 mIoU (%) | gamma1.5 mIoU (%) | 1.5−1（百分点） | gamma2 mIoU (%) | 1.5−2（百分点） |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

当前REL+−HHA为{gap:+.8f}个百分点；相对gamma1的{oldgap:+.8f}，差距缩小{oldgap-gap:.8f}个百分点。此差值不替代同臂提升判断。

## 执行与评估证据

两臂均完成200epochs、189000次更新、8rank、exit0、NaN替换0。HHA于{stamp(L['arm_times']['hha']['finished_at'])}完成，REL+于{stamp(L['arm_times']['relplus']['finished_at'])}完成；总训练{hours:.4f}小时，最终双臂评估及审计约{L['evaluation_minutes']:.2f}分钟。服务器最后检查时GPU空闲，原supervisor已退出。

原S2D Area1/2/3/4/6训练52903图，Area5a/5b测试17593图，13类、seed12345、batch56、lr0.00012及其余冻结参数不变。使用各臂固定epoch200权重，480×480整图、scale1/no-flip，不进行测试集checkpoint筛选。保持原criterion=None评估构造与原BN语义（decoder训练eps1e-3，评估eps1e-5）。

远端只读复核{L['checks_count']}项PASS：206文件bundle指纹、18个输入/前置证据文件指纹、两臂新权重与四个历史权重SHA保持一致；训练完成记录、实际gamma1.5和相同初始化、完整配置继承关系均通过。每臂17593唯一测试样本、精确有序8rank分片、3973620198有效像素、混淆矩阵求和、逐类IoU/mIoU/PA/mACC复算、GT直方图与gamma1/gamma2一致。

取回{T['output_files']}份完成输出与{T['input_files']}份列表，共{T['output_files']+T['input_files']}文件、{T['total_bytes']}字节，逐文件SHA传输校验PASS。本地独立标准库复算也已PASS；本地不含权重字节，权重实物SHA由本次远端只读复核确认。

## 逐类观察与科学边界

REL+相对gamma1，sofa IoU下降10.32797547个百分点，对13类mIoU贡献−0.79445965；board下降2.33923102，对mIoU贡献−0.17994085。door、clutter、bookcase等改善部分抵消下降，最终净差为−0.24540479。该算术分解不证明变化的因果机制。

这是单seed的开发性参数对照，不能据此认定跨seed稳定优劣；多次使用Area5指导参数选择后，该测试结果也不是完全独立于选参的泛化证据。执行PASS说明实验按协议完成；候选FAIL说明未达到预设性能条件，二者分开记录。没有新Git发布、额外seed、自动重试或额外参数实验。

## 路径与复核入口

- 原冻结基线：0920调参结果，提交c8ff74502cf6717fc554adc47db4d14e9ddd42a9。
- 本轮远端bundle：`{S['remote_source_root']}`。
- 本轮远端输出：`{S['output_root']}`。
- HHA epoch200 SHA256：`{weights['hha']['sha256']}`。
- REL+ epoch200 SHA256：`{weights['relplus']['sha256']}`。
- [最终原始对照](evidence/completed_20260927/evaluation_epoch200/dual_arm_comparison.json)
- [服务器最终只读复核](evidence/completed_20260927/LIVE_FINAL_RECHECK.json)
- [本地独立复算](evidence/completed_20260927/INDEPENDENT_OFFLINE_AUDIT.json)
- [传输校验](evidence/completed_20260927/TRANSFER_VERIFICATION.json)
- [总体指标CSV](RESULTS_metrics.csv) · [逐类IoU及差值CSV](RESULTS_per_class.csv)
- [完整实验协议](PROTOCOL.md)
"""
(P/'RESULTS.md').write_text(report)
old_status=(P/'status.json').read_bytes();(E/'local_status_before_completion.json').write_bytes(old_status)
status={'status':'PASS_TRAINING_EVALUATION_VERIFIED','suite_id':S['suite_id'],'focal_gamma':1.5,'sole_changed_scientific_parameter':'focal_gamma','baseline_commit':'c8ff74502cf6717fc554adc47db4d14e9ddd42a9','training_completion':'PASS','evaluation_completion':'PASS','candidate_screening':C['candidate_screening'],'finished_at':stamp(L['pipeline_finished_at']),'final_remote_recheck_at':stamp(L['checked_at']),'local_offline_audit_status':A['status'],'remote_checks_count':L['checks_count'],'active_arm':None,'queued_arms':[],'remote_bundle':S['remote_source_root'],'remote_output':S['output_root'],'metrics':{a:C['arms'][a]['metrics'] for a in ['hha','relplus']},'same_arm_differences_vs_gamma1_pp':C['gamma1_5_minus_gamma1_percentage_points'],'same_arm_differences_vs_gamma2_pp':C['gamma1_5_minus_gamma2_percentage_points'],'relplus_minus_hha_pp':gap,'weights':weights,'report':'RESULTS.md','evidence_root':'evidence/completed_20260927','github_published':False,'new_experiments_started':False}
(P/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
H=P.parents[1]/'HANDOFF.md';previous=H.read_text();h=hashlib.sha256(previous.encode()).hexdigest()
header=f"""# 最新完成交接：focal_gamma=1.5 双臂训练与评估完成，完整性PASS、候选晋级FAIL（2026-09-27）

本节覆盖下方历史RUNNING/QUEUED快照。用户授权的gamma1.5 HHA/REL+实验已于北京时间{stamp(L['pipeline_finished_at'])}全部完成；{stamp(L['checked_at'])}远端只读复核{L['checks_count']}项PASS，本地独立复算PASS。**不要重复启动训练或评估。**

| 模型 | gamma1 mIoU (%) | gamma1.5 mIoU (%) | 1.5−1（百分点） | gamma2 mIoU (%) | 1.5−2（百分点） |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

本轮REL+−HHA为+{gap:.8f}个百分点。HHA相对gamma1回升，但REL+下降，因此预设候选晋级FAIL；运行和评估均PASS。原冻结gamma1/gamma2配方及结果保持不变，未发布新Git归档，不自动开启下一实验。单seed开发性结果，不能宣称稳定优劣。

- 两臂各200轮/189000更新/8rank/exit0/NaN0；固定epoch200原17593测试样本、精确分片、混淆矩阵与指标独立复算PASS。
- 206bundle文件、18输入/前置证据文件、两新四旧权重SHA当前复核PASS；121份完成/列表证据传输SHA通过。
- 新HHA权重SHA `{weights['hha']['sha256']}`；新REL+权重SHA `{weights['relplus']['sha256']}`。
- 远端输出：`{S['output_root']}`，评估在`evaluation_epoch200`；该队列已退出，GPU最后检查空闲。
- 本地结果：`work/CMX_S2D_FOCAL15_DUAL_20260924/RESULTS.md`，总体/逐类CSV同目录；当前`status.json`已更新完成。
- 证据：`work/CMX_S2D_FOCAL15_DUAL_20260924/evidence/completed_20260927/`，含`LIVE_FINAL_RECHECK.json`、`INDEPENDENT_OFFLINE_AUDIT.json`、传输清单与原始逐rank证据。

---

<!-- PREVIOUS_HANDOFF_SHA256 {h} -->

"""
assert not previous.startswith('# 最新完成交接：focal_gamma=1.5')
H.write_text(header+previous)
for d in sorted(P.rglob('__pycache__'),key=lambda x:len(x.parts),reverse=True):shutil.rmtree(d)
print(json.dumps({'status':status['status'],'candidate':C['candidate_screening']['status'],'completed_at':status['finished_at'],'report':str(P/'RESULTS.md')},ensure_ascii=False))
