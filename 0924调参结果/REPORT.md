# focal_gamma=1.5：HHA / REL+ 双臂最终结果

**训练与评估完成并复核PASS；本轮参数候选晋级FAIL。** 两臂于北京时间2026-09-27 12:03:45完成全部流程，2026-09-27 14:41:49完成服务器最终只读复核，本地独立复算PASS。

以最新冻结0920调参结果的gamma1为主基线，只调整focal_gamma=1.5。HHA mIoU回升0.28744776个百分点，但REL+下降0.24540479个百分点，未满足“REL+高于gamma1且HHA不低于gamma1”的共同配方候选标准。本次将gamma1.5配方单独冻结归档；保留gamma1/gamma2归档与原候选筛选结论，不启动其他实验。

| 模型 | gamma1 mIoU (%) | gamma1.5 mIoU (%) | 1.5−1（百分点） | gamma2 mIoU (%) | 1.5−2（百分点） |
|---|---:|---:|---:|---:|---:|
| HHA | 60.18289885 | 60.47034661 | +0.28744776 | 60.98827182 | -0.51792521 |
| REL+ | 61.40637240 | 61.16096760 | -0.24540479 | 61.01689760 | +0.14407001 |

当前REL+−HHA为+0.69062100个百分点；相对gamma1的+1.22347355，差距缩小0.53285255个百分点。此差值不替代同臂提升判断。

## 执行与评估证据

两臂均完成200epochs、189000次更新、8rank、exit0、NaN替换0。HHA于2026-09-25 23:29:41完成，REL+于2026-09-27 11:58:01完成；总训练73.0106小时，最终双臂评估及审计约5.74分钟。服务器最后检查时GPU空闲，原supervisor已退出。

原S2D Area1/2/3/4/6训练52903图，Area5a/5b测试17593图，13类、seed12345、batch56、lr0.00012及其余冻结参数不变。使用各臂固定epoch200权重，480×480整图、scale1/no-flip，不进行测试集checkpoint筛选。保持原criterion=None评估构造与原BN语义（decoder训练eps1e-3，评估eps1e-5）。

远端只读复核309项PASS：206文件bundle指纹、18个输入/前置证据文件指纹、两臂新权重与四个历史权重SHA保持一致；训练完成记录、实际gamma1.5和相同初始化、完整配置继承关系均通过。每臂17593唯一测试样本、精确有序8rank分片、3973620198有效像素、混淆矩阵求和、逐类IoU/mIoU/PA/mACC复算、GT直方图与gamma1/gamma2一致。

取回119份完成输出与2份列表，共121文件、10436929字节，逐文件SHA传输校验PASS。本地独立标准库复算1073项检查、0失败，亦为PASS；本地不含权重字节，权重实物SHA由本次远端只读复核确认。

## 逐类观察与科学边界

REL+相对gamma1，sofa IoU下降10.32797547个百分点，对13类mIoU贡献−0.79445965；board下降2.33923102，对mIoU贡献−0.17994085。door、clutter、bookcase等改善部分抵消下降，最终净差为−0.24540479。该算术分解不证明变化的因果机制。

这是单seed的开发性参数对照，不能据此认定跨seed稳定优劣；多次使用Area5指导参数选择后，该测试结果也不是完全独立于选参的泛化证据。执行PASS说明实验按协议完成；候选FAIL说明未达到预设性能条件，二者分开记录。本目录按用户授权冻结参数、代码、结果与证据并用于GitHub发布；没有额外seed、自动重试或额外参数实验。

## 路径与复核入口

- 原冻结基线：0920调参结果，提交c8ff74502cf6717fc554adc47db4d14e9ddd42a9。
- 本轮远端bundle：`/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-FG15-20260924`。
- 本轮远端输出：`/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_FG15_20260924/attempt1`。
- HHA epoch200 SHA256：`cb5d1d9a456a32f98952f9b1eb429e1393493988f17218511436042e889ac405`。
- REL+ epoch200 SHA256：`1812ce2f4c1109b1ce5308a51e5a3aa06376a9f6abc8ffc43194ac6658358be6`。
- [最终原始对照](evidence/completed/evaluation_epoch200/dual_arm_comparison.json)
- [服务器最终只读复核](evidence/completed/LIVE_FINAL_RECHECK.json)
- [本地独立复算](evidence/completed/INDEPENDENT_OFFLINE_AUDIT.json)
- [传输校验](evidence/completed/TRANSFER_VERIFICATION.json)
- [总体指标CSV](results/gamma15_vs_baselines_metrics.csv) · [逐类IoU及差值CSV](results/gamma15_per_class.csv)
- [完整实验协议](code/training/PROTOCOL.md)
