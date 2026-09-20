"""Recompute downloaded confusion matrices and compare the frozen 0915 baseline."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
EVIDENCE = WORK / 'remote_evidence'
LABELS = {'rgbd': 'RGBD', 'hha': 'HHA', 'relplus': 'RELPlus'}


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recompute(folder):
    metrics = load(folder / 'metrics.json')
    with (folder / 'confusion_matrix.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r['true_class']) for r in rows] == list(range(13))
    matrix = [[int(r['pred_%d' % c]) for c in range(13)] for r in rows]
    assert all(v >= 0 for row in matrix for v in row)
    gt = [sum(row) for row in matrix]
    pred = [sum(row[c] for row in matrix) for c in range(13)]
    diagonal = [matrix[i][i] for i in range(13)]
    iou = [diagonal[i] / (gt[i] + pred[i] - diagonal[i]) for i in range(13)]
    values = {'mIoU': sum(iou) / 13, 'pixel_accuracy': sum(diagonal) / sum(gt),
              'mean_accuracy': sum(diagonal[i] / gt[i] for i in range(13)) / 13}
    assert sum(gt) == metrics['valid_pixel_count'] == 3973620198
    for key, expected in values.items():
        assert math.isclose(metrics[key], expected, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(metrics[key+'_percent'], expected*100, rel_tol=0, abs_tol=1e-10)
    assert all(math.isclose(x,y,rel_tol=0,abs_tol=1e-12) for x,y in zip(iou,metrics['per_class_iou']))
    return metrics, gt


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    status = load(EVIDENCE / 'status.json')
    assert status['status'] == 'PASS' and status['completed'] == list(LABELS)
    comparison = load(EVIDENCE / 'three_arm_comparison.json')
    before = load(EVIDENCE / 'preflight.json')
    baseline_archive = load(ROOT / 'work/HHA_epoch200_evaluation_20260914/remote_evidence/three_arm_comparison.json')
    rows, per_class, values = [], [], {}
    shared_gt = None
    for arm, label in LABELS.items():
        new, gt = recompute(EVIDENCE / arm / 'evaluation')
        prior_dir = ROOT / '0915三臂结果/results' / label
        old, old_gt = recompute(prior_dir)
        assert old_gt == gt
        if shared_gt is None: shared_gt = gt
        assert gt == shared_gt == comparison['arms'][arm]['gt_histogram']
        for fn in ['metrics.json','confusion_matrix.csv','per_class_iou.csv']:
            assert sha(prior_dir/fn) == baseline_archive['arms'][label]['evaluation_artifact_sha256'][fn]
        audit = load(EVIDENCE / arm / 'integrity_audit.json')
        assert audit['status'] == 'PASS' and audit['metrics'] == new
        assert audit['test_source_sha256'] == baseline_archive['test_source_sha256']
        assert (EVIDENCE / arm / 'exitcode').read_text().strip() == '0'
        runtimes = [load(EVIDENCE / arm / ('rank_%02d_runtime.json' % rank)) for rank in range(8)]
        for rank, runtime in enumerate(runtimes):
            assert runtime['status'] == 'COMPLETED'
            assert runtime['processed_samples'] == len(range(rank,17593,8))
            assert runtime['state_key_count'] == 837 and runtime['checkpoint_payload_epoch'] == 200
            assert next(b for b in runtime['batch_norm'] if b['name']=='decode_head.linear_fuse.1')['eps'] == 1e-5
        values[arm] = {'new':new,'old':old,'checkpoint_sha256':audit['checkpoint_sha256']}
        rows.append({'arm':label,'old_mIoU_percent':old['mIoU_percent'],
                     'new_mIoU_percent':new['mIoU_percent'],
                     'delta_mIoU_pp':new['mIoU_percent']-old['mIoU_percent'],
                     'pixel_accuracy_percent':new['pixel_accuracy_percent'],
                     'mean_accuracy_percent':new['mean_accuracy_percent']})
    names = comparison['class_names']
    for i, name in enumerate(names):
        row = {'class_id':i,'class_name':name}
        for arm,label in LABELS.items():
            new = values[arm]['new']['per_class_iou_percent'][i]
            old = values[arm]['old']['per_class_iou_percent'][i]
            row.update({label+'_new_IoU_percent':new,label+'_old_IoU_percent':old,label+'_delta_pp':new-old})
        row['RELPlus_minus_HHA_pp'] = row['RELPlus_new_IoU_percent']-row['HHA_new_IoU_percent']
        row['RELPlus_minus_RGBD_pp'] = row['RELPlus_new_IoU_percent']-row['RGBD_new_IoU_percent']
        per_class.append(row)
    write_csv(WORK/'new_vs_0915_metrics.csv',rows)
    write_csv(WORK/'new_vs_0915_per_class.csv',per_class)
    deltas=comparison['differences_percentage_points']
    improved=sum(r['delta_mIoU_pp']>0 for r in rows)
    desc=('三臂mIoU均高于旧0915配方。' if improved==3 else
          '三臂mIoU均低于旧0915配方，本轮没有支持这组参数提升性能。' if improved==0 else
          '三臂中%d臂mIoU高于旧配方，收益并不一致。' % improved)
    result={'status':'PASS_LOCAL_INDEPENDENT_COMPARISON','new_evaluation_status':status['status'],
            'rows':rows,'new_pairwise_differences_pp':deltas,'old_and_new_gt_histogram_equal':True,
            'gt_histogram':shared_gt,'test_source_sha256':baseline_archive['test_source_sha256'],
            'new_checkpoint_sha256':{a:v['checkpoint_sha256'] for a,v in values.items()},
            'summary':desc,'single_seed':12345,'batch_and_lr_changed_together':True,
            'decoder_BN_eps_eval':1e-5,'decoder_BN_eps_training':1e-3,
            'evaluation_elapsed_seconds':status['finished_at']-status['started_at']}
    (WORK/'comparison_verified.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# CMX S2D 新参数三臂：固定epoch200完整评估（2026-09-20）','',
           '**三臂全量评估及完整性复核均PASS。** '+desc,'',
           '新训练配方为global batch56、lr1.2e-4；旧0915为batch8、lr6e-5。两者都使用seed12345、固定200轮、同一17593张测试图。','',
           '## 1. 新旧主终点对照','',
           '| 模型 | 旧0915 mIoU（%） | 新配方mIoU（%） | 差值（百分点） | 新Pixel Accuracy（%） | 新Mean Accuracy（%） |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| {arm} | {old_mIoU_percent:.4f} | {new_mIoU_percent:.4f} | {delta_mIoU_pp:+.4f} | {pixel_accuracy_percent:.4f} | {mean_accuracy_percent:.4f} |'.format(**row))
    lines += ['', '新配方内，REL+−HHA为 **%+.4f个百分点**，REL+−RGBD为 **%+.4f个百分点**。' %
              (deltas['relplus-hha']['mIoU_percent'],deltas['relplus-rgbd']['mIoU_percent']), '',
              '## 2. 新配方逐类别结果','',
              '| 类别 | RGBD IoU（%） | HHA IoU（%） | REL+ IoU（%） | REL+−HHA（百分点） |','|---|---:|---:|---:|---:|']
    for row in per_class:
        lines.append('| {class_name} | {RGBD_new_IoU_percent:.4f} | {HHA_new_IoU_percent:.4f} | {RELPlus_new_IoU_percent:.4f} | {RELPlus_minus_HHA_pp:+.4f} |'.format(**row))
    lines += ['', '完整逐类新旧差值见CSV。', '', '## 3. 执行和完整性证据','',
              '- 三臂均只读取新训练的epoch-200.pth；评估过程中没有训练、反向传播、optimizer更新或权重覆盖。',
              '- 每臂exit0，8rank均完成；每臂17593张唯一样本，顺序分片严格等于test_ids[rank::8]，计数2200+7×2199。',
              '- 每臂3973620198有效像素；新三臂和旧三臂的13类GT像素直方图完全相同。',
              '- rank混淆矩阵求和与CSV一致，服务端和本地均独立重算mIoU、Pixel Accuracy、Mean Accuracy及逐类IoU。',
              '- 三个checkpoint的路径、size、mtime和SHA前后不变；训练bundle文件、评估实现及数据证据指纹核验通过。',
              '- 837个模型state keys完整匹配；浮点权重和推理logits均检查有限性。',
              '- 输入处理复用冻结loader和evaluator；RGBD/HHA adapter仅改变源码定位路径。实际import路径均来自本次训练source快照。',
              '- 推理：480×480整图、scale1/no-flip、align_corners=false、float32张量、未启用AMP、TF32保持原环境默认并记录、8卡×batch1、无补齐sampler。',
              '- 总评估与串行审计耗时约%.2f分钟。' % (result['evaluation_elapsed_seconds']/60), '',
              '## 4. 结论边界','',
              '这是单seed、固定测试协议的描述性比较。batch与学习率同时改变，优化器更新次数由1322600降至189000；不能把结果单独归因于某个参数，不能据此宣称跨seed稳定性或统计显著性。', '',
              '保留与0915相同的criterion=None评估构造：decoder BN eps实际为1e-5，而训练为1e-3。此差异未在本轮修正，影响量未知；没有重新前向评估或改写旧0915结果。HHA通道物理来源、K/gravity完整provenance的历史缺项仍保留。', '',
              '## 5. 权重与来源','', '| 臂 | epoch200 checkpoint SHA-256 |','|---|---|']
    for arm,label in LABELS.items():lines.append('| %s | `%s` |' % (label,values[arm]['checkpoint_sha256']))
    lines += ['', '远端评估证据根：`'+str(before['suite']['output_root'])+'/evaluation_epoch200_20260920_attempt1`。',
              '远端训练源码：`'+before['suite']['remote_source_root']+'/source`。', '',
              '## 6. 本地证据入口','']
    for label,path in [('验证后的新旧对照',WORK/'comparison_verified.json'),('新旧指标CSV',WORK/'new_vs_0915_metrics.csv'),
                       ('逐类新旧CSV',WORK/'new_vs_0915_per_class.csv'),('服务端三臂原始比较',EVIDENCE/'three_arm_comparison.json'),
                       ('全量评估状态',EVIDENCE/'status.json'),('数据、配置与权重预检',EVIDENCE/'preflight.json'),
                       ('本次评估协议',WORK/'PROTOCOL.json'),('传输核验',WORK/'TRANSFER_VERIFICATION.json')]:
        lines.append('- [%s](<%s>)' % (label,path))
    report=ROOT/'CMX_S2D_batch56_lr12_epoch200评估与三臂对照_20260920.md'
    report.write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False));print('REPORT',report)


if __name__=='__main__':main()
