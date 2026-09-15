#!/usr/bin/env python3
"""Check archived bytes and recompute the three fixed-epoch metrics (stdlib only)."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
errors=[]
count=0
listed=set()
for line in (ROOT/'SHA256SUMS').read_text().splitlines():
    expected,relative=line.split('  ',1)
    if relative in listed:errors.append('duplicate manifest entry: '+relative)
    listed.add(relative)
    path=ROOT/relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        errors.append('hash mismatch: '+relative)
    count+=1
actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
if actual!=listed:
    errors.append('unlisted or missing files: '+str(sorted(actual.symmetric_difference(listed))))
comparison=json.loads((ROOT/'results/three_arm_comparison.json').read_text())
metrics={}
histograms=[]
for arm in ('RGBD','HHA','RELPlus'):
    with (ROOT/'results'/arm/'confusion_matrix.csv').open() as stream:
        values=list(csv.reader(stream))
    # Evaluator writes a header and a leading true-class index column.
    if values[0][0]=='true_class':
        assert [int(row[0]) for row in values[1:]]==list(range(13))
        matrix=[[int(value) for value in row[1:]] for row in values[1:]]
    else:
        matrix=[[int(value) for value in row] for row in values]
    if len(matrix)!=13 or any(len(row)!=13 for row in matrix):
        raise ValueError('unexpected confusion matrix shape: '+arm)
    gt=[sum(row) for row in matrix]
    pred=[sum(matrix[i][j] for i in range(13)) for j in range(13)]
    diagonal=[matrix[i][i] for i in range(13)]
    ious=[diagonal[i]/(gt[i]+pred[i]-diagonal[i]) for i in range(13)]
    calculated=dict(mIoU=sum(ious)/13,pixel_accuracy=sum(diagonal)/sum(gt),
                    mean_accuracy=sum(diagonal[i]/gt[i] for i in range(13))/13)
    expected=comparison['arms'][arm]['metrics']
    for key,value in calculated.items():
        if not math.isclose(value,expected[key],rel_tol=0,abs_tol=1e-12):
            errors.append(arm+' metric mismatch: '+key)
    if sum(gt)!=3973620198:errors.append(arm+' valid pixel count mismatch')
    histograms.append(gt)
    metrics[arm]=calculated
if histograms[0]!=histograms[1] or histograms[1]!=histograms[2]:
    errors.append('ground-truth histograms differ')
print(json.dumps(dict(status='PASS' if not errors else 'FAIL',hashed_files=count,
    recomputed_metrics=metrics,errors=errors),indent=2))
raise SystemExit(1 if errors else 0)
