import json,os,subprocess,time
from pathlib import Path
root=Path('/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_FG15_20260924/attempt1')
result={'queried_at':time.time(),'root':str(root),'files':{}}
for rel in ['preparation_status.json','preflight.json','smoke_status.json','pipeline_status.json','train_status.json','prepare_supervisor_launch.json','execute_supervisor_launch.json']:
 f=root/rel
 if f.is_file():
  x=json.loads(f.read_text())
  if rel=='preflight.json': x={k:x[k] for k in ['status','suite_sha256','arms','environment'] if k in x}
  result['files'][rel]=x
for arm in ['hha','relplus']:
 for phase in ['smoke','runs']:
  d=root/phase/arm
  if not d.exists():continue
  a={'files':{}}
  for rel in ['arm_status.json','runtime_status.json','ddp_optimizer_smoke_summary.json','initialization_check.json','exitcode']:
   f=d/rel
   if f.is_file():
    try:a['files'][rel]=json.loads(f.read_text())
    except ValueError:a['files'][rel]=f.read_text()
  a['rank_init_count']=len(list(d.glob('rank_init_*.json')))
  a['rank_done_count']=len(list(d.glob('rank_done_*.json')))
  a['errors']={f.name:json.loads(f.read_text()) for f in d.glob('rank_error_*.json')}
  if a['rank_init_count']:
   a['initializations']=[{k:r[k] for k in ['rank','model_sha256','actual_criterion','local_batch','global_batch','initial_optimizer_lrs']} for r in [json.loads(f.read_text()) for f in sorted(d.glob('rank_init_*.json'))]]
  result[phase+'/'+arm]=a
result['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader'],text=True)
for name in ['prepare','execute']:
 f=root/(name+'_supervisor_launch.json')
 if f.exists():
  x=json.loads(f.read_text());pid=x.get('pid');proc=Path('/proc')/str(pid)
  result[name+'_process']={'pid':pid,'exists':proc.exists()}
  if proc.exists():
   result[name+'_process'].update(starttime=(proc/'stat').read_text().rsplit(')',1)[1].split()[19],cmdline=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode())
print(json.dumps(result,indent=2))
