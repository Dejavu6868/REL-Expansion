import json,sys,time,subprocess,hashlib
from pathlib import Path
B=Path('/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-FG15-20260924')
R=Path('/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_FG15_20260924/attempt1')
sys.path[:0]=[str(B),str(B/'evaluation')]
from suite_common import load_suite,configure_imports,build_config,digest
from run_suite import validate_train_prerequisites
from preflight import json_value
from audit_eval import audit_arm
from compare_results import validate_baseline,validate_gamma1_baseline,validate_resolved_config,build_comparison
J=lambda p:json.loads(Path(p).read_text())
checked=[]
def check(ok,label):
 if not ok:raise RuntimeError(label)
 checked.append(label)
suite=load_suite(B/'suite.json');configure_imports(suite)
manifest=J(B/'bundle_manifest.json');mh=digest(B/'bundle_manifest.json')
check(mh=='e488b4974ccf70095a3be6010a3c6ecc4f488e38b6d136475e690850c1a5605f','deployed manifest equals recorded launch manifest')
for rel,sha in manifest['files'].items():check(digest(B/rel)==sha,'bundle:'+rel)
pipeline=J(R/'pipeline_status.json');train=J(R/'train_status.json');eval_status=J(R/'evaluation_epoch200/status.json');before=J(R/'evaluation_epoch200/preflight.json');comparison=J(R/'evaluation_epoch200/dual_arm_comparison.json')
check(pipeline['status']==train['status']==eval_status['status']=='PASS','all stages PASS')
check(pipeline['completed_stages']==['training','evaluation_epoch200'],'exact completed stages')
check(train==before['training'] and train['suite_sha256']==digest(B/'suite.json'),'training records unchanged')
check(before['bundle_manifest_sha256']==mh,'evaluation bundle manifest unchanged')
check(train['prerequisites']==validate_train_prerequisites(suite,B/'suite.json'),'training prerequisites unchanged')
gamma2=validate_baseline(B,suite);gamma1=validate_gamma1_baseline(B,suite,gamma2)
check(gamma1==before['baseline_gamma1'] and gamma2==before['baseline_gamma2'],'both pinned baselines unchanged')
audits={};weights=[];input_hashes={};arm_times={}
for arm,completion in zip(suite['arms'],train['completed']):
 d=R/'runs'/arm;runtime=J(d/'runtime_status.json')
 check(completion==J(d/'arm_status.json') and completion['status']=='PASS','arm completed:'+arm)
 check(completion['epoch']==200 and completion['global_iteration']==189000 and completion['rank_done_count']==8,'training endpoint:'+arm)
 check((d/'exitcode').read_text().strip()=='0' and runtime['status']=='FORMAL_TRAINING_COMPLETED' and runtime['author_nan_replacement_count']==0,'training exit and NaN:'+arm)
 check(not list(d.glob('rank_error_*.json')),'no rank error:'+arm)
 for rank in range(8):
  init=J(d/('rank_init_%02d.json'%rank));done=J(d/('rank_done_%02d.json'%rank))
  check(init['rank']==rank and init['actual_criterion']=={'type':'FocalLoss2d','focal_gamma':1.5,'reduction':'none','ignore_index':255} and init['model_sha256']==suite['baseline_initial_model_sha256'],'actual rank initialization:'+arm+':'+str(rank))
  check(done['status']=='COMPLETED' and done['rank']==rank and done['epoch']==200 and done['global_iteration']==189000,'done:'+arm+':'+str(rank))
 cfg=json_value(build_config(suite,arm))
 check(cfg==J(R/'configs'/(arm+'.json'))==before['arms'][arm]['config'],'resolved config unchanged:'+arm)
 check(validate_resolved_config(B,suite,arm,cfg)==before['arms'][arm]['config_comparison'],'config baseline comparison:'+arm)
 for name,item in before['arms'][arm]['data_evidence'].items():
  if item.get('exists'):
   check(digest(item['path'])==item['sha256'],'input evidence:'+arm+':'+name);input_hashes[item['path']]=item['sha256']
 audits[arm]=audit_arm(R/'evaluation_epoch200'/arm,cfg,before['arms'][arm]['checkpoint'])
 check(audits[arm]==comparison['arms'][arm]==J(R/'evaluation_epoch200'/arm/'integrity_audit.json'),'current independent arm audit:'+arm)
 check(audits[arm]['checkpoint_sha256']==completion['checkpoint_sha256'],'training/evaluation checkpoint identity:'+arm)
 weights.append(dict(before['arms'][arm]['checkpoint'],arm=arm,gamma=1.5))
 arm_times[arm]={'started_at':completion['started_at'],'finished_at':completion['finished_at'],'hours':(completion['finished_at']-completion['started_at'])/3600}
 for rank in range(8):
  rr=J(R/'evaluation_epoch200'/arm/('rank_%02d_runtime.json'%rank))
  check(rr['status']=='COMPLETED' and rr['processed_samples']==len(range(rank,17593,8)),'evaluation rank completion:'+arm+':'+str(rank))
check(build_comparison(audits,before)==comparison,'comparison entirely reconstructed')
for gamma,baseline in [(1,gamma1),(2,gamma2)]:
 for arm in suite['arms']:
  f=Path(baseline['training_output_root'])/'runs'/arm/'checkpoints/epoch-200.pth';h=digest(f)
  check(h==baseline['checkpoint_sha256'][arm],'old scientific weight unchanged:'+str(gamma)+':'+arm)
  weights.append({'gamma':gamma,'arm':arm,'path':str(f),'sha256':h,'size':f.stat().st_size,'mtime_ns':str(f.stat().st_mtime_ns)})
launch=J(R/'execute_supervisor_launch.json');proc=Path('/proc')/str(launch['pid']);process={'pid':launch['pid'],'proc_exists':proc.exists(),'recorded_starttime':launch['starttime']}
if proc.exists():process['actual_starttime']=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
result={'status':'PASS','checked_at':time.time(),'checks_count':len(checked),'checks':checked,'bundle_file_count':len(manifest['files']),'input_file_count':len(input_hashes),'input_sha256':input_hashes,'weights':weights,'arm_times':arm_times,'pipeline_finished_at':pipeline['finished_at'],'evaluation_minutes':(eval_status['finished_at']-eval_status['started_at'])/60,'supervisor':process,'gpu_csv':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader'],text=True),'compute_processes_csv':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),'metrics':{a:audits[a]['metrics'] for a in suite['arms']},'candidate_screening':comparison['candidate_screening']}
print(json.dumps(result,indent=2))
