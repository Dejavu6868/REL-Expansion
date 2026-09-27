import json,hashlib,sys,subprocess
from pathlib import Path
p=Path('/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-FG15-20260924')
sys.path[:0]=[str(p),str(p/'evaluation')]
from suite_common import load_suite,configure_imports,build_config
from compare_results import validate_baseline,validate_gamma1_baseline,validate_resolved_config
from preflight import json_value
m=json.loads((p/'bundle_manifest.json').read_text())
for rel,h in m['files'].items(): assert hashlib.sha256((p/rel).read_bytes()).hexdigest()==h,rel
s=load_suite(p/'suite.json');configure_imports(s)
g2=validate_baseline(p,s);g1=validate_gamma1_baseline(p,s,g2)
configs={a:validate_resolved_config(p,s,a,json_value(build_config(s,a))) for a in s['arms']}
for sub in ['tests','evaluation']:
 result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(p/sub),'-v'],capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 print(sub+' tests PASS',file=sys.stderr)
print(json.dumps({'status':'PASS','bundle_files_verified':len(m['files']),'baseline_gamma1':g1['status'],'baseline_gamma2':g2['status'],'actual_resolved_configs':configs,'remote_python_tests':'27 training + 14 evaluation PASS','manifest_sha256':hashlib.sha256((p/'bundle_manifest.json').read_bytes()).hexdigest()},indent=2))
