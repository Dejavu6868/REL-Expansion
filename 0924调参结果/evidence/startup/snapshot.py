import argparse,hashlib,json,subprocess,time
from pathlib import Path
BUNDLE=Path(__file__).resolve().parents[1]
SSH=['ssh','-S','/tmp/codex-rel-expansion-master.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','zhuzhaoziao@172.20.50.33']
def main():
 parser=argparse.ArgumentParser();parser.add_argument('name');args=parser.parse_args()
 read=(BUNDLE/'evidence/read_remote_state.py').read_text()
 r=subprocess.run(SSH+['python3 -'],input=read,text=True,capture_output=True)
 (BUNDLE/'evidence'/(args.name+'.stderr')).write_text(r.stderr)
 if r.returncode:raise RuntimeError(r.stderr)
 data=json.loads(r.stdout)
 (BUNDLE/'evidence'/(args.name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
 result={'queried_at':data['queried_at']}
 for name,row in data['files'].items():
  result[name]={k:row[k] for k in ['status','phase','active_stage','active_arm','completed_stages','error','pid','starttime'] if k in row}
 for arm in ['hha','relplus']:
  for phase in ['smoke','runs']:
   key=phase+'/'+arm
   if key in data:
    row=data[key];result[key]={'rank_init_count':row['rank_init_count'],'rank_done_count':row['rank_done_count'],'errors':row['errors'],'files':{f:({k:v[k] for k in ['status','epoch','global_iteration','iteration','loss','author_nan_replacement_count','model_sha256'] if k in v} if isinstance(v,dict) else v) for f,v in row['files'].items()}}
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
