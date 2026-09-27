import hashlib,json,subprocess
from pathlib import Path
BUNDLE=Path(__file__).resolve().parents[1]
SUITE=json.loads((BUNDLE/'suite.json').read_text())
SSH=['ssh','-S','/tmp/codex-rel-expansion-master.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','zhuzhaoziao@172.20.50.33']
REMOTE=SUITE['output_root']
source="""import hashlib,json
from pathlib import Path
p=Path(%r)
assert json.loads((p/'preparation_status.json').read_text())['status']=='PASS'
files={}
for f in sorted(p.rglob('*')):
 r=f.relative_to(p)
 if f.is_symlink() or not f.is_file() or 'checkpoints' in r.parts or f.suffix=='.pth':continue
 if r.parts[0] in ['runs','evaluation_epoch200']:continue
 if f.suffix in ['.json','.jsonl','.log'] or f.name=='exitcode':
  b=f.read_bytes();files[str(r)]={'sha256':hashlib.sha256(b).hexdigest(),'size':len(b)}
print(json.dumps({'status':'PASS','root':str(p),'files':files}))
""" % REMOTE
r=subprocess.run(SSH+['python3 -'],input=source,text=True,capture_output=True)
assert r.returncode==0,r.stderr
manifest=json.loads(r.stdout)
out=BUNDLE/'evidence/remote_preparation';out.mkdir(exist_ok=True)
files_from=BUNDLE/'evidence/PREPARATION_FILES.txt'
files_from.write_text('\n'.join(manifest['files'])+'\n')
cmd=['rsync','-az','--files-from='+str(files_from.resolve()),'-e','ssh -S /tmp/codex-rel-expansion-master.sock -o BatchMode=yes -o ConnectTimeout=8','zhuzhaoziao@172.20.50.33:'+REMOTE+'/',str(out.resolve())+'/']
r=subprocess.run(cmd,text=True,capture_output=True);assert r.returncode==0,r.stderr
for rel,row in manifest['files'].items():
 b=(out/rel).read_bytes();assert len(b)==row['size'] and hashlib.sha256(b).hexdigest()==row['sha256'],rel
manifest['local_root']=str(out.resolve());manifest['file_count']=len(manifest['files'])
(BUNDLE/'evidence/PREPARATION_TRANSFER_VERIFICATION.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'PASS','files':len(manifest['files'])}))
