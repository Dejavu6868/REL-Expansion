import json,hashlib,time
from pathlib import Path
old=Path('/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-FG1-20260920')
manifest=json.loads((old/'bundle_manifest.json').read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for rel,h in manifest['files'].items():assert digest(old/rel)==h,rel
rows=[]
for gamma,root,expected in [(1,'/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_FG1_20260920/attempt2',{'hha':'d4d210ea92a473cc700b1660dff7da644e8d1b9934be759b216c0d4931ae2b4d','relplus':'6506a922b9d31919576b3f998864993a24ca7f07baca868d71a5ff3b6a581831'}),(2,'/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_20260915/attempt1',{'hha':'c5cf78b12d2458a48fe5505a4b7099ef84308dd4790f8283c02a605ba05ccd03','relplus':'bf17b065bbbedf01f78c698b039c384c9b220a2332b24b930ca685db552e1232'})]:
 for arm,h in expected.items():
  f=Path(root)/'runs'/arm/'checkpoints/epoch-200.pth';actual=digest(f);assert actual==h,str(f)
  rows.append({'gamma':gamma,'arm':arm,'path':str(f),'sha256':actual,'size':f.stat().st_size,'mtime_ns':str(f.stat().st_mtime_ns)})
print(json.dumps({'status':'PASS','time':time.time(),'gamma1_bundle_files_verified':len(manifest['files']),'checkpoints':rows},indent=2))
