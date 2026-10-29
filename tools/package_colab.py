"""Export the committed, clean Git tree as a small Colab upload archive."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
if status.strip():
    # Preserve interactive outputs; export only committed source.
    if status.strip() != 'M MEIO_YT_Colab.ipynb':
        raise RuntimeError('Commit code and documentation before packaging: '+status)
    committed=json.loads(subprocess.check_output(['git','show','HEAD:MEIO_YT_Colab.ipynb'],cwd=ROOT))
    working=json.loads((ROOT/'MEIO_YT_Colab.ipynb').read_text(encoding='utf-8'))
    def sources(nb): return [(c['cell_type'],''.join(c['source'])) for c in nb['cells']]
    if sources(committed)!=sources(working): raise RuntimeError('Notebook source edits must be committed')
    print('Preserving notebook execution outputs; packaging committed source.')
target=ROOT.parent/'meio_yt_colab.zip'
subprocess.run(['git','archive','--format=zip','--prefix=meio_yt/','--output='+str(target),'HEAD'],cwd=ROOT,check=True)
with zipfile.ZipFile(target) as archive:
    if archive.testzip() is not None: raise RuntimeError('ZIP integrity failure')
    names=archive.namelist()
    for required in ['MEIO_YT_Colab.ipynb','run.py','README.md','LICENSE','configs/yt.json']:
        if 'meio_yt/'+required not in names: raise RuntimeError('Missing '+required)
    if any('/.git/' in n or '/cache/' in n or '/runs/' in n or n.endswith(('.h5','.pt')) for n in names):
        raise RuntimeError('Runtime data must not enter the code archive')
print('Archive:',target,'bytes:',target.stat().st_size,'files:',len(names))
print('SHA256:',hashlib.sha256(target.read_bytes()).hexdigest())
