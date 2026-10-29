"""Export the committed, clean Git tree as a small Colab upload archive."""
import hashlib
import subprocess
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
if status.strip(): raise RuntimeError('Commit code and documentation before packaging: '+status)
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
