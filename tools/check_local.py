"""Exercise local train.py, automatic checkpoint resume, and test.py on CUDA."""
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from meio.common import ROOT, read_config, write_json, load_checkpoint


def main():
    out=ROOT/'runs/local_cli_check'
    cfg=read_config(ROOT/'configs/yt.json')
    cfg['data'].update(cache=str(out/'cache'),max_records=2,max_samples=8000,train_stride=200,valid_stride=200)
    cfg['train'].update(out=str(out/'fit'),batch=32,auto_batch=False,mse_epochs=1,transition_epochs=2,nll_epochs=1,limit_batches=3,log_every=1)
    config_path=out/'config.json'; write_json(config_path,cfg)
    def run(script,*args):
        # Run outside the repository to verify script/config path resolution.
        subprocess.run([sys.executable,'-u',str(ROOT/script),*map(str,args)],cwd=ROOT.parent,check=True)
    run('train.py','--config',config_path)
    cfg['train']['nll_epochs']=2; write_json(config_path,cfg)
    run('train.py','--config',config_path,'--resume')
    run('test.py','--checkpoint',out/'fit/best.pt','--out',out/'reports','--max-windows',100,'--batch',32)
    summary=json.loads((out/'reports/summary.json').read_text())
    epoch=load_checkpoint(out/'fit/last.pt')['epoch']+1
    if epoch!=5 or summary['final']['records']!=2 or summary['final']['windows']!=78:
        raise AssertionError('unexpected resume/evaluation output')
    write_json(out/'result.json',dict(passed=True,device='cuda',resume_epoch=epoch,
        records=2,test_windows=78,entrypoints=['train.py','train.py --resume','test.py'],
        note='real H5 subset; execution check only, not converged accuracy'))
    print('LOCAL CLI CHECK PASSED:',out)


if __name__=='__main__': main()
