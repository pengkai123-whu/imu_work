"""Real-H5 CUDA execution checks of both timing configs; no full training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from meio.common import ROOT,read_config,write_json,load_checkpoint
from meio.data import prepare
from meio.train import fit
from meio.evaluate import calibrate,evaluate

def main():
    results={}
    for name in ['yt_paper_1s','yt_1s_update05s']:
        out=ROOT/'runs'/('baseline_cadence_check_'+name)
        cfg=read_config(ROOT/'configs'/(name+'.json')); horizon=cfg['data']['horizon']
        cfg['data'].update(cache=str(out/'cache'),max_records=2,max_samples=14000,train_stride=horizon,valid_stride=horizon)
        cfg['train'].update(out=str(out/'fit'),batch=32,auto_batch=False,mse_epochs=1,transition_epochs=2,nll_epochs=1,limit_batches=3,log_every=1)
        write_json(out/'config.json',cfg); prepare(cfg); fit(cfg)
        cfg['train']['nll_epochs']=2; fit(cfg,out/'fit/last.pt')
        ckpt=out/'fit/best.pt'
        # Exercise both adapted validation calibration and paper-style raw train search.
        calibrate(ckpt,cfg['data']['cache'],out/'valid_policy.json','cuda',32,200)
        policy=out/'train_raw_policy.json'
        calibrate(ckpt,cfg['data']['cache'],policy,'cuda',32,200,.03,'train',False)
        final=evaluate(ckpt,cfg['data']['cache'],out/'final.json',device='cuda',batch=32)
        early=evaluate(ckpt,cfg['data']['cache'],out/'early.json',policy,device='cuda',batch=32)
        stride=cfg['data'].get('eval_stride',horizon)
        expected=2*((14000-horizon-1)//stride+1)
        assert final['windows']==expected and final['fixed_duration']['segments']>0
        assert final['trajectory_intervals']==expected-(2 if stride<horizon else 0)
        assert final['update_seconds']==stride/200
        assert load_checkpoint(out/'fit/last.pt')['epoch']==4
        assert final['primary_metrics']['displacement_seconds']==horizon/200
        results[name]=dict(passed=True,device='cuda',architecture=cfg['model']['architecture'],
            window=cfg['data']['window'],horizon=horizon,patch=cfg['model']['patch'],
            test_windows=final['windows'],trajectory_intervals=final['trajectory_intervals'],update_seconds=final['update_seconds'],segments=final['fixed_duration']['segments'],
            resume_epoch=5,policy_mode=early['mode'],note='small real-data execution check, no accuracy claim')
    write_json(ROOT/'validation/baseline_cadence_execution.json',results)
    print('BASELINE CHECK PASSED',results)

if __name__=='__main__': main()
