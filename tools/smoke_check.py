"""GPU/CPU end-to-end check on a small slice of the user's real H5. No full training."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from meio.common import read_config, write_json, load_checkpoint
from meio.data import prepare
from meio.train import fit
from meio.evaluate import calibrate, evaluate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda'); p.add_argument('--out',default='runs/smoke'); p.add_argument('--h5'); args=p.parse_args()
    cfg=read_config('configs/yt.json'); out=Path(args.out).resolve()
    cfg['data'].update(cache=str(out/'cache'),max_records=2,max_samples=8000,train_stride=200,valid_stride=200)
    if args.h5:cfg['data']['h5']=str(Path(args.h5).resolve())
    cfg['train'].update(out=str(out/'fit'),device=args.device,batch=16,auto_batch=False,mse_epochs=1,transition_epochs=1,nll_epochs=1,workers=0,limit_batches=3,log_every=1)
    write_json(out/'smoke_config.json',cfg); prepare(cfg)
    ckpt=fit(cfg)
    # Same full-size network, resume for one additional NLL epoch.
    cfg['train']['nll_epochs']=2; fit(cfg,str(out/'fit/last.pt'))
    ckpt=str(out/'fit/best.pt'); policy=out/'policy.json'
    calibrate(ckpt,cfg['data']['cache'],policy,args.device,batch=16,max_windows=100)
    final=evaluate(ckpt,cfg['data']['cache'],out/'final.json',device=args.device,batch=16)
    dynamic=evaluate(ckpt,cfg['data']['cache'],out/'early.json',policy,'test',args.device,16)
    write_json(out/'check_result.json',dict(passed=True,device=args.device,gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        checkpoint_resume_epoch=load_checkpoint(out/'fit/last.pt')['epoch']+1,records=final['records'],windows=final['windows'],policy_mode=dynamic['mode'],
        note='execution check only; tiny subset and three batches per epoch; not an accuracy benchmark'))
    print('SMOKE CHECK PASSED:',out,flush=True)

if __name__=='__main__':main()
