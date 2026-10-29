"""Validation-only uncertainty audit and batch-1 latency; no policy tuning."""
import argparse
import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from meio.common import ROOT,write_json,file_hash
from meio.data import Windows
from meio.evaluate import context

def correlation(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else None

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default=str(ROOT/'runs/yt/best.pt'))
    parser.add_argument('--policy',required=True)
    parser.add_argument('--out',default=str(ROOT/'validation/baseline_diagnostics.json'))
    parser.add_argument('--max-windows',type=int,default=5000)
    args=parser.parse_args()
    import json
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    device=torch.device('cuda'); torch.set_num_threads(4)
    cache=saved['config']['data']['cache']; saved,meta,model,mean,std=context(args.checkpoint,cache,device)
    policy=json.loads(Path(args.policy).read_text())
    if policy['checkpoint_sha256']!=file_hash(args.checkpoint) or policy['cache_identity']!=meta['identity']:
        raise ValueError('policy mismatch')
    cfg=saved['config']['data']; ds=Windows(cache,'valid',cfg['window'],cfg['horizon'],cfg['valid_stride'])
    idx=np.linspace(0,len(ds)-1,min(len(ds),args.max_windows),dtype=int)
    errors=[]; sigmas=[]; exits=[]; speeds=[]; residuals=[]; example=None
    with torch.inference_mode():
        for x,y in DataLoader(Subset(ds,idx.tolist()),batch_size=256):
            z=(x.to(device)-mean)/std
            if example is None: example=z[:1]
            p,l=model(z); e=(p-y.to(device)[:,None]).cpu().numpy()
            s=l.exp().cpu().numpy()*np.asarray(policy['sigma_scales'])[None]
            scores=np.linalg.norm(s,axis=2); stop=scores<=policy['threshold']; stop[:,-1]=True
            chosen=stop.argmax(1) if policy['mode']=='dynamic' else np.full(len(y),model.depth-1)
            errors.append(np.linalg.norm(e,axis=2)); sigmas.append(s); residuals.append(e)
            exits.append(chosen); speeds.append(y.norm(dim=1).numpy()/(cfg['horizon']/cfg['fs']))
        error=np.concatenate(errors); sigma=np.concatenate(sigmas); residual=np.concatenate(residuals)
        ex=np.concatenate(exits); speed=np.concatenate(speeds)
        per_exit=[]
        for i in range(model.depth):
            per_exit.append(dict(exit=i+1,vector_rmse=float(np.sqrt(np.mean(error[:,i]**2))),
                sigma_error_pearson=correlation(np.linalg.norm(sigma[:,i],axis=1),error[:,i]),
                per_axis_fraction_within_3sigma=np.mean(np.abs(residual[:,i])<=3*sigma[:,i],axis=0).tolist()))
        timings={}
        modes=dict(final=lambda:model.final(example),dynamic=lambda:model.dynamic(example,policy['threshold'],policy['sigma_scales']) if policy['mode']=='dynamic' else model.final(example))
        for name,fn in modes.items():
            for _ in range(10): fn()
            torch.cuda.synchronize(); samples=[]
            for _ in range(50):
                torch.cuda.synchronize(); start=time.perf_counter(); fn(); torch.cuda.synchronize()
                samples.append((time.perf_counter()-start)*1000)
            timings[name]=dict(batch=1,median_ms=float(np.median(samples)),p90_ms=float(np.quantile(samples,.9)))
    result=dict(checkpoint_sha256=file_hash(args.checkpoint),split='valid',windows=len(error),
        architecture=saved['model_cfg'].get('architecture','legacy'),input_seconds=cfg['window']/cfg['fs'],
        output_horizon_seconds=cfg['horizon']/cfg['fs'],per_exit=per_exit,
        exit_histogram=np.bincount(ex,minlength=model.depth).tolist(),
        exit_speed_pearson=correlation(ex+1,speed),latency=timings,
        latency_note='CUDA FP32 batch-1 wall latency including synchronization and Python routing; same first validation window repeated; excludes preprocessing; not a representative runtime benchmark',
        interpretation='Validation samples also participated in policy fitting; descriptive diagnostics only, no independent calibration guarantee. Speed is GT displacement/horizon, not annotated motion type. No semantic motion-awareness claim.')
    write_json(args.out,result); print(result)

if __name__=='__main__': main()
