"""Actual CUDA batch tuning and one real-data update at the selected batch."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from meio.common import read_config, write_json, seed_all, checked_tensor
from meio.data import prepare, Windows, augment
from meio.model import MEIO, losses, exit_weights
from meio.train import tune_batch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='runs/batch_check'); args=p.parse_args()
    cfg=read_config('configs/yt.json'); out=Path(args.out).resolve()
    cfg['data'].update(cache=str(out/'cache'),max_records=4,max_samples=40000)
    torch.set_num_threads(4); seed_all(6); device=torch.device('cuda')
    meta=prepare(cfg); model=MEIO(cfg['model']).to(device)
    chosen,rows=tune_batch(model,device,2048,8192,True)
    ds=Windows(cfg['data']['cache'],'train',200,200,10)
    if len(ds)<chosen: raise ValueError('not enough real windows to validate chosen batch')
    x,y=next(iter(DataLoader(ds,batch_size=chosen))); x=x.to(device); y=y.to(device)
    x,y=augment(x,y)
    mean=torch.tensor(meta['mean'],device=device,dtype=torch.float32)[None,:,None]
    std=torch.tensor(meta['std'],device=device,dtype=torch.float32)[None,:,None]
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['train']['lr'])
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    weights=exit_weights(model.depth,.5,device); phases=[]
    for alpha in [0.,.5,1.]:
        optimizer.zero_grad(set_to_none=True)
        before=model.embedding.weight.detach().clone()
        with torch.autocast('cuda',dtype=torch.float16): means,logs=model((x-mean)/std)
        loss=losses(means,logs,y,weights,alpha); checked_tensor(loss,'real batch loss')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
        delta=float((model.embedding.weight.detach()-before).abs().max())
        if delta==0: raise AssertionError('optimizer did not update parameters')
        phases.append(dict(nll_fraction=alpha,loss=float(loss.detach()),max_embedding_update=delta))
    result=dict(passed=True,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
        chosen=chosen,real_batch=len(x),probes=rows,updates=phases,
        note='Synthetic probe throughput excludes data loading; real-data updates check execution, not model accuracy.')
    write_json(out/'result.json',result); print(result,flush=True)

if __name__=='__main__': main()
