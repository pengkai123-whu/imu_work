import copy
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import device_for, seed_all, save_checkpoint, load_checkpoint, write_json, checked_tensor
from .data import Windows, BlockBatches, augment, read_meta
from .model import MEIO, exit_weights, losses

def phase_fraction(epoch, cfg):
    if epoch < cfg['mse_epochs']: return 0.0
    if cfg['transition_epochs'] == 0: return 1.0
    return min(1., (epoch-cfg['mse_epochs']+1)/cfg['transition_epochs'])

def _probe_one_batch(model, device, batch):
    # Function scope releases activation graphs before the caller handles OOM.
    x=torch.randn(batch,6,model.cfg['window'],device=device)
    target=torch.zeros(batch,3,device=device); w=exit_weights(model.depth,.5,device)
    for step in range(4):
        if step==1:
            torch.cuda.synchronize(device); started=time.perf_counter()
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.float16): means,logs=model(x)
        loss=losses(means.float(),logs.float(),target,w,1.); loss.backward()
    torch.cuda.synchronize(device)
    return (time.perf_counter()-started)/3

def tune_batch(model, device, requested=2048, maximum=8192, enabled=True):
    if device.type != 'cuda' or not enabled: return requested, []
    if requested < 1 or maximum < requested: raise ValueError('invalid batch search range')
    candidates=sorted(set([min(128,requested),min(512,requested),requested]+[b for b in [1024,2048,4096,8192] if requested<=b<=maximum]))
    rows=[]; original_training=model.training; model.train()
    for b in candidates:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            seconds=_probe_one_batch(model,device,b)
            peak=torch.cuda.max_memory_reserved(device); total=torch.cuda.get_device_properties(device).total_memory
            row=dict(batch=b,seconds=seconds,samples_per_second=b/seconds,peak_reserved_gib=peak/1024**3)
            rows.append(row); print('Batch probe:',row,flush=True)
            if peak > .8*total: break
        except torch.cuda.OutOfMemoryError:
            print('Batch probe OOM:',b,flush=True)
            model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache(); break
    model.zero_grad(set_to_none=True); model.train(original_training); torch.cuda.empty_cache()
    safe=[r for r in rows if r['peak_reserved_gib']<.8*torch.cuda.get_device_properties(device).total_memory/1024**3]
    if not safe: raise RuntimeError('No safe probed batch; reduce requested batch (e.g. 64)')
    return max(safe,key=lambda r:r['samples_per_second'])['batch'],rows

@torch.no_grad()
def validate(model, loader, mean, std, device, limit=None):
    model.eval(); sq=torch.zeros(model.depth,device=device); n=0
    for i,(x,y) in enumerate(loader):
        if limit and i>=limit: break
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            means,logs=model((x-mean)/std)
        checked_tensor(means,'validation prediction')
        sq += (means.float()-y[:,None]).square().sum((0,2)); n+=len(y)
    if not n: raise ValueError('empty validation')
    return (sq/n).sqrt().cpu().tolist()

def fit(cfg, resume=None):
    t,d=cfg['train'],cfg['data']; seed_all(t['seed']); torch.set_num_threads(t.get('cpu_threads',4))
    device=device_for(t['device']); meta=read_meta(d['cache'])
    out=Path(t['out']); out.mkdir(parents=True,exist_ok=True)
    if resume is None and (out/'last.pt').exists(): raise FileExistsError('run exists: set resume or choose a new out directory')
    model=MEIO(cfg['model']).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=t['lr'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda',init_scale=1024.) if hasattr(torch,'amp') and hasattr(torch.amp,'GradScaler') else torch.cuda.amp.GradScaler(enabled=device.type=='cuda',init_scale=1024.)
    start=0; best=float('inf'); saved=None
    if resume:
        saved=load_checkpoint(resume)
        if saved['cache_identity']!=meta['identity'] or saved['model_cfg']!=cfg['model']:
            raise ValueError('checkpoint/cache/model mismatch')
        for key in ['window','horizon','fs','train_stride','valid_stride']:
            if saved['config']['data'][key]!=d[key]: raise ValueError('resume data mismatch: '+key)
        for key in ['seed','lr','mse_epochs','transition_epochs','gamma','augment']:
            if saved['config']['train'][key]!=t[key]: raise ValueError('resume training mismatch: '+key)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        if saved.get('scaler'): scaler.load_state_dict(saved['scaler'])
        start=saved['epoch']+1; best=saved['best_rmse']
    chosen,probes=tune_batch(model,device,t['batch'],t.get('max_batch',8192),t.get('auto_batch',False))
    seed_all(t['seed'])
    if saved:
        torch.set_rng_state(saved['torch_rng']); np.random.set_state(saved['numpy_rng'])
        if device.type=='cuda' and len(saved.get('cuda_rng',[]))==torch.cuda.device_count(): torch.cuda.set_rng_state_all(saved['cuda_rng'])
    data=Windows(d['cache'],'train',d['window'],d['horizon'],d['train_stride'])
    valid=Windows(d['cache'],'valid',d['window'],d['horizon'],d['valid_stride'])
    weights=exit_weights(model.depth,t['gamma'],device)
    mean=torch.tensor(meta['mean'],device=device,dtype=torch.float32)[None,:,None]
    std=torch.tensor(meta['std'],device=device,dtype=torch.float32)[None,:,None]
    total=t['mse_epochs']+t['transition_epochs']+t['nll_epochs']
    write_json(out/'config.json',cfg); write_json(out/'batch_probe.json',dict(chosen=chosen,probes=probes))
    print(f'Device {device}; batch {chosen}; train {len(data)}; valid {len(valid)}; epochs {start}..{total-1}',flush=True)
    for epoch in range(start,total):
        epoch_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        epoch_optimizer=copy.deepcopy(optimizer.state_dict()); epoch_scaler=copy.deepcopy(scaler.state_dict())
        epoch_rng=torch.get_rng_state(); epoch_numpy=np.random.get_state()
        epoch_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else []
        while True:
            try:
                sampler=BlockBatches(data,chosen,t['seed']); sampler.epoch=epoch
                loader=DataLoader(data,batch_sampler=sampler,num_workers=t['workers'],pin_memory=device.type=='cuda')
                val_loader=DataLoader(valid,batch_size=chosen,num_workers=t['workers'],pin_memory=device.type=='cuda')
                alpha=phase_fraction(epoch,t); model.train(); loss_sum=0.; count=0; skipped=0; steps=0; started=time.perf_counter()
                for i,(x,y) in enumerate(loader):
                    if t.get('limit_batches') and i>=t['limit_batches']: break
                    x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
                    if t['augment']: x,y=augment(x,y)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                        means,logs=model((x-mean)/std)
                    loss=losses(means.float(),logs.float(),y,weights,alpha); checked_tensor(loss,'training loss')
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),t['clip_grad'],error_if_nonfinite=device.type!='cuda')
                    old_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
                    skipped+=int(scaler.get_scale()<old_scale); steps+=1
                    loss_sum+=float(loss.detach())*len(y); count+=len(y)
                    if (i+1)%t['log_every']==0: print(f'epoch {epoch+1} batch {i+1}/{len(loader)} loss {loss_sum/count:.6f}',flush=True)
                rmse=validate(model,val_loader,mean,std,device,t.get('limit_batches'))
                break
            except torch.cuda.OutOfMemoryError:
                if chosen<=16: raise
                chosen=max(16,chosen//2); print('OOM: restarting epoch with batch',chosen,flush=True)
                x=y=means=logs=loss=None
                optimizer.zero_grad(set_to_none=True); model.load_state_dict(epoch_model); optimizer.load_state_dict(epoch_optimizer); scaler.load_state_dict(epoch_scaler)
                torch.set_rng_state(epoch_rng); np.random.set_state(epoch_numpy)
                if epoch_cuda: torch.cuda.set_rng_state_all(epoch_cuda)
                gc.collect(); torch.cuda.empty_cache()
        if not count: raise ValueError('no training batches')
        if skipped==steps: raise FloatingPointError('all optimizer steps skipped due to AMP overflow')
        improved=alpha>=1 and rmse[-1]<best
        if improved: best=rmse[-1]
        row=dict(epoch=epoch+1,nll_fraction=alpha,loss=loss_sum/count,valid_rmse=rmse,batch=chosen,amp_skipped_steps=skipped,seconds=time.perf_counter()-started)
        print(json.dumps(row),flush=True)
        with open(out/'metrics.jsonl','a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        checkpoint=dict(format_version=1,model=model.state_dict(),model_cfg=cfg['model'],optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            epoch=epoch,best_rmse=best,config=cfg,cache_identity=meta['identity'],mean=meta['mean'],std=meta['std'],
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        save_checkpoint(out/'last.pt',checkpoint)
        if improved: save_checkpoint(out/'best.pt',checkpoint)
        if alpha==0: save_checkpoint(out/'mse_last.pt',checkpoint)
    return str(out/'best.pt')
