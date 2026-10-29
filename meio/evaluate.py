import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .common import load_checkpoint, device_for, write_json, file_hash, checked_tensor
from .data import Windows, read_meta
from .model import MEIO

def context(checkpoint, cache, device):
    saved=load_checkpoint(checkpoint); meta=read_meta(cache)
    if saved['cache_identity']!=meta['identity']: raise ValueError('checkpoint and cache identity mismatch')
    model=MEIO(saved['model_cfg']).to(device); model.load_state_dict(saved['model']); model.eval()
    mean=torch.tensor(saved['mean'],device=device,dtype=torch.float32)[None,:,None]
    std=torch.tensor(saved['std'],device=device,dtype=torch.float32)[None,:,None]
    return saved,meta,model,mean,std

@torch.no_grad()
def calibrate(checkpoint, cache, output, device='auto', batch=512, max_windows=50000, tolerance=.03):
    if max_windows<4 or tolerance<0: raise ValueError('require at least four calibration windows and nonnegative tolerance')
    device=device_for(device); saved,meta,model,mean,std=context(checkpoint,cache,device)
    cfg=saved['config']['data']; ds=Windows(cache,'valid',cfg['window'],cfg['horizon'],cfg['valid_stride'])
    indices=np.linspace(0,len(ds)-1,min(len(ds),max_windows),dtype=np.int64)
    if len(indices)<4: raise ValueError('validation split too small to calibrate')
    predictions=[]; sigmas=[]; targets=[]
    for x,y in DataLoader(Subset(ds,indices.tolist()),batch_size=batch):
        p,s=model((x.to(device)-mean)/std); checked_tensor(p,'calibration mean'); checked_tensor(s,'calibration log sigma')
        predictions.append(p.cpu().numpy()); sigmas.append(s.exp().cpu().numpy()); targets.append(y.numpy())
    p=np.concatenate(predictions); sigma=np.concatenate(sigmas); y=np.concatenate(targets)
    # Two disjoint window subsets of validation. No test samples used for decisions.
    cut=len(p)//2; error=p-y[:,None,:]
    scales=np.sqrt(np.mean((error[:cut]/sigma[:cut])**2,axis=0)).clip(.05,20)
    p=p[cut:]; y=y[cut:]; sigma=sigma[cut:]*scales[None]
    scores=np.linalg.norm(sigma,axis=-1); costs=model.operation_costs()
    baseline=float(np.sqrt(np.mean(np.sum((p[:,-1]-y)**2,axis=1))))
    rows=[]; allowed=baseline*(1+tolerance)
    for threshold in np.unique(np.r_[0.,np.quantile(scores,np.linspace(0,1,81))]):
        mask=scores<=threshold; mask[:,-1]=True; exits=mask.argmax(1)
        pred=p[np.arange(len(p)),exits]
        rmse=float(np.sqrt(np.mean(np.sum((pred-y)**2,axis=1))))
        cost=float(np.mean(np.asarray(costs['dynamic'])[exits]))
        rows.append(dict(threshold=float(threshold),rmse=rmse,linear_conv_flops=cost,mean_exit_1based=float(exits.mean()+1)))
    eligible=[r for r in rows if r['rmse']<=allowed and r['linear_conv_flops']<costs['final']]
    selected=min(eligible,key=lambda r:r['linear_conv_flops']) if eligible else None
    policy=dict(format_version=1,checkpoint_sha256=file_hash(checkpoint),cache_identity=meta['identity'],
        selection_split='valid',scale_windows=cut,selection_windows=len(p),max_windows=max_windows,tolerance=tolerance,
        sigma_scales=scales.tolist(),score='L2 norm of calibrated per-axis standard deviations in meters',
        mode='dynamic' if selected else 'final',threshold=selected['threshold'] if selected else 0.,
        baseline_rmse=baseline,selected=selected,costs=costs,curve=rows)
    write_json(output,policy); print('Calibration:',policy['mode'],'threshold',policy['threshold'],'baseline RMSE',baseline,flush=True)
    return policy

def trajectory_metrics(pred, target, seconds):
    # Non-overlap intervals: endpoint k is exactly k*horizon/fs seconds.
    est=np.vstack([np.zeros(3),np.cumsum(pred.astype(np.float64),axis=0)])
    gt=np.vstack([np.zeros(3),np.cumsum(target.astype(np.float64),axis=0)])
    error=est-gt; lag=int(round(30/seconds))
    row={}
    for label,dim in [('xyz',3),('xy',2)]:
        row['ate_'+label]=float(np.sqrt(np.mean(np.sum(error[:,:dim]**2,axis=1))))
        row['rte30_'+label]=float(np.sqrt(np.mean(np.sum((error[lag:,:dim]-error[:-lag,:dim])**2,axis=1)))) if lag>=1 and len(error)>lag else None
    row['rte_actual_seconds']=lag*seconds
    row['final_error_xyz']=float(np.linalg.norm(error[-1])); return row

def fixed_duration_metrics(pred, target, seconds, segment_seconds=30.):
    """All complete, non-overlapping segments from t=0; never rank by error.

    Each segment is translated to its own origin. No rotation/scale alignment.
    The incomplete tail is excluded from these supplementary metrics only.
    """
    steps=segment_seconds/seconds
    if not np.isfinite(steps) or segment_seconds<30 or not np.isclose(steps,round(steps),rtol=0,atol=1e-7):
        raise ValueError('segment duration must be >=30s and an integer number of prediction intervals')
    steps=int(round(steps))
    if steps<1: raise ValueError('empty segment')
    if len(pred)!=len(target): raise ValueError('prediction/target length mismatch')
    rows=[]
    for start in range(0,len(pred)-steps+1,steps):
        row=dict(start_seconds=start*seconds,end_seconds=(start+steps)*seconds,
                 **trajectory_metrics(pred[start:start+steps],target[start:start+steps],seconds))
        rows.append(row)
    keys=['ate_xyz','ate_xy','rte30_xyz','rte30_xy','final_error_xyz']
    macro={key:float(np.mean([r[key] for r in rows])) if rows else None for key in keys}
    return dict(segment_seconds=segment_seconds,segments=len(rows),covered_windows=len(rows)*steps,
                excluded_tail_windows=len(pred)-len(rows)*steps,macro=macro,per_segment=rows)

@torch.no_grad()
def evaluate(checkpoint,cache,output,policy_path=None,split='test',device='auto',batch=512,segment_seconds=30.):
    device=device_for(device); saved,meta,model,mean,std=context(checkpoint,cache,device)
    policy=json.loads(Path(policy_path).read_text(encoding='utf-8')) if policy_path else None
    if policy and (policy['checkpoint_sha256']!=file_hash(checkpoint) or policy['cache_identity']!=meta['identity']):
        raise ValueError('policy belongs to another checkpoint/cache')
    cfg=saved['config']['data']; records=[r for r in meta['records'] if r['split']==split]
    seconds=cfg['horizon']/cfg['fs']
    # Reject an invalid protocol before expensive inference.
    fixed_duration_metrics(np.empty((0,3)),np.empty((0,3)),seconds,segment_seconds)
    if not records: raise ValueError('no records for evaluation')
    costs=model.operation_costs(); per_record=[]; total_sq=0.; total_sq_xy=0.; total_n=0; histogram=np.zeros(model.depth,dtype=np.int64); total_cost=0.
    dynamic=policy is not None and policy['mode']=='dynamic'
    for r in records:
        ds=Windows(cache,split,cfg['window'],cfg['horizon'],cfg['horizon'],r['name'])
        predictions=[]; targets=[]
        for x,y in DataLoader(ds,batch_size=batch):
            x=(x.to(device)-mean)/std
            if dynamic:
                pred,sigma,exits=model.dynamic(x,policy['threshold'],policy['sigma_scales'])
                ex=exits.cpu().numpy(); total_cost+=float(np.sum(np.asarray(costs['dynamic'])[ex]))
            else:
                pred,_=model.final(x); ex=np.full(len(x),model.depth-1); total_cost+=len(x)*costs['final']
            checked_tensor(pred,'test prediction'); histogram+=np.bincount(ex,minlength=model.depth)
            predictions.append(pred.cpu().numpy()); targets.append(y.numpy())
        p=np.concatenate(predictions); y=np.concatenate(targets); sq=np.sum((p-y)**2,axis=1)
        total_sq+=float(sq.sum()); total_sq_xy+=float(np.sum((p[:,:2]-y[:,:2])**2)); total_n+=len(sq)
        row=dict(record=r['name'],windows=len(p),displacement_rmse=float(np.sqrt(sq.mean())),**trajectory_metrics(p,y,cfg['horizon']/cfg['fs']))
        row['fixed_duration']=fixed_duration_metrics(p,y,seconds,segment_seconds)
        per_record.append(row)
        print(f"{r['name']} displacement={row['displacement_rmse']:.4f} full_ATE_XYZ={row['ate_xyz']:.4f}",flush=True)
    macro={key:float(np.mean([r[key] for r in per_record if r[key] is not None])) if any(r[key] is not None for r in per_record) else None
           for key in ['ate_xyz','ate_xy','rte30_xyz','rte30_xy']}
    eligible=[r['fixed_duration'] for r in per_record if r['fixed_duration']['segments']>0]
    segment_macro={key:float(np.mean([r['macro'][key] for r in eligible])) if eligible else None
                   for key in ['ate_xyz','ate_xy','rte30_xyz','rte30_xy','final_error_xyz']}
    covered=sum(r['fixed_duration']['covered_windows'] for r in per_record)
    fixed=dict(segment_seconds=segment_seconds,segments=sum(r['segments'] for r in eligible),
        eligible_records=len(eligible),excluded_short_records=len(per_record)-len(eligible),
        covered_windows=covered,excluded_tail_windows=total_n-covered,coverage_fraction=covered/total_n,
        aggregation='arithmetic mean of segment RMSEs within each record, then equal-weight mean over eligible records',
        selection='all complete non-overlapping segments starting at record time zero; incomplete tails excluded; no error-based selection',
        alignment='initial translation at every segment start only; no rotation or scale fitting',
        rte_protocol='30-second endpoint differences within each segment; pairs crossing segment boundaries excluded',
        macro=segment_macro)
    primary=dict(units='meters',ate_segment_seconds=segment_seconds,displacement_seconds=seconds,
        ate_xyz=segment_macro['ate_xyz'],ate_xy=segment_macro['ate_xy'],
        rte30_xyz=macro['rte30_xyz'],rte30_xy=macro['rte30_xy'],
        displacement_rmse_xyz=float(np.sqrt(total_sq/total_n)),displacement_rmse_xy=float(np.sqrt(total_sq_xy/total_n)),
        ate_protocol='all complete fixed-duration segments; initial translation per segment; record-balanced mean',
        rte_protocol='all legal 30-second endpoint pairs within full records; record-balanced RMSE mean; includes record tails',
        displacement_protocol='pooled vector RMSE across all predicted displacement intervals; includes record tails')
    report=dict(split=split,records=len(records),windows=total_n,mode='dynamic' if dynamic else 'final',
        checkpoint_sha256=file_hash(checkpoint),cache_identity=meta['identity'],data_limits=meta['limits'],
        trajectory_protocol='non-overlapping predicted displacements, initial translation only, no heading alignment',
        horizon_seconds=cfg['horizon']/cfg['fs'],displacement_rmse=float(np.sqrt(total_sq/total_n)),macro=macro,
        exit_histogram=histogram.tolist(),linear_conv_flops_per_window=total_cost/total_n,
        cost_note='multiply-add counted as 2 FLOPs; all visited heads included; excludes activation, routing, memory; no latency claim',
        primary_metrics=primary,fixed_duration=fixed,per_record=per_record)
    write_json(output,report); print('Evaluation:',json.dumps({k:report[k] for k in ['mode','primary_metrics','exit_histogram']}),flush=True)
    return report
