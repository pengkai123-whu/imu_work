"""Local valid-only calibration and complete final/dynamic test evaluation."""
import argparse
from pathlib import Path
from meio.common import ROOT, load_checkpoint, device_for, write_json


def main():
    parser=argparse.ArgumentParser(description='Calibrate on valid, then evaluate final and dynamic exits on test.')
    parser.add_argument('--checkpoint',default=str(ROOT/'runs/yt/best.pt'))
    parser.add_argument('--cache',help='Override the checkpoint cache path')
    parser.add_argument('--out',help='Report directory; default: checkpoint directory')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch',type=int,default=512)
    parser.add_argument('--max-windows',type=int,default=50000,help='Validation calibration window cap, never a test cap')
    parser.add_argument('--tolerance',type=float,default=.03)
    parser.add_argument('--segment-seconds',type=float,default=30.,help='ATE segment duration, default 30s; all complete segments, no best-segment selection')
    parser.add_argument('--policy',help='Reuse a frozen policy instead of recalibrating on valid')
    args=parser.parse_args()
    if args.batch<1: parser.error('--batch must be positive')
    if args.max_windows<4 or args.tolerance<0: parser.error('require --max-windows >= 4 and --tolerance >= 0')
    import math
    if not math.isfinite(args.segment_seconds) or args.segment_seconds<30:
        parser.error('--segment-seconds must be finite and >=30')
    checkpoint=Path(args.checkpoint).resolve()
    if not checkpoint.is_file(): raise FileNotFoundError('Train first; missing checkpoint: '+str(checkpoint))
    saved=load_checkpoint(checkpoint)
    cache=Path(args.cache or saved['config']['data']['cache']).resolve()
    out=Path(args.out).resolve() if args.out else checkpoint.parent
    import torch
    device=device_for(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Activate the spk environment with CUDA-enabled PyTorch.')
    torch.set_num_threads(saved['config']['train'].get('cpu_threads',4))
    from meio.evaluate import calibrate, evaluate
    policy=Path(args.policy).resolve() if args.policy else out/'policy.json'
    if args.policy:
        if not policy.is_file(): raise FileNotFoundError(policy)
    else:
        calibrate(checkpoint,cache,policy,args.device,args.batch,args.max_windows,args.tolerance)
    reports={}
    for name,path in [('final',None),('early',policy)]:
        report=evaluate(checkpoint,cache,out/(name+'_test.json'),path,'test',args.device,args.batch,args.segment_seconds)
        reports[name]={key:report[key] for key in ['mode','records','windows','displacement_rmse','macro','exit_histogram','linear_conv_flops_per_window','primary_metrics','fixed_duration']}
    write_json(out/'summary.json',reports)
    seconds=reports['final']['primary_metrics']['displacement_seconds']
    print('\nPrimary metrics (XYZ vector errors, meters); full-record and XY metrics retained in JSON.')
    print(f'{"Mode":<12} {"ATE"+format(args.segment_seconds,"g")+"s XYZ":>15} {"RTE30s XYZ":>15} {"Disp"+format(seconds,"g")+"s RMSE":>15}')
    for name,report in reports.items():
        metrics=report['primary_metrics']
        values=[f'{metrics[k]:.4f}' if metrics[k] is not None else 'N/A'
                for k in ['ate_xyz','rte30_xyz','displacement_rmse_xyz']]
        print(f'{name:<12} '+ ' '.join(f'{v:>15}' for v in values))
    fixed=reports['final']['fixed_duration']
    print(f'Fixed-duration coverage: {fixed["segments"]} segments, {fixed["eligible_records"]}/{reports["final"]["records"]} records, '
          f'{fixed["coverage_fraction"]:.2%} windows; {fixed["excluded_tail_windows"]} tail windows excluded from segment ATE only.')
    print('Segment scores are averaged within each record, then across records. No rotation fitting or best-segment selection.')
    print('RTE30 uses all legal 30s pairs within full records. Displacement RMSE uses all 1-step intervals, including tails.')
    print('Evaluation complete. Reports:',out,flush=True)


if __name__=='__main__': main()
