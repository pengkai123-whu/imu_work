"""Local valid-only calibration and complete final/dynamic test evaluation."""
import argparse
from pathlib import Path
from meio.common import ROOT, load_checkpoint, device_for, write_json


def main():
    parser=argparse.ArgumentParser(description='Calibrate on valid, then evaluate final and dynamic exits on test.')
    parser.add_argument('--checkpoint',default=str(ROOT/'runs/yt/best.pt'))
    parser.add_argument('--cache',help='Override the checkpoint cache path')
    parser.add_argument('--out',help='Report directory; default: checkpoint directory/evaluation_full')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch',type=int,default=512)
    parser.add_argument('--max-windows',type=int,default=50000,help='Validation calibration window cap, never a test cap')
    parser.add_argument('--tolerance',type=float,default=.03)
    parser.add_argument('--segment-seconds',type=float,default=30.,help='Supplementary segment ATE only; primary ATE always uses full records')
    parser.add_argument('--policy',help='Reuse a frozen policy instead of recalibrating on valid')
    parser.add_argument('--recalibrate',action='store_true',help='Explicitly select a new policy on valid instead of reusing checkpoint-directory policy.json')
    args=parser.parse_args()
    if args.batch<1: parser.error('--batch must be positive')
    if args.policy and args.recalibrate: parser.error('--policy and --recalibrate are mutually exclusive')
    if args.max_windows<4 or args.tolerance<0: parser.error('require --max-windows >= 4 and --tolerance >= 0')
    import math
    if not math.isfinite(args.segment_seconds) or args.segment_seconds<30:
        parser.error('--segment-seconds must be finite and >=30')
    checkpoint=Path(args.checkpoint).resolve()
    if not checkpoint.is_file(): raise FileNotFoundError('Train first; missing checkpoint: '+str(checkpoint))
    saved=load_checkpoint(checkpoint)
    cache=Path(args.cache or saved['config']['data']['cache']).resolve()
    out=Path(args.out).resolve() if args.out else checkpoint.parent/'evaluation_full'
    import torch
    device=device_for(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Activate the spk environment with CUDA-enabled PyTorch.')
    torch.set_num_threads(saved['config']['train'].get('cpu_threads',4))
    from meio.evaluate import calibrate, evaluate
    frozen=checkpoint.parent/'policy.json'
    policy=Path(args.policy).resolve() if args.policy else (frozen if frozen.is_file() and not args.recalibrate else out/'policy.json')
    if args.policy:
        if not policy.is_file(): raise FileNotFoundError(policy)
    elif frozen.is_file() and not args.recalibrate:
        print('Reusing frozen policy:',policy,flush=True)
    else:
        calibrate(checkpoint,cache,policy,args.device,args.batch,args.max_windows,args.tolerance)
    reports={}
    for name,path in [('final',None),('early',policy)]:
        report=evaluate(checkpoint,cache,out/(name+'_test.json'),path,'test',args.device,args.batch,args.segment_seconds)
        reports[name]={key:report[key] for key in ['mode','records','windows','displacement_rmse','macro','exit_histogram','linear_conv_flops_per_window','linear_conv_flops_per_second','primary_metrics','fixed_duration','update_seconds','trajectory_protocol','rte30_eligible_records','rte30_excluded_short_records']}
    write_json(out/'summary.json',reports)
    seconds=reports['final']['primary_metrics']['displacement_seconds']
    print(f'Input/target horizon: {seconds:g}s; non-overlapping update interval: {seconds:g}s')
    print('\nPrimary metrics (XYZ vector errors, meters); XY retained in JSON.')
    print(f'{"Mode":<12} {"Full ATE XYZ":>15} {"RTE30s XYZ":>15} {"Disp"+format(seconds,"g")+"s RMSE":>15}')
    for name,report in reports.items():
        metrics=report['primary_metrics']
        values=[f'{metrics[k]:.4f}' if metrics[k] is not None else 'N/A'
                for k in ['ate_xyz','rte30_xyz','displacement_rmse_xyz']]
        print(f'{name:<12} '+ ' '.join(f'{v:>15}' for v in values))
    fixed=reports['final']['fixed_duration']
    print('Full ATE includes all trajectory endpoints and tails, one initial translation per record; no intermediate reset, rotation or scale fitting.')
    print(f'RTE30 eligible records: {reports["final"]["rte30_eligible_records"]}; excluded short records: {reports["final"]["rte30_excluded_short_records"]}.')
    print(f'\nSupplementary segment ATE ({args.segment_seconds:g}s; not a primary metric):')
    for name,report in reports.items():
        value=report['fixed_duration']['macro']['ate_xyz']
        print(f'{name}: {value:.4f} m' if value is not None else f'{name}: N/A')
    print(f'Supplementary coverage: {fixed["segments"]} segments, {fixed["eligible_records"]}/{reports["final"]["records"]} records, '
          f'{fixed["coverage_fraction"]:.2%} windows; {fixed["excluded_tail_windows"]} tail windows excluded from segment ATE only.')
    print('Segment scores are averaged within each record, then across records. No rotation fitting or best-segment selection.')
    print('RTE30 uses all legal 30s pairs within full records. Displacement RMSE uses all 1-step intervals, including tails.')
    print('Evaluation complete. Reports:',out,flush=True)


if __name__=='__main__': main()
