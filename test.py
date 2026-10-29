"""Local valid-only calibration and complete final/dynamic test evaluation."""
import argparse
from pathlib import Path
from meio.common import ROOT, load_checkpoint, device_for, write_json


def main():
    parser=argparse.ArgumentParser(description='Calibrate on valid, then evaluate final and dynamic exits on test.')
    parser.add_argument('--checkpoint',default=str(ROOT/'runs/yt_1s_update05s/best.pt'))
    parser.add_argument('--cache',help='Override the checkpoint cache path')
    parser.add_argument('--out',help='Report directory; default: checkpoint directory')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch',type=int,default=512)
    parser.add_argument('--max-windows',type=int,default=50000,help='Validation calibration window cap, never a test cap')
    parser.add_argument('--tolerance',type=float,default=.03)
    parser.add_argument('--segment-seconds',type=float,default=30.,help='Supplementary segment ATE duration only; primary ATE always uses full records')
    parser.add_argument('--policy',help='Reuse a frozen policy instead of recalibrating on valid')
    parser.add_argument('--selection-split',choices=['train','valid'],default='valid')
    parser.add_argument('--raw-sigma',action='store_true',help='Disable scale calibration; closer to paper threshold search')
    parser.add_argument('--reference-rmse',type=float,help='Optional reference error on identical selection windows')
    parser.add_argument('--reference-checkpoint',help='Measure a reference model on identical selection windows for accuracy matching')
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
    final_only=saved['config']['train'].get('supervision','all')=='final'
    if final_only and args.policy: parser.error('final-only baseline cannot use an early-exit policy')
    policy=Path(args.policy).resolve() if args.policy else out/'policy.json'
    if not final_only:
        if args.policy:
            if not policy.is_file(): raise FileNotFoundError(policy)
        else:
            calibrate(checkpoint,cache,policy,args.device,args.batch,args.max_windows,args.tolerance,
                      args.selection_split,not args.raw_sigma,args.reference_rmse,args.reference_checkpoint)
    reports={}
    modes=[('final',None)] if final_only else [('final',None),('early',policy)]
    for name,path in modes:
        report=evaluate(checkpoint,cache,out/(name+'_test.json'),path,'test',args.device,args.batch,args.segment_seconds)
        reports[name]={key:report[key] for key in ['mode','records','windows','displacement_rmse','macro','exit_histogram','linear_conv_flops_per_window','linear_conv_flops_per_second','primary_metrics','fixed_duration','update_seconds','trajectory_intervals','trajectory_protocol','integration_oracle','rte30_eligible_records','rte30_excluded_short_records']}
    write_json(out/'summary.json',reports)
    seconds=reports['final']['primary_metrics']['displacement_seconds']
    print('Prediction horizon:',seconds,'s; update interval:',reports['final']['update_seconds'],'s')
    print('\nPrimary metrics (XYZ vector errors, meters): full-record ATE, RTE30s, displacement RMSE. XY retained in JSON.')
    print(f'{"Mode":<12} {"Full ATE XYZ":>15} {"RTE30s XYZ":>15} {"Disp"+format(seconds,"g")+"s RMSE":>15}')
    for name,report in reports.items():
        metrics=report['primary_metrics']
        values=[f'{metrics[k]:.4f}' if metrics[k] is not None else 'N/A'
                for k in ['ate_xyz','rte30_xyz','displacement_rmse_xyz']]
        print(f'{name:<12} '+ ' '.join(f'{v:>15}' for v in values))
    fixed=reports['final']['fixed_duration']
    print('Full-record ATE includes all reconstructed trajectory endpoints and tails, with one initial translation per record; no intermediate reset, rotation or scale fitting.')
    print('ATE and RTE are RMSEs per record, then averaged equally over eligible records. Displacement RMSE is pooled over all prediction windows.')
    print(f'RTE30 eligible records: {reports["final"]["rte30_eligible_records"]}; excluded short records: {reports["final"]["rte30_excluded_short_records"]}.')
    print(f'\nSupplementary segment ATE ({args.segment_seconds:g}s; excluded from primary table):')
    for name,report in reports.items():
        score=report['fixed_duration']['macro']['ate_xyz']
        print(f'{name}: XYZ ATE = {score:.4f} m' if score is not None else f'{name}: XYZ ATE = N/A')
    print(f'Supplementary coverage: {fixed["segments"]} segments, {fixed["eligible_records"]}/{reports["final"]["records"]} records, '
          f'{fixed["coverage_fraction"]:.2%} trajectory intervals; {fixed["excluded_tail_windows"]} tail intervals excluded from segment ATE only.')
    print('Segment scores are averaged within each record, then across records. No rotation fitting or best-segment selection.')
    print(f'RTE30 uses all legal 30s pairs on the reconstructed trajectory. Displacement RMSE uses all {seconds:g}s outputs, including tails.')
    print('Trajectory:',reports['final']['trajectory_protocol'])
    print('Exact-label integration diagnostic:',reports['final']['integration_oracle'])
    print('Evaluation complete. Reports:',out,flush=True)


if __name__=='__main__': main()
