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
    args=parser.parse_args()
    if args.batch<1: parser.error('--batch must be positive')
    if args.max_windows<4 or args.tolerance<0: parser.error('require --max-windows >= 4 and --tolerance >= 0')
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
    policy=out/'policy.json'
    calibrate(checkpoint,cache,policy,args.device,args.batch,args.max_windows,args.tolerance)
    reports={}
    for name,path in [('final',None),('early',policy)]:
        report=evaluate(checkpoint,cache,out/(name+'_test.json'),path,'test',args.device,args.batch)
        reports[name]={key:report[key] for key in ['mode','records','windows','displacement_rmse','macro','exit_histogram','linear_conv_flops_per_window']}
    write_json(out/'summary.json',reports)
    print('Evaluation complete. Reports:',out,flush=True)


if __name__=='__main__': main()
