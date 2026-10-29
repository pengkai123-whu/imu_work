"""Local training: python train.py; resume: python train.py --resume."""
import argparse
from pathlib import Path
from meio.common import ROOT, read_config, device_for


def main():
    parser=argparse.ArgumentParser(description='Prepare local H5 cache and train on the local GPU.')
    parser.add_argument('--config',default=str(ROOT/'configs/yt_1s_update05s.json'))
    parser.add_argument('--resume',nargs='?',const='auto',help='Resume a checkpoint; no path means out/last.pt')
    parser.add_argument('--h5',help='Override local H5 path')
    parser.add_argument('--out',help='Override run directory')
    parser.add_argument('--device',help='Override configured device, e.g. cuda or cuda:0')
    parser.add_argument('--batch',type=int,help='Requested batch; use --no-auto-batch for a fixed batch')
    parser.add_argument('--no-auto-batch',action='store_true')
    args=parser.parse_args()
    cfg=read_config(args.config)
    for section,key,value in [('data','h5',args.h5),('train','out',args.out)]:
        if value is not None: cfg[section][key]=str(Path(value).resolve())
    if args.device: cfg['train']['device']=args.device
    if args.batch is not None:
        if args.batch<1: parser.error('--batch must be positive')
        cfg['train']['batch']=args.batch
        cfg['train']['max_batch']=max(args.batch,cfg['train'].get('max_batch',8192))
    if args.no_auto_batch: cfg['train']['auto_batch']=False
    import torch
    device=device_for(cfg['train']['device'])
    if device.type=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Activate the spk environment with CUDA-enabled PyTorch.')
        print('GPU:',torch.cuda.get_device_name(device),'VRAM GiB:',round(torch.cuda.get_device_properties(device).total_memory/1024**3,2),flush=True)
    print('PyTorch:',torch.__version__,'Device:',device,flush=True)
    out=Path(cfg['train']['out'])
    resume=str(out/'last.pt') if args.resume=='auto' else args.resume
    if resume is not None and not Path(resume).is_file(): raise FileNotFoundError(resume)
    if resume is None and (out/'last.pt').exists():
        raise FileExistsError('Run already exists. Use --resume or choose a different --out directory.')
    from meio.data import prepare
    from meio.train import fit
    prepare(cfg)
    checkpoint=fit(cfg,resume)
    print('Training complete. Best checkpoint:',checkpoint,flush=True)
    print('Next: python test.py --checkpoint "'+checkpoint+'"',flush=True)


if __name__=='__main__': main()
