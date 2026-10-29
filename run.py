"""Individual local prepare/train/calibrate/evaluate commands."""
import argparse
from meio.common import ROOT, read_config

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    for name in ['prepare','train']:
        s=sub.add_parser(name); s.add_argument('--config',default=str(ROOT/'configs/yt_1s_update05s.json'))
        if name=='train': s.add_argument('--resume')
    for name in ['calibrate','evaluate']:
        s=sub.add_parser(name); s.add_argument('--checkpoint',required=True); s.add_argument('--cache',required=True)
        s.add_argument('--out',required=True); s.add_argument('--device',default='auto'); s.add_argument('--batch',type=int,default=512)
        if name=='calibrate': s.add_argument('--max-windows',type=int,default=50000); s.add_argument('--tolerance',type=float,default=.03)
        else: s.add_argument('--policy'); s.add_argument('--split',choices=['train','valid','test'],default='test')
    args=p.parse_args()
    if args.command=='prepare':
        from meio.data import prepare
        prepare(read_config(args.config))
    elif args.command=='train':
        from meio.train import fit
        fit(read_config(args.config),args.resume)
    elif args.command=='calibrate':
        from meio.evaluate import calibrate
        calibrate(args.checkpoint,args.cache,args.out,args.device,args.batch,args.max_windows,args.tolerance)
    else:
        from meio.evaluate import evaluate
        evaluate(args.checkpoint,args.cache,args.out,args.policy,args.split,args.device,args.batch)

if __name__=='__main__': main()
