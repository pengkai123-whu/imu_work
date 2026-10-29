"""Create controlled paper-core ablation configs without changing the test split."""
import argparse
import copy
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'configs/yt_1s_update05s.json'))
    args=parser.parse_args(); path=Path(args.config).resolve()
    cfg=json.loads(path.read_text(encoding='utf-8')); stem=path.stem
    for suffix,updates in [('equal_weights',dict(gamma=0.)),('final_only',dict(supervision='final'))]:
        new=copy.deepcopy(cfg); new['train'].update(updates)
        new['train']['out']='../runs/'+stem+'_'+suffix
        new['experiment']=stem+' controlled ablation: '+suffix
        # Preserve path meaning when writing next to the input configuration.
        dest=path.with_name(stem+'_'+suffix+'.json')
        dest.write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(dest)

if __name__=='__main__': main()
