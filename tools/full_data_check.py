"""A few CUDA updates using the complete cache; not a convergence experiment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from meio.common import read_config,write_json
from meio.data import prepare
from meio.train import fit

cfg=read_config('configs/yt.json')
cfg['train'].update(out=str(Path('runs/full_data_check').resolve()),device='cuda',
    batch=2048,auto_batch=True,max_batch=8192,mse_epochs=1,transition_epochs=2,
    nll_epochs=1,limit_batches=3,log_every=1)
meta=prepare(cfg)
write_json('runs/full_data_check/input_config.json',cfg)
best=fit(cfg)
write_json('runs/full_data_check/result.json',dict(passed=True,checkpoint=best,
    records=len(meta['records']),train_samples=meta['train_samples'],
    note='Full cache, only three train and validation batches per epoch; not full training.'))
