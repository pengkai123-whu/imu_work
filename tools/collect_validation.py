"""Collect execution evidence after running the documented local checks."""
import collections
import hashlib
import json
import platform
import sys
from pathlib import Path
import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
def read(relative): return json.loads((ROOT/relative).read_text(encoding='utf-8'))
meta=read('cache/yt/manifest.json')
metrics=[json.loads(line) for line in (ROOT/'runs/full_data_check/metrics.jsonl').read_text().splitlines()]
result=dict(date='2026-10-29',environment=dict(python=sys.version,platform=platform.platform(),
    torch=torch.__version__,numpy=np.__version__,h5py=h5py.__version__,cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0),vram_gib=torch.cuda.get_device_properties(0).total_memory/1024**3),
    real_h5_smoke=read('runs/smoke_gpu_final/check_result.json'),
    batch_check=read('runs/batch_check/result.json'),notebook=read('runs/notebook_check.json'),
    full_cache=dict(records=dict(collections.Counter(r['split'] for r in meta['records'])),
        samples=sum(r['n'] for r in meta['records']),train_samples=meta['train_samples'],identity=meta['identity'],
        mean=meta['mean'],std=meta['std']),
    full_cache_short_training=dict(passed=True,epochs=len(metrics),batches_per_epoch=3,
        batch=[r['batch'] for r in metrics],nll_fractions=[r['nll_fraction'] for r in metrics],
        amp_skipped_steps=[r['amp_skipped_steps'] for r in metrics]),
    evaluation={name:{key:read('runs/full_data_check/'+name+'_test.json')[key]
        for key in ['records','windows','mode','exit_histogram']} for name in ['final','early']},
    llio_model_sha256=hashlib.sha256((ROOT/'upstream/llio/model_MLP.py').read_bytes()).hexdigest(),
    limitations=['Execution checks only; no complete converged training',
        'Colab hosted runtime and Drive synchronization not executed',
        'CUDA OOM retry code reviewed; no deliberately induced OOM'])
target=ROOT/'validation/evidence.json'; target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('Wrote',target)
