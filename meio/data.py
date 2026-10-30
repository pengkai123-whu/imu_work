"""Train-only normalization and disk-backed navigation-frame cache."""
import hashlib
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from .common import write_json

CACHE_VERSION = 1

def rotate(v, q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    xyz = q[:, 1:]
    t = 2 * np.cross(xyz, v)
    return v + q[:, :1] * t + np.cross(xyz, t)

def window_count(n, window, horizon, stride):
    return max(0, min(n-window, n-horizon-1)//stride+1)

def cache_id(meta):
    payload = {k: meta[k] for k in ['version', 'source_bytes', 'fs', 'limits', 'records', 'mean', 'std']}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def prepare(cfg):
    d = cfg['data']; root = Path(d['cache']); root.mkdir(parents=True, exist_ok=True)
    path = Path(d['h5'])
    limits = {k: d.get(k) for k in ['max_records', 'max_samples']}
    records = []; offset = 0
    with h5py.File(path, 'r') as f:
        for split in ['train', 'valid', 'test']:
            names = sorted(f[split])
            if limits['max_records']: names = names[:limits['max_records']]
            for name in names:
                g = f[split][name]
                sizes = [len(g[k]) for k in ['acc', 'gyr', 'qua', 'pos', 'ts']]
                if len(set(sizes)) != 1: raise ValueError(f'{split}/{name}: inconsistent lengths')
                n = min(sizes[0], limits['max_samples'] or sizes[0])
                if n < max(d['window'], d['horizon']+1): raise ValueError(f'{name}: too short')
                records.append(dict(split=split, name=name, n=n, offset=offset,
                                    t0=float(g['ts'][0]), t1=float(g['ts'][n-1])))
                offset += n
        expected = dict(version=CACHE_VERSION, source_bytes=path.stat().st_size, fs=d['fs'], limits=limits, records=records)
        marker = root/'manifest.json'
        if marker.exists():
            meta = json.loads(marker.read_text(encoding='utf-8'))
            if any(meta.get(k) != v for k,v in expected.items()):
                raise ValueError('cache does not match source/limits/fs; choose a new cache directory')
            for key, shape in [('signals', (offset,6)), ('positions',(offset,3))]:
                a = np.load(root/(key+'.npy'), mmap_mode='r')
                if a.shape != shape or a.dtype != np.float32: raise ValueError('invalid cache array '+key)
            if meta.get('identity') != cache_id(meta): raise ValueError('invalid cache identity')
            print('Using complete cache:', root, flush=True)
            return meta
        signals = np.lib.format.open_memmap(root/'signals.npy', mode='w+', dtype='float32', shape=(offset,6))
        positions = np.lib.format.open_memmap(root/'positions.npy', mode='w+', dtype='float32', shape=(offset,3))
        mean = np.zeros(6); m2 = np.zeros(6); total = 0
        for i, r in enumerate(records):
            g = f[r['split']][r['name']]; n = r['n']; sl = slice(0,n)
            acc, gyr, q, pos, ts = [np.asarray(g[k][sl], dtype=np.float64) for k in ['acc','gyr','qua','pos','ts']]
            if not all(np.isfinite(a).all() for a in [acc,gyr,q,pos,ts]): raise ValueError('non-finite data: '+r['name'])
            if np.max(np.abs(np.linalg.norm(q,axis=1)-1)) > .01: raise ValueError('invalid wxyz quaternion: '+r['name'])
            if not np.allclose(np.diff(ts), 1/d['fs'], rtol=.02, atol=1e-5): raise ValueError('irregular timestamps: '+r['name'])
            nav = np.concatenate([rotate(acc,q), rotate(gyr,q)], axis=1)
            start = r['offset']; signals[start:start+n] = nav; positions[start:start+n] = pos
            if r['split'] == 'train':
                part_mean = nav.mean(0); delta = part_mean-mean; new_total=total+n
                m2 += ((nav-part_mean)**2).sum(0) + delta**2*total*n/new_total
                mean += delta*n/new_total; total=new_total
            if i % 100 == 0 or i+1 == len(records):
                print(f'Cache {i+1}/{len(records)} records', flush=True)
        if total == 0: raise ValueError('no training samples')
        signals.flush(); positions.flush()
        meta = dict(expected, mean=mean.tolist(), std=np.sqrt(np.maximum(m2/total,1e-8)).tolist(),
                    normalization_source='train only, navigation frame, gravity retained', train_samples=total)
        meta['identity'] = cache_id(meta)
        write_json(marker, meta)
        return meta

def read_meta(cache):
    meta = json.loads((Path(cache)/'manifest.json').read_text(encoding='utf-8'))
    if meta.get('identity') != cache_id(meta): raise ValueError('cache identity mismatch')
    return meta

class Windows(Dataset):
    def __init__(self, cache, split, window, horizon, stride, record_name=None,
                 max_records=None, max_windows=None):
        self.cache=str(cache); self.meta=read_meta(cache); self.window=window; self.horizon=horizon; self.stride=stride
        self.records=[r for r in self.meta['records'] if r['split']==split and (record_name is None or r['name']==record_name)]
        if max_records is not None and record_name is None:
            self.records=self.records[:int(max_records)]
        counts=[window_count(r['n'],window,horizon,stride) for r in self.records]
        if max_windows is not None and record_name is None:
            remaining=int(max_windows); selected=[]; selected_counts=[]
            for record, count in zip(self.records, counts):
                if remaining <= 0: break
                take=min(int(count), remaining)
                if take:
                    selected.append(record); selected_counts.append(take); remaining-=take
            self.records, counts=selected, selected_counts
        self.counts=np.array(counts,dtype=np.int64)
        self.ends=np.cumsum(self.counts); self.starts=np.r_[0,self.ends[:-1]]
        self.offsets=np.array([r['offset'] for r in self.records],dtype=np.int64)
        self._pid=None; self._signals=None; self._positions=None
        if len(self)==0: raise ValueError('no windows for '+split)
    def __len__(self): return int(self.ends[-1]) if len(self.ends) else 0
    def _open(self):
        if self._pid != os.getpid():
            self._signals=np.load(Path(self.cache)/'signals.npy',mmap_mode='r')
            self._positions=np.load(Path(self.cache)/'positions.npy',mmap_mode='r'); self._pid=os.getpid()
    def __getstate__(self):
        state=dict(self.__dict__); state.update(_pid=None,_signals=None,_positions=None); return state
    def indices(self, indices):
        indices=np.asarray(indices,dtype=np.int64)
        if (indices<0).any() or (indices>=len(self)).any(): raise IndexError('window index out of range')
        r=np.searchsorted(self.ends,indices,side='right')
        return self.offsets[r]+(indices-self.starts[r])*self.stride
    def __getitems__(self, indices):
        self._open(); starts=self.indices(indices)
        raw=self._signals[starts[:,None]+np.arange(self.window)[None,:]]
        x=torch.from_numpy(np.ascontiguousarray(raw.transpose(0,2,1)))
        y=torch.from_numpy(np.asarray(self._positions[starts+self.horizon]-self._positions[starts],dtype=np.float32))
        return list(zip(x,y))
    def __getitem__(self,index): return self.__getitems__([index])[0]

class BlockBatches(Sampler):
    """Shuffle batches of nearby windows, keeping disk reads local. No windows dropped."""
    def __init__(self, dataset, batch_size, seed):
        self.n=len(dataset); self.batch=batch_size; self.seed=seed; self.epoch=0
    def __len__(self): return (self.n+self.batch-1)//self.batch
    def __iter__(self):
        order=np.random.default_rng(self.seed+self.epoch).permutation(len(self))
        for b in order:
            yield list(range(int(b)*self.batch,min((int(b)+1)*self.batch,self.n)))

def augment(x,y):
    """GPU augmentation in physical units; yaw rotates inputs and target consistently."""
    b=len(x); angle=torch.rand(b,device=x.device)*2*torch.pi
    c,s=angle.cos(),angle.sin()
    def yaw(v):
        a,z=v[:,0].clone(),v[:,1].clone()
        v[:,0]=c[:,None]*a-s[:,None]*z; v[:,1]=s[:,None]*a+c[:,None]*z
    x=x.clone(); y=y.clone()
    yaw(x[:,:3]); yaw(x[:,3:]); yaw(y[:,:,None])
    bias=x.new_tensor([.2,.2,.2,.00872665,.00872665,.00872665])[None,:,None]
    noise=x.new_tensor([.05,.05,.05,.000174533,.000174533,.000174533])[None,:,None]
    x += (torch.rand(b,6,1,device=x.device)*2-1)*bias+torch.randn_like(x)*noise
    return x,y
