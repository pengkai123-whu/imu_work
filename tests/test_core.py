import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from meio.common import read_config, write_json
from meio.data import prepare, Windows, rotate, window_count, augment
from meio.model import MEIO, losses, exit_weights
from meio.evaluate import trajectory_metrics, fixed_duration_metrics

def tiny_config(root, device='cpu'):
    cfg=read_config(Path(__file__).resolve().parents[1]/'configs/yt.json')
    cfg['data'].update(h5=str(root/'tiny.h5'),cache=str(root/'cache'),window=20,horizon=20,fs=20,train_stride=10,valid_stride=20)
    cfg['model'].update(window=20,patch=5,dim=16,depth=3,dropout=0.)
    cfg['train'].update(out=str(root/'run'),batch=8,auto_batch=False,device=device,mse_epochs=1,transition_epochs=0,nll_epochs=1,workers=0,log_every=100)
    return cfg

def make_h5(path):
    with h5py.File(path,'w') as f:
        for i,split in enumerate(['train','valid','test']):
            g=f.create_group(split+'/record'); n=180
            ts=np.arange(n)/20.; q=np.zeros((n,4)); q[:,0]=1
            acc=np.ones((n,3))*(i*100); acc[:,2]+=9.81
            acc[:,0]+=np.sin(ts); gyr=np.column_stack([np.sin(ts),np.cos(ts),ts/20])*.1
            pos=np.column_stack([ts*.7,np.sin(ts)*.1,np.zeros(n)])
            for k,v in dict(acc=acc,gyr=gyr,qua=q,pos=pos,ts=ts).items():g[k]=v

class CoreTests(unittest.TestCase):
    def test_rotation_and_window_boundaries(self):
        q=np.array([[np.cos(np.pi/4),0,0,np.sin(np.pi/4)]])
        np.testing.assert_allclose(rotate([[1.,0,0]],q),[[0,1,0]],atol=1e-12)
        self.assertEqual(window_count(201,200,200,10),1)
        self.assertEqual(window_count(200,200,10,10),1)
        self.assertEqual(window_count(199,200,10,10),0)

    def test_train_only_stats_and_short_horizon(self):
        root=Path(__file__).resolve().parents[1]/'runs'/'unit_fixture'
        root.mkdir(parents=True,exist_ok=True)
        cfg=tiny_config(root); make_h5(root/'tiny.h5'); meta=prepare(cfg)
        self.assertLess(abs(meta['mean'][0]),2)
        ds=Windows(root/'cache','train',20,2,1)
        x,y=ds[len(ds)-1]; self.assertEqual(tuple(x.shape),(6,20))
        self.assertTrue(torch.isfinite(y).all())
        self.assertEqual(prepare(cfg)['identity'],meta['identity'])
        self.assertEqual(len(ds.__getitems__([0,1])),2)

    def test_dynamic_routes_match_all_exit_predictions(self):
        cfg=dict(window=20,patch=5,dim=16,depth=3,expansion=2,dropout=0.,min_sigma=.01,max_sigma=10.)
        model=MEIO(cfg).eval(); x=torch.randn(7,6,20)
        with torch.no_grad(): all_means,logs=model(x)
        pred,sigma,ex=model.dynamic(x,0.)
        torch.testing.assert_close(pred,all_means[:,-1]); self.assertTrue((ex==2).all())
        pred,sigma,ex=model.dynamic(x,100.)
        torch.testing.assert_close(pred,all_means[:,0]); self.assertTrue((ex==0).all())
        # Perturb the initially constant sigma head so the batch really splits.
        torch.manual_seed(9)
        torch.nn.init.normal_(model.heads[0].log_sigma[-1].weight,std=.7)
        with torch.no_grad(): all_means,logs=model(x)
        scores=logs[:,0].exp().norm(dim=1); threshold=float(scores.sort().values[2])
        visited=[]
        hooks=[b.register_forward_pre_hook(lambda module,args: visited.append(len(args[0]))) for b in model.blocks]
        pred,sigma,ex=model.dynamic(x,threshold)
        for hook in hooks: hook.remove()
        self.assertEqual(int((ex==0).sum()),3)
        self.assertEqual(visited[0],7); self.assertEqual(visited[1],4)
        torch.testing.assert_close(pred,all_means[torch.arange(len(x)),ex])
        model.train()
        with self.assertRaises(RuntimeError):model.dynamic(x,1.)

    def test_weighted_nll_and_gradients(self):
        mean=torch.tensor([[[1.,2.,3.],[2.,1.,0.]]],requires_grad=True)
        logs=torch.zeros_like(mean,requires_grad=True); y=torch.zeros(1,3); w=torch.tensor([.2,.8])
        loss=losses(mean,logs,y,w,1.)
        expected=(.5*mean.square().mean((0,2))*w).sum()
        torch.testing.assert_close(loss,expected); loss.backward()
        self.assertTrue(torch.isfinite(mean.grad).all()); self.assertTrue(torch.isfinite(logs.grad).all())
        self.assertAlmostEqual(float(exit_weights(8,.5,'cpu').sum()),1.,places=6)

    def test_oracle_trajectory(self):
        target=np.random.default_rng(2).normal(size=(80,3))
        row=trajectory_metrics(target,target,1.)
        self.assertEqual(row['ate_xyz'],0.); self.assertEqual(row['rte30_xyz'],0.)
        shifted=trajectory_metrics(target+np.array([1.,0.,0.]),target,1.)
        self.assertAlmostEqual(shifted['ate_xyz'],float(np.sqrt(np.mean(np.arange(81.)**2))))
        self.assertAlmostEqual(shifted['rte30_xyz'],30.)

    def test_augmentation_shapes(self):
        x,y=augment(torch.randn(4,6,20),torch.randn(4,3))
        self.assertEqual(tuple(x.shape),(4,6,20)); self.assertEqual(tuple(y.shape),(4,3))
        self.assertTrue(torch.isfinite(x).all())

    def test_fixed_duration_keeps_all_segments_and_counts_tail(self):
        target=np.zeros((125,3)); pred=np.zeros_like(target)
        pred[:60,0]=1.; pred[60:120,0]=2.; pred[120:,0]=100.
        result=fixed_duration_metrics(pred,target,1.,60.)
        self.assertEqual(result['segments'],2)
        self.assertEqual(result['covered_windows'],120)
        self.assertEqual(result['excluded_tail_windows'],5)
        self.assertEqual([r['start_seconds'] for r in result['per_segment']],[0.,60.])
        unit_ate=float(np.sqrt(np.mean(np.arange(61.)**2)))
        self.assertAlmostEqual(result['macro']['ate_xyz'],1.5*unit_ate)
        self.assertAlmostEqual(result['macro']['rte30_xyz'],45.)
        self.assertAlmostEqual(result['macro']['final_error_xyz'],90.)
        # Full-record metrics still include the high-error tail.
        self.assertGreater(trajectory_metrics(pred,target,1.)['ate_xyz'],result['macro']['ate_xyz'])
        short=fixed_duration_metrics(pred[:59],target[:59],1.,60.)
        self.assertEqual(short['segments'],0); self.assertIsNone(short['macro']['ate_xyz'])
        exact=fixed_duration_metrics(target[:60],target[:60],1.,60.)
        self.assertEqual(exact['segments'],1); self.assertEqual(exact['macro']['rte30_xyz'],0.)
        with self.assertRaises(ValueError): fixed_duration_metrics(pred,target,1.,60.5)
        with self.assertRaises(ValueError): fixed_duration_metrics(pred,target,1.,20.)

if __name__=='__main__': unittest.main()
