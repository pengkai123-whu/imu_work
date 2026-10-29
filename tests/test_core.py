import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from meio.common import read_config, write_json
from meio.data import prepare, Windows, rotate, window_count, augment
from meio.model import MEIO, losses, exit_weights, Residual
from meio.evaluate import trajectory_metrics, fixed_duration_metrics, trajectory_increments

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

    def test_30s_ate_and_relative_endpoint_error(self):
        target=np.zeros((95,3)); pred=np.zeros_like(target); pred[:,0]=.02
        result=fixed_duration_metrics(pred,target,1.)
        self.assertEqual(result['segment_seconds'],30.)
        self.assertEqual(result['segments'],3)
        self.assertEqual(result['excluded_tail_windows'],5)
        expected=.02*float(np.sqrt(np.mean(np.arange(31.)**2)))
        self.assertAlmostEqual(result['macro']['ate_xyz'],expected)
        self.assertAlmostEqual(result['macro']['rte30_xyz'],.6)
        self.assertAlmostEqual(trajectory_metrics(pred,target,1.)['rte30_xyz'],.6)
        self.assertGreater(trajectory_metrics(pred,target,1.)['ate_xyz'],expected)

    def test_paper_figure_shared_head_and_residual(self):
        cfg=dict(architecture='paper_fig3',window=100,patch=25,dim=16,depth=3,expansion=2,dropout=0.,min_sigma=.01,max_sigma=10.)
        model=MEIO(cfg).eval(); x=torch.randn(5,6,100)
        with torch.no_grad(): means,logs=model(x)
        self.assertEqual(tuple(means.shape),(5,3,3))
        self.assertEqual(sum(isinstance(m,torch.nn.Linear) for m in model.heads[0].modules()),3)
        p,_,ex=model.dynamic(x,100.)
        torch.testing.assert_close(p,means[:,0]); self.assertTrue((ex==0).all())
        p,_,ex=model.dynamic(x,0.)
        torch.testing.assert_close(p,means[:,-1]); self.assertTrue((ex==2).all())
        # Conv patch embedding equals rearrange + Linear in the documented order.
        patches=x.unfold(2,25,25).permute(0,2,1,3).reshape(5,4,150)
        explicit=torch.nn.functional.linear(patches,model.embedding.weight[:,:, :].reshape(16,150),model.embedding.bias)
        torch.testing.assert_close(model.embed(x),explicit)
        branch=Residual(2,torch.nn.Identity(),'paper_fig3')
        with torch.no_grad():
            branch.affine_out.b.fill_(2.); branch.scale.fill_(.1)
        z=torch.ones(1,1,2)
        torch.testing.assert_close(branch(z),z+.1*(z+2.))

    def test_paper_and_half_second_update_configs(self):
        root=Path(__file__).resolve().parents[1]
        for file,n,patch in [('yt_paper_1s.json',200,50),('yt_1s_update05s.json',200,50)]:
            cfg=read_config(root/'configs'/file)
            self.assertEqual(cfg['data']['window'],n); self.assertEqual(cfg['data']['horizon'],n)
            self.assertEqual(cfg['model']['patch'],patch); self.assertEqual(n//patch,4)
            self.assertEqual(cfg['model']['architecture'],'paper_fig3')
        cfg=read_config(root/'configs/yt_1s_update05s.json')
        self.assertEqual(cfg['data']['eval_stride'],100)
        self.assertEqual(cfg['data']['valid_stride'],100)
        delta=np.full((120,3),.01)
        row=fixed_duration_metrics(delta,delta,.5,30.)
        self.assertEqual(row['segments'],2); self.assertEqual(row['macro']['ate_xyz'],0.)
        self.assertEqual(row['macro']['rte30_xyz'],0.)

    def test_overlapping_causal_integration_and_actual_truth(self):
        # Constant speed: 1s outputs of 2m must advance by 1m per 0.5s.
        positions=np.zeros((1000,3)); positions[:,0]=np.arange(1000)*.01
        starts=np.arange(0,701,100); target=positions[starts+200]-positions[starts]
        pred,truth,oracle=trajectory_increments(target,target,positions,starts,200,100)
        self.assertEqual(len(pred),7)
        np.testing.assert_allclose(pred,truth,atol=1e-12)
        self.assertAlmostEqual(float(pred[:,0].sum()),7.)
        self.assertEqual(trajectory_metrics(pred,truth,.5)['ate_xyz'],0.)
        # Acceleration: exact 1s labels are not exact most-recent-half labels.
        positions[:,0]=(np.arange(1000)/200.)**2
        target=positions[starts+200]-positions[starts]
        pred,truth,oracle=trajectory_increments(target,target,positions,starts,200,100)
        np.testing.assert_allclose(truth,np.diff(positions[starts+200],axis=0))
        self.assertGreater(trajectory_metrics(oracle,truth,.5)['ate_xyz'],0.)
        # Editing future positions cannot change an earlier output increment.
        changed=positions.copy(); changed[601:]+=1000.
        pp,tt,_=trajectory_increments(target,target,changed,starts,200,100)
        np.testing.assert_array_equal(pp,pred); np.testing.assert_array_equal(tt[:4],truth[:4])
        with self.assertRaises(ValueError): trajectory_increments(target,target,positions,starts,200,201)

    def test_all_exit_training_and_whole_nll_weight(self):
        cfg=dict(architecture='paper_fig3',window=20,patch=5,dim=16,depth=3,expansion=2,dropout=0.,min_sigma=.01,max_sigma=10.)
        model=MEIO(cfg); p,s=model(torch.randn(4,6,20))
        losses(p,s,torch.randn(4,3),exit_weights(3,.5,'cpu'),1.).backward()
        for head in model.heads:
            self.assertGreater(float(head.mean.weight.grad.norm()),0.)
            self.assertGreater(float(head.log_sigma.weight.grad.norm()),0.)
        means=torch.tensor([[[1.,2.,3.],[3.,1.,2.]]]); logs=torch.tensor([[[.2,-.3,.4],[.5,-.1,.1]]])
        w=torch.tensor([.2,.8]); target=torch.zeros(1,3)
        expected=(w*(.5*means.square()*torch.exp(-2*logs)+logs).mean((0,2))).sum()
        torch.testing.assert_close(losses(means,logs,target,w,1.),expected)

if __name__=='__main__': unittest.main()
