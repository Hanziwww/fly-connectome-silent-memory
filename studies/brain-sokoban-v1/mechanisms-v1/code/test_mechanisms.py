import unittest
from types import SimpleNamespace
import tempfile
from pathlib import Path
import numpy as np
import torch
import mechanism_base as base


class ReadoutAlgebra(unittest.TestCase):
    def heads(self):
        rng=np.random.default_rng(10)
        def head():return {'mean':rng.normal(size=512),'scale':rng.uniform(.1,2,size=512),
            'proposal.weight':rng.normal(size=(4,512)),'proposal.bias':rng.normal(size=4)}
        return rng,head(),head()
    def test_clipping_standardization_bias_and_interaction_reconstruct(self):
        rng,a,b=self.heads();x=rng.normal(0,20,(11,512));y=rng.normal(0,20,(11,512))
        d=base.contribution_decomposition(x,y,a,b)
        np.testing.assert_allclose(d['brain']+d['head']+d['interaction'],d['q11']-d['q00'],atol=1e-10)
        self.assertGreater(np.abs(d['interaction']).max(),1.)
    def test_centered_rank_and_nullspace(self):
        rng,h,_=self.heads();q,_=base.decision_basis(h);self.assertEqual(q.shape,(512,3))
        z=base.standardized(rng.normal(size=(6,512)),h);null=z-z@q@q.T
        w=h['proposal.weight']-h['proposal.weight'].mean(0)
        np.testing.assert_allclose(null@w.T,0,atol=1e-10)
        h['proposal.weight'][:]=h['proposal.weight'][0]
        self.assertEqual(base.decision_basis(h)[0].shape,(512,0))


class StateControls(unittest.TestCase):
    def setup_policy(self):
        from mechanism_model import NEURAL
        b=SimpleNamespace(t=160.,step_count=160,buf_pos=1,gen=torch.Generator(),_rate_np_key=None)
        fields=NEURAL+('drive','rate','poisson_p')
        for name in fields:
            v=torch.tensor([[1.,2.,3.],[7.,8.,9.]])
            if name in ('g_slow_cls','spike_buf'):v=v[None]
            setattr(b,name,v)
        def capture(detach=True):
            return dict(tensors={k:getattr(b,k).clone() for k in fields},t=b.t,step_count=b.step_count,
                        buf_pos=b.buf_pos,generator=b.gen.get_state())
        def restore(state):
            for k,v in state['tensors'].items():setattr(b,k,v.clone())
            for k in ('t','step_count','buf_pos'):setattr(b,k,state[k])
            b.gen.set_state(state['generator'])
        e=SimpleNamespace(brain=b,params=SimpleNamespace(v_rest=-52.),capture_state=capture,restore_state=restore)
        return SimpleNamespace(encoder=e)
    def test_clear_and_restore_preserve_clock_rng_drive(self):
        from mechanism_model import state_intervention
        p=self.setup_policy();b=p.encoder.brain;old=p.encoder.capture_state();cases=[{'cue':0,'context':0},{'cue':1,'context':0}]
        state_intervention(p,'restore',cases)
        for k,v in old['tensors'].items():self.assertTrue(torch.equal(v,getattr(b,k)))
        state_intervention(p,'clear',cases)
        self.assertTrue((b.v==-52).all());self.assertTrue((b.res==1).all());self.assertTrue((b.adapt==0).all())
        self.assertTrue(torch.equal(b.drive,old['tensors']['drive']));self.assertEqual((b.t,b.step_count,b.buf_pos),(160.,160,1))
        self.assertTrue(torch.equal(b.gen.get_state(),old['generator']))


class Storage(unittest.TestCase):
    def test_metadata_does_not_drop_acceptance_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'verification.json'
            payload=dict(status='verified',results={'rollouts':[dict(status='solved',cue=0,features=[1.,2.],decisions=[dict(action=1,features=[3.,4.])])]})
            base.write(path,payload);small=base.read_metadata(path)
            self.assertEqual(small['results']['rollouts'][0]['status'],'solved')
            self.assertEqual(small['results']['rollouts'][0]['features'],[])
            self.assertEqual(small['results']['rollouts'][0]['decisions'],[])
    def test_lossless_and_budget_reservation(self):
        original=base.SESSION
        try:
            with tempfile.TemporaryDirectory() as folder:
                base.SESSION=Path(folder);a=np.arange(240,dtype='float32').reshape(3,8,10)
                base.save_npz(Path(folder)/'event.npz',values=a)
                with np.load(Path(folder)/'event.npz') as f:np.testing.assert_array_equal(f['values'],a)
                ledger=base.read(Path(folder)/'storage.json');self.assertEqual(ledger['reserved'],{})
        finally:base.SESSION=original


if __name__=='__main__':unittest.main(verbosity=2)
