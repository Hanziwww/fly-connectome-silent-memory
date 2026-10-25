"""Tests for experimental validity and continuation decisions; no training mocks."""
import unittest
import numpy as np
import torch
import bootstrap
from game import restore,state_from,step,solved,canonical
from registry import STUDY,read
from tasks import transform
from training import extension_allowed,objective


class StudyTests(unittest.TestCase):
    def test_behavior_feature_archive_is_lossless(self):
        import tempfile
        from pathlib import Path
        from behavior import archive_behavior,restore_behavior
        f=np.arange(512,dtype='float32').tolist()
        value=dict(records=[dict(decisions=[dict(features=f,action=1),dict(features=f,action=2)])])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);packed=archive_behavior(path,value,1);unpacked=restore_behavior(path,packed)
            self.assertEqual(unpacked['records'],value['records'])
            self.assertEqual(np.load(path/packed['feature_table'])['features'].shape,(1,512))

    def test_segmented_propagation_and_transpose_gradient(self):
        from backend import SegmentedMatrix,_SegmentMultiply
        w=torch.tensor([[0.,2.,0.],[-1.,0.,3.],[4.,0.,0.]])
        m=SegmentedMatrix(w.to_sparse_csr());x=torch.arange(18,dtype=torch.float32).reshape(3,6).requires_grad_(True)
        y=_SegmentMultiply.apply(x,m)
        torch.testing.assert_close(y,w@x)
        y.square().sum().backward();torch.testing.assert_close(x.grad,w.T@(2*(w@x.detach())))

    def test_memory_cues_require_hidden_history(self):
        from memory_tasks import prepare_memory,hidden_board,frames,prefix_states
        from progressive_tasks import reverse_distances
        memory=prepare_memory();self.assertEqual(len(memory),3)
        for task in memory:
            qs=state_from(task['query_state'])
            self.assertTrue(np.array_equal(hidden_board(task['levels'][0],qs),hidden_board(task['levels'][1],qs)))
            for cue,l in enumerate(task['levels']):
                ds,_=reverse_distances(l)
                self.assertNotIn((34,(35 if cue==0 else 33,)),ds)
                for prefix in (1,2,4,8):
                    seq=frames(task,cue,40,prefix)
                    self.assertTrue(all(not f['board'][1].any() for f in seq[1:]))
                    self.assertEqual(seq[0]['ms'],40)
                    self.assertEqual(len(prefix_states(prefix)),prefix)

    def test_locked_task_families_and_solutions(self):
        levels=read(STUDY/'tasks/levels.json')
        self.assertEqual(len(levels),34);self.assertEqual(len({l['family_id'] for l in levels}),34)
        self.assertEqual(len({canonical(dict(walls=set(l['walls']),goals=[])) for l in levels}),34)
        for l in levels:
            m=restore(l);s=state_from(l['start'])
            for a in l['reference_actions']:
                s=step(m,s,a);self.assertIsNotNone(s)
            self.assertTrue(solved(m,s));self.assertEqual(len(l['reference_actions']),l['optimal_steps'])
            for k in range(4):
                for flip in (False,True):
                    v=transform(l,k,flip);m=restore(v);s=state_from(v['start'])
                    for a in v['reference_actions']:s=step(m,s,a)
                    self.assertTrue(solved(m,s));self.assertEqual(canonical(m),canonical(restore(l)))

    def test_extension_uses_same_entry_and_retention(self):
        s=dict(promotions=1,phase_promotions=1,depth=4,phase_depth=4,phase_margin=-.1)
        self.assertTrue(extension_allowed(s,dict(retained=True,bridge_margin=-.07)))
        self.assertFalse(extension_allowed(s,dict(retained=False,bridge_margin=1)))
        self.assertFalse(extension_allowed(dict(s,depth=3),dict(retained=True,bridge_margin=1)))
        self.assertTrue(extension_allowed(dict(s,promotions=2),dict(retained=False,bridge_margin=-1)))

    def test_recovery_repairs_negative_anchor(self):
        data=dict(legal=np.ones((2,4),bool),clauses=[dict(atoms=[(0,(0,))]),dict(atoms=[(1,(1,))])],
                  bridge_clause=0,protected_clauses=[1])
        q=torch.tensor([[0.,0.,0.,0.],[1.,-1.,0.,0.]],requires_grad=True)
        loss=objective(q,data,'recovery')[0];loss.backward();self.assertLess(q.grad[1,1].item(),0)
        q.grad=None;objective(q,data,'entry_only')[0].backward();self.assertEqual(q.grad[1].abs().sum(),0)

    def test_schedule_pairing_and_no_seed_repeats(self):
        jobs=read(STUDY/'schedule.json');self.assertEqual({j['seed'] for j in jobs},{10})
        self.assertEqual(len({j['id'] for j in jobs}),len(jobs))
        self.assertEqual(sum(j['module']=='difficulty' for j in jobs),24)
        for j in jobs:
            if j['module']=='controls':self.assertIsNone(j['parent'])


if __name__=='__main__':unittest.main()
