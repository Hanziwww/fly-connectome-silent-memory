import unittest
import numpy as np
import torch
from supp_replay import matched_control,paired_indices,timing_frames
from supp_base import SOURCE,read


class Controls(unittest.TestCase):
    def test_norm_and_orthogonality(self):
        gen=torch.Generator().manual_seed(10)
        x=torch.randn((4,200),generator=gen);d=torch.randn((4,200),generator=gen)
        c=matched_control(x,d)
        torch.testing.assert_close(c.norm(dim=1),d.norm(dim=1))
        torch.testing.assert_close((c*d).sum(1),torch.zeros(4),atol=2e-5,rtol=0)
    def test_adaptation_control_respects_domain(self):
        gen=torch.Generator().manual_seed(10)
        x=torch.rand((4,200),generator=gen)
        d=(x.flip(0)-x)*.5;c=matched_control(x,d,True)
        self.assertTrue(((x+c)>=-1e-7).all())
        torch.testing.assert_close(c.norm(dim=1),d.norm(dim=1),atol=1e-5,rtol=1e-5)
        self.assertTrue(torch.equal(matched_control(x,torch.zeros_like(x),True),torch.zeros_like(x)))
    def test_cue_pairing_and_equal_time_context_blind_distractors(self):
        tasks=read(SOURCE/'inputs/memory-tasks.json')
        for task in tasks:
            if task['id'] not in ('M0-audit','M0-visibility','S1-ring','S2-context'):continue
            cases=task['cases'];indices=paired_indices(cases)
            self.assertEqual([indices[j] for j in indices],list(range(len(cases))))
            for mode in ('neutral_stationary','neutral_walk_0','neutral_walk_1'):
                histories=timing_frames(task,240,4,mode)
                self.assertTrue(all(sum(f['ms'] for f in h)==280 for h in histories))
                for h in histories:
                    for frame in h[1:]:self.assertEqual(float(frame['board'][1].sum()),0.)


if __name__=='__main__':unittest.main(verbosity=2)
