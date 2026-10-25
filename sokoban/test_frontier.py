"""CPU contracts for isolated frontier losses, response fitting and behavior gates."""
from workspace import ARCHIVE, RUNS
import json
import unittest
import numpy as np
import torch

from bootstrap import HERE
from frontier_tasks import ARMS, build_data, loss, numpy_objective, response_coefficients, torch_objective
from joint_tasks import can_replace, decorate


class FrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = ARCHIVE/'joint/20261001-retention01'
        cls.level = json.loads((folder/'level.json').read_text())
        cls.snapshot = json.loads((folder/'lessons.json').read_text())
        cls.data = {a:build_data(cls.level,cls.snapshot,**spec)[1] for a,spec in ARMS.items()}

    def test_frontier_excludes_other_loss_terms_but_keeps_measured_anchors(self):
        for arm in ('A','C'):
            d=self.data[arm]
            self.assertEqual((len(d['boards']),len(d['objective_indices'])),(14,22))
        for arm in ('B','D'):
            d=self.data[arm]
            self.assertEqual(len(d['objective_indices']),1)
            self.assertEqual(len(d['protected_clauses']),5)
            self.assertEqual(len(d['boards']),6)
            self.assertEqual(d['states'][d['clauses'][d['bridge_clause']]['atoms'][0][0]],(55,(54,)))

    def test_current_objective_reproduces_frozen_audit(self):
        audit=json.loads((ARCHIVE/'research/20261001-training-loss/local-loss-audit.json').read_text())
        margins=np.array([r['margin'] for r in audit['loss_terms']])
        value,_=numpy_objective(margins,self.data['A'])
        self.assertAlmostEqual(value,audit['final_loss'],places=12)

    def test_torch_and_fit_objectives_and_gradients_match(self):
        rng=np.random.default_rng(10)
        for arm,data in self.data.items():
            values=rng.uniform(-.6,.7,len(data['clauses']))
            margins=torch.tensor(values,dtype=torch.float64,requires_grad=True)
            value=torch_objective(margins,data)
            value.backward()
            expected,gradient=numpy_objective(values,data)
            self.assertAlmostEqual(float(value.detach()),expected,places=12)
            np.testing.assert_allclose(margins.grad.numpy(),gradient,rtol=1e-12,atol=1e-12)
            for i in data['objective_indices']:
                a,b=values.copy(),values.copy()
                a[i]+=1e-6;b[i]-=1e-6
                numerical=(numpy_objective(a,data)[0]-numpy_objective(b,data)[0])/2e-6
                self.assertAlmostEqual(gradient[i],numerical,places=7)

    def test_inactive_losses_cannot_dominate_frontier(self):
        for arm in ('B','D'):
            data=self.data[arm]
            base=np.zeros(len(data['clauses']))
            changed=base.copy()
            for i in data['protected_clauses']:
                changed[i]=-100.
            self.assertEqual(numpy_objective(base,data)[0],numpy_objective(changed,data)[0])
            gradient=numpy_objective(changed,data)[1]
            self.assertTrue(all(gradient[i]==0 for i in data['protected_clauses']))

    def test_hinge_has_zero_pressure_above_target(self):
        data=self.data['D']
        value,gradient=numpy_objective(np.full(len(data['clauses']),.2),data)
        self.assertEqual(value,0.)
        self.assertTrue(np.array_equal(gradient,np.zeros_like(gradient)))

    def test_response_combines_real_directions_and_respects_anchor(self):
        for penalty in ('softplus','huber'):
            data=dict(objective_indices=[1],weights=[1.,1.],protected_clauses=[0],penalty=penalty)
            base=np.array([.05,-1.]);matrix=np.array([[-1.,-1.],[2.,-2.]])
            coefficients,fit=response_coefficients(base,matrix,data)
            self.assertTrue(fit['fit_success'])
            self.assertTrue(fit['task_function_matches_selection'])
            prediction=base+matrix@coefficients
            self.assertGreaterEqual(prediction[0],.03-1e-6)
            self.assertGreater(prediction[1],0.)

    def test_allowed_probability_mass_does_not_hide_wrong_argmax(self):
        data=dict(legal=np.array([[1,1,1,0]],dtype=bool),
                  clauses=[dict(atoms=[(0,(0,1))])],objective_indices=[0],weights=[1.],penalty='huber')
        _,margins,correct=loss(torch.tensor([[.3,.3,.4,100.]]),data)
        self.assertFalse(bool(correct[0]))
        self.assertAlmostEqual(float(margins[0]),-.1,places=6)

    def test_all_legal_actions_allowed_has_zero_penalty(self):
        data=dict(legal=np.array([[1,1,0,0]],dtype=bool),
                  clauses=[dict(atoms=[(0,(0,1))])],objective_indices=[0],weights=[1.],penalty='huber')
        value,_,correct=loss(torch.tensor([[0.,1.,10.,10.]]),data)
        self.assertEqual(float(value),0.)
        self.assertTrue(bool(correct[0]))

    def test_original_solution_overrides_loss_margin_and_auxiliary_regression(self):
        def obs(solved,critical,loss):
            statuses=['solved' if solved else 'repeated_state','repeated_state','repeated_state']
            return decorate(dict(solved=solved,lesson_solved=False,complete=True,critical_margin=critical,
                loss=loss,records=[dict(level_id=str(i),status=s) for i,s in enumerate(statuses)]))
        candidate=obs(True,-20.,99.)
        self.assertTrue(candidate['joint_passed'])
        self.assertTrue(can_replace(candidate,obs(False,.1,0.)))

    def test_dataset_construction_does_not_mutate_frozen_snapshot(self):
        before=json.dumps(self.snapshot,sort_keys=True)
        build_data(self.level,self.snapshot,'frontier','huber')
        self.assertEqual(before,json.dumps(self.snapshot,sort_keys=True))


if __name__=='__main__':
    unittest.main()

