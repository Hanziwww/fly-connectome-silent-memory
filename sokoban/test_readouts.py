"""Meaningful CPU contracts for the linear and E/I readout replacements."""
import unittest

import torch

from bootstrap import HERE
from common import configure, cpu_state
from heads import ActivityHead
from readout_models import ActivityEI, ActivityLinear, MODELS, make_readout


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        configure(10)
        self.old = ActivityHead()
        with torch.no_grad():
            self.old.mean.normal_()
            self.old.scale.uniform_(.1, 3)
            self.old.logit_scale.fill_(2.7)
        self.state = cpu_state(self.old)
        self.x = torch.randn(128, 512) * 4

    def test_all_initializations_preserve_source_function_with_shared_clipping(self):
        target = self.old(self.x, correction=False)['brain_logits']
        for kind in MODELS:
            with self.subTest(kind=kind):
                head = make_readout(kind, self.state)
                q = head(self.x)['final_logits']
                torch.testing.assert_close(q, target, atol=3e-6, rtol=3e-6)

    def test_parameter_counts_and_normalization_are_comparable(self):
        for kind, count in [('linear', 2052), ('mlp', 33092), ('ei', 33092)]:
            head = make_readout(kind, self.state)
            self.assertEqual(sum(p.numel() for p in head.parameters()), count)
            self.assertTrue(all(p.requires_grad for p in head.parameters()))
            self.assertFalse(head.mean.requires_grad or head.scale.requires_grad)
            torch.testing.assert_close(head.mean, self.state['mean'])
            torch.testing.assert_close(head.scale, self.state['scale'])

    def test_ei_output_signs_survive_gradient_updates(self):
        head = ActivityEI.from_linear(self.state)
        optimizer = torch.optim.Adam(head.parameters(), lr=.05)
        target = torch.randn(128, 4)
        for _ in range(12):
            optimizer.zero_grad()
            (head(self.x)['final_logits'] - target).square().mean().backward()
            optimizer.step()
            audit = head.sign_audit()
            self.assertTrue(audit['fixed_signs_valid'] and audit['all_outgoing_signs_valid'])
        self.assertGreater(audit['minimum_magnitude'], 0)

    def test_ei_paired_extra_units_are_not_dead_at_initialization(self):
        head = ActivityEI.from_linear(self.state)
        target = torch.randn(128, 4)
        (head(self.x)['final_logits'] - target).square().mean().backward()
        self.assertTrue(bool((head.hidden.weight.grad[8:32].abs().sum(1) > 0).all()))
        self.assertTrue(bool((head.hidden.weight.grad[40:64].abs().sum(1) > 0).all()))
        self.assertTrue(bool((head.raw_magnitude.grad.abs().sum(0) > 0).all()))

    def test_linear_is_affine_inside_fixed_preprocessing_region(self):
        head = ActivityLinear.from_linear(self.state)
        x = self.state['mean'] + .2 * self.state['scale'] * torch.randn(2, 512)
        q = head(x)['final_logits']
        midpoint = head(x.mean(0, keepdim=True))['final_logits'][0]
        torch.testing.assert_close(midpoint, q.mean(0), atol=1e-6, rtol=1e-5)

    def test_activity_only_shape_contract_and_checkpoint_roundtrip(self):
        for kind in MODELS:
            head = make_readout(kind, self.state)
            with self.assertRaises(ValueError):
                head(torch.zeros(1, 4, 10, 10))
            fresh = make_readout(kind)
            fresh.load_state_dict(cpu_state(head), strict=True)
            self.assertTrue(torch.equal(head(self.x)['final_logits'], fresh(self.x)['final_logits']))


if __name__ == '__main__':
    unittest.main()
