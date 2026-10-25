"""CPU contracts for the MLP boundary, recovery objective and activity cache."""
from workspace import ARCHIVE, RUNS
import json
import unittest

import numpy as np
import torch

from bootstrap import HERE
from common import cpu_state
from heads import ActivityHead
from hybrid_model import ActivityMLP, padded_forward
from hybrid_runtime import FeatureBank
from hybrid_tasks import dataset, loss, numpy_objective, objective, response_coefficients
from joint_tasks import can_replace


class HybridTests(unittest.TestCase):
    def test_initialization_preserves_linear_function_and_all_parameters_can_train(self):
        torch.manual_seed(10)
        old = ActivityHead()
        old.logit_scale.fill_(2.)
        old.mean.normal_()
        old.scale.uniform_(.1, 2.)
        model = ActivityMLP.from_linear(cpu_state(old))
        x = torch.randn(29, 512) * 20
        expected = old(x, correction=False)['final_logits']
        actual = model(x)['final_logits']
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 33092)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(set(dict(model.named_buffers())), {'mean', 'scale'})

    def test_model_rejects_board_or_wrong_feature_inputs(self):
        model = ActivityMLP()
        for shape in [(1, 4, 10, 10), (1, 400), (512,)]:
            with self.assertRaises(ValueError):
                model(torch.zeros(shape))
        with self.assertRaises(ValueError):
            padded_forward(model, torch.zeros(129, 512))

    def test_mlp_gradients_reach_features_without_changing_normalization(self):
        torch.manual_seed(10)
        model = ActivityMLP()
        x = torch.randn(3, 512, requires_grad=True)
        q = padded_forward(model, x)
        (q[:, 0] - q[:, 1]).square().sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertIsNone(model.mean.grad)
        self.assertIsNone(model.scale.grad)

    def test_recovery_objective_torch_numpy_and_finite_differences_agree(self):
        data = dict(bridge_clause=0, protected_clauses=[1, 2, 3])
        values = np.array([-.4, -.3, .01, .7])
        x = torch.tensor(values, requires_grad=True)
        y = objective(x, data)
        y.backward()
        expected, derivative = numpy_objective(values, data)
        self.assertAlmostEqual(float(y.detach()), expected, places=12)
        np.testing.assert_allclose(x.grad.numpy(), derivative, atol=1e-12, rtol=1e-12)
        for i in range(4):
            a, b = values.copy(), values.copy()
            a[i] += 1e-6
            b[i] -= 1e-6
            finite = (numpy_objective(a, data)[0] - numpy_objective(b, data)[0]) / 2e-6
            self.assertAlmostEqual(finite, derivative[i], places=7)

    def test_broken_dependencies_have_recovery_signal_and_healthy_ones_do_not(self):
        data = dict(bridge_clause=0, protected_clauses=[1, 2])
        _, derivative = numpy_objective([-.2, -.8, .7], data)
        self.assertLess(derivative[1], 0)
        self.assertEqual(derivative[2], 0)

    def test_response_fit_can_repair_negative_anchor(self):
        data = dict(bridge_clause=0, protected_clauses=[1, 2])
        base = np.array([-.3, -.2, .05])
        response = np.array([[.6, 0], [0, .6], [0, 0]])
        coefficients, info = response_coefficients(base, response, data)
        predicted = base + response @ coefficients
        self.assertTrue(info['usable'])
        self.assertGreater(predicted[0], base[0])
        self.assertGreater(predicted[1], base[1] + .1)
        self.assertGreaterEqual(predicted[2], .03)

    def test_dataset_uses_current_connection_and_required_dependencies(self):
        folder = ARCHIVE / 'frontier/20261001-directions01/trials/G'
        level = json.loads((folder / 'level.json').read_text())
        snapshot = json.loads((folder / 'lessons.json').read_text())
        before = json.dumps(snapshot, sort_keys=True)
        _, data, levels = dataset(level, snapshot, 6)
        self.assertEqual(len(data['boards']), 7)
        self.assertEqual(len(data['protected_clauses']), 6)
        self.assertEqual(data['states'][data['clauses'][data['bridge_clause']]['atoms'][0][0]], (45, (54,)))
        self.assertEqual(levels[0]['id'], 'P01')
        self.assertEqual(before, json.dumps(snapshot, sort_keys=True))
        value, _, _ = loss(torch.zeros(len(data['boards']), 4), data)
        self.assertTrue(torch.isfinite(value))

    def test_feature_cache_reuses_same_brain_and_rejects_stale_brain(self):
        class FakeEncoder:
            def __init__(self):
                self.core, self.calls = torch.nn.Linear(1, 1, bias=False), 0

            def set_core(self, state):
                self.core.load_state_dict(state)

            def encode(self, boards, check):
                self.calls += 1
                return np.full((len(boards), 512), float(self.core.weight.detach()[0, 0]), dtype='float32')

        encoder = FakeEncoder()
        bank = FeatureBank(encoder, cpu_state(encoder.core))
        boards = np.zeros((2, 4, 10, 10), dtype='uint8')
        first = bank.encode(boards)
        np.testing.assert_array_equal(first, bank.encode(boards))
        self.assertEqual(encoder.calls, 1)
        with torch.no_grad():
            encoder.core.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'invalidate'):
            bank.encode(boards)
        new_bank = FeatureBank(encoder, cpu_state(encoder.core))
        np.testing.assert_allclose(new_bank.encode(boards), first + 1)

    def test_original_solution_overrides_auxiliary_regression(self):
        candidate = dict(solved=True, retained=False, loss=100.)
        champion = dict(solved=False, retained=True, loss=0.)
        self.assertTrue(can_replace(candidate, champion))
        self.assertFalse(can_replace(dict(solved=False, retained=False, loss=-1.), champion))


if __name__ == '__main__':
    unittest.main()
