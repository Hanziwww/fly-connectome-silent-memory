"""Verify generic multi-box lessons and simultaneous gradient accumulation."""
from workspace import ARCHIVE, RUNS
import json
import unittest

import numpy as np
import torch

from bootstrap import HERE
from common import configure
from game import solved, state_from
from hybrid_model import padded_forward
from hybrid_tasks import loss
from lesson_tasks import trace
from linear_joint_tasks import SuffixCurriculum
from linear_joint_training import can_replace, joint_gradients
from progressive_tasks import forward_certificate
from readout_models import ActivityLinear


def p03():
    return next(l for l in json.loads((ARCHIVE / 'progressive/20261001-single-seed01/levels.json').read_text()) if l['id'] == 'P03')


class JointLinearTests(unittest.TestCase):
    def test_p03_is_independently_certified_and_harder_on_all_selected_dimensions(self):
        level = p03()
        self.assertEqual((level['size'], level['boxes'], level['optimal_steps']), (8, 2, 22))
        self.assertTrue(forward_certificate(level)['verified'])
        self.assertEqual(level['floor_cells'], 31)

    def test_every_suffix_has_a_valid_witness_and_no_p01_state(self):
        lessons = SuffixCurriculum(p03())
        for depth in range(1, 23):
            level = lessons.at_depth(depth)
            rows, end = trace(lessons.game, state_from(level['start']), level['reference_actions'][-depth:])
            self.assertTrue(solved(lessons.game, end))
            self.assertEqual(len(rows), depth)
            data = lessons.data(depth)
            self.assertTrue(all(len(s[1]) == 2 for s in data['states']))
            self.assertEqual(len(data['states']), 1)
            self.assertEqual(len(lessons.levels(depth)), depth + 1)

    def test_successful_detour_is_accepted_and_snapshot_restores_anchors(self):
        lessons = SuffixCurriculum(p03())
        # P03 starts in a horizontal corridor; right-left is a valid detour.
        actions = [3, 2] + p03()['reference_actions']
        lessons.observe_success(p03(), dict(status='solved', actions=actions))
        restored = SuffixCurriculum(p03(), lessons.snapshot())
        self.assertEqual(lessons.snapshot(), restored.snapshot())
        self.assertGreater(len(restored.data(22)['protected_clauses']), 1)
        with self.assertRaises(ValueError):
            lessons.observe_success(p03(), dict(status='repeated_state', actions=[3, 2]))

    def test_two_pass_gradients_match_direct_joint_chain_rule(self):
        configure(10)
        class ToyEncoder:
            def __init__(self):
                self.batch = 2
                self.core = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(400, 512), torch.nn.Tanh())
            def forward(self, boards):
                return self.core(boards)
            def encode(self, boards, check):
                with torch.no_grad():
                    parts = []
                    for start in range(0, len(boards), self.batch):
                        padded = np.zeros((self.batch, 4, 10, 10), dtype='float32')
                        chunk = boards[start:start + self.batch]
                        padded[:len(chunk)] = chunk
                        parts.append(self.forward(torch.from_numpy(padded))[:len(chunk)].numpy())
                    return np.concatenate(parts)
            def release_graph(self):
                for p in self.core.parameters():
                    p.grad = None
        encoder, head = ToyEncoder(), ActivityLinear()
        data = dict(boards=np.random.normal(size=(3, 4, 10, 10)).astype('float32'),
            legal=np.ones((3, 4), dtype=bool),
            clauses=[dict(atoms=[(i, (i % 4,))]) for i in range(3)], bridge_clause=0, protected_clauses=[1, 2])
        loss(padded_forward(head, encoder.forward(torch.from_numpy(data['boards']))), data)[0].backward()
        expected = [p.grad.clone() for p in list(encoder.core.parameters()) + list(head.parameters())]
        info = joint_gradients(encoder, head, data)
        for p, gradient in zip(list(encoder.core.parameters()) + list(head.parameters()), expected):
            torch.testing.assert_close(p.grad, gradient, rtol=1e-4, atol=2e-6)
        self.assertGreater(info['gradient_norms']['core'], 0)
        self.assertGreater(info['gradient_norms']['head'], 0)
        optimizer = torch.optim.Adam(list(encoder.core.parameters()) + list(head.parameters()), lr=.001)
        before_core = next(encoder.core.parameters()).detach().clone()
        before_head = next(head.parameters()).detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before_core, next(encoder.core.parameters())))
        self.assertFalse(torch.equal(before_head, next(head.parameters())))

    def test_original_success_overrides_loss_and_old_suffix_regression(self):
        before = dict(solved=False, retained=True, stage_passed=False, loss=.1)
        self.assertTrue(can_replace(dict(solved=True, retained=False, stage_passed=True, loss=100.), before))
        self.assertFalse(can_replace(dict(solved=False, retained=False, stage_passed=False, loss=0.), before))


if __name__ == '__main__':
    unittest.main()
