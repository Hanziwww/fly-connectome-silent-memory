"""CPU contracts for intervention isolation and experiment-specific viewer input."""
import copy
import unittest
import numpy as np
import torch
from verify_repair import FixedSignal, same_rollout, substituted_core
from viewer_catalog import repair_bundle
from serve import Player


class RepairViewerTests(unittest.TestCase):
    def test_substitutions_preserve_source_and_only_revert_target_family(self):
        anchor = {key: torch.arange(12, dtype=torch.float32) / 100 for key in
                  ('pre_log_gain', 'post_log_gain', 'threshold_mv')}
        learned = {key: value + torch.linspace(0, .01, 12) for key, value in anchor.items()}
        saved = copy.deepcopy(learned)
        changed = substituted_core(learned, anchor, 'undo_threshold_mv', None)
        self.assertTrue(torch.equal(changed['threshold_mv'], anchor['threshold_mv']))
        self.assertTrue(torch.equal(changed['pre_log_gain'], learned['pre_log_gain']))
        for key in learned:
            self.assertTrue(torch.equal(learned[key], saved[key]))
        shuffled = substituted_core(learned, anchor, 'shuffle_delta', 381)
        for key in learned:
            torch.testing.assert_close(torch.sort(shuffled[key] - anchor[key]).values,
                                       torch.sort(learned[key] - anchor[key]).values, atol=1e-7, rtol=0)

    def test_signal_control_is_independent_of_the_board(self):
        feature = np.arange(512, dtype='float32')
        source = FixedSignal(feature)
        feature[:] = -1
        boards = np.stack([np.zeros((4, 10, 10)), np.ones((4, 10, 10))])
        output = source.encode(boards)
        np.testing.assert_array_equal(output[0], np.arange(512))
        np.testing.assert_array_equal(output[0], output[1])

    def test_restore_requires_logits_and_actions_not_only_success(self):
        first = dict(complete=True, records=[dict(status='solved', actions=[0], decisions=[dict(brain_logits=[1, 2, 3, 4])])])
        second = copy.deepcopy(first)
        self.assertTrue(same_rollout(first, second))
        second['records'][0]['decisions'][0]['brain_logits'][0] = 1.001
        self.assertFalse(same_rollout(first, second))

    def test_registry_and_suite_restriction(self):
        paths, trials, level = repair_bundle()
        self.assertEqual(len(trials), 3)
        self.assertEqual(len({str(p) for p in paths.values()}), 4)
        player = Player.__new__(Player)
        player.levels = {'B00': level}
        msg = dict(level_id='B00', state=level['start'], model='repair10')
        self.assertEqual(player.validate(msg)[-1], 'repair10')
        with self.assertRaises(ValueError):
            player.validate(dict(msg, model='stage1'))


if __name__ == '__main__':
    unittest.main()
