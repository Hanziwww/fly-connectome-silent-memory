"""Unit tests for record extractors, the status vocabulary and the CSV conventions."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msa import csvout, memory, taxonomy  # noqa: E402


class FirstPush(unittest.TestCase):
    def test_push_detected(self):
        # Player at 44 under the box at 34; moving right (3) then up (0) then left (2) pushes the box from 35 into 34-1.
        decisions = [dict(state=[44, [34]], action=3), dict(state=[45, [34]], action=0), dict(state=[35, [34]], action=2)]
        self.assertEqual(memory.derive_first_push(decisions), 2)

    def test_no_push(self):
        decisions = [dict(state=[44, [34]], action=3), dict(state=[45, [34]], action=1)]
        self.assertIsNone(memory.derive_first_push(decisions))


class Vocabulary(unittest.TestCase):
    def test_irreversible_labels_merge(self):
        self.assertEqual(taxonomy.behavior('wrong_irreversible_push'), taxonomy.behavior('irreversible_deadlock'))

    def test_qa_checks(self):
        self.assertEqual(taxonomy.QA_CHECKS, {'sham', 'restore', 'same_cue_swap'})


class CuePairs(unittest.TestCase):
    def test_classes(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'panels'))
        from panels import p4_memory
        H = pd.DataFrame([dict(model='m', family='M0', learner='l', kind='connectome', context=0, visible=False, delay_ms=d, prefix_windows=0,
                               split='train', cue=c, solved=s, first_push_correct=s, first_push_action=pd.NA)
                          for d, (s0, s1) in zip((0, 40, 160), ((True, True), (True, False), (False, False))) for c, s in ((0, s0), (1, s1))])
        P = p4_memory.cue_pairs(H)
        self.assertEqual(sorted(P.pair_class.tolist()), ['both', 'none', 'one'])


class Conventions(unittest.TestCase):
    def test_format(self):
        self.assertEqual(csvout._fmt(0.1), '0.1')
        self.assertEqual(csvout._fmt(True), 'true')
        self.assertEqual(csvout._fmt(np.float64('nan')), '')
        self.assertEqual(csvout._fmt(np.int64(7)), '7')
        self.assertEqual(csvout._fmt(1e-07), '1e-07')

    def test_undocumented_column_rejected(self):
        with self.assertRaises(ValueError):
            csvout.write_table(pd.DataFrame({'mystery': [1]}), 'qa/_unit_test.csv', title='t', describe={})


if __name__ == '__main__':
    unittest.main()
