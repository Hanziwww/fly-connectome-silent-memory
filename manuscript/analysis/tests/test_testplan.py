"""Unit tests for the test-family declarations (msa/testplan.py) and the map bootstrap of the consistency tests."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msa import testplan  # noqa: E402
from panels import p3_brain_readout as p3  # noqa: E402


class Families(unittest.TestCase):
    def test_posthoc_family_is_never_primary(self):
        bad = [f['family'] for f in testplan.FAMILIES if f['posthoc'] and f['primary']]
        self.assertEqual(bad, [])

    def test_declaring_a_primary_posthoc_family_fails(self):
        n = len(testplan.FAMILIES)
        with self.assertRaises(ValueError):
            testplan.family('X.test', 'none', 'B', 'holm', True, 'test only', posthoc=True)
        self.assertEqual(len(testplan.FAMILIES), n)

    def test_corrected_families(self):
        fams = {f['family']: f for f in testplan.FAMILIES}
        for name in ('F4e.hidden', 'F5.memory_use.M0', 'F6g.ensemble', 'F6g.sign'):
            with self.subTest(name=name):
                self.assertTrue(fams[name]['posthoc'])
                self.assertFalse(fams[name]['primary'])


class MapBootstrap(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        shared = rng.normal(size=500)
        X = np.abs(shared[None, :] + rng.normal(scale=1.5, size=(12, 500)))
        R, self.Z = p3._standardized_ranks(X)
        k = 5
        self.T = (R > 500 - k).astype(np.float64)
        self.kk = self.T.sum(1)
        M, n = self.Z.shape
        rho = (np.sum(self.Z.sum(0) ** 2) - M * n) / (M * (M - 1) * n)
        S = self.T.sum(0)
        ov = (np.sum(S ** 2) - self.kk.sum()) / (M * (M - 1) * self.kk.mean())
        self.observed = (rho, ov)

    def test_interval_brackets_the_observed_value(self):
        (r_lo, r_hi), (o_lo, o_hi) = p3._map_bootstrap(self.Z, self.T, self.kk, self.observed, 'unit.test')
        self.assertLessEqual(r_lo, self.observed[0])
        self.assertGreaterEqual(r_hi, self.observed[0])
        self.assertLessEqual(o_lo, self.observed[1])
        self.assertGreaterEqual(o_hi, self.observed[1])
        self.assertGreater(r_lo, 0.0)                       # a shared component gives positive consistency

    def test_deterministic_and_checks_the_observed_statistic(self):
        a = p3._map_bootstrap(self.Z, self.T, self.kk, self.observed, 'unit.test')
        b = p3._map_bootstrap(self.Z, self.T, self.kk, self.observed, 'unit.test')
        np.testing.assert_array_equal(a[0], b[0])
        with self.assertRaises(RuntimeError):
            p3._map_bootstrap(self.Z, self.T, self.kk, (self.observed[0] + 0.1, self.observed[1]), 'unit.test')


if __name__ == '__main__':
    unittest.main()
