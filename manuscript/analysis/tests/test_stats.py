"""Unit tests for msa.stats against hand-checked values and scipy references."""
import itertools
import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msa import paths, stats  # noqa: E402


class Proportions(unittest.TestCase):
    def test_clopper_pearson_known(self):
        self.assertAlmostEqual(stats.cp_interval(24, 24)[0], 0.8575264022774745, places=12)
        self.assertAlmostEqual(stats.cp_interval(0, 8)[1], 0.3694166475528192, places=12)

    def test_wilson_bounds(self):
        lo, hi = stats.wilson(5, 10)
        self.assertTrue(0 < lo < 0.5 < hi < 1)


class PairedBinary(unittest.TestCase):
    def test_mcnemar_six_zero(self):
        r = stats.mcnemar_counts(0, 6, 0, 0)
        self.assertAlmostEqual(r['p_value'], 0.03125)
        self.assertAlmostEqual(r['min_attainable_p'], 0.03125)
        self.assertTrue(r['informative'])

    def test_mcnemar_no_discordance(self):
        r = stats.mcnemar([True, True, False], [True, True, False])
        self.assertEqual(r['p_value'], 1.0)
        self.assertFalse(r['informative'])

    def test_mcnemar_orientation(self):
        r = stats.mcnemar([True] * 9 + [False] * 9, [True] * 18)
        self.assertEqual((r['a_only'], r['b_only']), (0, 9))
        self.assertAlmostEqual(r['p_value'], 2 * 0.5 ** 9)
        self.assertLess(r['effect'], 0)

    def test_newcombe_contains_estimate(self):
        lo, hi = stats.newcombe_paired(10, 6, 1, 3)
        self.assertLess(lo, (6 - 1) / 20)
        self.assertGreater(hi, (6 - 1) / 20)

    def test_stratified(self):
        r = stats.stratified_mcnemar([(6, 0), (5, 0), (7, 0), (6, 0)])
        self.assertAlmostEqual(r['p_value'], 2 * 0.5 ** 24)


class PairedContinuous(unittest.TestCase):
    def test_sign(self):
        r = stats.sign_test([1, 2, 3, 4, 5, 6, 0])
        self.assertAlmostEqual(r['p_value'], 0.03125)
        self.assertEqual(r['n_zero'], 1)

    def test_signed_rank_matches_scipy_exact(self):
        x = np.array([1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30])
        y = np.array([0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29])
        d = x - y
        ref = st.wilcoxon(d, method='exact').pvalue
        r = stats.signed_rank(d, 'unit:wilcoxon')
        self.assertAlmostEqual(r['p_value'], ref, places=12)
        self.assertEqual(r['p_method'], 'exact')

    def test_hl_interval_inverts_test(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0.4, 1.0, 12)
        hl, lo, hi, method = stats.hodges_lehmann(x)
        self.assertEqual(method, 'walsh_exact')
        self.assertTrue(lo <= hl <= hi)
        eps = 1e-7
        inside = [st.wilcoxon(x - v, method='exact').pvalue for v in (lo + eps, hi - eps)]
        outside = [st.wilcoxon(x - v, method='exact').pvalue for v in (lo - eps, hi + eps)]
        self.assertTrue(all(p >= 0.05 for p in inside), inside)
        self.assertTrue(all(p < 0.05 for p in outside), outside)

    def test_signed_rank_zero_handling(self):
        r = stats.signed_rank([0, 0, 1, 2, 3], 'unit:zeros')
        self.assertEqual(r['n_zero'], 2)
        self.assertEqual(r['n_used'], 3)


class Trend(unittest.TestCase):
    def test_cochran_armitage_matches_enumeration(self):
        scores = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        outcomes = [1, 0, 0, 1, 1, 0, 1, 1, 1]
        r = stats.cochran_armitage(scores, outcomes)
        s = np.array(scores)
        k = sum(outcomes)
        tobs = float(np.dot(s, outcomes))
        ts = []
        for idx in itertools.combinations(range(len(s)), k):
            ts.append(s[list(idx)].sum())
        ts = np.array(ts, float)
        e = ts.mean()
        ref = np.mean(np.abs(ts - e) >= abs(tobs - e) - 1e-9)
        self.assertAlmostEqual(r['p_value'], ref, places=12)

    def test_page_two_sided(self):
        data = np.array([[1, 2, 3], [1, 3, 2], [0, 2, 4], [1, 2, 5]], float)
        r = stats.page(data)
        self.assertEqual(r['direction'], 'increasing')
        self.assertLessEqual(r['p_value'], 1)

    def test_spearman_perfect(self):
        x = np.arange(10.0)
        r = stats.spearman(x, x * 2 + 1, 'unit:spearman')
        self.assertAlmostEqual(r['statistic'], 1.0)
        self.assertLess(r['p_value'], 0.001)

    def test_jonckheere_increasing(self):
        vals = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        grp = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        r = stats.jonckheere(vals, grp, 'unit:jt')
        self.assertEqual(r['statistic'], 27.0)
        self.assertLess(r['p_value'], 0.01)


class Multiplicity(unittest.TestCase):
    def test_holm(self):
        np.testing.assert_allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_bh(self):
        np.testing.assert_allclose(stats.bh([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])


class Guard(unittest.TestCase):
    def _run(self, code):
        return subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
                              cwd=str(paths.ANALYSIS), timeout=120)

    def test_blocks_write_outside(self):
        target = (paths.ROOT / 'README.md').as_posix()
        code = ("import msa.guard as g; g.install()\n"
                f"open({target!r}, 'a')\n")
        r = self._run(code)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('read-only guard', r.stderr)

    def test_allows_work(self):
        target = paths.work('guard-probe.txt').as_posix()
        code = ("import msa.guard as g, os; g.install()\n"
                "from pathlib import Path\n"
                f"p = Path({target!r}); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('ok'); p.unlink()\n")
        r = self._run(code)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == '__main__':
    unittest.main()
