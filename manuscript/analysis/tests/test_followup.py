"""Unit tests for the follow-up analyses (panels/p11_followup.py)."""
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panels import p11_followup as p11  # noqa: E402


def dense_ledoit_wolf(X):
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    b2 = min(sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / n ** 2 / p, d2)
    a = b2 / d2
    return a * mu * np.eye(p) + (1 - a) * S


class ShrinkageLDA(unittest.TestCase):
    def test_gram_solve_equals_dense_ledoit_wolf(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(20, 60)) * np.linspace(0.5, 2.0, 60)
        X -= X.mean(0)
        v = rng.normal(size=60)
        np.testing.assert_allclose(p11._lw_solve(X, v), np.linalg.solve(dense_ledoit_wolf(X), v), rtol=1e-8, atol=1e-10)

    def test_separable_cues_are_decoded_across_timings(self):
        rng = np.random.default_rng(6)
        axis = rng.normal(size=40)
        Z, y, fold = [], [], []
        for t in range(12):
            drift = rng.normal(scale=3.0, size=40)          # a timing-specific offset shared by both cues
            for cue in (0, 1):
                Z.append(drift + (1 if cue else -1) * axis + rng.normal(scale=0.3, size=40))
                y.append(cue)
                fold.append(t)
        self.assertEqual(p11._loto_accuracy(np.array(Z), np.array(y), np.array(fold)), 1.0)

    def test_no_cue_information_gives_chance(self):
        rng = np.random.default_rng(7)
        Z = rng.normal(size=(70, 30))
        y = np.tile([0, 1], 35)
        fold = np.repeat(np.arange(35), 2)
        self.assertLess(abs(p11._loto_accuracy(Z, y, fold) - 0.5), 0.25)


class PairedEquivalence(unittest.TestCase):
    CASES = [(5, 5, 26, 0.1), (5, 5, 26, -0.1), (2, 12, 26, 0.1), (14, 9, 52, -0.1), (3, 0, 14, 0.1), (0, 4, 14, -0.1)]

    def test_restricted_mle_maximises_the_likelihood(self):
        for b, c, n, d0 in self.CASES:
            with self.subTest(b=b, c=c, n=n, d0=d0):
                def nll(q):
                    rest = 1 - 2 * q - d0
                    if q <= 0 or q + d0 <= 0 or rest <= 0:
                        return 1e18
                    return -(b * np.log(q + d0) + c * np.log(q) + (n - b - c) * np.log(rest))
                r = optimize.minimize_scalar(nll, bounds=(max(1e-12, -d0 + 1e-12), (1 - d0) / 2 - 1e-12), method='bounded',
                                             options={'xatol': 1e-12})
                A, B, C = 2 * n, -b - c + (2 * n - b + c) * d0, -c * d0 * (1 - d0)
                q = (np.sqrt(B * B - 4 * A * C) - B) / (2 * A)
                self.assertAlmostEqual(q, r.x, places=6)

    def test_zero_difference_null_is_the_mcnemar_score(self):
        self.assertAlmostEqual(p11._tango_z(9, 3, 40, 0.0), (9 - 3) / np.sqrt(9 + 3), places=12)

    def test_interval_matches_the_one_sided_tests(self):
        t = p11.tost_paired(14, 9, 52)
        self.assertAlmostEqual(p11._tango_z(14, 9, 52, t['ci90_low']), p11.Z_TOST, places=6)
        self.assertAlmostEqual(p11._tango_z(14, 9, 52, t['ci90_high']), -p11.Z_TOST, places=6)
        self.assertEqual(t['equivalent'], t['ci90_low'] > -0.1 and t['ci90_high'] < 0.1)

    def test_large_balanced_sample_is_equivalent_small_is_not(self):
        self.assertTrue(p11.tost_paired(60, 58, 400)['equivalent'])
        small = p11.tost_paired(0, 0, 14)
        self.assertFalse(small['equivalent'])
        self.assertGreater(small['margin_supported'], 0.1)


class DenseRecords(unittest.TestCase):
    def test_streamed_rows_equal_the_array(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(47, 3, 11)).astype(np.float32)
        buf = io.BytesIO()
        np.savez_compressed(buf, values=x, other=np.arange(3))
        data = buf.getvalue()
        parts = [(s, a.copy()) for s, a in p11._npz_rows(data, 'values', 10)]
        self.assertEqual([s for s, _ in parts], [0, 10, 20, 30, 40])
        np.testing.assert_array_equal(np.concatenate([a for _, a in parts]), x)

    def test_voltage_split_is_exact_without_spikes(self):
        rng = np.random.default_rng(9)
        a = np.exp(-1 / 20)
        n, T = 50, 120
        dg, dad = rng.normal(size=(T, n)), np.cumsum(rng.normal(size=(T, n)), 0) * 0.1
        dv = rng.normal(size=n)
        split = p11.VoltageSplit(dv, dad[0], a)
        for t in range(1, T):
            dv = a * dv + (1 - a) * (dg[t] - dad[t - 1])     # the subthreshold LIF update of the cue difference
            parts = split.step(dg[t], dad[t])
            np.testing.assert_allclose(sum(parts), dv, rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(p11._shares(dv, list(parts))[-1], 0.0, places=10)

    def test_decay_time_is_recovered(self):
        ms = np.arange(1, 400)
        tau, r2 = p11._decay_tau(ms, 7.0 * np.exp(-2 * ms / 200.0), 50, 300)
        self.assertAlmostEqual(tau, 200.0, places=6)
        self.assertAlmostEqual(r2, 1.0, places=10)


class PartialSpearman(unittest.TestCase):
    def test_shared_dependence_on_the_control_is_removed(self):
        rng = np.random.default_rng(12)
        z = rng.normal(size=60)
        x, y = z + 0.5 * rng.normal(size=60), z + 0.5 * rng.normal(size=60)
        r = p11.partial_spearman(x, y, z, 'unit.partial.a')
        self.assertLess(abs(r['statistic']), 0.3)
        self.assertGreater(abs(p11.stats.spearman(x, y, 'unit.plain')['statistic']), 0.6)

    def test_direct_association_survives_and_is_deterministic(self):
        rng = np.random.default_rng(13)
        z = rng.normal(size=40)
        x = rng.normal(size=40)
        y = x + 0.3 * rng.normal(size=40) + z
        a = p11.partial_spearman(x, y, z, 'unit.partial.b')
        b = p11.partial_spearman(x, y, z, 'unit.partial.b')
        self.assertGreater(a['statistic'], 0.7)
        self.assertLess(a['p_value'], 0.01)
        self.assertEqual(a, b)


class EnsembleCounts(unittest.TestCase):
    def test_within_set_count_equals_the_submatrix(self):
        rng = np.random.default_rng(10)
        A = (sp.random(300, 300, density=0.03, random_state=11, format='csr') != 0).tocsr()
        mask = np.zeros(300, bool)
        for _ in range(5):
            idx = np.sort(rng.choice(300, 40, replace=False))
            self.assertEqual(p11._within(A, idx, mask), A[idx][:, idx].nnz)
            self.assertFalse(mask.any())

    def test_pools_group_cells_by_stratum(self):
        codes = np.array([2, 0, 2, 1, 0, 2])
        pools = p11._pools(codes)
        self.assertEqual({k: v.tolist() for k, v in pools.items()}, {0: [1, 4], 1: [3], 2: [0, 2, 5]})


class SignPropagation(unittest.TestCase):
    def setUp(self):
        W = sp.random(60, 60, density=0.1, random_state=21, format='csr')
        W.data = np.round(W.data * 20) + 1                       # synapse counts
        W.data *= np.where(np.random.default_rng(22).random(W.nnz) < 0.4, -1.0, 1.0)
        W[5, :] = 0                                              # a cell without input
        self.W = W.tocsr()

    def test_rows_are_normalised_by_their_absolute_input(self):
        Wn = p11._row_normalised(self.W)
        total = np.asarray(abs(Wn).sum(1)).ravel()
        has = np.asarray(abs(self.W).sum(1)).ravel() > 0
        np.testing.assert_allclose(total[has], 1.0, rtol=1e-12)
        self.assertTrue(np.all(total[~has] == 0))
        np.testing.assert_array_equal(np.sign(Wn.toarray()), np.sign(self.W.toarray()))

    def test_propagation_is_the_neumann_series(self):
        Wn = p11._row_normalised(self.W)
        U = np.random.default_rng(23).normal(size=(60, 3))
        for gain in p11.PROP_GAINS:
            exact = np.linalg.solve(np.eye(60) - gain * Wn.toarray(), U) - U
            np.testing.assert_allclose(p11._propagate(Wn, U, gain), exact, atol=1e-7)

    def test_motif_counts(self):
        # cells 0, 1 prefer +, cell 2 prefers -; 1<-0 excitatory and 2<-0 inhibitory (rows postsynaptic)
        W = sp.csr_matrix(np.array([[0, 0, 0], [3.0, 0, 0], [-2.0, 0, 0]]))
        stat, same, exc = p11._motif(W, np.arange(3), np.array([1.0, 2.0, -1.0]))
        self.assertEqual(stat, 1.0)
        self.assertEqual((same.tolist(), exc.tolist()), ([True, False], [True, False]))
        self.assertTrue(np.isnan(p11._motif(W, np.arange(2), np.array([1.0, 2.0]))[0]))   # no opposite-sign connection

    def test_single_input_propagation_gives_the_full_motif(self):
        # a random tree: every cell has one input, so its propagated sign is its parent's sign times the synapse sign
        rng = np.random.default_rng(24)
        parent = [rng.integers(0, i) for i in range(1, 80)]
        W = sp.csr_matrix((rng.choice([-1.0, 1.0], 79) * rng.integers(1, 9, 79), (np.arange(1, 80), parent)), shape=(80, 80))
        U = np.zeros((80, 1))
        U[0] = 1.0
        x = p11._propagate(p11._row_normalised(W), U, 0.5)[:, 0] + U[:, 0]
        stat, _, _ = p11._motif(W, np.arange(80), x)
        self.assertEqual(stat, 1.0)


class DriveSplit(unittest.TestCase):
    def test_shares_sum_to_one(self):
        rng = np.random.default_rng(25)
        dg, da = rng.normal(size=200), rng.normal(size=200)
        e, s, a = p11._drive_split(dg, da)
        self.assertAlmostEqual(e, float((dg - da) @ (dg - da)), places=9)
        self.assertAlmostEqual(s + a, 1.0, places=12)
        self.assertEqual(p11._drive_split(dg, np.zeros(200))[1:], (1.0, 0.0))
        self.assertTrue(np.isnan(p11._drive_split(np.zeros(5), np.zeros(5))[1]))

    def test_trace_decay_recovers_the_time_constant(self):
        d = np.array([0, 20, 40, 80, 160, 320, 640])
        tau, r2, used = p11._trace_decay(d, 3.0 * np.exp(-2 * d / 200.0))
        self.assertAlmostEqual(tau, 200.0, places=6)
        self.assertAlmostEqual(r2, 1.0, places=9)
        self.assertEqual(used, 7)


if __name__ == '__main__':
    unittest.main()
