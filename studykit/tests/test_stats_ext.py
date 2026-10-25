"""studykit.stats_ext against scipy references, brute-force enumeration and closed forms."""
import itertools
import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from studykit import stats_ext as sx  # noqa: E402


def brute_sign_flip(d, alternative):
    d = np.asarray(d, float)
    total = d.sum()
    null = np.array([np.dot(s, d) for s in itertools.product((1, -1), repeat=len(d))])
    if alternative == 'two-sided':
        return np.mean(np.abs(null) >= abs(total) - 1e-12)
    if alternative == 'greater':
        return np.mean(null >= total - 1e-12)
    return np.mean(null <= total + 1e-12)


class SignFlip(unittest.TestCase):
    def test_exact_p_matches_enumeration_and_scipy(self):
        rng = np.random.default_rng(1)
        for trial in range(6):
            d = rng.normal(0.4, 1.0, size=8 + trial)
            for alternative in ('two-sided', 'greater', 'less'):
                r = sx.sign_flip(d, alternative=alternative, ci=False)
                self.assertEqual(r['p_method'], 'exact')
                self.assertAlmostEqual(r['p_value'], brute_sign_flip(d, alternative), places=12)
                ref = st.permutation_test((d,), np.mean, permutation_type='samples', n_resamples=np.inf,
                                          alternative=alternative)
                self.assertAlmostEqual(r['p_value'], ref.pvalue, places=12)

    def test_paired_input_zeros_and_smallest_p(self):
        x, y = np.array([3.0, 4, 5, 6, 7, 8]), np.array([1.0, 2, 3, 4, 5, 8])
        r = sx.sign_flip(x, y, ci=False)
        self.assertEqual((r['n'], r['n_used'], r['n_zero']), (6, 5, 1))
        self.assertAlmostEqual(r['p_value'], 2 / 2 ** 5)
        self.assertAlmostEqual(r['min_attainable_p'], 2 / 2 ** 5)
        ten = sx.sign_flip(np.arange(1, 11), ci=False)
        self.assertAlmostEqual(ten['p_value'], 2 / 1024)  # the plan's n = 10: smallest two-sided p is 2 / 2**10
        none = sx.sign_flip([0.0, 0.0])
        self.assertEqual((none['p_value'], none['informative']), (1.0, False))
        with self.assertRaises(ValueError):
            sx.sign_flip([1, 2], [1])

    def test_monte_carlo_above_max_exact(self):
        d = np.random.default_rng(2).normal(0.3, 1, size=21)
        r = sx.sign_flip(d, ci=False, test_id='mc')
        self.assertEqual((r['p_method'], r['n_resamples']), ('permutation', sx.N_RESAMPLES))
        exact = sx.sign_flip(d, ci=False, max_exact=21)
        self.assertLess(abs(r['p_value'] - exact['p_value']), 4 * r['mc_se'] + 1e-3)
        self.assertEqual(r['p_value'], sx.sign_flip(d, ci=False, test_id='mc')['p_value'], 'not reproducible')

    def test_interval_inverts_the_test(self):
        d = np.random.default_rng(3).normal(1.0, 1.0, size=10)
        r = sx.sign_flip(d, level=0.95)
        lo, hi = r['effect_ci_low'], r['effect_ci_high']
        self.assertLess(lo, d.mean())
        self.assertGreater(hi, d.mean())
        span = d.max() - d.min()
        for bound, outward in ((lo, -1), (hi, 1)):
            inside = sx.sign_flip(d - (bound - outward * 1e-6 * span), ci=False)['p_value']
            outside = sx.sign_flip(d - (bound + outward * 1e-6 * span), ci=False)['p_value']
            self.assertGreater(inside, 0.05)
            self.assertLessEqual(outside, 0.05)
        small = sx.sign_flip([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(small['effect_ci_method'], 'unavailable_small_n')
        self.assertTrue(math.isnan(small['effect_ci_low']))


class Equivalence(unittest.TestCase):
    def test_t_tost_matches_one_sided_t_tests(self):
        d = np.random.default_rng(4).normal(0.02, 0.05, size=12)
        r = sx.tost(d, -0.1, 0.1)
        self.assertAlmostEqual(r['p_lower'], st.ttest_1samp(d, -0.1, alternative='greater').pvalue, places=12)
        self.assertAlmostEqual(r['p_upper'], st.ttest_1samp(d, 0.1, alternative='less').pvalue, places=12)
        self.assertAlmostEqual(r['p_difference'], st.ttest_1samp(d, 0.0).pvalue, places=12)
        self.assertEqual(r['equivalent'], -0.1 < r['effect_ci_low'] and r['effect_ci_high'] < 0.1)
        self.assertEqual(r['conclusion'], 'equivalent')
        wide = sx.tost(np.random.default_rng(5).normal(0, 1, size=6), -0.1, 0.1)
        self.assertEqual(wide['conclusion'], 'inconclusive')
        self.assertGreater(wide['minimal_detectable_effect'], 0.1)
        far = sx.tost(np.random.default_rng(6).normal(1, 0.1, size=10), -0.1, 0.1)
        self.assertEqual(far['conclusion'], 'different')
        with self.assertRaises(ValueError):
            sx.tost(d, 0.1, -0.1)

    def test_mde_has_the_stated_power(self):
        sd, n = 1.3, 10
        mde = sx.mde_t(sd, n, alpha=0.05, power=0.8)
        nc, df = mde / (sd / math.sqrt(n)), n - 1
        tcrit = st.t.ppf(0.975, df)
        self.assertAlmostEqual(st.nct.sf(tcrit, df, nc) + st.nct.cdf(-tcrit, df, nc), 0.8, places=6)

    def test_sign_flip_tost(self):
        d = np.random.default_rng(7).normal(0.0, 0.03, size=10)
        r = sx.tost(d, -0.1, 0.1, method='sign_flip')
        self.assertAlmostEqual(r['p_lower'], sx.sign_flip(d + 0.1, alternative='greater', ci=False)['p_value'])
        self.assertAlmostEqual(r['p_upper'], sx.sign_flip(d - 0.1, alternative='less', ci=False)['p_value'])
        self.assertTrue(r['equivalent'])
        self.assertAlmostEqual(r['p_value'], 1 / 1024)

    def test_binomial_tost(self):
        k, n = 104, 200  # 90% CI about 0.46-0.58, inside 0.4-0.6
        r = sx.tost_binomial(k, n, 0.4, 0.6)
        self.assertAlmostEqual(r['p_lower'], st.binomtest(k, n, 0.4, alternative='greater').pvalue, places=12)
        self.assertAlmostEqual(r['p_upper'], st.binomtest(k, n, 0.6, alternative='less').pvalue, places=12)
        self.assertEqual(r['conclusion'], 'equivalent')
        ci = st.binomtest(k, n).proportion_ci(confidence_level=0.9, method='exact')
        self.assertAlmostEqual(r['effect_ci_low'], ci.low, places=12)
        small = sx.tost_binomial(5, 10, 0.4, 0.6)
        self.assertEqual(small['conclusion'], 'inconclusive')
        mde = small['minimal_detectable_effect']
        xs = np.arange(11)
        reject = np.array([st.binomtest(int(x), 10, 0.5).pvalue < 0.05 for x in xs])
        power = lambda p: st.binom.pmf(xs[reject], 10, p).sum()  # noqa: E731
        self.assertGreaterEqual(min(power(0.5 - mde), power(0.5 + mde)), 0.8)
        self.assertLess(min(power(0.5 - mde + 0.001), power(0.5 + mde - 0.001)), 0.8)


class PairedBinaryEquivalence(unittest.TestCase):
    CASES = [(5, 5, 26, 0.1), (5, 5, 26, -0.1), (2, 12, 26, 0.1), (14, 9, 52, -0.1), (3, 0, 14, 0.1), (0, 4, 14, -0.1)]

    def test_restricted_mle_maximises_the_likelihood(self):
        from scipy import optimize
        for b, c, n, d0 in self.CASES:
            with self.subTest(b=b, c=c, n=n, d0=d0):
                def nll(q):
                    rest = 1 - 2 * q - d0
                    if q <= 0 or q + d0 <= 0 or rest <= 0:
                        return 1e18
                    return -(b * np.log(q + d0) + c * np.log(q) + (n - b - c) * np.log(rest))
                r = optimize.minimize_scalar(nll, bounds=(max(1e-12, -d0 + 1e-12), (1 - d0) / 2 - 1e-12), method='bounded',
                                             options={'xatol': 1e-12})
                q = r.x
                var = n * (2 * q + d0 * (1 - d0))
                self.assertAlmostEqual(sx.tango_z(b, c, n, d0), (b - c - n * d0) / math.sqrt(var), places=4)

    def test_zero_difference_is_the_mcnemar_score(self):
        self.assertAlmostEqual(sx.tango_z(9, 3, 40, 0.0), (9 - 3) / math.sqrt(9 + 3), places=12)

    def test_interval_matches_the_one_sided_tests(self):
        r = sx.tost_paired_binary(14, 9, 52, 0.1)
        z = st.norm.ppf(0.95)
        self.assertAlmostEqual(sx.tango_z(14, 9, 52, r['effect_ci_low']), z, places=6)
        self.assertAlmostEqual(sx.tango_z(14, 9, 52, r['effect_ci_high']), -z, places=6)
        self.assertEqual(r['equivalent'], r['effect_ci_low'] > -0.1 and r['effect_ci_high'] < 0.1)

    def test_no_discordant_pairs_in_a_large_sample_is_equivalent(self):
        self.assertTrue(sx.tost_paired_binary(0, 0, 126, 0.05)['equivalent'])
        self.assertFalse(sx.tost_paired_binary(0, 0, 14, 0.05)['equivalent'])
        self.assertEqual(sx.tost_paired_binary(30, 2, 126, 0.05)['conclusion'], 'different')

    @unittest.skipUnless((ROOT / 'manuscript' / 'analysis' / 'panels').is_dir(), 'manuscript code is not pushed to the GPU hosts')
    def test_agrees_with_the_manuscript_follow_up(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'manuscript' / 'analysis'))
        from panels import p11_followup as p11
        for b, c, n, _ in self.CASES:
            mine, theirs = sx.tost_paired_binary(b, c, n, 0.1), p11.tost_paired(b, c, n, 0.1)
            self.assertAlmostEqual(mine['p_value'], theirs['p_tost'], places=12)
            self.assertAlmostEqual(mine['effect_ci_low'], theirs['ci90_low'], places=9)
            self.assertAlmostEqual(mine['effect_ci_high'], theirs['ci90_high'], places=9)


class Power(unittest.TestCase):
    def test_simulated_power_matches_the_noncentral_t(self):
        effect, n = 0.8, 15

        def simulate(rng, size):
            return rng.normal(effect, 1.0, size=size)

        def test(data):
            return st.ttest_1samp(data, 0.0).pvalue

        r = sx.simulate_power(simulate, test, n, n_sim=4000, test_id='t-power')
        df, tcrit = n - 1, st.t.ppf(0.975, n - 1)
        exact = st.nct.sf(tcrit, df, effect * math.sqrt(n)) + st.nct.cdf(-tcrit, df, effect * math.sqrt(n))
        self.assertLess(abs(r['power'] - exact), 4 * r['mc_se'])
        self.assertLessEqual(r['power_ci_low'], r['power'])
        self.assertEqual(r, sx.simulate_power(simulate, test, n, n_sim=4000, test_id='t-power'))

    def test_sample_size_with_the_sign_flip(self):
        def simulate(rng, size):
            return rng.normal(1.0, 1.0, size=size)

        def test(data):
            return sx.sign_flip(data, ci=False)

        result = sx.sample_size(simulate, test, n_values=range(4, 16, 2), target=0.8, n_sim=300)
        powers = [c['power'] for c in result['curve']]
        self.assertEqual(powers[0], 0.0, 'n = 4 cannot reach p < 0.05 (smallest p = 0.125)')
        self.assertIsNotNone(result['n'])
        self.assertEqual(result['n'], next(c['n'] for c in result['curve'] if c['power'] >= 0.8))


class DoseResponse(unittest.TestCase):
    dose = np.repeat([0.0, 0.5, 1.0, 1.5, 2.0, 3.0], 4)
    truth = dict(bottom=0.2, top=0.9, ed50=1.1, hill=2.5)

    def test_4pl_recovers_noise_free_parameters(self):
        y = sx.logistic4(self.dose, **self.truth)
        fit = sx.fit_logistic4(self.dose, y)
        for key, value in self.truth.items():
            self.assertAlmostEqual(fit['params'][key], value, delta=1e-5 * max(1, abs(value)))
        self.assertAlmostEqual(sx.logistic4(0.0, **self.truth), self.truth['top'])
        self.assertAlmostEqual(sx.logistic4(self.truth['ed50'], **self.truth), 0.55)

    def test_4pl_matches_scipy_curve_fit_and_gaussian_aic(self):
        rng = np.random.default_rng(8)
        y = sx.logistic4(self.dose, **self.truth) + rng.normal(0, 0.03, size=len(self.dose))
        fit = sx.fit_logistic4(self.dose, y)
        ref, cov = optimize.curve_fit(sx.logistic4, self.dose, y, p0=list(self.truth.values()),
                                      bounds=([-np.inf, -np.inf, 1e-3, 0.05], [np.inf, np.inf, 300, 50]))
        for i, key in enumerate(('bottom', 'top', 'ed50', 'hill')):
            self.assertAlmostEqual(fit['params'][key], ref[i], delta=1e-4 * max(1, abs(ref[i])))
            self.assertAlmostEqual(fit['se'][key], math.sqrt(cov[i, i]), delta=1e-2 * math.sqrt(cov[i, i]))
        residual = y - sx.logistic4(self.dose, *ref)
        loglik = st.norm.logpdf(residual, scale=math.sqrt(np.mean(residual ** 2))).sum()
        self.assertAlmostEqual(fit['log_likelihood'], loglik, places=4)
        self.assertAlmostEqual(fit['aic'], 2 * 5 - 2 * loglik, places=4)
        self.assertAlmostEqual(fit['aicc'], fit['aic'] + 2 * 5 * 6 / (len(y) - 6), places=6)

    def test_linear_and_threshold(self):
        rng = np.random.default_rng(9)
        y = 0.7 - 0.2 * self.dose + rng.normal(0, 0.02, size=len(self.dose))
        fit = sx.fit_linear(self.dose, y)
        ref = st.linregress(self.dose, y)
        self.assertAlmostEqual(fit['params']['slope'], ref.slope, places=10)
        self.assertAlmostEqual(fit['params']['intercept'], ref.intercept, places=10)
        hinge = 0.8 + np.where(self.dose > 1.2, -0.3 * (self.dose - 1.2), 0.0)
        th = sx.fit_threshold(self.dose, hinge)
        self.assertAlmostEqual(th['params']['threshold'], 1.2, delta=1e-4)
        self.assertAlmostEqual(th['params']['slope'], -0.3, delta=1e-4)
        self.assertAlmostEqual(th['params']['level'], 0.8, delta=1e-4)

    def test_model_comparison(self):
        rng = np.random.default_rng(10)
        sigmoid = sx.logistic4(self.dose, **self.truth) + rng.normal(0, 0.02, size=len(self.dose))
        result = sx.compare_dose_models(self.dose, sigmoid)
        self.assertEqual((result['criterion'], result['best']), ('aicc', 'logistic4'))
        self.assertAlmostEqual(sum(m['weight'] for m in result['models'].values()), 1.0)
        line = 0.7 - 0.1 * self.dose + rng.normal(0, 0.05, size=len(self.dose))
        self.assertEqual(sx.compare_dose_models(self.dose, line)['best'], 'linear')
        few = sx.compare_dose_models([0.0, 1, 2], [1.0, 0.5, 0.2])
        self.assertIn('error', few['models']['logistic4'])


class HierarchicalBootstrap(unittest.TestCase):
    def test_one_level_matches_scipy_bootstrap(self):
        x = np.random.default_rng(11).gamma(2.0, 1.0, size=40)
        r = sx.hierarchical_bootstrap([[v] for v in x], n_resamples=20000, test_id='one-level')
        ref = st.bootstrap((x,), np.mean, n_resamples=20000, method='percentile', confidence_level=0.95,
                           rng=np.random.default_rng(12))
        scale = x.std() / math.sqrt(len(x))
        self.assertAlmostEqual(r['effect'], x.mean())
        self.assertLess(abs(r['effect_ci_low'] - ref.confidence_interval.low), 0.1 * scale)
        self.assertLess(abs(r['effect_ci_high'] - ref.confidence_interval.high), 0.1 * scale)
        self.assertLess(abs(r['se'] - x.std(ddof=0) / math.sqrt(len(x))), 0.05 * scale)

    def test_two_levels_and_failures(self):
        rng = np.random.default_rng(13)
        clusters = [rng.normal(m, 0.5, size=5) for m in rng.normal(0, 1, size=10)]
        two = sx.hierarchical_bootstrap(clusters, n_resamples=4000, test_id='two-level')
        cluster_only = sx.hierarchical_bootstrap(clusters, n_resamples=4000, resample_within=False,
                                                 test_id='two-level')
        means = np.array([c.mean() for c in clusters])
        self.assertAlmostEqual(two['effect'], means.mean())
        expected = math.sqrt(means.var(ddof=0) / len(means) + np.mean([c.var(ddof=0) / len(c) for c in clusters])
                             / len(clusters))
        self.assertLess(abs(two['se'] - expected), 0.1 * expected)
        self.assertLess(abs(cluster_only['se'] - means.std(ddof=0) / math.sqrt(len(means))), 0.1 * expected)
        self.assertEqual(two, sx.hierarchical_bootstrap(clusters, n_resamples=4000, test_id='two-level'))

        def fragile(groups):
            if len({id(g) for g in groups}) < 6:
                raise ValueError('degenerate')
            return float(np.mean([g.mean() for g in groups]))

        r = sx.hierarchical_bootstrap(clusters, statistic=fragile, n_resamples=500, resample_within=False)
        self.assertGreater(r['failures'], 0)
        self.assertLess(r['failures'], 500)

    def test_ed50_from_replicate_dose_response(self):
        rng = np.random.default_rng(14)
        dose = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        truth = dict(bottom=0.1, top=0.9, ed50=1.0, hill=3.0)
        clusters = [np.column_stack([dose, sx.logistic4(dose, **truth) + rng.normal(0, 0.03, 5)]) for _ in range(8)]

        def ed50(groups):
            data = np.vstack(groups)
            return sx.fit_logistic4(data[:, 0], data[:, 1])['params']['ed50']

        r = sx.hierarchical_bootstrap(clusters, statistic=ed50, n_resamples=200, test_id='ed50')
        self.assertLess(r['effect_ci_low'], 1.0)
        self.assertGreater(r['effect_ci_high'], 1.0)
        self.assertEqual(r['n'], 8)


class PartialSpearman(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(15)
        self.z = rng.normal(size=(30, 2))
        self.x = self.z[:, 0] + rng.normal(size=30)
        self.y = self.z[:, 0] + 0.5 * self.x + rng.normal(size=30)

    def test_no_covariate_is_spearman(self):
        r = sx.partial_spearman(self.x, self.y)
        ref = st.spearmanr(self.x, self.y)
        self.assertAlmostEqual(r['statistic'], ref.statistic, places=12)
        self.assertAlmostEqual(r['p_value'], ref.pvalue, places=10)

    def test_one_covariate_is_the_first_order_formula(self):
        r = sx.partial_spearman(self.x, self.y, self.z[:, 0])
        rxy, rxz, ryz = (st.spearmanr(a, b).statistic for a, b in ((self.x, self.y), (self.x, self.z[:, 0]),
                                                                    (self.y, self.z[:, 0])))
        expected = (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
        self.assertAlmostEqual(r['statistic'], expected, places=12)
        df = 30 - 3
        t = expected * math.sqrt(df / (1 - expected ** 2))
        self.assertAlmostEqual(r['p_value'], 2 * st.t.sf(abs(t), df), places=12)
        self.assertEqual(r['df'], df)

    def test_two_covariates_are_correlated_rank_residuals(self):
        r = sx.partial_spearman(self.x, self.y, self.z)
        design = np.column_stack([np.ones(30), st.rankdata(self.z[:, 0]), st.rankdata(self.z[:, 1])])
        res = [v - design @ np.linalg.lstsq(design, v, rcond=None)[0] for v in (st.rankdata(self.x), st.rankdata(self.y))]
        self.assertAlmostEqual(r['statistic'], st.pearsonr(*res).statistic, places=12)
        perm = sx.partial_spearman(self.x, self.y, self.z, permutations=2000, test_id='ps')
        self.assertEqual(perm['p_method'], 'permutation')
        self.assertEqual(perm['p_value'], sx.partial_spearman(self.x, self.y, self.z, permutations=2000,
                                                              test_id='ps')['p_value'])
        self.assertLess(abs(perm['p_value'] - r['p_value']), 0.05 + 4 * perm['mc_se'])
        self.assertLess(r['effect_ci_low'], r['statistic'])
        with self.assertRaises(ValueError):
            sx.partial_spearman([1, 2, 3], [3, 2, 1], [[1, 2], [2, 1], [3, 3]])


class Multiplicity(unittest.TestCase):
    def test_holm_and_bh(self):
        p = np.array([0.01, 0.04, 0.03, 0.005, 0.2])
        self.assertTrue(np.allclose(sx.holm(p), [0.04, 0.09, 0.09, 0.025, 0.2]))
        self.assertTrue(np.allclose(sx.bh(p), st.false_discovery_control(p, method='bh')))
        self.assertEqual(len(sx.bh([])), 0)


if __name__ == '__main__':
    unittest.main()
