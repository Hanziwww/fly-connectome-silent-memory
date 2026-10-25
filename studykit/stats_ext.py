"""Statistics added for the plan v2 studies: equivalence (TOST), exact sign-flip, simulated power, 4-parameter
logistic dose-response with model comparison, hierarchical bootstrap and partial Spearman correlation.

Results are dicts in the column vocabulary of manuscript/analysis/msa/stats.py (test, statistic_name, statistic,
p_value, p_method, n, n_used, effect_name, effect, effect_ci_low, effect_ci_high, effect_ci_method, informative,
min_attainable_p, n_resamples, mc_se), so one statistics table can hold rows from both. Every resampling step draws
from its own generator, derived from the seed and a test ID, so results do not depend on execution order. Monte Carlo
p-values use (k + 1) / (N + 1); exact enumeration is used whenever it is feasible (n <= MAX_EXACT).
"""
import math
import warnings
import zlib

import numpy as np
from scipy import optimize
from scipy import stats as st

SEED = 20261019
ALPHA = 0.05
N_RESAMPLES = 10_000
MAX_EXACT = 20
_CHUNK = 1 << 16
_NAN = float('nan')


# ---------------------------------------------------------------- helpers

def rng_for(test_id, seed=None):
    """Generator for one test ID: independent of every other test and of execution order."""
    return np.random.default_rng(np.random.SeedSequence(SEED if seed is None else seed,
                                                        spawn_key=(zlib.crc32(test_id.encode('utf-8')),)))


def mc_se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else _NAN


def cp_interval(k, n, level=0.95):
    if n == 0:
        return (_NAN, _NAN)
    ci = st.binomtest(int(k), int(n)).proportion_ci(confidence_level=level, method='exact')
    return (float(ci.low), float(ci.high))


def _paired(x, y=None):
    """Differences x - y (or x alone), pairs with a NaN dropped."""
    x = np.asarray(x, float).ravel()
    if y is None:
        return x[~np.isnan(x)]
    y = np.asarray(y, float).ravel()
    if x.shape != y.shape:
        raise ValueError('paired samples differ in length')
    keep = ~(np.isnan(x) | np.isnan(y))
    return x[keep] - y[keep]


def _xy(x, y):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.shape != y.shape:
        raise ValueError('dose and response differ in length')
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


# ---------------------------------------------------------------- exact sign-flip

def _signs(n, start, stop):
    """Sign patterns start..stop-1 of all 2**n; bit i of the pattern index flips observation i."""
    codes = np.arange(start, stop, dtype=np.int64)[:, None]
    return 1.0 - 2.0 * ((codes >> np.arange(n, dtype=np.int64)) & 1)


def _flip_sums(d, exact, rng, n_resamples):
    """For each sign pattern s: A = s . d and B = s . 1, over all 2**n patterns or n_resamples random draws."""
    n = len(d)
    if exact:
        total = 1 << n
        A, B = np.empty(total), np.empty(total)
        for start in range(0, total, _CHUNK):
            s = _signs(n, start, min(start + _CHUNK, total))
            A[start:start + len(s)] = s @ d
            B[start:start + len(s)] = s.sum(axis=1)
        return A, B
    s = 1.0 - 2.0 * rng.integers(0, 2, size=(n_resamples, n))
    return s @ d, s.sum(axis=1)


def _flip_p(A, B, d, mu, alternative, exact):
    """p of the sign-flip test of mean(d) = mu from the pattern sums (the null statistic is A - mu * B)."""
    observed = d.sum() - mu * len(d)
    null = A - mu * B
    tol = 1e-9 * (np.abs(d - mu).sum() + np.finfo(float).tiny)
    if alternative == 'two-sided':
        k = np.count_nonzero(np.abs(null) >= abs(observed) - tol)
    elif alternative == 'greater':
        k = np.count_nonzero(null >= observed - tol)
    else:
        k = np.count_nonzero(null <= observed + tol)
    return k / len(A) if exact else (k + 1) / (len(A) + 1)


def sign_flip(x, y=None, alternative='two-sided', test_id='sign_flip', seed=None, max_exact=MAX_EXACT,
              n_resamples=N_RESAMPLES, ci=True, level=0.95):
    """Exact sign-flip (one-sample or paired permutation) test that the mean difference is zero.

    Under H0 the differences are symmetric about zero, so all 2**m sign patterns are equally likely. Zero differences
    are dropped (they cannot change the statistic); with m non-zero differences the smallest two-sided p is 2 / 2**m,
    which is why primary families use n = 10 replicates. Above max_exact non-zero differences, n_resamples random
    patterns are drawn. The interval inverts the test over all n differences (``sign_flip_ci``).
    """
    if alternative not in ('two-sided', 'greater', 'less'):
        raise ValueError(f'unknown alternative {alternative!r}')
    d = _paired(x, y)
    nonzero = d[d != 0]
    n, m = len(d), len(nonzero)
    out = dict(test='sign_flip', statistic_name='sum_difference', n=n, n_used=m, n_zero=n - m,
               alternative=alternative, effect_name='mean_difference', effect=float(d.mean()) if n else _NAN,
               effect_ci_low=_NAN, effect_ci_high=_NAN, effect_ci_method='none')
    if m == 0:
        out.update(statistic=0.0, p_value=1.0, p_method='exact', n_resamples=None, mc_se=None, informative=False,
                   min_attainable_p=1.0)
        return out
    exact = m <= max_exact
    A, B = _flip_sums(nonzero, exact, None if exact else rng_for(test_id, seed), n_resamples)
    p = min(1.0, _flip_p(A, B, nonzero, 0.0, alternative, exact))
    smallest = min(1.0, (2.0 if alternative == 'two-sided' else 1.0) / 2 ** m) if exact else 1 / (n_resamples + 1)
    out.update(statistic=float(nonzero.sum()), p_value=float(p), p_method='exact' if exact else 'permutation',
               n_resamples=None if exact else n_resamples, mc_se=None if exact else mc_se(p, n_resamples),
               informative=True, min_attainable_p=smallest)
    if ci:
        lo, hi, method = sign_flip_ci(d, level=level, test_id=test_id, seed=seed, max_exact=max_exact,
                                      n_resamples=n_resamples)
        out.update(effect_ci_low=lo, effect_ci_high=hi, effect_ci_method=method)
    return out


def sign_flip_ci(x, y=None, level=0.95, test_id='sign_flip', seed=None, max_exact=MAX_EXACT, n_resamples=N_RESAMPLES):
    """Interval for the mean difference: every mu that the two-sided sign-flip test of mean = mu does not reject.

    Bisection from the mean outwards on each side. Returns (nan, nan, 'unavailable_small_n') when even the most
    extreme sign pattern cannot reach 1 - level (n <= 5 at 95%).
    """
    d = _paired(x, y)
    n = len(d)
    if n == 0:
        return _NAN, _NAN, 'none'
    exact = n <= max_exact
    A, B = _flip_sums(d, exact, None if exact else rng_for(test_id + ':ci', seed), n_resamples)
    alpha = 1.0 - level
    centre = float(d.mean())
    span = float(d.max() - d.min()) + abs(centre) + 1.0

    def accepted(mu):
        return _flip_p(A, B, d, mu, 'two-sided', exact) > alpha

    far_low, far_high = float(d.min()) - span, float(d.max()) + span
    if accepted(far_low) or accepted(far_high):
        return _NAN, _NAN, 'unavailable_small_n'

    def boundary(inside, outside):
        for _ in range(200):
            mid = 0.5 * (inside + outside)
            if mid in (inside, outside) or abs(outside - inside) <= 1e-12 * span:
                break
            if accepted(mid):
                inside = mid
            else:
                outside = mid
        return 0.5 * (inside + outside)

    method = 'sign_flip_inversion_exact' if exact else 'sign_flip_inversion_permutation'
    return float(boundary(centre, far_low)), float(boundary(centre, far_high)), method


# ---------------------------------------------------------------- equivalence (TOST)

def mde_t(sd, n, alpha=ALPHA, power=0.8):
    """Smallest |mean| that a two-sided one-sample t-test detects with the given power (sigma = sd)."""
    if n < 2 or not sd > 0:
        return _NAN
    df = n - 1
    tcrit = st.t.ppf(1 - alpha / 2, df)

    def gap(nc):
        value = st.nct.sf(tcrit, df, nc) + st.nct.cdf(-tcrit, df, nc)
        return (value if math.isfinite(value) else 1.0) - power  # scipy's nct is unstable far out, where power is 1

    high = 1.0
    while gap(high) < 0 and high < 1e4:  # bracket the root before the unstable region
        high *= 2
    return float(optimize.brentq(gap, 0.0, high) * sd / math.sqrt(n))


def tost(x, low, high, y=None, alpha=ALPHA, method='t', power=0.8, test_id='tost', seed=None):
    """Equivalence of a mean (or paired difference, with y) to the interval (low, high) by two one-sided tests.

    H0a: mean <= low and H0b: mean >= high; p = max(p_a, p_b) and the mean is 'equivalent' when p < alpha, i.e. when
    the 1 - 2 alpha interval lies inside (low, high). method: 't' (Student) or 'sign_flip' (exact; differences
    symmetric about the bound). Otherwise the conclusion is 'different' when the two-sided test of zero rejects and
    'inconclusive' when it does not; the minimal detectable effect (two-sided t-test, sample SD, given power) is
    reported either way.
    """
    if not low < high:
        raise ValueError('the equivalence interval needs low < high')
    d = _paired(x, y)
    n = len(d)
    if n < 2:
        raise ValueError('TOST needs at least two observations')
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    if method == 't':
        se, df = sd / math.sqrt(n), n - 1
        if se == 0:
            p_low, p_high = float(mean <= low), float(mean >= high)
            p_diff, ci = float(mean == 0), (mean, mean)
        else:
            p_low = float(st.t.sf((mean - low) / se, df))
            p_high = float(st.t.cdf((mean - high) / se, df))
            p_diff = float(2 * st.t.sf(abs(mean) / se, df))
            half = float(st.t.ppf(1 - alpha, df)) * se
            ci = (mean - half, mean + half)
        name, p_method, ci_method = 'tost_t', 'student_t', 't'
    elif method == 'sign_flip':
        p_low = sign_flip(d - low, alternative='greater', test_id=test_id + ':low', seed=seed, ci=False)['p_value']
        p_high = sign_flip(d - high, alternative='less', test_id=test_id + ':high', seed=seed, ci=False)['p_value']
        p_diff = sign_flip(d, test_id=test_id, seed=seed, ci=False)['p_value']
        lo, hi, _ = sign_flip_ci(d, level=1 - 2 * alpha, test_id=test_id, seed=seed)
        ci = (lo, hi)
        name, p_method, ci_method = 'tost_sign_flip', ('exact' if n <= MAX_EXACT else 'permutation'), 'sign_flip_inversion'
    else:
        raise ValueError(f'unknown TOST method {method!r}')
    p = max(p_low, p_high)
    equivalent = p < alpha
    return dict(test=name, statistic_name='mean_difference', statistic=mean, p_value=float(p), p_method=p_method, n=n,
                n_used=n, p_lower=float(p_low), p_upper=float(p_high), margin_low=float(low), margin_high=float(high),
                alpha=alpha, effect_name='mean_difference', effect=mean, effect_ci_low=float(ci[0]),
                effect_ci_high=float(ci[1]), effect_ci_method=f'{100 * (1 - 2 * alpha):g}% {ci_method}',
                p_difference=float(p_diff), equivalent=bool(equivalent),
                conclusion='equivalent' if equivalent else ('different' if p_diff < alpha else 'inconclusive'),
                minimal_detectable_effect=mde_t(sd, n, alpha, power), mde_power=power, informative=True)


def mde_binomial(n, p0, alpha=ALPHA, power=0.8, step=0.001):
    """Smallest |p - p0| (grid ``step``) that the exact two-sided binomial test of p0 detects with the given power."""
    xs = np.arange(n + 1)
    reject = np.array([st.binomtest(int(x), int(n), p0).pvalue < alpha for x in xs])
    if not reject.any():
        return _NAN
    for delta in np.arange(step, 1.0, step):
        powers = [st.binom.pmf(xs[reject], n, p).sum() for p in (p0 - delta, p0 + delta) if 0.0 <= p <= 1.0]
        if powers and min(powers) >= power:
            return float(round(delta, 10))
    return _NAN


def tost_binomial(k, n, low, high, alpha=ALPHA, power=0.8):
    """Equivalence of a proportion k/n to (low, high), e.g. chance 0.5 +/- 0.10, by two exact one-sided binomial tests."""
    if not 0 < low < high < 1:
        raise ValueError('the equivalence interval needs 0 < low < high < 1')
    k, n = int(k), int(n)
    p_low = float(st.binomtest(k, n, low, alternative='greater').pvalue)
    p_high = float(st.binomtest(k, n, high, alternative='less').pvalue)
    centre = 0.5 * (low + high)
    p_diff = float(st.binomtest(k, n, centre).pvalue)
    lo, hi = cp_interval(k, n, 1 - 2 * alpha)
    p = max(p_low, p_high)
    equivalent = p < alpha
    return dict(test='tost_binomial', statistic_name='k', statistic=float(k), p_value=p, p_method='exact', n=n,
                n_used=n, p_lower=p_low, p_upper=p_high, margin_low=float(low), margin_high=float(high), alpha=alpha,
                effect_name='proportion', effect=k / n if n else _NAN, effect_ci_low=lo, effect_ci_high=hi,
                effect_ci_method=f'{100 * (1 - 2 * alpha):g}% clopper_pearson_exact', p_difference=p_diff,
                equivalent=bool(equivalent),
                conclusion='equivalent' if equivalent else ('different' if p_diff < alpha else 'inconclusive'),
                minimal_detectable_effect=mde_binomial(n, centre, alpha, power), mde_power=power, informative=n > 0)


def tango_z(b, c, n, d0):
    """Score statistic for H0: p_b - p_c = d0 from paired binary outcomes (b and c the two discordant counts of n pairs),
    with the restricted maximum-likelihood estimate of p_c (Tango 1998, Stat Med 17:891-908; as in
    manuscript/analysis/panels/p11_followup.py)."""
    A = 2.0 * n
    B = -b - c + (2.0 * n - b + c) * d0
    C = -c * d0 * (1.0 - d0)
    q = (math.sqrt(max(B * B - 4.0 * A * C, 0.0)) - B) / (2.0 * A)
    var = n * (2.0 * q + d0 * (1.0 - d0))
    num = b - c - n * d0
    if var <= 0:
        return math.copysign(math.inf, num) if num else 0.0
    return num / math.sqrt(var)


def tost_paired_binary(b, c, n, margin, alpha=ALPHA):
    """Equivalence of a paired difference of proportions (b - c) / n to zero within +-margin: two one-sided Tango score
    tests, the matching (1 - 2 alpha) score interval and the smallest symmetric margin the data support. b: pairs where
    only the first condition succeeds, c: only the second."""
    b, c, n = int(b), int(c), int(n)
    if n <= 0 or b + c > n or margin <= 0:
        raise ValueError('need n > 0, b + c <= n and a positive margin')
    z = st.norm.ppf(1 - alpha)
    d = (b - c) / n
    p_low = float(st.norm.sf(tango_z(b, c, n, -margin)))          # H0: difference <= -margin
    p_up = float(st.norm.cdf(tango_z(b, c, n, margin)))           # H0: difference >= +margin
    eps = 1e-9
    f_lo = lambda x: tango_z(b, c, n, x) - z                      # noqa: E731 (the statistic decreases in x)
    f_hi = lambda x: tango_z(b, c, n, x) + z                      # noqa: E731
    lo = optimize.brentq(f_lo, -1 + eps, d, xtol=1e-10) if d > -1 + eps and f_lo(-1 + eps) > 0 else -1.0
    hi = optimize.brentq(f_hi, d, 1 - eps, xtol=1e-10) if d < 1 - eps and f_hi(1 - eps) < 0 else 1.0
    p = max(p_low, p_up)
    p_diff = float(2 * st.norm.sf(abs(tango_z(b, c, n, 0.0))))
    equivalent = p < alpha
    return dict(test='tost_paired_binary', statistic_name='difference_of_paired_proportions', statistic=d, p_value=p,
                p_method='asymptotic score (Tango), two one-sided tests', n=n, n_used=n, p_lower=p_low, p_upper=p_up,
                margin_low=-float(margin), margin_high=float(margin), alpha=alpha, effect_name='difference', effect=d,
                effect_ci_low=float(lo), effect_ci_high=float(hi), effect_ci_method=f'{100 * (1 - 2 * alpha):g}% Tango score interval',
                p_difference=p_diff, equivalent=bool(equivalent),
                conclusion='equivalent' if equivalent else ('different' if p_diff < alpha else 'inconclusive'),
                margin_supported=float(max(abs(lo), abs(hi))), discordant_first=b, discordant_second=c, informative=True)


# ---------------------------------------------------------------- simulated power

def simulate_power(simulate, test, n, n_sim=2000, alpha=ALPHA, test_id='power', seed=None, level=0.95):
    """Share of ``n_sim`` simulated data sets that ``test`` rejects at ``alpha``.

    ``simulate(rng, n)`` returns one data set; ``test(data)`` returns a p-value or a result dict with 'p_value'.
    Each n uses its own generator stream, so a power curve is reproducible point by point.
    """
    rng = rng_for(f'{test_id}:n={n}', seed)
    rejections = 0
    for _ in range(n_sim):
        result = test(simulate(rng, n))
        p = result['p_value'] if isinstance(result, dict) else float(result)
        rejections += p < alpha
    power = rejections / n_sim
    lo, hi = cp_interval(rejections, n_sim, level)
    return dict(n=int(n), n_sim=int(n_sim), alpha=alpha, rejections=int(rejections), power=power,
                power_ci_low=lo, power_ci_high=hi, mc_se=mc_se(power, n_sim))


def sample_size(simulate, test, n_values, target=0.8, n_sim=2000, alpha=ALPHA, test_id='sample_size', seed=None):
    """Smallest n in ``n_values`` whose simulated power reaches ``target`` (None if none does), with the curve."""
    curve = [simulate_power(simulate, test, n, n_sim, alpha, test_id, seed) for n in n_values]
    chosen = next((c['n'] for c in curve if c['power'] >= target), None)
    return dict(target=target, alpha=alpha, n=chosen, curve=curve)


# ---------------------------------------------------------------- dose-response models

def _information(rss, n, n_params):
    """Gaussian log-likelihood (ML variance) and AIC/AICc/BIC; the residual variance counts as a parameter."""
    k = n_params + 1
    loglik = -0.5 * n * (math.log(2 * math.pi * max(rss, np.finfo(float).tiny) / n) + 1)
    aic = 2 * k - 2 * loglik
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else _NAN
    return dict(log_likelihood=loglik, aic=aic, aicc=aicc, bic=k * math.log(n) - 2 * loglik)


def logistic4(dose, bottom, top, ed50, hill):
    """Four-parameter logistic (Hill) curve: ``top`` at dose 0, ``bottom`` at large doses, half-way at ``ed50``."""
    d = np.asarray(dose, float)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        ratio = np.power(d / ed50, hill)
    return bottom + (top - bottom) / (1.0 + ratio)


def fit_logistic4(dose, response):
    """Least-squares 4PL fit (multi-start, bounded: ed50 within 1/100..100x the positive doses, hill 0.05..50)."""
    d, y = _xy(dose, response)
    levels = np.unique(d)
    positive = d[d > 0]
    if len(d) < 5 or len(levels) < 4 or not len(positive):
        raise ValueError('a 4PL fit needs at least five observations at four or more doses, some positive')
    lower = [-np.inf, -np.inf, positive.min() / 100, 0.05]
    upper = [np.inf, np.inf, positive.max() * 100, 50.0]
    first, last = y[d == levels[0]].mean(), y[d == levels[-1]].mean()
    best = None
    for ed50 in np.geomspace(positive.min(), positive.max(), 5):
        for hill in (0.5, 1.0, 2.0, 4.0):
            try:
                with warnings.catch_warnings():  # an exact fit has no covariance estimate; se is then nan
                    warnings.simplefilter('ignore', optimize.OptimizeWarning)
                    params, cov = optimize.curve_fit(logistic4, d, y, p0=(last, first, ed50, hill),
                                                     bounds=(lower, upper), maxfev=20000)
            except (RuntimeError, ValueError):
                continue
            rss = float(np.sum((y - logistic4(d, *params)) ** 2))
            if best is None or rss < best[0]:
                best = (rss, params, cov)
    if best is None:
        raise RuntimeError('the 4PL fit did not converge from any start')
    rss, params, cov = best
    se = np.sqrt(np.clip(np.diag(cov), 0, None)) if np.all(np.isfinite(cov)) else np.full(4, _NAN)
    names = ('bottom', 'top', 'ed50', 'hill')
    at_bound = [names[i] for i in (2, 3) if np.isclose(params[i], lower[i]) or np.isclose(params[i], upper[i])]
    return dict(model='logistic4', n=len(d), k=4, params={k: float(v) for k, v in zip(names, params)},
                se={k: float(v) for k, v in zip(names, se)}, rss=rss, at_bound=at_bound,
                **_information(rss, len(d), 4))


def fit_linear(dose, response):
    d, y = _xy(dose, response)
    if len(d) < 3:
        raise ValueError('a linear fit needs at least three observations')
    X = np.column_stack([np.ones(len(d)), d])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    rss = float(np.sum((y - X @ coef) ** 2))
    return dict(model='linear', n=len(d), k=2, params=dict(intercept=float(coef[0]), slope=float(coef[1])), rss=rss,
                **_information(rss, len(d), 2))


def _hinge(theta, d, y):
    X = np.column_stack([np.ones(len(d)), np.maximum(d - theta, 0.0)])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.sum((y - X @ coef) ** 2)), coef


def fit_threshold(dose, response, grid=200):
    """Threshold (hinge) model: flat at ``level`` up to ``threshold``, then linear with ``slope``."""
    d, y = _xy(dose, response)
    levels = np.unique(d)
    if len(d) < 4 or len(levels) < 3:
        raise ValueError('a threshold fit needs at least four observations at three or more doses')
    thetas = np.linspace(levels[0], levels[-2], grid)
    rss = np.array([_hinge(t, d, y)[0] for t in thetas])
    i = int(np.argmin(rss))
    theta = float(thetas[i])
    lo, hi = thetas[max(i - 1, 0)], thetas[min(i + 1, grid - 1)]
    if hi > lo:
        refined = optimize.minimize_scalar(lambda t: _hinge(t, d, y)[0], bounds=(lo, hi), method='bounded',
                                           options=dict(xatol=1e-10 * (levels[-1] - levels[0])))
        if refined.fun <= rss[i]:
            theta = float(refined.x)
    value, coef = _hinge(theta, d, y)
    return dict(model='threshold', n=len(d), k=3,
                params=dict(level=float(coef[0]), slope=float(coef[1]), threshold=theta), rss=value,
                **_information(value, len(d), 3))


def compare_dose_models(dose, response):
    """4PL vs linear vs threshold by AICc (AIC if n is too small for AICc), with Akaike weights."""
    fits = {}
    for name, fit in (('logistic4', fit_logistic4), ('linear', fit_linear), ('threshold', fit_threshold)):
        try:
            fits[name] = fit(dose, response)
        except (ValueError, RuntimeError) as error:
            fits[name] = dict(model=name, error=str(error))
    ok = {k: f for k, f in fits.items() if 'error' not in f}
    if not ok:
        raise ValueError('no dose-response model could be fitted')
    criterion = 'aicc' if all(math.isfinite(f['aicc']) for f in ok.values()) else 'aic'
    floor = min(f[criterion] for f in ok.values())
    raw = {k: math.exp(-0.5 * (f[criterion] - floor)) for k, f in ok.items()}
    for k, f in ok.items():
        f.update(delta=f[criterion] - floor, weight=raw[k] / sum(raw.values()))
    return dict(criterion=criterion, best=min(ok, key=lambda k: ok[k][criterion]), models=fits)


# ---------------------------------------------------------------- hierarchical bootstrap

def _mean_of_means(groups):
    return float(np.mean([np.mean(g) for g in groups]))


def hierarchical_bootstrap(clusters, statistic=None, n_resamples=2000, level=0.95, resample_within=True,
                           test_id='hierarchical_bootstrap', seed=None, return_distribution=False):
    """Two-stage percentile bootstrap: clusters (e.g. replicates) with replacement, then units (e.g. random draws)
    with replacement within each drawn cluster.

    ``clusters``: one array per cluster, units along axis 0 (rows may be (dose, response) pairs). ``statistic``
    maps a list of such arrays to a float (default: mean of cluster means, the replicate being the unit). Draws
    where the statistic fails or is not finite are counted in ``failures`` and left out of the interval.
    """
    groups = [np.asarray(c) for c in clusters]
    if not groups or any(len(g) == 0 for g in groups):
        raise ValueError('every cluster needs at least one unit')
    statistic = statistic or _mean_of_means
    estimate = float(statistic(groups))
    rng = rng_for(test_id, seed)
    values = np.full(n_resamples, _NAN)
    for b in range(n_resamples):
        picks = rng.integers(0, len(groups), size=len(groups))
        sample = [groups[i][rng.integers(0, len(groups[i]), size=len(groups[i]))] if resample_within else groups[i]
                  for i in picks]
        try:
            values[b] = statistic(sample)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
            pass
    ok = values[np.isfinite(values)]
    alpha = 1.0 - level
    lo, hi = (float(q) for q in np.quantile(ok, [alpha / 2, 1 - alpha / 2])) if len(ok) else (_NAN, _NAN)
    out = dict(test='hierarchical_bootstrap', statistic_name='statistic', statistic=estimate, p_value=None,
               p_method=None, n=len(groups), n_used=len(groups), n_units=int(sum(len(g) for g in groups)),
               n_resamples=n_resamples, failures=int(n_resamples - len(ok)), effect_name='statistic', effect=estimate,
               effect_ci_low=lo, effect_ci_high=hi,
               effect_ci_method='hierarchical_percentile_bootstrap' if resample_within else 'cluster_percentile_bootstrap',
               se=float(ok.std(ddof=1)) if len(ok) > 1 else _NAN, bias=float(ok.mean() - estimate) if len(ok) else _NAN,
               informative=len(ok) > 0)
    if return_distribution:
        out['distribution'] = values
    return out


# ---------------------------------------------------------------- partial Spearman

def _residuals(v, design):
    return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / den if den > 0 else _NAN


def partial_spearman(x, y, covariates=None, permutations=0, test_id='partial_spearman', seed=None, level=0.95):
    """Spearman correlation of x and y controlling for covariates (e.g. log(updates)).

    Ranks (mid-ranks for ties) of x, y and each covariate; residuals of the x and y ranks on the covariate ranks
    (OLS with intercept); Pearson correlation of the residuals. p: Student t with n - 2 - k df (k covariates; equal to
    scipy's spearmanr when k = 0), or, with ``permutations`` > 0, residuals of y permuted and re-residualised.
    Interval: Fisher z with standard error 1 / sqrt(n - 3 - k).
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    z = np.empty((len(x), 0)) if covariates is None else np.asarray(covariates, float).reshape(len(x), -1)
    keep = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(z), axis=1)
    x, y, z = x[keep], y[keep], z[keep]
    n, k = len(x), z.shape[1]
    df = n - 2 - k
    if df < 1:
        raise ValueError('partial Spearman needs n > k + 2')
    design = np.column_stack([np.ones(n)] + [st.rankdata(c) for c in z.T])
    res_x = _residuals(st.rankdata(x), design)
    res_y = _residuals(st.rankdata(y), design)
    r = _corr(res_x, res_y)
    if not math.isfinite(r):
        p_t = _NAN
    elif abs(r) >= 1.0:
        p_t = 0.0
    else:
        p_t = float(2 * st.t.sf(abs(r) * math.sqrt(df / (1 - r * r)), df))
    lo = hi = _NAN
    if math.isfinite(r) and n - 3 - k > 0:
        zc = math.atanh(min(max(r, -1 + 1e-15), 1 - 1e-15))
        half = st.norm.ppf(1 - (1 - level) / 2) / math.sqrt(n - 3 - k)
        lo, hi = math.tanh(zc - half), math.tanh(zc + half)
    out = dict(test='partial_spearman', statistic_name='partial_rho', statistic=r, p_value=p_t, p_method='student_t',
               p_t=p_t, n=n, n_used=n, covariates=k, df=df, n_resamples=None, mc_se=None,
               effect_name='partial_spearman_rho', effect=r, effect_ci_low=lo, effect_ci_high=hi,
               effect_ci_method='fisher_z', informative=math.isfinite(r))
    if permutations and math.isfinite(r):
        rng = rng_for(test_id, seed)
        null = np.array([_corr(res_x, _residuals(res_y[rng.permutation(n)], design)) for _ in range(permutations)])
        p = (np.count_nonzero(np.abs(null) >= abs(r) - 1e-12) + 1) / (permutations + 1)
        out.update(p_value=float(p), p_method='permutation', n_resamples=permutations, mc_se=mc_se(p, permutations))
    return out


# ---------------------------------------------------------------- multiplicity (as in msa.stats)

def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p, kind='mergesort')
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def bh(pvalues):
    p = np.asarray(pvalues, float)
    return p if len(p) == 0 else st.false_discovery_control(p, method='bh')
