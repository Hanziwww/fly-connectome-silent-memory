"""Exact and resampling tests used by the manuscript analysis, plus the test registry (ST2).

Every resampling test draws from its own generator derived from the analysis seed and its
test ID, so results do not depend on execution order. Permutation p-values use
(k + 1) / (N + 1); exact enumeration is used whenever it is feasible.
"""
import json
import math
import zlib
from itertools import combinations

import numpy as np
from scipy import stats as st

from . import SEED, paths

N_RESAMPLES = 10_000
ALPHA = 0.05
_registry = []


# ---------------------------------------------------------------- helpers

def rng_for(test_id):
    return np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(zlib.crc32(test_id.encode('utf-8')),)))


def cp_interval(k, n, level=0.95):
    if n == 0:
        return (float('nan'), float('nan'))
    ci = st.binomtest(int(k), int(n)).proportion_ci(confidence_level=level, method='exact')
    return (float(ci.low), float(ci.high))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_p(k, n):
    if n == 0:
        return 1.0
    return float(st.binomtest(int(k), int(n), 0.5).pvalue)


def min_attainable_p(n_informative):
    return 1.0 if n_informative <= 0 else min(1.0, 2 * 0.5 ** n_informative)


def mc_se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float('nan')


def proportion(k, n):
    lo, hi = cp_interval(k, n)
    return dict(numerator=int(k), denominator=int(n), estimate=(k / n if n else float('nan')),
                ci_low=lo, ci_high=hi, ci_method='clopper_pearson_exact')


# ---------------------------------------------------------------- paired binary

def mcnemar(x, y):
    """Exact McNemar for paired booleans x (condition A) and y (condition B).

    b = A true & B false, c = A false & B true. Effect sizes: paired difference
    P(A) - P(B) with Newcombe's score interval, and b / (b + c) with a CP interval.
    """
    x = np.asarray(x, bool)
    y = np.asarray(y, bool)
    a = int(np.sum(x & y)); b = int(np.sum(x & ~y)); c = int(np.sum(~x & y)); d = int(np.sum(~x & ~y))
    return mcnemar_counts(a, b, c, d)


def mcnemar_counts(a, b, c, d):
    n = a + b + c + d
    disc = b + c
    p = binom_p(b, disc)
    diff = (b - c) / n if n else float('nan')
    lo, hi = newcombe_paired(a, b, c, d)
    flo, fhi = cp_interval(b, disc)
    return dict(test='mcnemar_exact', statistic_name='b_discordant', statistic=float(b), p_value=p,
                p_method='exact', n=n, n_used=disc, both=a, a_only=b, b_only=c, neither=d,
                effect_name='paired_difference', effect=diff, effect_ci_low=lo, effect_ci_high=hi,
                effect_ci_method='newcombe_score_paired', share_a_only=(b / disc if disc else float('nan')),
                share_ci_low=flo, share_ci_high=fhi, informative=disc > 0,
                min_attainable_p=min_attainable_p(disc))


def newcombe_paired(a, b, c, d):
    n = a + b + c + d
    if n == 0:
        return (float('nan'), float('nan'))
    p1 = (a + b) / n
    p2 = (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(denom) if denom > 0 else 0.0
    dd = p1 - p2
    delta = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    eps = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return (max(-1.0, dd - delta), min(1.0, dd + eps))


def stratified_mcnemar(pairs):
    """Combined exact test over strata of discordant counts [(b, c), ...]."""
    b = sum(int(p[0]) for p in pairs)
    c = sum(int(p[1]) for p in pairs)
    return dict(test='stratified_mcnemar_exact', statistic_name='b_discordant_total', statistic=float(b),
                p_value=binom_p(b, b + c), p_method='exact', n_used=b + c, a_only=b, b_only=c,
                effect_name='share_a_only', effect=(b / (b + c) if b + c else float('nan')),
                effect_ci_low=cp_interval(b, b + c)[0], effect_ci_high=cp_interval(b, b + c)[1],
                effect_ci_method='clopper_pearson_exact', informative=(b + c) > 0,
                min_attainable_p=min_attainable_p(b + c))


# ---------------------------------------------------------------- paired continuous

def sign_test(x, tol=0.0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    pos = int(np.sum(x > tol)); neg = int(np.sum(x < -tol)); zero = len(x) - pos - neg
    lo, hi = cp_interval(pos, pos + neg)
    return dict(test='sign_test_exact', statistic_name='n_positive', statistic=float(pos),
                p_value=binom_p(pos, pos + neg), p_method='exact', n=len(x), n_used=pos + neg,
                n_positive=pos, n_negative=neg, n_zero=zero, effect_name='share_positive',
                effect=(pos / (pos + neg) if pos + neg else float('nan')), effect_ci_low=lo,
                effect_ci_high=hi, effect_ci_method='clopper_pearson_exact',
                informative=(pos + neg) > 0, min_attainable_p=min_attainable_p(pos + neg))


def _signed_rank_exact_untied(t_plus, n):
    """Exact two-sided p for untied ranks 1..n from the subset-sum distribution."""
    m = n * (n + 1) // 2
    counts = [0] * (m + 1)
    counts[0] = 1
    for r in range(1, n + 1):
        for t in range(m, r - 1, -1):
            counts[t] += counts[t - r]
    centre = m / 2
    tail = sum(c for t, c in enumerate(counts) if abs(t - centre) >= abs(t_plus - centre) - 1e-9)
    return tail / 2 ** n


def _signed_rank_null(ranks, test_id):
    n = len(ranks)
    if n <= 20:
        codes = np.arange(2 ** n, dtype=np.int64)[:, None]
        bits = ((codes >> np.arange(n, dtype=np.int64)) & 1).astype(np.float64)
        return bits @ ranks, 'exact', 2 ** n
    rng = rng_for(test_id)
    bits = rng.integers(0, 2, size=(N_RESAMPLES, n)).astype(np.float64)
    return bits @ ranks, 'permutation', N_RESAMPLES


def signed_rank(x, test_id, tol=0.0):
    """Wilcoxon signed-rank with mid-ranks; zeros dropped and counted; exact for n <= 20."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    nz = x[np.abs(x) > tol]
    n_zero = len(x) - len(nz)
    out = dict(test='wilcoxon_signed_rank', statistic_name='T_plus', n=len(x), n_used=len(nz), n_zero=n_zero)
    if len(nz) == 0:
        out.update(statistic=float('nan'), p_value=1.0, p_method='exact', informative=False,
                   effect_name='hodges_lehmann', effect=0.0 if len(x) else float('nan'),
                   effect_ci_low=float('nan'), effect_ci_high=float('nan'), effect_ci_method='none')
        return out
    ranks = st.rankdata(np.abs(nz))
    ties = len(np.unique(np.abs(nz))) < len(nz)
    t_plus = float(ranks[nz > 0].sum())
    centre = ranks.sum() / 2
    if not ties:
        p, method, draws = _signed_rank_exact_untied(t_plus, len(nz)), 'exact', None
    else:
        null, method, draws = _signed_rank_null(ranks, test_id)
        extreme = np.abs(null - centre) >= abs(t_plus - centre) - 1e-9
        if method == 'exact':
            p = float(extreme.mean())
        else:
            p = float((extreme.sum() + 1) / (draws + 1))
    hl, lo, hi, ci_method = hodges_lehmann(nz, ties)
    rbc = (2 * t_plus - ranks.sum()) / ranks.sum()
    out.update(statistic=t_plus, p_value=min(1.0, p), p_method=method, n_resamples=(draws if method != 'exact' else None),
               mc_se=(mc_se(p, draws) if method != 'exact' and draws else None), ties=bool(ties),
               effect_name='hodges_lehmann', effect=hl, effect_ci_low=lo, effect_ci_high=hi,
               effect_ci_method=ci_method, rank_biserial=float(rbc), informative=True,
               min_attainable_p=min_attainable_p(len(nz)))
    return out


def _signed_rank_cdf(n):
    """Exact null CDF of T+ for untied ranks 1..n (subset-sum counts)."""
    m = n * (n + 1) // 2
    counts = [0] * (m + 1)
    counts[0] = 1
    for r in range(1, n + 1):
        for t in range(m, r - 1, -1):
            counts[t] += counts[t - r]
    total = 2 ** n
    running = 0
    cdf = []
    for c in counts:
        running += c
        cdf.append(running / total)
    return np.asarray(cdf)


def hodges_lehmann(x, ties=False, level=0.95):
    x = np.asarray(x, float)
    n = len(x)
    i, j = np.triu_indices(n)
    walsh = np.sort((x[i] + x[j]) / 2)
    hl = float(np.median(walsh))
    cdf = _signed_rank_cdf(n)
    alpha = 1 - level
    ok = np.nonzero(cdf <= alpha / 2)[0]
    if len(ok) == 0:
        return hl, float('nan'), float('nan'), 'unavailable_small_n'
    c = int(ok[-1])
    m = len(walsh)
    lo = float(walsh[c]); hi = float(walsh[m - c - 1])
    return hl, lo, hi, ('walsh_exact_approx_ties' if ties else 'walsh_exact')


# ---------------------------------------------------------------- association and trend

def _rank_last(a):
    return st.rankdata(a, axis=-1)


def _pearson_last(a, b):
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return (a * b).sum(-1) / den


def _spearman_vec(x, y, axis=-1):
    return _pearson_last(_rank_last(x), _rank_last(y))


def bootstrap_ci(data, statistic, test_id, vectorized=True):
    rng = rng_for(test_id + ':bootstrap')
    try:
        res = st.bootstrap(data, statistic, n_resamples=N_RESAMPLES, paired=True, vectorized=vectorized,
                           method='BCa', rng=rng)
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        if math.isnan(lo) or math.isnan(hi):
            raise ValueError
        return lo, hi, 'bca_bootstrap'
    except Exception:
        rng = rng_for(test_id + ':bootstrap')
        res = st.bootstrap(data, statistic, n_resamples=N_RESAMPLES, paired=True, vectorized=vectorized,
                           method='percentile', rng=rng)
        return float(res.confidence_interval.low), float(res.confidence_interval.high), 'percentile_bootstrap'


def spearman(x, y, test_id):
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    rho = float(st.spearmanr(x, y).statistic)
    rng = rng_for(test_id)
    ry = st.rankdata(y)
    rx = st.rankdata(x)
    perm = np.argsort(rng.random((N_RESAMPLES, len(y))), axis=1)
    null = _pearson_last(np.broadcast_to(rx, perm.shape), ry[perm])
    k = int(np.sum(np.abs(null) >= abs(rho) - 1e-12))
    p = (k + 1) / (N_RESAMPLES + 1)
    lo, hi, method = bootstrap_ci((x, y), _spearman_vec, test_id)
    return dict(test='spearman_permutation', statistic_name='rho', statistic=rho, p_value=p,
                p_method='permutation', n=len(x), n_used=len(x), n_resamples=N_RESAMPLES, mc_se=mc_se(p, N_RESAMPLES),
                effect_name='spearman_rho', effect=rho, effect_ci_low=lo, effect_ci_high=hi,
                effect_ci_method=method, informative=True)


def theil_sen(x, y, test_id):
    x = np.asarray(x, float); y = np.asarray(y, float)
    res = st.theilslopes(y, x)

    def slope(xx, yy):
        return st.theilslopes(yy, xx).slope

    lo, hi, method = bootstrap_ci((x, y), slope, test_id, vectorized=False)
    return dict(test='theil_sen', statistic_name='slope', statistic=float(res.slope), p_value=None,
                p_method=None, n=len(x), n_used=len(x), effect_name='theil_sen_slope', effect=float(res.slope),
                effect_ci_low=lo, effect_ci_high=hi, effect_ci_method=method,
                kendall_ci_low=float(res.low_slope), kendall_ci_high=float(res.high_slope),
                intercept=float(res.intercept), informative=True)


def jonckheere(values, groups, test_id):
    """Two-sided Jonckheere-Terpstra trend test over ordered groups (permutation null)."""
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    levels = sorted(np.unique(groups))

    def stat(v):
        total = 0.0
        for gi, gj in combinations(levels, 2):
            a = v[..., groups == gi][..., :, None]
            b = v[..., groups == gj][..., None, :]
            total = total + (a < b).sum(axis=(-1, -2)) + 0.5 * (a == b).sum(axis=(-1, -2))
        return total

    obs = float(stat(values))
    sizes = np.array([np.sum(groups == g) for g in levels])
    n = len(values)
    expected = (n * n - np.sum(sizes ** 2)) / 4
    rng = rng_for(test_id)
    perm = np.argsort(rng.random((N_RESAMPLES, n)), axis=1)
    null = stat(values[perm])
    k = int(np.sum(np.abs(null - expected) >= abs(obs - expected) - 1e-9))
    p = (k + 1) / (N_RESAMPLES + 1)
    return dict(test='jonckheere_terpstra_permutation', statistic_name='J', statistic=obs, p_value=p,
                p_method='permutation', n=n, n_used=n, n_resamples=N_RESAMPLES, mc_se=mc_se(p, N_RESAMPLES),
                effect_name='J_minus_expected_over_max', effect=float((obs - expected) / (2 * expected)) if expected else float('nan'),
                informative=True)


def cochran_armitage(scores, outcomes):
    """Exact (conditional) Cochran-Armitage trend test; scores are ordinal integers."""
    scores = np.asarray(scores, int)
    outcomes = np.asarray(outcomes, bool)
    levels = [int(s) for s in sorted(np.unique(scores))]  # Python ints: weights are exact big integers
    n_g = [int(np.sum(scores == s)) for s in levels]
    k_g = [int(np.sum(outcomes[scores == s])) for s in levels]
    K = int(outcomes.sum()); N = len(outcomes)
    t_obs = sum(s * k for s, k in zip(levels, k_g))
    dist = {(0, 0): 1}
    for s, n in zip(levels, n_g):
        new = {}
        for (used, t), w in dist.items():
            for k in range(0, min(n, K - used) + 1):
                key = (used + k, t + s * k)
                new[key] = new.get(key, 0) + w * math.comb(n, k)
        dist = new
    final = {t: w for (used, t), w in dist.items() if used == K}
    total = math.comb(N, K)
    expected = sum(t * w for t, w in final.items()) / total
    tail = sum(w for t, w in final.items() if abs(t - expected) >= abs(t_obs - expected) - 1e-9)
    p = tail / total
    direction = 'increasing' if t_obs > expected else ('decreasing' if t_obs < expected else 'none')
    return dict(test='cochran_armitage_exact', statistic_name='T', statistic=float(t_obs), p_value=float(min(1.0, p)),
                p_method='exact', n=N, n_used=N, effect_name='T_minus_expected', effect=float(t_obs - expected),
                direction=direction, informative=K not in (0, N), min_attainable_p=None)


def page(data, predicted_increasing=True):
    """Two-sided Page trend test on a subjects x conditions array (columns in predicted order)."""
    data = np.asarray(data, float)
    up = st.page_trend_test(data, method='auto')
    down = st.page_trend_test(data[:, ::-1], method='auto')
    p = min(1.0, 2 * min(float(up.pvalue), float(down.pvalue)))
    return dict(test='page_trend', statistic_name='L', statistic=float(up.statistic), p_value=p,
                p_method=str(up.method), n=data.shape[0], n_used=data.shape[0],
                effect_name='direction', effect=float('nan'),
                direction=('increasing' if up.pvalue < down.pvalue else 'decreasing'), informative=True)


def fisher(table):
    from scipy.stats.contingency import odds_ratio
    table = np.asarray(table, int)
    p = float(st.fisher_exact(table).pvalue)
    orr = odds_ratio(table, kind='conditional')
    ci = orr.confidence_interval()
    return dict(test='fisher_exact', statistic_name='conditional_odds_ratio', statistic=float(orr.statistic),
                p_value=p, p_method='exact', n=int(table.sum()), n_used=int(table.sum()),
                effect_name='conditional_odds_ratio', effect=float(orr.statistic), effect_ci_low=float(ci.low),
                effect_ci_high=float(ci.high), effect_ci_method='conditional_mle_exact', informative=True)


def perm_p(observed, null, two_sided=True, center=None):
    null = np.asarray(null, float)
    if two_sided:
        c = np.mean(null) if center is None else center
        k = int(np.sum(np.abs(null - c) >= abs(observed - c) - 1e-12))
    else:
        k = int(np.sum(null >= observed - 1e-12))
    return (k + 1) / (len(null) + 1)


# ---------------------------------------------------------------- multiplicity

def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p, kind='mergesort')
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def bh(pvalues):
    p = np.asarray(pvalues, float)
    if len(p) == 0:
        return p
    return st.false_discovery_control(p, method='bh')


# ---------------------------------------------------------------- registry

REQUIRED = ('test_id', 'panel', 'family', 'contrast', 'unit', 'level', 'analysis_status')
_phase = 'unset'


def set_phase(name):
    global _phase
    _phase = name


def record(result, **meta):
    missing = [k for k in REQUIRED if k not in meta]
    if missing:
        raise ValueError(f'test metadata missing {missing}')
    if any(r['test_id'] == meta['test_id'] for r in _registry):
        raise ValueError(f'duplicate test_id {meta["test_id"]}')
    row = {'phase': _phase, **meta, **result}
    _registry.append(row)
    return row


def registry():
    return list(_registry)


def save_phase(phase):
    target = paths.work(f'tests/{phase}.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in _registry if r.get('phase') == phase]
    target.write_text(json.dumps(rows, indent=0, default=_json_default), encoding='utf-8')


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    raise TypeError(type(v))


def load_all():
    rows = []
    folder = paths.work('tests')
    if folder.exists():
        for f in sorted(folder.glob('*.json')):
            rows.extend(json.loads(f.read_text(encoding='utf-8')))
    return rows
