"""T-DY2 analysis (local, after the pulls; F14.DY.robust, .secondary, .descriptive). Written before any T-DY2 result existed.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\analyze_dy2.py

Inputs: runs/dy2-eval/<model>.json (dy2_evaluate.py) for the 45 healthy and 5 C01-trained T-DY2 runs, the X1 readout-only
replicates 0-9 and regime B C01 dose 0.5 readout-only replicates 1-6, and each run's result.json. Measures as in
tasks/dy2-design.json:
- net use = (cue pairs both solved - pairs none solved) / pairs; trained timings 9 pairs, untrained 26; lock = difference;
- C01 ED50: the shift magnitude at which trained net use first falls to <= 0.5 along -3e-5 ... -3.22 mV (as-trained at 0),
  log-linear interpolation; 3e-5 if already <= 0.5 there, 3.22 if never; 0 (the lowest rank: no memory to lose) if the
  as-trained trained net use is <= 0.5. Removal ED50 likewise over the shares 0.0002-0.005 (mean over the 3 draws);
- own perturbation: trained net use averaged over the two held-out draws of the model's training condition (X1: of each
  condition in turn), and for T-DY2 models untrained net use and the lock under them; restoration loss; mechanism drops
  at the prefix-0 trained timings; updates.
Tests: exact Mann-Whitney (rank sum of the first group over all splits of the pooled values, mid-ranks for ties).
Writes results/dy2_models.csv, dy2_panel.csv, dy2_conditions.csv and dy2_tests.csv (collect_tests.py adjusts the families).
"""
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
EVAL = STUDY / 'runs' / 'dy2-eval'
sys.path.insert(0, str(STUDY / 'code'))
import dy2_perturbation as dy2  # noqa: E402
import dy2_evaluate as ev  # noqa: E402

CONDITIONS = list(dy2.CONDITIONS)
TRAINED = {(d, p) for d in (0, 40, 160) for p in (0, 1, 4)}
UNTRAINED = {(d, p) for d in (0, 20, 40, 80, 160, 320, 640) for p in (0, 1, 2, 4, 8)} - TRAINED
PREFIX0 = {(d, 0) for d in ev.MECHANISM['delays']}
C01_GRID = sorted(-s for s in ev.SHIFTS_MV if s < 0)
SHARES = (0.0002, 0.0005, 0.001, 0.002, 0.005)
THRESHOLD = 0.5
BOOT, SEED = 10000, 20261020


def models():
    out = [dict(model=f'dy2-{c}-r{r}', group='dy2', condition=c, replicate=r, folder=STUDY / 'runs' / f'dy2-{c}-r{r}')
           for c in CONDITIONS for r in range(5)]
    out += [dict(model=f'x1-M0-head_only-r{r}', group='x1', condition=None, replicate=r, folder=SUPP / 'runs' / f'x1-M0-head_only-r{r}')
            for r in range(10)]
    out += [dict(model=f'dy2-C01-dose0.5-th0.5-r{r}', group='dy2-c01', condition='th0.5', replicate=r,
                 folder=STUDY / 'runs' / f'dy2-C01-dose0.5-th0.5-r{r}') for r in range(1, 6)]
    out += [dict(model=f'b-C01-dose0.5-M0-head_only-r{r}', group='regime-b', condition=None, replicate=r,
                 folder=STUDY / 'runs' / f'b-C01-dose0.5-M0-head_only-r{r}') for r in range(1, 7)]
    return out


def net_use(rollouts, timings):
    groups = {}
    for r in rollouts:
        key = (r['delay_ms'], r['prefix_windows'])
        if key in timings:
            groups.setdefault(key, []).append(r['status'] == 'solved')
    if not groups:
        return None
    both = sum(all(v) for v in groups.values())
    none = sum(not any(v) for v in groups.values())
    return (both - none) / len(groups)


def ed50(start, points, low, high):
    """(value, kind): kind interpolated, left (<= low), right (>= high) or none (start <= threshold: value 0)."""
    if start is None or start <= THRESHOLD:
        return 0.0, 'none'
    prev_m, prev_u = None, start
    for m, u in points:
        if u <= THRESHOLD:
            if prev_m is None:
                return low, 'left'
            f = (prev_u - THRESHOLD) / (prev_u - u)
            return 10 ** (math.log10(prev_m) + f * (math.log10(m) - math.log10(prev_m))), 'interpolated'
        prev_m, prev_u = m, u
    return high, 'right'


def ranks(values):
    v = np.asarray(values, float)
    order = np.argsort(v, kind='mergesort')
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def mann_whitney(x, y, alternative):
    """Exact: rank sum of x over all len(x)-subsets of the pooled sample (mid-ranks)."""
    x, y = [float(a) for a in x], [float(b) for b in y]
    if not x or not y:
        return None, None
    r = ranks(x + y)
    obs = r[:len(x)].sum()
    null = np.array([r[list(c)].sum() for c in itertools.combinations(range(len(r)), len(x))])
    mean = len(x) * (len(r) + 1) / 2
    if alternative == 'greater':
        p = np.mean(null >= obs - 1e-9)
    elif alternative == 'less':
        p = np.mean(null <= obs + 1e-9)
    else:
        p = np.mean(np.abs(null - mean) >= abs(obs - mean) - 1e-9)
    return float(obs), float(p)


def geo_ratio(x, y, rng):
    x = np.array([a for a in x if a > 0], float)
    y = np.array([b for b in y if b > 0], float)
    if not len(x) or not len(y):
        return None, None, None
    est = float(np.exp(np.log(x).mean() - np.log(y).mean()))
    boot = [np.exp(np.log(rng.choice(x, len(x))).mean() - np.log(rng.choice(y, len(y))).mean()) for _ in range(BOOT)]
    return est, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def measure(m, rec):
    cond = rec['conditions']
    res = json.loads((m['folder'] / 'result.json').read_text(encoding='utf-8'))
    a = cond['as-trained']['rollouts']
    row = dict(model=m['model'], group=m['group'], condition=m['condition'], replicate=m['replicate'], status=res.get('status'),
               updates=res.get('updates'), training_seconds=res.get('training_seconds'),
               reproduces_verification=rec['reproduces_verification'],
               trained_net_use=net_use(a, TRAINED), untrained_net_use=net_use(a, UNTRAINED))
    row['lock'] = row['trained_net_use'] - row['untrained_net_use']
    base0 = net_use(a, PREFIX0)
    for kind in ev.MECHANISM['interventions']:
        row[f'{kind}_drop'] = base0 - net_use(cond[f'mechanism-{kind}']['rollouts'], PREFIX0)
    if 'restored' in cond:
        row['restored_trained_net_use'] = net_use(cond['restored']['rollouts'], TRAINED)
        row['restoration_loss'] = row['trained_net_use'] - row['restored_trained_net_use']
        row['th0.5_heldout_trained_net_use'] = float(np.mean([net_use(cond[f'th0.5-s{k}']['rollouts'], TRAINED) for k in range(2)]))
        return row
    points = [(g, net_use(cond[f'shift{-g:+g}']['rollouts'], TRAINED)) for g in C01_GRID]
    row['ed50_c01_mv'], row['ed50_c01_kind'] = ed50(row['trained_net_use'], points, C01_GRID[0], C01_GRID[-1])
    removal = []
    for q in SHARES:
        names = [k for k in cond if k.startswith('remove-random-') and abs(float(k.split('-')[2]) - q) < 1e-12]
        if len(names) != 3:
            raise RuntimeError(f"{m['model']}: {len(names)} removal draws at share {q}")
        removal.append((q, float(np.mean([net_use(cond[k]['rollouts'], TRAINED) for k in names]))))
    row['ed50_removal_share'], row['ed50_removal_kind'] = ed50(row['trained_net_use'], removal, SHARES[0], SHARES[-1])
    for c in CONDITIONS:
        row[f'perturbed_{c}'] = float(np.mean([net_use(cond[f'{c}-s{k}']['rollouts'], TRAINED) for k in range(2)]))
    row['own_perturbation'] = row[f"perturbed_{m['condition']}"] if m['condition'] else None
    if m['condition']:
        row['own_untrained_net_use'] = float(np.mean([net_use(cond[f"{m['condition']}-s{k}"]['rollouts'], UNTRAINED) for k in range(2)]))
        row['own_lock'] = row['own_perturbation'] - row['own_untrained_net_use']
    return row


def panel(m, rec):
    rows = []
    for name, c in rec['conditions'].items():
        timings = {(r['delay_ms'], r['prefix_windows']) for r in c['rollouts']}
        rows.append(dict(model=m['model'], group=m['group'], condition=m['condition'], evaluation=name, solved=c['solved'], total=c['total'],
                         trained_net_use=net_use(c['rollouts'], TRAINED & timings),
                         untrained_net_use=net_use(c['rollouts'], timings - TRAINED) if timings - TRAINED else None,
                         seconds=rec['seconds'].get(name)))
    return rows


def main():
    ms = models()
    missing = [m['model'] for m in ms if not (EVAL / f"{m['model']}.json").exists()]
    if missing:
        print(f'pending: {len(missing)} of {len(ms)} evaluations missing ({", ".join(missing[:5])}{" ..." if len(missing) > 5 else ""})')
        return 0
    out = STUDY / 'results'
    rows, prows = [], []
    for m in ms:
        rec = json.loads((EVAL / f"{m['model']}.json").read_text(encoding='utf-8'))
        if not rec['reproduces_verification']:
            raise SystemExit(f"{m['model']}: evaluation does not reproduce its verification")
        rows.append(measure(m, rec))
        prows += panel(m, rec)
    x1 = [r for r in rows if r['group'] == 'x1']
    tests, conds = [], []
    rng = np.random.default_rng(SEED)
    for c in CONDITIONS:
        g = [r for r in rows if r['group'] == 'dy2' and r['condition'] == c]
        spec = [('F14.DY.robust', 'ed50-c01', 'ed50_c01_mv', 'greater', None),
                ('F14.DY.robust.secondary', 'trained-net-use', 'trained_net_use', 'two-sided', None),
                ('F14.DY.robust.secondary', 'untrained-net-use', 'untrained_net_use', 'two-sided', None),
                ('F14.DY.robust.secondary', 'own-perturbation', 'own_perturbation', 'greater', f'perturbed_{c}'),
                ('F14.DY.robust.secondary', 'removal-ed50', 'ed50_removal_share', 'greater', None),
                ('F14.DY.robust.secondary', 'updates', 'updates', 'two-sided', None)]
        for fam, name, key, alt, xkey in spec:
            xs, ys = [r[key] for r in g], [r[xkey or key] for r in x1]
            stat, p = mann_whitney(xs, ys, alt)
            tests.append(dict(family=fam, test=f'{c}:{name}', n=len(xs), n_reference=len(ys), alternative=alt, statistic=stat,
                              mean=float(np.mean(xs)), mean_reference=float(np.mean(ys)), p_value=p))
        ratio, lo, hi = geo_ratio([r['ed50_c01_mv'] for r in g], [r['ed50_c01_mv'] for r in x1], rng)
        rratio, rlo, rhi = geo_ratio([r['ed50_removal_share'] for r in g], [r['ed50_removal_share'] for r in x1], rng)
        drops = {k: (float(np.mean([r[f'{k}_drop'] for r in g])), float(np.mean([r[f'{k}_drop'] for r in x1])))
                 for k in ev.MECHANISM['interventions']}
        memory_kept = sum(r['own_perturbation'] > 0 for r in g)
        mechanism = all(d >= 0.5 * ref for d, ref in drops.values())
        conds.append(dict(condition=c, replicates=len(g), learned=sum(r['status'] == 'learned_solved' for r in g),
                          own_perturbation_positive=memory_kept, own_perturbation_mean=float(np.mean([r['own_perturbation'] for r in g])),
                          x1_under_same_mean=float(np.mean([r[f'perturbed_{c}'] for r in x1])),
                          adaptation_drop=drops['adaptation'][0], adaptation_drop_x1=drops['adaptation'][1],
                          swap_drop=drops['swap'][0], swap_drop_x1=drops['swap'][1], mechanism_retained=mechanism,
                          gate1_criterion=memory_kept >= 4 and mechanism,
                          ed50_c01_ratio=ratio, ed50_c01_ratio_lo=lo, ed50_c01_ratio_hi=hi,
                          ed50_c01_ratio_at_least_10=None if ratio is None else ratio >= 10,
                          ed50_removal_ratio=rratio, ed50_removal_ratio_lo=rlo, ed50_removal_ratio_hi=rhi,
                          updates_median=float(np.median([r['updates'] for r in g])),
                          seconds_median=float(np.median([r['training_seconds'] for r in g]))))
    dc = [r for r in rows if r['group'] == 'dy2-c01']
    rb = [r for r in rows if r['group'] == 'regime-b']
    stat, p = mann_whitney([r['restoration_loss'] for r in dc], [r['restoration_loss'] for r in rb], 'less')
    tests.append(dict(family='F14.DY.robust', test='state-dependence:restoration-loss', n=len(dc), n_reference=len(rb), alternative='less',
                      statistic=stat, mean=float(np.mean([r['restoration_loss'] for r in dc])),
                      mean_reference=float(np.mean([r['restoration_loss'] for r in rb])), p_value=p))
    write(out / 'dy2_models.csv', rows)
    write(out / 'dy2_panel.csv', prows)
    write(out / 'dy2_conditions.csv', conds)
    write(out / 'dy2_tests.csv', tests)
    for c in conds:
        print(f"{c['condition']:10s} learned {c['learned']}/{c['replicates']}, own-perturbation use > 0 in {c['own_perturbation_positive']}, "
              f"mechanism {c['mechanism_retained']}, Gate 1 {c['gate1_criterion']}, ED50 ratio {c['ed50_c01_ratio']}")
    for t in tests:
        if t['family'] == 'F14.DY.robust':
            print(f"{t['test']:34s} n {t['n']} vs {t['n_reference']}: p {t['p_value']:.4g} (raw; Holm in collect_tests.py)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
