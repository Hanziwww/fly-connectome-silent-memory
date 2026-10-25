"""T-SC6 analysis (local, after the hash-checked pulls of runs/sc-variant-* and runs/sc-variant-eval). Written before any
T-SC6 result existed; families F14.SC.variant, .secondary, .descriptive (tasks/sc-panels-v1.json, amendments 0005, 0012).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\analyze_sc_variant.py

Per run (from its evaluation, whose as-trained assessment reproduces the run's verification): trained and untrained net
use (cue-pair classes of analyze_campaign.classes over the 9 trained and 26 untrained timings), lock = trained - untrained,
adaptation drop = prefix-0 trained net use as trained minus with adaptation cleared at decision onset, swap drop likewise,
and for M0 the C01 ED50 (analyze_dy2.ed50 over the shift ladder, trained net use threshold 0.5).
Tests:
- F14.SC.variant (Holm, 3): variant brain, M0 r0-9: trained net use > 0, lock > 0, adaptation drop > 0; exact one-sided
  sign tests (zeros dropped);
- F14.SC.variant.secondary (BH, 11): variant vs X1 M0 r0-9 (results/dy2_models.csv) and variant vs null brain M0 (exact
  two-sided Mann-Whitney): trained net use, untrained net use, lock, C01 ED50 (8); S1 variant r0-4: the three sign tests (3).
Descriptive: per run values, the null brain's own sign tests, updates and statuses.
Outputs (results/): sc_variant_tests.csv, sc_variant_runs.csv.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
RUNS = STUDY / 'runs'
RESULTS = STUDY / 'results'
sys.path.insert(0, str(STUDY / 'code'))
sys.path.insert(0, str(ROOT / 'studies' / 'brain-supplement-v2' / 'code'))
from analyze_campaign import TRAINED, classes  # noqa: E402
from analyze_dy2 import C01_GRID, ed50, mann_whitney  # noqa: E402
from sc_variant import runs  # noqa: E402

PREFIX0 = {(d, 0) for d in (0, 40, 160)}


def use(rows, timings, inside=True):
    sel = [r for r in rows if ((r['delay_ms'], r['prefix_windows']) in timings) == inside]
    return classes(sel)[2] if sel else None


def measures(ev):
    cond = ev['conditions']
    rows = cond['as-trained']['rollouts']
    m = dict(trained_net_use=use(rows, TRAINED), untrained_net_use=use(rows, TRAINED, False))
    m['lock'] = m['trained_net_use'] - m['untrained_net_use']
    base0 = use(rows, PREFIX0)
    for kind in ('adaptation', 'swap'):
        m[f'{kind}_drop'] = base0 - use(cond[f'mechanism-{kind}']['rollouts'], PREFIX0)
    if 'shift-3.22' in cond:
        points = [(g, use(cond[f'shift{-g:+g}']['rollouts'], TRAINED)) for g in C01_GRID]
        m['ed50_c01_mv'], m['ed50_c01_kind'] = ed50(m['trained_net_use'], points, C01_GRID[0], C01_GRID[-1])
    return m


def sign_test(values):
    v = [x for x in values if x is not None and x != 0]
    k = sum(x > 0 for x in v)
    return (float(st.binomtest(k, len(v), 0.5, alternative='greater').pvalue) if v else 1.0), k, len(v)


def main():
    rows = []
    for brain, family, model, rep, out in runs():
        name = Path(out).name
        ev_path = RUNS / 'sc-variant-eval' / f'{name}.json'
        if not ev_path.exists():
            raise SystemExit(f'{name}: evaluation missing (pull runs/sc-variant-eval and the run folders)')
        ev = json.loads(ev_path.read_text(encoding='utf-8'))
        if not ev.get('reproduces_verification'):
            raise SystemExit(f'{name}: the evaluation does not reproduce the verification')
        res = json.loads((ROOT / out / 'result.json').read_text(encoding='utf-8'))
        rows.append(dict(run=name, brain=brain, family=family, replicate=rep, status=res.get('status'), updates=res.get('updates'),
                         **measures(ev)))
    x1 = list(csv.DictReader(open(RESULTS / 'dy2_models.csv', encoding='utf-8')))
    x1 = [r for r in x1 if r['group'] == 'x1']
    if len(x1) != 10:
        raise SystemExit(f'expected the 10 X1 M0 replicates in dy2_models.csv, found {len(x1)}')
    sel = lambda b, f: [r for r in rows if r['brain'] == b and r['family'] == f]   # noqa: E731
    vm, nm, vs = sel('variant', 'M0'), sel('null', 'M0'), sel('variant', 'S1')
    tests = []
    for key, label in (('trained_net_use', 'trained net use > 0'), ('lock', 'lock > 0'), ('adaptation_drop', 'adaptation drop > 0')):
        p, k, n = sign_test([r[key] for r in vm])
        tests.append(dict(family='F14.SC.variant', test=f'variant M0: {label}', n=n, statistic=k, alternative='greater', p_value=p))
    for key in ('trained_net_use', 'untrained_net_use', 'lock', 'ed50_c01_mv'):
        v = [float(r[key]) for r in vm]
        for ref_name, ref in (('X1', [float(r[key]) for r in x1]), ('null brain', [float(r[key]) for r in nm])):
            stat, p = mann_whitney(v, ref, 'two-sided')
            tests.append(dict(family='F14.SC.variant.secondary', test=f'variant vs {ref_name}, M0: {key}', n=len(v), n_reference=len(ref),
                              statistic=stat, mean=float(np.mean(v)), mean_reference=float(np.mean(ref)), alternative='two-sided', p_value=p))
    for key, label in (('trained_net_use', 'trained net use > 0'), ('lock', 'lock > 0'), ('adaptation_drop', 'adaptation drop > 0')):
        p, k, n = sign_test([r[key] for r in vs])
        tests.append(dict(family='F14.SC.variant.secondary', test=f'variant S1: {label}', n=n, statistic=k, alternative='greater', p_value=p))
    for path, data in (('sc_variant_tests.csv', tests), ('sc_variant_runs.csv', rows)):
        keys = list(dict.fromkeys(k for r in data for k in r))
        with open(RESULTS / path, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
            w.writeheader()
            w.writerows(data)
    for t in tests:
        print(f"{t['family']:26s} {t['test']:44s} p {t['p_value']:.4g}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
