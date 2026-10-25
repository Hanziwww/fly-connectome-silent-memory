"""Analysis of X2 (new scenes, as X1) (local, after the pulls). Written before any X2 result existed; families declared
in code/testplan.py.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-supplement-v2\\code\\analyze_x2.py

Per run: analyze_campaign.run_row (net memory use at the 9 trained and 26 untrained timings). Per scene, once all 7
replicates are verified: F13.X2.trained (> 0), F13.X2.untrained (< +0.10), F13.X2.lock (trained - untrained > 0), exact
one-sided sign-flip. Per family (original scene = the X1 learner of that family, M0 readout-only / S1 / S2, plus the two
new scenes): the lock effect's Cochran Q and I^2 from the scene means and their standard errors over replicates (a
scene whose replicates all agree gets the smallest positive variance over the family's scenes, floor 1e-6).
Writes results/x2_runs.csv and x2_tests.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUPP / 'code'))
from studykit import stats_ext as sx  # noqa: E402
from analyze_campaign import run_row  # noqa: E402

N = 7
FAMILIES = {'M': ('x1-M0-head_only', ('M1', 'M2')), 'S1': ('x1-S1-head_only', ('S1-mirror', 'S1-minus45')),
            'S2': ('x1-S2-head_only', ('S2-mirror', 'S2-minus48'))}


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def scene_runs(prefix, n):
    rows = []
    for r in range(n):
        folder = SUPP / 'runs' / f'{prefix}-r{r}'
        if (folder / 'result.json').exists() and (folder / 'verification.json').exists():
            row = run_row(folder, folder.name)
            if row['verified'] == 'verified':
                rows.append(row)
    return rows


def main():
    out = SUPP / 'results'
    tests, rows = [], []
    for fam, (original, scenes) in FAMILIES.items():
        effects = {}
        orig = scene_runs(original, 10 if fam == 'M' else 7)
        if orig:
            effects['original'] = np.array([r['trained_net_use'] - r['untrained_net_use'] for r in orig])
        for scene in scenes:
            reps = scene_runs(f'x2-{scene}', N)
            for r in reps:
                rows.append(dict(r, scene=scene, family=fam))
            if len(reps) < N:
                print(f'{scene}: {len(reps)} of {N} replicates verified; tests pending')
                continue
            tr = np.array([r['trained_net_use'] for r in reps])
            un = np.array([r['untrained_net_use'] for r in reps])
            effects[scene] = tr - un
            for fam_name, x, mu, alt in (('F13.X2.trained', tr, 0.0, 'greater'), ('F13.X2.untrained', un - 0.10, 0.0, 'less'),
                                         ('F13.X2.lock', tr - un, 0.0, 'greater')):
                tests.append(dict(family=fam_name, test=scene, n=N, estimate=float(np.mean(x + (0.10 if fam_name.endswith('untrained') else 0))),
                                  p_value=sx.sign_flip(x - mu, alternative=alt, ci=False)['p_value']))
        if len(effects) == 3:
            means = np.array([v.mean() for v in effects.values()])
            var = np.array([v.var(ddof=1) / len(v) for v in effects.values()])
            positive = var[var > 0]
            var = np.where(var > 0, var, max(positive.min() if len(positive) else 1e-6, 1e-6))
            w = 1 / var
            pooled = float((w * means).sum() / w.sum())
            Q = float((w * (means - pooled) ** 2).sum())
            I2 = max(0.0, (Q - 2) / Q) if Q > 0 else 0.0
            tests.append(dict(family='F13.X2.heterogeneity', test=f'{fam}: lock effect across the original and two new scenes', n=3,
                              estimate=pooled, Q=Q, I2=I2, scene_means=';'.join(f'{k}={v.mean():.3f}' for k, v in effects.items()), p_value=None))
    write(out / 'x2_runs.csv', rows)
    write(out / 'x2_tests.csv', [t for t in tests if t['p_value'] is not None] + [t for t in tests if t['p_value'] is None])
    for t in tests:
        print(t['family'], '|', t['test'], '|', round(t['estimate'], 4), '|', t['p_value'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
