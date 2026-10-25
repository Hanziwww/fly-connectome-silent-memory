"""Analysis of X3 (jittered-timing training) against the fixed-timing X1 replicates (local, after the pulls). Written
before any X3 result existed; families declared in code/testplan.py.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-supplement-v2\\code\\analyze_x3.py

Per run: the X1 run summary (analyze_campaign.run_row): net memory use at the 9 trained and the 26 untrained timings,
plus the untrained timings split by whether the jitter range covers them (delay <= 320 ms) or not (640 ms).
F13.X3.lock: net use at the 26 untrained timings, X3 replicate r minus X1 replicate r, exact one-sided sign-flip (> 0),
computed only when all 10 pairs are verified. Writes results/x3_runs.csv and x3_tests.csv.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUPP / 'code'))
from studykit import stats_ext as sx  # noqa: E402
from analyze_campaign import TRAINED, classes, run_row  # noqa: E402

N = 10


def split_use(folder):
    ver = json.loads((folder / 'verification.json').read_text(encoding='utf-8'))
    rolls = [r for r in ver['results']['rollouts'] if (r['delay_ms'], r['prefix_windows']) not in TRAINED]
    inside = [r for r in rolls if r['delay_ms'] <= 320]
    outside = [r for r in rolls if r['delay_ms'] > 320]
    return classes(inside)[2], classes(outside)[2]


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    out = SUPP / 'results'
    rows, pairs = [], []
    for r in range(N):
        pair = {}
        for arm, name in (('jittered', f'x3-M0-head_only-r{r}'), ('fixed', f'x1-M0-head_only-r{r}')):
            folder = SUPP / 'runs' / name
            if (folder / 'result.json').exists() and (folder / 'verification.json').exists():
                row = run_row(folder, name)
                row['arm'], row['replicate'] = arm, r
                row['untrained_inside_net_use'], row['untrained_outside_net_use'] = split_use(folder)
                rows.append(row)
                if row['verified'] == 'verified':
                    pair[arm] = row
        if len(pair) == 2:
            pairs.append(pair)
    write(out / 'x3_runs.csv', rows)
    tests = []
    if len(pairs) == N:
        d = np.array([p['jittered']['untrained_net_use'] - p['fixed']['untrained_net_use'] for p in pairs])
        sf = sx.sign_flip(d, alternative='greater', ci=False)
        tests.append(dict(family='F13.X3.lock', test='untrained-timing net memory use, jittered minus fixed training', n=N,
                          estimate=float(d.mean()), p_value=sf['p_value']))
        for field in ('trained_net_use', 'untrained_inside_net_use', 'untrained_outside_net_use', 'updates'):
            x = np.array([p['jittered'][field] - p['fixed'][field] for p in pairs], float)
            tests.append(dict(family='F13.X3.descriptive', test=f'{field}: jittered minus fixed (two-sided sign-flip)', n=N,
                              estimate=float(x.mean()), p_value=sx.sign_flip(x, alternative='two-sided', ci=False)['p_value']))
    else:
        print(f'{len(pairs)} of {N} replicate pairs verified; tests pending')
    write(out / 'x3_tests.csv', tests)
    for t in tests:
        print(t['family'], '|', t['test'], '|', round(t['estimate'], 4), '|', round(t['p_value'], 5))
    return 0


if __name__ == '__main__':
    sys.exit(main())
