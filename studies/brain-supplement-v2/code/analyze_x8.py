"""Analysis of X8 (Sokoban training from a neutral brain, readout-only vs joint) (local, after the pulls). Written before
any X8 run; families declared in code/testplan.py.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-supplement-v2\\code\\analyze_x8.py

Per run (result.json, verification.json of the frozen worker): status, updates, training seconds, solved after
training, and whether the verification reproduced it. Updates to solve: a run that ends 'learned_solved' or
'already_solved' is an event at its update count; any other end (stagnation, segment limit) is censored at its count.
Replicate 0 of the joint arm is compared with the frozen control-D<t>-neutral run of the same map.
Tests (only once every run of the final design is verified; n per arm and map from amendments/x8-n.json, written after
the pilot; before that only the pilot table is written):
- F13.X8.primary: stratified log-rank statistic (sum over maps of observed minus expected events of the readout-only
  arm), two-sided permutation p with the arm labels permuted within maps (exact when feasible, else 100,000 draws);
- F13.X8.map: per map, exact permutation Mann-Whitney on updates with unsolved runs ranked last (ties shared), and
  Fisher's exact test on runs solved.
Writes results/x8_runs.csv and, when complete, x8_tests.csv.
"""
import csv
import itertools
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
FROZEN_RUNS = ROOT / 'studies' / 'brain-sokoban-v1' / 'runs'
MAPS = ('D1-T0', 'D3-T0', 'D5-T0', 'D7-T0')
ARMS = ('head_only', 'joint')
EVENT = ('learned_solved', 'already_solved')
N_MC = 100_000


def runs():
    rows = []
    for folder in sorted((SUPP / 'runs').glob('x8-*')):
        if not (folder / 'result.json').exists():
            continue
        res = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        ver = json.loads((folder / 'verification.json').read_text(encoding='utf-8')) if (folder / 'verification.json').exists() else {}
        cfg = res['config']
        arm = 'head_only' if not cfg['condition']['train_core'] else 'joint'
        rows.append(dict(run=folder.name, map=cfg['task'], arm=arm, replicate=cfg['replicate'], status=res.get('status'),
                         updates=res.get('updates'), training_seconds=res.get('training_seconds'), final_solved=res.get('final_solved'),
                         event=res.get('status') in EVENT, verified=ver.get('status')))
    for r in rows:
        if r['arm'] == 'joint' and r['replicate'] == 0:
            f = FROZEN_RUNS / f"control-{r['map'].split('-')[0]}-neutral" / 'result.json'
            if f.exists():
                fr = json.loads(f.read_text(encoding='utf-8'))
                r.update(frozen_status=fr.get('status'), frozen_updates=fr.get('updates'),
                         reproduces_frozen=fr.get('status') == r['status'] and fr.get('updates') == r['updates'])
    return rows


def logrank_stat(times, events, group, strata):
    """Sum over strata of (observed - expected) events in group 1 (Mantel's stratified log-rank numerator)."""
    total = 0.0
    for s in np.unique(strata):
        m = strata == s
        t, e, g = times[m], events[m], group[m]
        for u in np.unique(t[e]):
            at_risk = t >= u
            d = (t == u) & e
            n, n1, dd = at_risk.sum(), (at_risk & g).sum(), d.sum()
            total += (d & g).sum() - dd * n1 / n
    return float(total)


def stratified_perm(times, events, group, strata, test_id):
    obs = logrank_stat(times, events, group, strata)
    blocks = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    combos = [list(itertools.combinations(range(len(b)), int(group[b].sum()))) for b in blocks]
    count = math.prod(len(c) for c in combos)
    stats_ = []
    if count <= 200_000:
        for choice in itertools.product(*combos):
            g = np.zeros_like(group)
            for b, ch in zip(blocks, choice):
                g[b[list(ch)]] = True
            stats_.append(logrank_stat(times, events, g, strata))
        null, exact = np.array(stats_), True
    else:
        rng = np.random.default_rng(np.random.SeedSequence(20261019, spawn_key=(zlib.crc32(test_id.encode('utf-8')),)))
        null = np.empty(N_MC)
        for i in range(N_MC):
            g = np.zeros_like(group)
            for b in blocks:
                g[rng.permutation(b)[:int(group[b].sum())]] = True
            null[i] = logrank_stat(times, events, g, strata)
        exact = False
    k = int(np.sum(np.abs(null) >= abs(obs) - 1e-12))
    p = k / len(null) if exact else (k + 1) / (len(null) + 1)
    return obs, p, exact


def mann_whitney_exact(x, y):
    """Two-sided exact permutation p of the rank-sum of x (unsolved runs carry rank ties at the top)."""
    allv = np.concatenate([x, y])
    ranks = st.rankdata(allv)
    obs = ranks[:len(x)].sum()
    mean = len(x) * (len(allv) + 1) / 2
    sums = np.array([ranks[list(c)].sum() for c in itertools.combinations(range(len(allv)), len(x))])
    return float(np.mean(np.abs(sums - mean) >= abs(obs - mean) - 1e-12))


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    out = SUPP / 'results'
    out.mkdir(exist_ok=True)
    rows = runs()
    write(out / 'x8_runs.csv', rows)
    amendment = SUPP / 'amendments' / 'x8-n.json'
    if not amendment.exists():
        print(f'{len(rows)} runs; no final n yet (pilot)')
        return 0
    n = int(json.loads(amendment.read_text(encoding='utf-8'))['replicates_per_arm_and_map'])
    final = [r for r in rows if r['replicate'] < n and r['verified'] == 'verified']
    if len(final) < n * len(ARMS) * len(MAPS):
        print(f'{len(final)} of {n * len(ARMS) * len(MAPS)} final runs verified; tests pending')
        return 0
    times = np.array([r['updates'] for r in final], float)
    events = np.array([r['event'] for r in final], bool)
    group = np.array([r['arm'] == 'head_only' for r in final], bool)
    strata = np.array([MAPS.index(r['map']) for r in final])
    obs, p, exact = stratified_perm(times, events, group, strata, 'x8.primary')
    tests = [dict(family='F13.X8.primary', test='readout-only vs joint from a neutral brain: updates to solve, stratified log-rank', n=len(final),
                  estimate=obs, exact=exact, p_value=p)]
    for m in MAPS:
        sub = [r for r in final if r['map'] == m]
        worst = max(r['updates'] for r in sub) + 1
        val = lambda r: r['updates'] if r['event'] else worst          # noqa: E731
        x = np.array([val(r) for r in sub if r['arm'] == 'head_only'], float)
        y = np.array([val(r) for r in sub if r['arm'] == 'joint'], float)
        tests.append(dict(family='F13.X8.map', test=f'{m}: updates to solve, readout-only vs joint (unsolved ranked last)', n=len(sub),
                          estimate=float(np.median(x) - np.median(y)), p_value=mann_whitney_exact(x, y)))
        a = sum(r['event'] for r in sub if r['arm'] == 'head_only')
        b = sum(r['event'] for r in sub if r['arm'] == 'joint')
        tests.append(dict(family='F13.X8.map', test=f'{m}: runs solved, readout-only vs joint', n=len(sub), estimate=(a - b) / n,
                          p_value=float(st.fisher_exact([[a, n - a], [b, n - b]])[1])))
    write(out / 'x8_tests.csv', tests)
    for t in tests:
        print(t['family'], '|', t['test'], '|', t['estimate'], '|', round(t['p_value'], 5))
    return 0


if __name__ == '__main__':
    sys.exit(main())
