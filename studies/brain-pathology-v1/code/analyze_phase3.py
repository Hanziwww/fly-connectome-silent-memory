"""Phase 3 analysis (local, after the pull): regime A (lesion after training) and the olfactory memory curves.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\analyze_phase3.py

Regime A, per condition and memory model, against the healthy brain on the same histories:
- histories solved at the 9 trained and the 26 untrained timings; cue-pair classes (both, one, neither solved) and net
  memory use (both - neither) / pairs;
- exact McNemar on solved, paired by history (trained timings), and on first-push correctness (all timings);
- C01 dose-response: exact Page trend of trained-timing histories solved over doses 0, 0.5, 1, 1.5, 2 with the five
  memory models as blocks;
- movement: giant-fibre spikes and latency under the looming ramp; activity: mean rate under the M0 task input.
Olfactory curves, per condition, odour pair, delay and read-out set: decoding accuracy and d'; the change from healthy is
tested by exact McNemar on the held-out correctness of the same 96 trials (identical Poisson noise).
Families (declared in code/testplan.py before this runs):
- F12.C01.primary (Holm over its three tests): the memory dose trend is computed here; the two learning dose trends come
  from the training campaign (studies/brain-supplement-v2/code/analyze_campaign.py). The Holm adjustment of this family is
  therefore applied only by collect_tests.py, once all three p-values exist; here p_adjusted stays empty.
- F12.olfactory.primary (Holm, 3 tests): targeted vs size-matched random lesion at dose 1, MBON read-out, similar odour
  pair, 160-ms delay, paired by trial (exact McNemar).
- F12.regimeA.solved and F12.olfactory (BH, exploratory).
Writes results/phase3_regime_a.csv, results/phase3_olfactory.csv, results/phase3_tests.csv.
"""
import csv
import json
import math
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy import stats as st

STUDY = Path(__file__).resolve().parents[1]
RA = STUDY / 'runs' / 'phase3-regime-a'
OL = STUDY / 'runs' / 'phase3-olfactory'
RES = STUDY / 'results'
TRAINED = {(d, p) for d in (0, 40, 160) for p in (0, 1, 4)}


def mcnemar(a, b):
    """Exact two-sided McNemar for paired booleans: a = healthy, b = lesioned. Returns (b_only, c_only, p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n10 + n01
    p = 1.0 if n == 0 else min(1.0, 2 * st.binom.cdf(min(n10, n01), n, 0.5))
    return n10, n01, p


def page_exact(data):
    """Exact Page test for an increasing trend over k ordered treatments (columns) in n blocks (rows), one-sided.
    The null distribution of L = sum_j j R_ij is the convolution over blocks of each block's distribution over the k!
    permutations of its ranks (ties keep their midranks), held as value -> count tables."""
    data = np.asarray(data, float)
    n, k = data.shape
    w = np.arange(1, k + 1)
    ranks = np.vstack([st.rankdata(r) for r in data])
    L = float(np.sum(ranks * w))
    dist = {0.0: 1}
    for r in ranks:
        block = {}
        for perm in permutations(range(k)):
            v = round(float(np.dot(r[list(perm)], w)), 6)
            block[v] = block.get(v, 0) + 1
        new = {}
        for x, cx in dist.items():
            for v, cv in block.items():
                key = round(x + v, 6)
                new[key] = new.get(key, 0) + cx * cv
        dist = new
    total = sum(dist.values())
    p = sum(c for v, c in dist.items() if v >= L - 1e-6) / total
    return L, float(p)


def bh(p):
    p = np.asarray(p, float)
    if len(p) == 0:
        return p
    return st.false_discovery_control(p, method='bh')


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind='mergesort')
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def pair_classes(rows):
    by = {}
    for r in rows:
        by.setdefault((r['context'], r['visible'], r['delay_ms'], r['prefix_windows']), []).append(r['status'] == 'solved')
    out = {'both': 0, 'one': 0, 'none': 0}
    for v in by.values():
        k = sum(v)
        out['both' if k == len(v) else 'none' if k == 0 else 'one'] += 1
    return out, len(by)


def regime_a():
    conds = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in sorted(RA.glob('*.json')) if p.stem not in ('conditions', 'done')}
    healthy = conds['healthy']
    rows, tests = [], []
    for name, c in conds.items():
        for mid, m in c['memory'].items():
            h = {(r['case'], r['delay_ms'], r['prefix_windows']): r for r in healthy['memory'][mid]['rows']}
            cur = m['rows']
            for split in ('train', 'heldout'):
                sel = [r for r in cur if r['history_split'] == split]
                classes, pairs = pair_classes(sel)
                rows.append(dict(condition=name, case=c['case'], dose=c['dose'], model=mid, split=split, histories=len(sel),
                                 solved=sum(r['status'] == 'solved' for r in sel), pairs=pairs, both=classes['both'], one=classes['one'],
                                 none=classes['none'], net_memory_use=(classes['both'] - classes['none']) / pairs if pairs else None,
                                 first_push_correct=sum(bool(r['first_push_correct']) for r in sel),
                                 gf_spikes=c['loom']['gf_spikes'], gf_first_ms=c['loom']['gf_first_spike_ms'],
                                 task_mean_hz=c['activity']['task_mean_hz']))
            if name == 'healthy':
                continue
            tr = [r for r in cur if r['history_split'] == 'train']
            a = [h[(r['case'], r['delay_ms'], r['prefix_windows'])]['status'] == 'solved' for r in tr]
            b = [r['status'] == 'solved' for r in tr]
            n10, n01, p = mcnemar(a, b)
            tests.append(dict(family='F12.regimeA.solved', test=f'{name}|{mid}', statistic='solved healthy only vs lesioned only (trained histories)',
                              a_only=n10, b_only=n01, n=len(tr), p_value=p))
    # C01 dose-response (Page, decreasing trend in trained-timing solved over doses), five models as blocks
    doses = [('healthy', 0.0)] + [(f'C01-dose{d:g}', d) for d in (0.5, 1.0, 1.5, 2.0)]
    if all(n in conds for n, _ in doses):
        mids = list(healthy['memory'])
        mat = [[sum(r['status'] == 'solved' for r in conds[n]['memory'][mid]['rows'] if r['history_split'] == 'train') for n, _ in doses] for mid in mids]
        L, p = page_exact(np.asarray(mat)[:, ::-1])
        tests.append(dict(family='F12.C01.primary', test='memory: C01 dose trend (trained-timing histories solved, five models as blocks)',
                          statistic='Page L (decreasing in dose)',
                          a_only=None, b_only=None, n=len(mids), p_value=p, value=L, matrix=json.dumps(mat)))
        gf = [conds[n]['loom']['gf_spikes'] for n, _ in doses]
        tests.append(dict(family='descriptive', test='C01 giant-fibre spikes by dose', statistic='GF spikes', a_only=None, b_only=None,
                          n=len(gf), p_value=None, value=None, matrix=json.dumps(gf)))
    return rows, tests


def olfactory():
    rows, tests = [], []
    files = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in sorted(OL.glob('*.json')) if p.stem != 'done'}
    if 'healthy' not in files:
        return rows, tests
    for name, c in files.items():
        for pair, items in c['pairs'].items():
            for it in items:
                for k in ('readout512', 'mbon', 'kc', 'pn'):
                    rows.append(dict(condition=name, case=c['case'], dose=c['dose'], pair=pair, delay_ms=it['delay_ms'], readout=k,
                                     accuracy=it[k]['accuracy'], dprime=it[k]['dprime'], features=it[k]['informative_features']))
                    if name == 'healthy':
                        continue
                    h = next(x for x in files['healthy']['pairs'][pair] if x['delay_ms'] == it['delay_ms'])[k]
                    if h.get('correct') is None or it[k].get('correct') is None:
                        continue
                    n10, n01, p = mcnemar(h['correct'], it[k]['correct'])
                    tests.append(dict(family='F12.olfactory', test=f'{name}|{pair}|d{it["delay_ms"]}|{k}',
                                      statistic='trials decoded correctly only when healthy vs only when lesioned', a_only=n10, b_only=n01,
                                      n=len(h['correct']), p_value=p))
    # primary: targeted vs size-matched random lesion at dose 1 (MBON read-out, similar pair, 160-ms delay)
    for case in ('C04', 'C13', 'C16'):
        tg, rd = files.get(f'{case}-dose1'), files.get(f'{case}-random')
        if tg is None or rd is None:
            continue
        a = next(x for x in tg['pairs']['apple-grape'] if x['delay_ms'] == 160)['mbon']
        b = next(x for x in rd['pairs']['apple-grape'] if x['delay_ms'] == 160)['mbon']
        if a.get('correct') is None or b.get('correct') is None:
            continue
        n10, n01, p = mcnemar(b['correct'], a['correct'])           # a_only: correct only after the random lesion
        tests.append(dict(family='F12.olfactory.primary', test=f'{case}: targeted vs random lesion (MBON, apple-grape, 160 ms)',
                          statistic='trials correct only after the random lesion vs only after the targeted lesion', a_only=n10,
                          b_only=n01, n=len(a['correct']), p_value=p))
    return rows, tests


def write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    RES.mkdir(exist_ok=True)
    ra, ta = regime_a()
    ol, to = olfactory()
    tests = ta + to
    for fam, adj in (('F12.regimeA.solved', bh), ('F12.olfactory', bh), ('F12.olfactory.primary', holm)):
        idx = [i for i, t in enumerate(tests) if t['family'] == fam and t['p_value'] is not None]
        for i, q in zip(idx, adj([tests[i]['p_value'] for i in idx])):
            tests[i]['p_adjusted'] = float(q)
    write(RES / 'phase3_regime_a.csv', ra)
    write(RES / 'phase3_olfactory.csv', ol)
    write(RES / 'phase3_tests.csv', tests)
    print(f'regime A rows {len(ra)}, olfactory rows {len(ol)}, tests {len(tests)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
