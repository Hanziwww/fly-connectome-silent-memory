"""Analysis of X9 (replays for gaps G06 and G10; the G01 D7 controls) and X10 (Fig. 2 replicates) (local, after the
pulls). Written before any of their results existed; families declared in code/testplan.py.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-supplement-v2\\code\\analyze_x9_x10.py

- G06 (F13.X9.pairing): the 4 frozen pairings (before/after brain x before/after readout) must reproduce
  mechanisms-v1/interpretation/four-combination-behavior.csv (status and steps) on all 24 maps, else the tests are
  withheld; then 11 vs each of the 5 pairings involving mid: exact McNemar on solved over the 24 maps.
- G10 (descriptive): share of the 24 maps solved from the original start by curriculum-stage fraction (depth / final
  depth, in fifths).
- G01 (descriptive): status, updates and solved of each D7 control next to the frozen D1/D3/D5 controls.
- X10 (F13.X10.fig2, descriptive): per replicate set (r = 0 the frozen runs, r = 1-5 the replicates), Spearman rho of
  log(updates) with the certified optimal moves over the 24 maps; median and range of rho; rho of the per-map median.
Writes results/x9_pairings.csv, x9_stages.csv, x9_d7_controls.csv, x10_maps.csv and x9_x10_tests.csv.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
FROZEN = ROOT / 'studies' / 'brain-sokoban-v1'
FOUR = FROZEN / 'mechanisms-v1' / 'interpretation' / 'four-combination-behavior.csv'
CODE = {'before': '0', 'after': '1', 'mid': 'm'}


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def mcnemar(a, b):
    x = sum(p and not q for p, q in zip(a, b))
    y = sum(q and not p for p, q in zip(a, b))
    return (1.0 if x + y == 0 else float(st.binomtest(x, x + y, 0.5).pvalue)), x, y


def stage_log(model_id):
    """The frozen trainer's own original-start assessment of each saved stage checkpoint: the last history.jsonl row of
    that depth with stage_passed (training.py saves suffix-<depth>.pt right after it, with the same parameters)."""
    logged = {}
    with open(FROZEN / 'runs' / model_id / 'history.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('stage_passed'):
                logged[r['depth']] = (r['original_status'], r['original_steps'])
    return logged


def x9(tests, pending, out):
    folder = SUPP / 'runs' / 'x9-replays'
    if not (folder / 'done.json').exists():
        pending.append('X9 replays not finished')
        return
    frozen = {}
    with open(FOUR, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            frozen[(r['model'], r['combination'].zfill(2))] = (r['status'], int(r['steps']))
    prs, stages, solved = [], [], {}
    mismatch = checked = stage_mismatch = 0
    for p in sorted(folder.glob('D*.json')):
        if p.name == 'done.json':                    # Windows globs ignore case: 'D*' matches done.json
            continue
        rec = json.loads(p.read_text(encoding='utf-8'))
        for x in rec['pairings']:
            combo = CODE[x['brain']] + CODE[x['readout']]
            solved.setdefault(combo, {})[rec['map']] = x['status'] == 'solved'
            prs.append(dict(map=rec['map'], combination=combo, brain=x['brain'], readout=x['readout'], status=x['status'], steps=x['steps']))
            if (rec['model'], combo) in frozen:
                checked += 1
                mismatch += frozen[(rec['model'], combo)] != (x['status'], x['steps'])
        final = max((s['depth'] for s in rec['stages']), default=0)
        final_first = next(x['first_action'] for x in rec['pairings'] if x['brain'] == 'after' and x['readout'] == 'after')
        logged = stage_log(rec['model'])
        for s in rec['stages']:
            t = logged.get(s['depth'], (None, None))
            stage_mismatch += t != (s['status'], s['steps'])
            stages.append(dict(map=rec['map'], depth=s['depth'], final_depth=final, fraction=s['depth'] / final if final else None,
                               status=s['status'], steps=s['steps'], trainer_status=t[0], trainer_steps=t[1],
                               first_action=s['first_action'], final_first_action=final_first))
    write(out / 'x9_pairings.csv', prs)
    write(out / 'x9_stages.csv', stages)
    if checked != 96 or mismatch:
        pending.append(f'G06: frozen pairings reproduced {checked - mismatch} of {checked} (96 expected); tests withheld')
    else:
        maps = sorted(solved['11'])
        for combo in ('m1', '1m', 'mm', 'm0', '0m'):
            p, a, b = mcnemar([solved['11'][m] for m in maps], [solved[combo][m] for m in maps])
            tests.append(dict(family='F13.X9.pairing', test=f'new brain + new readout (11) vs pairing {combo}', n=len(maps), b=a, c=b,
                              estimate=(a - b) / len(maps), p_value=p))
    by_fifth = {}
    for s in stages:
        if s['fraction'] is not None:
            by_fifth.setdefault(min(4, int(s['fraction'] * 5)), []).append(s)
    for k, v in sorted(by_fifth.items()):
        tests.append(dict(family='F13.X9.descriptive', test=f'G10: original-start solved at stage fraction {k / 5:.1f}-{(k + 1) / 5:.1f}', n=len(v),
                          estimate=float(np.mean([s['status'] == 'solved' for s in v])), p_value=None))
    # Post hoc (F13.X9.posthoc): the share above is 0 by construction; graded measures, only if the replays reproduce
    # the trainer's own logged original-start outcome of every stage checkpoint.
    print(f'G10 check: stage replays reproduce the trainer log for {len(stages) - stage_mismatch} of {len(stages)}')
    if stage_mismatch:
        pending.append(f'G10: stage replays reproduce the trainer log for {len(stages) - stage_mismatch} of {len(stages)}; post hoc descriptives withheld')
        return
    for k, v in sorted(by_fifth.items()):
        band = f'{k / 5:.1f}-{(k + 1) / 5:.1f}'
        tests.append(dict(family='F13.X9.posthoc', test=f"G10: original-start first decision equals the final model's, stage fraction {band}",
                          n=len(v), estimate=float(np.mean([s['first_action'] == s['final_first_action'] for s in v])), p_value=None))
        tests.append(dict(family='F13.X9.posthoc', test=f'G10: median steps before the original-start rollout ends, stage fraction {band}',
                          n=len(v), estimate=float(np.median([s['steps'] for s in v])), p_value=None))


def d7_controls(pending, out):
    rows = []
    for folder in sorted(list((SUPP / 'runs').glob('controls-D7-T0-*')) + list((FROZEN / 'runs').glob('control-D[135]-*'))):
        if not (folder / 'result.json').exists():
            continue
        r = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        rows.append(dict(run=folder.name, source='X9 G01' if folder.parent.parent == SUPP else 'frozen', task=r.get('task') or r['config']['task'],
                         condition=r['config']['condition']['name'], status=r.get('status'), updates=r.get('updates'),
                         training_seconds=r.get('training_seconds'), final_solved=r.get('final_solved')))
    write(out / 'x9_d7_controls.csv', rows)
    if not any(r['source'] == 'X9 G01' for r in rows):
        pending.append('G01 D7 controls not run yet')


def x10(tests, pending, out):
    levels = {l['id']: l for name in ('levels.json', 'variants.json') for l in json.loads((FROZEN / 'tasks' / name).read_text(encoding='utf-8'))}
    maps = sorted(f'D{t}-T{k}' for t in range(8) for k in range(3))
    sets, rows = {}, []
    for m in maps:
        optimal = levels[m].get('optimal_steps') or len(levels[m]['reference_actions'])
        for r in range(6):
            folder = FROZEN / 'runs' / f'difficulty-{m}' if r == 0 else SUPP / 'runs' / f'difficulty-{m}-joint-r{r}'
            if not (folder / 'result.json').exists():
                continue
            res = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
            rows.append(dict(map=m, replicate=r, optimal_moves=optimal, status=res.get('status'), updates=res.get('updates'),
                             training_seconds=res.get('training_seconds'), final_solved=res.get('final_solved')))
            if res.get('status') in ('learned_solved', 'already_solved'):
                sets.setdefault(r, {})[m] = (optimal, res['updates'])
    write(out / 'x10_maps.csv', rows)
    complete = [r for r in range(6) if len(sets.get(r, {})) == len(maps)]
    if len(complete) < 6:
        pending.append(f'X10: complete replicate sets {complete} (all 24 maps learned); 0-5 expected')
    rhos = []
    for r in complete:
        x = [sets[r][m][0] for m in maps]
        y = [math.log(sets[r][m][1]) for m in maps]
        rho, p = st.spearmanr(x, y)
        rhos.append(rho)
        tests.append(dict(family='F13.X10.fig2', test=f'replicate set {r}: Spearman rho of log(updates) with optimal moves', n=len(maps),
                          estimate=float(rho), p_value=float(p)))
    if len(complete) == 6:
        med = [float(np.median([sets[r][m][1] for r in complete])) for m in maps]
        rho, p = st.spearmanr([sets[0][m][0] for m in maps], np.log(med))
        tests.append(dict(family='F13.X10.fig2', test='per-map median updates over the 6 sets: Spearman rho with optimal moves', n=len(maps),
                          estimate=float(rho), p_value=float(p), rho_median=float(np.median(rhos)), rho_min=float(min(rhos)), rho_max=float(max(rhos))))


def main():
    out = SUPP / 'results'
    tests, pending = [], []
    x9(tests, pending, out)
    d7_controls(pending, out)
    x10(tests, pending, out)
    write(out / 'x9_x10_tests.csv', [t for t in tests if t.get('p_value') is not None] + [t for t in tests if t.get('p_value') is None])
    for t in tests:
        print(t['family'], '|', t['test'], '|', t.get('estimate'), '|', t.get('p_value'))
    for p in pending:
        print('pending:', p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
