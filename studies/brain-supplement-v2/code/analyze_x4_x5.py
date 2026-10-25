"""Analysis of X4 (scrambled-shift null) and X5 (swaps at the trained timings) (local, after the pulls). Written before
any of their results existed; families declared in code/testplan.py.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-supplement-v2\\code\\analyze_x4_x5.py

Margin of a history under a condition: first-decision logit of the case's reference (correct) first action minus the
strongest other legal action, as in the paper (Fig. 4f; mechanisms-supplement-v1 supp_analysis.readout_margin).
Units: M0 (readout-only), M0joint, M0vis.hidden, M0vis.visible (shared readout), S1, S2. A history is a case x timing.
X4:
- reproduction: the frozen scrambled draw (k = 0) at doses 0.25, 0.5 and 1 must give the frozen statuses and margins
  (mechanisms-supplement-v1/analysis/condition-outcomes.csv, read window, continuous state); reported, and the tests
  are withheld if it fails;
- F13.X4.content: D = cue effect - mean of the 20 scrambled effects at dose 1, one-sided sign-flip (D < 0) per unit;
- F13.X4.dose: exact Page trend of D over the 5 doses (decreasing), histories as blocks, per unit;
- F13.X4.rank: rank of the unit-mean cue effect among the 20 unit-mean scrambled effects, p = (k + 1) / 21 (descriptive).
X5:
- F13.X5.swap: baseline vs opposite-cue swap, histories solved, exact McNemar per unit;
- F13.X5.visibility: shared readout, disrupted (baseline solved, swap unsolved) hidden vs visible on pairs matched by cue
  and timing, exact McNemar;
- F13.X5.cross: S2, baseline vs each cross-orientation swap, and same-cue vs opposite-cue cross swap (exact McNemar).
Writes results/x4_x5_tests.csv, x4_histories.csv, x4_units.csv, x5_histories.csv.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
FROZEN_OUTCOMES = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-supplement-v1' / 'analysis' / 'condition-outcomes.csv'
TASKS = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1' / 'inputs' / 'memory-tasks.json'
SUPP_MODELS = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-supplement-v1' / 'models.json'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUPP / 'code'))
from studykit import stats_ext as sx  # noqa: E402
from analyze_campaign import page_exact  # noqa: E402

MODEL_UNITS = {'remote-dynamic-M0-head_only': 'M0', 'remote-dynamic-M0-joint': 'M0joint', 'supp-M0-visibility': 'M0vis',
               'supp-S1-ring-brain': 'S1', 'supp-S2-context-brain': 'S2'}
DOSES = (0.125, 0.25, 0.5, 0.75, 1.0)


def truth(x):
    return str(x).strip().lower() in ('1', 'true')


def unit(model, visible):
    u = MODEL_UNITS[model]
    return f'{u}.{"visible" if visible else "hidden"}' if u == 'M0vis' else u


def reference_actions():
    """{(model, case id): the case's reference first action} from the frozen task definitions."""
    tasks = {t['id']: t for t in json.loads(TASKS.read_text(encoding='utf-8'))}
    out = {}
    for m in json.loads(SUPP_MODELS.read_text(encoding='utf-8')):
        for c in tasks[m['task']]['cases']:
            out[(m['id'], c['id'])] = int(c['level']['reference_actions'][0])
    return out


def margin(logits, action, legal):
    if logits is None or action is None:
        return np.nan
    others = [logits[a] for a in range(len(logits)) if legal[a] and a != action]
    return logits[action] - max(others) if others else np.nan


def mcnemar(b, c):
    return 1.0 if b + c == 0 else float(st.binomtest(b, b + c, 0.5).pvalue)


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def x4(tests, pending, out):
    folder = SUPP / 'runs' / 'x4-scrambles'
    if not (folder / 'done.json').exists():
        pending.append('X4 not finished')
        return
    hist = {}                       # (unit, case, delay, prefix) -> {('cue'|draw k, dose): (effect, solved)} + baseline
    reference = reference_actions()
    for p in sorted(folder.glob('*-d*-p*.json')):
        rec = json.loads(p.read_text(encoding='utf-8'))
        conds = rec['conditions']
        base = conds[0]['rows']
        for i, b in enumerate(base):
            key = (unit(rec['model'], b['visible']), b['case'], rec['delay'], rec['prefix'])
            ref = reference[(rec['model'], b['case'])]
            m0 = margin(b['first_logits'], ref, b['first_legal'])
            h = hist.setdefault(key, dict(base_margin=m0, base_solved=b['status'] == 'solved', model=rec['model'], cond={}))
            for c in conds[1:]:
                r = c['rows'][i]
                if r['case'] != b['case']:
                    raise RuntimeError('row order differs between conditions')
                label = 'cue' if c['kind'] == 'cue' else c['draw']
                h['cond'][(label, c['dose'])] = (margin(r['first_logits'], ref, r['first_legal']) - m0, r['status'] == 'solved')
    # reproduction of the frozen scrambled draw
    frozen = {}
    with open(FROZEN_OUTCOMES, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['module'] == 'selective' and r['kind'] == 'scrambled' and r['window'] == 'read' and r['part'] == 'continuous':
                frozen[(unit(r['model'], truth(r['visible'])), r['case'], int(float(r['delay'])), int(float(r['prefix'])), float(r['fraction']))] = (
                    truth(r['solved']), float(r['margin_effect']) if r['margin_effect'] not in ('', 'nan') else np.nan)
    checked = mism = 0
    for (u, case, d, pfx), h in hist.items():
        for dose in (0.25, 0.5, 1.0):
            fz = frozen.get((u, case, d, pfx, dose))
            if fz is None:
                continue
            eff, solved = h['cond'][(0, dose)]
            checked += 1
            same_margin = (np.isnan(eff) and np.isnan(fz[1])) or abs(eff - fz[1]) < 1e-3
            mism += int(solved != fz[0] or not same_margin)
    reproduced = checked > 0 and mism == 0
    rows, urows = [], []
    for key, h in sorted(hist.items()):
        for dose in DOSES:
            null = np.array([h['cond'][(k, dose)][0] for k in range(1, 21)])
            cue_eff, cue_solved = h['cond'][('cue', dose)]
            rows.append(dict(unit=key[0], case=key[1], delay_ms=key[2], prefix_windows=key[3], dose=dose, baseline_margin=h['base_margin'],
                             baseline_solved=h['base_solved'], cue_effect=cue_eff, cue_solved=cue_solved, null_mean=float(np.nanmean(null)),
                             null_min=float(np.nanmin(null)), null_max=float(np.nanmax(null)),
                             null_solved_share=float(np.mean([h['cond'][(k, dose)][1] for k in range(1, 21)])), D=cue_eff - float(np.nanmean(null))))
    write(out / 'x4_histories.csv', rows)
    if not reproduced:
        pending.append(f'X4 frozen-draw reproduction failed ({mism} of {checked} histories differ): tests withheld')
        return
    for u in sorted({r['unit'] for r in rows}):
        sub = [r for r in rows if r['unit'] == u]
        D1 = np.array([r['D'] for r in sub if r['dose'] == 1.0 and not np.isnan(r['D'])])
        sf = sx.sign_flip(D1, alternative='less', ci=False)
        tests.append(dict(family='F13.X4.content', test=f'{u}: cue minus mean scrambled margin effect at dose 1 < 0', n=len(D1),
                          estimate=float(D1.mean()), p_value=sf['p_value']))
        keys = sorted({(r['case'], r['delay_ms'], r['prefix_windows']) for r in sub})
        mat = np.array([[next(r['D'] for r in sub if (r['case'], r['delay_ms'], r['prefix_windows']) == k and r['dose'] == a) for a in DOSES] for k in keys])
        mat = mat[~np.isnan(mat).any(1)]
        L, p = page_exact(mat[:, ::-1])
        tests.append(dict(family='F13.X4.dose', test=f'{u}: cue minus scrambled margin effect decreasing over doses 0.125-1', n=len(mat), estimate=None,
                          value=L, p_value=p))
        for a in DOSES:
            cue_mean = float(np.nanmean([r['cue_effect'] for r in sub if r['dose'] == a]))
            hk = [h for k, h in hist.items() if k[0] == u]
            null_means = np.array([np.nanmean([h['cond'][(k, a)][0] for h in hk]) for k in range(1, 21)])
            k_le = int((null_means <= cue_mean).sum())
            tests.append(dict(family='F13.X4.rank', test=f'{u}: rank of the cue effect among 20 scrambles at dose {a:g}', n=20,
                              estimate=cue_mean, value=k_le, p_value=(k_le + 1) / 21))
            urows.append(dict(unit=u, dose=a, cue_mean_effect=cue_mean, null_mean_effect=float(null_means.mean()),
                              null_min=float(null_means.min()), null_max=float(null_means.max()), scrambles_at_or_below_cue=k_le,
                              cue_disrupted=sum(1 for r in sub if r['dose'] == a and r['baseline_solved'] and not r['cue_solved']),
                              scrambled_disrupted_mean=float(sum(1.0 - r['null_solved_share'] for r in sub if r['dose'] == a and r['baseline_solved'])),
                              baseline_solved=sum(1 for r in sub if r['dose'] == a and r['baseline_solved'])))
    urows.append(dict(unit='reproduction', dose=None, frozen_histories_checked=checked, frozen_histories_differing=mism))
    write(out / 'x4_units.csv', urows)


def x5(tests, pending, out):
    folder = SUPP / 'runs' / 'x5-swaps'
    if not (folder / 'done.json').exists():
        pending.append('X5 not finished')
        return
    rows = []
    for p in sorted(folder.glob('*-d*-p*.json')):
        rec = json.loads(p.read_text(encoding='utf-8'))
        conds = {c['kind']: c['rows'] for c in rec['conditions']}
        for i, b in enumerate(conds['baseline']):
            row = dict(unit=unit(rec['model'], b['visible']), case=b['case'], cue=b['cue'], context=b['context'], delay_ms=rec['delay'],
                       prefix_windows=rec['prefix'], baseline_solved=b['status'] == 'solved')
            for kind, rr in conds.items():
                if kind != 'baseline':
                    row[f'{kind}_solved'] = rr[i]['status'] == 'solved'
            rows.append(row)
    write(out / 'x5_histories.csv', rows)
    for u in sorted({r['unit'] for r in rows}):
        sub = [r for r in rows if r['unit'] == u]
        b = sum(r['baseline_solved'] and not r['opposite_cue_solved'] for r in sub)
        c = sum(r['opposite_cue_solved'] and not r['baseline_solved'] for r in sub)
        tests.append(dict(family='F13.X5.swap', test=f'{u}: opposite-cue swap vs baseline (solved)', n=len(sub), b=b, c=c,
                          estimate=(c - b) / len(sub), p_value=mcnemar(b, c)))
    hid = {(r['cue'], r['delay_ms'], r['prefix_windows']): r for r in rows if r['unit'] == 'M0vis.hidden'}
    vis = {(r['cue'], r['delay_ms'], r['prefix_windows']): r for r in rows if r['unit'] == 'M0vis.visible'}
    keys = sorted(set(hid) & set(vis))
    dis = lambda r: r['baseline_solved'] and not r['opposite_cue_solved']        # noqa: E731
    b = sum(dis(hid[k]) and not dis(vis[k]) for k in keys)
    c = sum(dis(vis[k]) and not dis(hid[k]) for k in keys)
    tests.append(dict(family='F13.X5.visibility', test='shared readout: disrupted by the opposite-cue swap, hidden vs visible goal', n=len(keys),
                      b=b, c=c, estimate=(b - c) / len(keys) if keys else None, p_value=mcnemar(b, c)))
    s2 = [r for r in rows if r['unit'] == 'S2']
    for kind in ('cross_same_cue', 'cross_opposite_cue'):
        b = sum(r['baseline_solved'] and not r[f'{kind}_solved'] for r in s2)
        c = sum(r[f'{kind}_solved'] and not r['baseline_solved'] for r in s2)
        tests.append(dict(family='F13.X5.cross', test=f'S2: {kind} swap vs baseline (solved)', n=len(s2), b=b, c=c,
                          estimate=(c - b) / len(s2), p_value=mcnemar(b, c)))
    b = sum(r['cross_same_cue_solved'] and not r['cross_opposite_cue_solved'] for r in s2)
    c = sum(r['cross_opposite_cue_solved'] and not r['cross_same_cue_solved'] for r in s2)
    tests.append(dict(family='F13.X5.cross', test='S2: same-cue vs opposite-cue cross-orientation swap (solved)', n=len(s2), b=b, c=c,
                      estimate=(b - c) / len(s2), p_value=mcnemar(b, c)))


def main():
    out = SUPP / 'results'
    out.mkdir(exist_ok=True)
    tests, pending = [], []
    x4(tests, pending, out)
    x5(tests, pending, out)
    write(out / 'x4_x5_tests.csv', tests)
    for t in tests:
        print(t['family'], '|', t['test'], '|', t.get('estimate'), '|', round(t['p_value'], 5))
    for p in pending:
        print('pending:', p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
