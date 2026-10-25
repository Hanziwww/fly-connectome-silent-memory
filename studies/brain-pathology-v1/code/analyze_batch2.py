"""Analysis of the second GPU batch (local, after the pulls): G3a dependency map, late rescue, regime C; and the post hoc
equivalence of the regime A nulls. Written before any of the batch's results existed.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\analyze_batch2.py

Unit and tests (families declared in code/testplan.py; corrections applied by collect_tests.py once a family is complete):
- History pairs: a trained-timing history (model, case, delay, prefix) solved under one condition and not the other.
- F12.regimeA.equivalence (post hoc) and F12.dependency.equivalence: Tango score TOST (studykit.stats_ext.
  tost_paired_binary, margin 5 percentage points) of the condition vs healthy over the 126 trained histories of the five
  memory models.
- F12.dependency.null: per model, D = (change in solved after the targeted removal) - (median change over the 10 matched
  sets), changes as shares of the model's trained histories; exact one-sided sign-flip over the five models (D > 0:
  the targeted removal is less harmful). Descriptive: dependency index D / IQR of the null changes.
- F12.rescue: trained histories solved with excitability restored vs as trained under the lesion (the run's own
  verification), paired by history over replicates 1-3; exact two-sided McNemar per learner and dose.
- F12.regimeC: recovery fraction per job; when the lesion leaves all 18 trained histories solved there is nothing to
  recover and the fraction is set to 1. Exact Page trend (decreasing in dose; replicates as blocks) per retraining mode,
  and the exact two-sided sign-flip of joint minus readout-only over the 12 dose x replicate pairs.
Outputs (studies/brain-pathology-v1/results/): batch2_tests.csv, dependency_conditions.csv, dependency_models.csv,
rescue.csv, regime_c.csv; a part whose runs are incomplete is reported as pending and not tested.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'studies' / 'brain-supplement-v2' / 'code'))
from studykit import stats_ext as sx  # noqa: E402
from analyze_campaign import page_exact  # noqa: E402

MODELS = ['remote-dynamic-M0-head_only', 'remote-dynamic-M0-joint', 'supp-M0-visibility', 'supp-S1-ring-brain', 'supp-S2-context-brain']
TRAINED = {(d, p) for d in (0, 40, 160) for p in (0, 1, 4)}
MARGIN = 0.05
DOSES = (0.5, 1.0, 1.5, 2.0)


def trained_outcomes(res):
    """{(model, case, delay, prefix): solved} over the trained timings of a regime A or dependency record."""
    out = {}
    for m, v in res['memory'].items():
        for r in v['rows']:
            if (r['delay_ms'], r['prefix_windows']) in TRAINED:
                out[(m, r['case'], r['delay_ms'], r['prefix_windows'])] = r['status'] == 'solved'
    return out


def discordant(a, b):
    if set(a) != set(b):
        raise RuntimeError('history sets differ')
    k = sorted(a)
    return sum(a[x] and not b[x] for x in k), sum(b[x] and not a[x] for x in k), len(k)


def mcnemar(b, c):
    return 1.0 if b + c == 0 else float(st.binomtest(b, b + c, 0.5).pvalue)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def regime_a_equivalence(tests, pending):
    folder = RUNS / 'phase3-regime-a'
    if not (folder / 'done.json').exists():
        pending.append('regime A not pulled')
        return
    healthy = trained_outcomes(load(folder / 'healthy.json'))
    for case in ('C04', 'C13', 'C16'):
        for dose in (1, 2):
            b, c, n = discordant(trained_outcomes(load(folder / f'{case}-dose{dose}.json')), healthy)
            r = sx.tost_paired_binary(b, c, n, MARGIN)
            tests.append(dict(family='F12.regimeA.equivalence', test=f'{case} dose {dose} vs healthy: trained histories solved', n=n,
                              estimate=r['effect'], ci90_low=r['effect_ci_low'], ci90_high=r['effect_ci_high'], b=b, c=c,
                              conclusion=r['conclusion'], p_value=r['p_value']))


def per_model(outcomes):
    out = {}
    for (m, *_), v in outcomes.items():
        s = out.setdefault(m, [0, 0])
        s[0] += int(v)
        s[1] += 1
    return out


def dependency(tests, pending, out):
    folder = RUNS / 'phase3-dependency'
    if not (folder / 'done.json').exists():
        pending.append('dependency map not finished')
        return
    spec = load(STUDY / 'tasks' / 'phase3-dependency-conditions.json')
    recs = {c['name']: load(folder / f"{c['name']}.json") for c in spec['conditions']}
    outcomes = {k: trained_outcomes(v) for k, v in recs.items()}
    healthy = outcomes['healthy']
    hm = per_model(healthy)
    first = {k: {(m, r['case'], r['delay_ms'], r['prefix_windows']): r['first_logits'] for m, v in recs[k]['memory'].items() for r in v['rows']}
             for k in ('healthy', 'monoamine-1')}
    rows, mrows = [], []
    for c in spec['conditions']:
        k = c['name']
        b, cc, n = discordant(outcomes[k], healthy)
        pm = per_model(outcomes[k])
        rows.append(dict(condition=k, population=c['population'], level=c['level'], kind=c['kind'], cells=len(c['cells']),
                         solved=sum(outcomes[k].values()), histories=n, only_lesion=b, only_healthy=cc,
                         gf_spikes=recs[k]['loom']['gf_spikes'], task_mean_hz=recs[k]['activity']['task_mean_hz'],
                         identical_first_logits=first['healthy'] == first['monoamine-1'] if k == 'monoamine-1' else None,
                         **{f'solved_{m}': pm[m][0] for m in MODELS}))
    for pop in ('kc', 'alln_gaba'):
        for level in (0.5, 1.0):
            name = f'{pop}-{level:g}'
            b, c, n = discordant(outcomes[name], healthy)
            r = sx.tost_paired_binary(b, c, n, MARGIN)
            tests.append(dict(family='F12.dependency.equivalence', test=f'{name} removal vs healthy: trained histories solved', n=n,
                              estimate=r['effect'], ci90_low=r['effect_ci_low'], ci90_high=r['effect_ci_high'], b=b, c=c,
                              conclusion=r['conclusion'], p_value=r['p_value']))
            D = []
            for m in MODELS:
                n_m = hm[m][1]
                dt = (per_model(outcomes[name])[m][0] - hm[m][0]) / n_m
                dn = np.array([(per_model(outcomes[f'{name}-null{i}'])[m][0] - hm[m][0]) / n_m for i in range(10)])
                d = dt - float(np.median(dn))
                iqr = float(np.subtract(*np.quantile(dn, [0.75, 0.25])))
                D.append(d)
                mrows.append(dict(condition=name, model=m, histories=n_m, change_targeted=dt, null_median=float(np.median(dn)),
                                  null_min=float(dn.min()), null_max=float(dn.max()), D=d, dependency_index=d / iqr if iqr > 0 else None))
            sf = sx.sign_flip(np.array(D), alternative='greater', ci=False)
            tests.append(dict(family='F12.dependency.null', test=f'{name}: targeted removal less harmful than matched random removal', n=len(D),
                              estimate=float(np.mean(D)), p_value=sf['p_value']))
    write(out / 'dependency_conditions.csv', rows)
    write(out / 'dependency_models.csv', mrows)


def rescue(tests, pending, out):
    folder = RUNS / 'rescue'
    per_host = all((folder / f'done-{x}.json').exists() for x in ('M0-head_only', 'M0-joint'))   # each host rescued its own learner
    if not ((folder / 'done.json').exists() or per_host):
        pending.append('rescue not finished')
        return
    rows = []
    for learner in ('M0-head_only', 'M0-joint'):
        for dose in DOSES:
            b = c = n = 0
            restored_all = trained_all = 0
            for rep in (1, 2, 3):
                run = f'b-C01-dose{dose:g}-{learner}-r{rep}'
                res = load(folder / f'{run}.json')['results']['rollouts']
                ver = load(RUNS / run / 'verification.json')['results']['rollouts']
                key = lambda r: (r['cue'], r['delay_ms'], r['prefix_windows'])          # noqa: E731
                a = {key(r): r['status'] == 'solved' for r in res if (r['delay_ms'], r['prefix_windows']) in TRAINED}
                t = {key(r): r['status'] == 'solved' for r in ver if (r['delay_ms'], r['prefix_windows']) in TRAINED}
                bb, cc, nn = discordant(a, t)
                b, c, n = b + bb, c + cc, n + nn
                restored_all += sum(r['status'] == 'solved' for r in res)
                trained_all += sum(r['status'] == 'solved' for r in ver)
                rows.append(dict(run=run, learner=learner, dose=dose, replicate=rep, trained_solved_restored=sum(a.values()),
                                 trained_solved_as_trained=sum(t.values()), all_solved_restored=sum(r['status'] == 'solved' for r in res),
                                 all_solved_as_trained=sum(r['status'] == 'solved' for r in ver)))
            tests.append(dict(family='F12.rescue', test=f'{learner} dose {dose:g}: restored vs as trained (trained histories solved)', n=n,
                              estimate=(b - c) / n, b=b, c=c, p_value=mcnemar(b, c)))
    controls = sorted(folder.glob('*-lesion-control.json'))
    for p in controls:
        rows.append(dict(run=p.stem, learner=None, dose=None, replicate=None, reproduces_verification=load(p)['reproduces_verification']))
    write(out / 'rescue.csv', rows)


def regime_c(tests, pending, out):
    folder = RUNS / 'regime-c'
    if not (folder / 'done.json').exists():
        pending.append('regime C not finished')
        return
    rows, R = [], {}
    for rep in (1, 2, 3):
        for dose in DOSES:
            for mode in ('head_only', 'joint'):
                rec = load(folder / f'C01-dose{dose:g}-{mode}-r{rep}.json')
                a = rec['assessments']
                frac = rec['recovery_fraction'] if rec['recovery_fraction'] is not None else 1.0
                R[(mode, dose, rep)] = frac
                rows.append(dict(job=rec['job'], replicate=rep, dose=dose, mode=mode, solved_0=a['0']['solved'], solved_50=a['50']['solved'],
                                 solved_100=a['100']['solved'], total=a['0']['total'], recovery_fraction=frac,
                                 nothing_to_recover=rec['recovery_fraction'] is None, seconds=rec['seconds']))
    for mode in ('head_only', 'joint'):
        mat = np.array([[R[(mode, d, r)] for d in DOSES] for r in (1, 2, 3)])
        L, p = page_exact(mat[:, ::-1])                       # decreasing in dose
        tests.append(dict(family='F12.regimeC', test=f'recovery fraction decreasing in C01 dose ({mode} retraining)', n=3, estimate=None,
                          value=L, matrix=json.dumps(mat.tolist()), p_value=p))
    diff = np.array([R[('joint', d, r)] - R[('head_only', d, r)] for d in DOSES for r in (1, 2, 3)])
    sf = sx.sign_flip(diff, alternative='two-sided', ci=False)
    tests.append(dict(family='F12.regimeC', test='recovery fraction: joint minus readout-only retraining', n=len(diff),
                      estimate=float(diff.mean()), p_value=sf['p_value']))
    write(out / 'regime_c.csv', rows)


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    out = STUDY / 'results'
    out.mkdir(exist_ok=True)
    tests, pending = [], []
    regime_a_equivalence(tests, pending)
    dependency(tests, pending, out)
    rescue(tests, pending, out)
    regime_c(tests, pending, out)
    write(out / 'batch2_tests.csv', tests)
    for t in tests:
        print(t['family'], '|', t['test'], '|', t.get('estimate'), '|', round(t['p_value'], 5), '|', t.get('conclusion', ''))
    for p in pending:
        print('pending:', p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
