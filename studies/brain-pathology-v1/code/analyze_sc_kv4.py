"""T-SC4 analysis (local, after the hash-checked pull of runs/sc-kv4): Kv4-weighted vs uniform C01. Written before any
T-SC4 result existed; families F14.SC.kv4, .equivalence, .descriptive (tasks/sc-panels-v1.json, amendments 0005-0006).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\analyze_sc_kv4.py

- Identity check: check-uniform-dose0.5 (w = 1 through the weighted code path) must reproduce
  runs/phase3-regime-a/C01-dose0.5.json (every memory row's status, steps, first action, first logits and
  first-push correctness, the activity and the looming response). If it fails, no test is written.
- Tests (as analyze_batch2.py): history pairs over the 126 trained-timing histories of the five memory models,
  kv4-<imputation>-dose<d> vs runs/phase3-regime-a/C01-dose<d>.json; exact two-sided McNemar (F14.SC.kv4) and Tango score
  TOST with margin 5 percentage points (F14.SC.kv4.equivalence); 8 tests each.
- Descriptive: held-out timings; the dose series (trained solved share per model; ED50 = the dose at which the share
  first falls to half the model's healthy share, log-linear interpolation, uniform from runs/phase4-dose-low,
  phase4-dose and phase3-regime-a); the regime B runs (uniform control vs each run's verification.json, kv4 and the
  restored assessment of runs/rescue); leave-one-superclass-out at dose 1; weights, activity and loom.
Outputs (results/): sc_kv4_tests.csv, sc_kv4_conditions.csv, sc_kv4_doseseries.csv, sc_kv4_regime_b.csv,
sc_kv4_identity.json.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
RUNS = STUDY / 'runs'
RESULTS = STUDY / 'results'
KV4 = RUNS / 'sc-kv4'
sys.path.insert(0, str(STUDY / 'code'))
import analyze_batch2 as b2  # noqa: E402  (trained_outcomes, discordant, mcnemar, MARGIN, MODELS; sets the paths)
from studykit import stats_ext as sx  # noqa: E402

DOSES = (0.5, 1.0, 1.5, 2.0)
SERIES = (0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4)
LOW, MAIN = (0.003125, 0.00625, 0.0125, 0.025), (0.05, 0.1, 0.2, 0.3, 0.4)
ROW_KEYS = ('case', 'cue', 'delay_ms', 'prefix_windows', 'status', 'steps', 'first_push_correct', 'first_action', 'first_logits')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity():
    new, old = load(KV4 / 'check-uniform-dose0.5.json'), load(RUNS / 'phase3-regime-a' / 'C01-dose0.5.json')
    diffs = []
    for m in b2.MODELS:
        a, b = new['memory'][m]['rows'], old['memory'][m]['rows']
        if len(a) != len(b):
            diffs.append(f'{m}: {len(a)} vs {len(b)} rows')
            continue
        bad = sum(any(x.get(k) != y.get(k) for k in ROW_KEYS) for x, y in zip(a, b))
        if bad:
            diffs.append(f'{m}: {bad} rows differ')
    for part in ('activity', 'loom'):
        for k, v in old[part].items():
            if new[part].get(k) != v:
                diffs.append(f'{part}.{k}: {new[part].get(k)} vs {v}')
    return dict(ok=not diffs, differences=diffs, rows=sum(len(new['memory'][m]['rows']) for m in b2.MODELS))


def share(outcomes):
    """Trained solved share per model from {(model, case, delay, prefix): solved}."""
    return {m: float(np.mean([v for (mm, *_), v in outcomes.items() if mm == m])) for m in b2.MODELS}


def heldout_solved(res):
    return {m: sum(r['status'] == 'solved' for r in v['rows'] if r.get('history_split') == 'heldout') for m, v in res['memory'].items()}


def ed50(healthy, points):
    """First dose at which the share falls to <= half the healthy share (log-linear interpolation); None if never."""
    target = 0.5 * healthy
    prev = (None, healthy)
    for d, s in points:
        if s <= target:
            if prev[0] is None:
                return d
            (d0, s0), (d1, s1) = prev, (d, s)
            f = (s0 - target) / (s0 - s1) if s0 != s1 else 0.0
            return float(np.exp(np.log(d0) + f * (np.log(d1) - np.log(d0))))
        prev = (d, s)
    return None


def main():
    names = [c['name'] for c in load(STUDY / 'tasks' / 'sc-kv4-conditions.json')['conditions']]
    missing = [n for n in names if not (KV4 / f'{n}.json').exists()]
    if missing:
        raise SystemExit(f'{len(missing)} T-SC4 conditions not pulled yet: {missing[:5]}')
    check = identity()
    (RESULTS / 'sc_kv4_identity.json').write_text(json.dumps(check, indent=1) + '\n', encoding='utf-8', newline='\n')
    if not check['ok']:
        raise SystemExit(f'identity check failed: {check["differences"][:5]}')
    regime_a = {d: load(RUNS / 'phase3-regime-a' / f'C01-dose{d:g}.json') for d in DOSES}
    healthy = load(RUNS / 'phase3-regime-a' / 'healthy.json')
    tests, conds = [], []
    for imp in ('class', 'global'):
        for d in DOSES:
            res = load(KV4 / f'kv4-{imp}-dose{d:g}.json')
            b, c, n = b2.discordant(b2.trained_outcomes(res), b2.trained_outcomes(regime_a[d]))
            r = sx.tost_paired_binary(b, c, n, b2.MARGIN)
            name = f'{imp} dose {d:g}: Kv4-weighted vs uniform, trained histories solved'
            tests.append(dict(family='F14.SC.kv4', test=name, n=n, b=b, c=c, estimate=(b - c) / n, p_value=b2.mcnemar(b, c)))
            tests.append(dict(family='F14.SC.kv4.equivalence', test=name, n=n, b=b, c=c, estimate=r['effect'],
                              ci90_low=r['effect_ci_low'], ci90_high=r['effect_ci_high'], conclusion=r['conclusion'],
                              p_value=r['p_value']))
    for name in names:
        res = load(KV4 / f'{name}.json')
        c = res['condition']
        row = dict(condition=name, protocol=c['protocol'], weights=c['weights'], dose=c['dose'],
                   uniform_superclass=c.get('uniform_superclass'), run=c.get('run'), seconds=round(res['seconds']),
                   w_min=res['weights']['min'], w_max=res['weights']['max'], distinct_offsets=res['weights']['distinct'])
        if 'memory' in res:
            row.update({f'trained_solved_{m}': int(sum(v for (mm, *_), v in b2.trained_outcomes(res).items() if mm == m))
                        for m in b2.MODELS})
            if c['protocol'] == 'regime-a':
                row.update({f'heldout_solved_{m}': v for m, v in heldout_solved(res).items()})
            row.update(task_mean_hz=res['activity']['task_mean_hz'], fraction_above_100hz=res['activity']['fraction_above_100hz'],
                       gf_spikes=res['loom']['gf_spikes'], gf_first_spike_ms=res['loom']['gf_first_spike_ms'])
        else:
            row.update(solved=res['results']['solved'], total=res['results']['total'])
        conds.append(row)
    # dose series (trained solved share per model)
    base = share(b2.trained_outcomes(healthy))
    series = []
    for kind in ('uniform', 'kv4-class'):
        points = {m: [] for m in b2.MODELS}
        for d in SERIES + DOSES:
            if kind == 'uniform':
                folder = 'phase4-dose-low' if d in LOW else 'phase4-dose' if d in MAIN else 'phase3-regime-a'
                res = load(RUNS / folder / f'C01-dose{d:g}.json')
            else:
                res = load(KV4 / f'kv4-class-dose{d:g}.json')
            s = share(b2.trained_outcomes(res))
            for m in b2.MODELS:
                points[m].append((d, s[m]))
                series.append(dict(operator=kind, model=m, dose=d, trained_share=s[m], healthy_share=base[m]))
        for m in b2.MODELS:
            series.append(dict(operator=kind, model=m, dose='ED50', trained_share=ed50(base[m], points[m]), healthy_share=base[m]))
    # regime B: uniform control vs verification, kv4, restored
    regime_b = []
    for run in sorted({c['run'] for c in (load(KV4 / f'{n}.json')['condition'] for n in names) if c.get('run')}):
        ver = load(RUNS / run / 'verification.json')['results']
        uni, kv4 = load(KV4 / f'{run}-uniform.json')['results'], load(KV4 / f'{run}-kv4.json')['results']
        restored = load(RUNS / 'rescue' / f'{run}.json')['results']
        key = lambda rows: [(r['cue'], r['delay_ms'], r['prefix_windows'], r['status']) for r in rows]   # noqa: E731
        trained = lambda rows: sum(r['status'] == 'solved' for r in rows                                 # noqa: E731
                                   if (r['delay_ms'], r['prefix_windows']) in b2.TRAINED)
        regime_b.append(dict(run=run, control_reproduces_verification=key(uni['rollouts']) == key(ver['rollouts']),
                             solved_as_trained=ver['solved'], solved_uniform=uni['solved'], solved_kv4=kv4['solved'],
                             solved_restored=restored['solved'], total=ver['total'],
                             trained_as_trained=trained(ver['rollouts']), trained_kv4=trained(kv4['rollouts']),
                             trained_restored=trained(restored['rollouts'])))
    for path, rows in (('sc_kv4_tests.csv', tests), ('sc_kv4_conditions.csv', conds), ('sc_kv4_doseseries.csv', series),
                       ('sc_kv4_regime_b.csv', regime_b)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(RESULTS / path, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
    for t in tests:
        print(f"{t['family']:24s} {t['test']:60s} b={t['b']:3d} c={t['c']:3d} est {t['estimate']:+.3f} p {t['p_value']:.4g}")
    for r in regime_b:
        print(r)
    return 0


if __name__ == '__main__':
    sys.exit(main())
