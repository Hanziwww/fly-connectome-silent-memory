"""Phase 4 dose series (local, after the pull): estimates, not tests. Written before any dose-series result existed.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\analyze_phase4_dose.py

Response: share of the trained-timing histories solved (the five memory models; 126 histories), per condition.
- C01: doses 0.05-0.4 (runs/phase4-dose) with regime A's healthy, 0.5, 1, 1.5 and 2 (runs/phase3-regime-a, the same
  trained timings); offset = -3.22 mV x dose.
- Random removal: shares 0.002-0.05 of the non-input, non-readout cells (3 draws each) and the healthy brain.
For each series: 4PL, linear and threshold fits by AICc (studykit.stats_ext.compare_dose_models) on (dose, history
solved); ED50 of the 4PL with a hierarchical percentile bootstrap (models, then histories within a model; 2,000
resamples). Also per condition: task activity, share of cells above 100 Hz and the giant-fibre looming response.
Compared with: the runaway onset of the mean-field scan (-6.0 to -6.5 mV, PLAN.md section 19), the calibrated C01 dose
(-3.22 mV), and the structural threshold (none up to 90% removal, phase4_graph.py).
Low extension (post hoc, amendment 0001; runs/phase4-dose-low): C01 at 0.003125-0.025 and random removal of
0.0002-0.001. Its C01 dose 0 must reproduce regime A's healthy outcome history by history, else its fits are withheld;
the declared fits of the main series are unchanged, and fits of main plus low are added with set = 'main + low (post
hoc)'.
Writes results/phase4_dose_conditions.csv (one row per condition, with its set) and phase4_dose_fits.csv.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
sys.path.insert(0, str(ROOT))
from studykit import stats_ext as sx  # noqa: E402

TRAINED = {(d, p) for d in (0, 40, 160) for p in (0, 1, 4)}
C01_MV = -3.22


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def histories(rec):
    """{model: [solved flags]} over the trained timings, in a fixed order."""
    out = {}
    for m, v in sorted(rec['memory'].items()):
        rows = sorted((r for r in v['rows'] if (r['delay_ms'], r['prefix_windows']) in TRAINED),
                      key=lambda r: (r['case'], r['delay_ms'], r['prefix_windows']))
        out[m] = [r['status'] == 'solved' for r in rows]
    return out


def ed50_statistic(clusters):
    data = np.concatenate(clusters)
    fit = sx.fit_logistic4(data[:, 0], data[:, 1])
    return fit['params']['ed50']


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def collect(folder):
    """{'C01': [(dose, record)], 'random removal': [...]} of one dose-series folder."""
    out = {'C01': [], 'random removal': []}
    for p in sorted(folder.glob('*.json')):
        if p.name == 'done.json':
            continue
        rec = load(p)
        out['C01' if rec['kind'] == 'threshold_offset' else 'random removal'].append((rec['dose'], rec))
    return out


def condition_row(label, name, dose, rec):
    h = histories(rec)
    return dict(set=label, series=name, condition=rec.get('condition', 'healthy'), dose=dose,
                offset_mv=C01_MV * dose if name == 'C01' else None,
                solved_share=float(np.mean([f for v in h.values() for f in v])),
                task_mean_hz=rec['activity']['task_mean_hz'], fraction_above_100hz=rec['activity'].get('fraction_above_100hz'),
                gf_spikes=rec['loom']['gf_spikes'])


def fit_series(series, label, fits):
    for name, items in series.items():
        clusters = {}
        for dose, rec in sorted(items, key=lambda x: x[0]):
            for m, flags in histories(rec).items():
                clusters.setdefault(m, []).extend((dose, float(f)) for f in flags)
        data = np.array([x for c in clusters.values() for x in c])
        comp = sx.compare_dose_models(data[:, 0], data[:, 1])
        test_id = f'phase4.ed50.{name}' if label == MAIN else f'phase4.ed50.{name}.low'
        boot = sx.hierarchical_bootstrap([np.array(c) for c in clusters.values()], statistic=ed50_statistic, n_resamples=2000,
                                         test_id=test_id)
        fit = comp['models']['logistic4']
        fits.append(dict(set=label, series=name, best_model=comp['best'], criterion=comp['criterion'], ed50=boot['statistic'],
                         ed50_ci_low=boot['effect_ci_low'], ed50_ci_high=boot['effect_ci_high'], bootstrap_failures=boot['failures'],
                         ed50_mv=C01_MV * boot['statistic'] if name == 'C01' else None,
                         ed50_mv_ci=(C01_MV * boot['effect_ci_high'], C01_MV * boot['effect_ci_low']) if name == 'C01' else None,
                         top=fit.get('params', {}).get('top'), bottom=fit.get('params', {}).get('bottom'), hill=fit.get('params', {}).get('hill'),
                         at_bound=';'.join(fit.get('at_bound', [])), fit_error=fit.get('error'),
                         **{f'{k}_{c}': v[c] for k, v in comp['models'].items() for c in (comp['criterion'],) if c in v}))


MAIN, LOW = 'main', 'main + low (post hoc)'


def main():
    out = STUDY / 'results'
    dose_dir, low_dir, ra = STUDY / 'runs' / 'phase4-dose', STUDY / 'runs' / 'phase4-dose-low', STUDY / 'runs' / 'phase3-regime-a'
    if not (dose_dir / 'done.json').exists():
        print('pending: the dose series is not finished')
        return 0
    healthy = load(ra / 'healthy.json')
    found = collect(dose_dir)
    series = {'C01': [(0.0, healthy)] + [(d, load(ra / f'C01-dose{d:g}.json')) for d in (0.5, 1.0, 1.5, 2.0)] + found['C01'],
              'random removal': [(0.0, healthy)] + found['random removal']}
    rows = [condition_row(MAIN, name, dose, rec) for name, items in series.items() for dose, rec in sorted(items, key=lambda x: x[0])]
    fits = []
    fit_series(series, MAIN, fits)
    if (low_dir / 'done.json').exists():
        low = collect(low_dir)
        zero = next(rec for d, rec in low['C01'] if d == 0.0)
        same = histories(zero) == histories(healthy)
        print(f'low extension: C01 dose 0 reproduces the healthy outcome history by history: {same}')
        rows += [dict(condition_row('low (post hoc)', name, dose, rec), reproduces_healthy=same if dose == 0.0 else None)
                 for name, items in low.items() for dose, rec in sorted(items, key=lambda x: x[0])]
        if same:
            fit_series({k: v + [(d, rec) for d, rec in low[k] if d > 0] for k, v in series.items()}, LOW, fits)
        else:
            print('pending: the low extension does not reproduce the healthy outcome at dose 0; its fits are withheld')
    write(out / 'phase4_dose_conditions.csv', rows)
    write(out / 'phase4_dose_fits.csv', fits)
    for f in fits:
        print(f)
    return 0


if __name__ == '__main__':
    sys.exit(main())
