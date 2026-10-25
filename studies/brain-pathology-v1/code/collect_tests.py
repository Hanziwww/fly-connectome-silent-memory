"""Collect every Phase 3 and X1 test into one table and apply each family's declared correction (local).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\collect_tests.py

Inputs: studies/brain-pathology-v1/results/phase3_tests.csv (analyze_phase3.py), batch2_tests.csv (analyze_batch2.py) and
studies/brain-supplement-v2/results/campaign_tests.csv (analyze_campaign.py) and x4_x5_tests.csv (analyze_x4_x5.py), the other
X-batch tables, dy2_tests.csv (analyze_dy2.py, T-DY2), the single-cell tables sc_{trace,adfca,kv4,variant}_tests.csv
(T-SC3, T-SC4b, T-SC4, T-SC6) and lesion_link_tests.csv (analyze_lesion_link.py). Families and corrections come from the two
studies' code/testplan.py (declared before any result). A test whose family is not declared stops the collection; a
declared family with missing tests is reported as incomplete and left unadjusted (never adjusted over a partial set).
Writes studies/brain-pathology-v1/results/all_tests.csv and prints the family summary.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[3]
STUDIES = {'brain-pathology-v1': ROOT / 'studies' / 'brain-pathology-v1', 'brain-supplement-v2': ROOT / 'studies' / 'brain-supplement-v2'}
EXPECTED = {'F12.C01.primary': 3, 'F12.olfactory.primary': 3, 'F12.regimeB': 24, 'F12.dependency.equivalence': 4,
            'F12.dependency.null': 4, 'F12.regimeC': 3, 'F12.rescue': 8, 'F12.regimeA.equivalence': 6, 'F12.comparison': 3,
            'F13.X1.trained': 4, 'F13.X1.untrained': 4, 'F13.X1.lock': 4, 'F13.X4.content': 6, 'F13.X4.dose': 6,
            'F13.X4.rank': 30, 'F13.X5.swap': 6, 'F13.X5.visibility': 1, 'F13.X5.cross': 3, 'F13.X8.primary': 1, 'F13.X8.map': 8,
            'F13.X6.ensemble': 4, 'F13.X6.rank': 4, 'F13.X7.edges': 4, 'F13.X7.rank': 4,
            'F13.X3.lock': 1, 'F13.X3.descriptive': 4, 'F13.X9.pairing': 5, 'F13.X10.fig2': 7,
            'F13.X2.trained': 6, 'F13.X2.untrained': 6, 'F13.X2.lock': 6,
            'F14.DY.robust': 10, 'F14.DY.robust.secondary': 45,
            'F14.SC.trace': 2, 'F14.SC.trace.secondary': 22, 'F14.SC.adfca.pan': 4, 'F14.SC.adfca': 192,
            'F14.SC.kv4': 8, 'F14.SC.kv4.equivalence': 8, 'F14.SC.variant': 3, 'F14.SC.variant.secondary': 11,
            'F14.SC.trace.v2': 4, 'F14.SC.trace.v2.secondary': 12, 'F14.SC.trace.v2.vnc': 4, 'F14.SC.trace.sensitivity': 4,
            'F14.SC.adfca.sensitivity': 140, 'F15.link': 3, 'F15.link.secondary': 9}


def load_plan(name, folder):
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(f'plan_{name.replace("-", "_")}', folder / 'code' / 'testplan.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {f['family']: f for f in module.FAMILIES}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind='mergesort')
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def main():
    plans = {}
    for name, folder in STUDIES.items():
        plans.update({k: dict(v, study=name) for k, v in load_plan(name, folder).items()})
    rows = []
    for path in (STUDIES['brain-pathology-v1'] / 'results' / 'phase3_tests.csv', STUDIES['brain-pathology-v1'] / 'results' / 'batch2_tests.csv',
                 STUDIES['brain-pathology-v1'] / 'results' / 'phase3_comparison_tests.csv',
                 STUDIES['brain-supplement-v2'] / 'results' / 'campaign_tests.csv',
                 STUDIES['brain-supplement-v2'] / 'results' / 'x4_x5_tests.csv', STUDIES['brain-supplement-v2'] / 'results' / 'x8_tests.csv',
                 STUDIES['brain-supplement-v2'] / 'results' / 'x6_x7_tests.csv',
                 STUDIES['brain-supplement-v2'] / 'results' / 'x3_tests.csv',
                 STUDIES['brain-supplement-v2'] / 'results' / 'x9_x10_tests.csv',
                 STUDIES['brain-supplement-v2'] / 'results' / 'x2_tests.csv',
                 STUDIES['brain-pathology-v1'] / 'results' / 'dy2_tests.csv',
                 *(STUDIES['brain-pathology-v1'] / 'results' / f'sc_{t}_tests.csv'
                   for t in ('trace', 'adfca', 'kv4', 'variant', 'trace_v2', 'trace_sensitivity', 'adfca_sensitivity')),
                 STUDIES['brain-pathology-v1'] / 'results' / 'lesion_link_tests.csv'):
        if path.exists():
            rows += list(csv.DictReader(open(path, encoding='utf-8')))
    tests = [r for r in rows if r['family'] != 'descriptive' and r.get('p_value') not in (None, '')]
    unknown = sorted({r['family'] for r in tests} - set(plans))
    if unknown:
        raise SystemExit(f'tests in undeclared families: {unknown}')
    summary = []
    for fam, decl in plans.items():
        idx = [i for i, r in enumerate(tests) if r['family'] == fam]
        complete = len(idx) >= EXPECTED.get(fam, len(idx)) and len(idx) > 0
        if idx and complete:
            p = [float(tests[i]['p_value']) for i in idx]
            adj = holm(p) if decl['correction'] == 'holm' else p if decl['correction'] == 'none' else st.false_discovery_control(np.asarray(p), method='bh')
            for i, q in zip(idx, adj):
                tests[i]['p_adjusted_family'] = float(q)
                tests[i]['correction'] = decl['correction']
                tests[i]['primary'] = decl['primary']
        summary.append(dict(family=fam, study=decl['study'], correction=decl['correction'], primary=decl['primary'], tests=len(idx),
                            expected=EXPECTED.get(fam), status='adjusted' if idx and complete else ('incomplete' if idx else 'no tests yet'),
                            significant=sum(1 for i in idx if tests[i].get('p_adjusted_family') not in (None, '') and float(tests[i]['p_adjusted_family']) < 0.05)))
    out = STUDIES['brain-pathology-v1'] / 'results'
    out.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(k for r in tests for k in r))
    with open(out / 'all_tests.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(tests)
    for s in summary:
        print(f"{s['family']:24s} {s['study']:20s} {s['correction']:5s} primary={s['primary']!s:5s} tests {s['tests']:3d}/{s['expected']} "
              f"{s['status']:13s} significant {s['significant']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
