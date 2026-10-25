"""Analysis of X6 and X7 (causal tests of the cue ensemble) (local, after the pull). Written before any of their
results existed; families declared in code/testplan.py.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-supplement-v2\\code\\analyze_x6_x7.py

History = case x trained timing (18 per scene). Margin = first-decision logit of the case's reference first action minus
the strongest other legal action (as X4 and the paper's Fig. 4f); effect = margin under the condition - margin at
baseline. Per (kind x6 or x7, ensemble source, test scene):
- F13.X6.ensemble / F13.X7.edges: D = ensemble effect - mean over the 20 null sets, exact one-sided sign-flip (D < 0);
- F13.X6.rank / F13.X7.rank: p = (null sets whose scene-mean effect is at or below the ensemble's + 1) / 21.
Descriptive: histories solved under each condition. Writes results/x6_x7_tests.csv, x6_x7_effects.csv.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
TASKS = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1' / 'inputs' / 'memory-tasks.json'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUPP / 'code'))
from studykit import stats_ext as sx  # noqa: E402
from analyze_x4_x5 import margin  # noqa: E402

FAMILY = {'x6': ('F13.X6.ensemble', 'F13.X6.rank'), 'x7': ('F13.X7.edges', 'F13.X7.rank')}


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    folder = SUPP / 'runs' / 'x6-x7'
    out = SUPP / 'results'
    out.mkdir(exist_ok=True)
    if not (folder / 'done.json').exists():
        print('pending: X6/X7 not finished')
        write(out / 'x6_x7_tests.csv', [])
        return 0
    tasks = {t['id']: t for t in json.loads(TASKS.read_text(encoding='utf-8'))}
    tests, effects = [], []
    for scene_file in sorted(folder.glob('*.json')):
        if scene_file.name == 'done.json':
            continue
        rec = json.loads(scene_file.read_text(encoding='utf-8'))
        ref = {c['id']: int(c['level']['reference_actions'][0]) for c in tasks[rec['task']]['cases']}
        conds = rec['conditions']
        eff = [[] for _ in conds]
        solved = [0] * len(conds)
        base_solved = 0
        for entry in rec['results']:
            for i, b in enumerate(entry['baseline']):
                m0 = margin(b['first_logits'], ref[b['case']], b['first_legal'])
                base_solved += b['status'] == 'solved'
                for j, rows in enumerate(entry['outcomes']):
                    r = rows[i]
                    eff[j].append(margin(r['first_logits'], ref[r['case']], r['first_legal']) - m0)
                    solved[j] += r['status'] == 'solved'
        eff = np.array(eff)
        for kind in ('x6', 'x7'):
            for src in sorted({c['source'] for c in conds}):
                idx = [j for j, c in enumerate(conds) if c['kind'] == kind and c['source'] == src]
                ens = next(j for j in idx if conds[j]['set'] == 'ensemble')
                nulls = [j for j in idx if conds[j]['set'] != 'ensemble']
                D = eff[ens] - eff[nulls].mean(0)
                sf = sx.sign_flip(D, alternative='less', ci=False)
                fam, rank_fam = FAMILY[kind]
                label = f"{kind.upper()}: ensemble from {src} tested in {rec['scene']}"
                tests.append(dict(family=fam, test=f'{label}: margin effect minus mean null effect < 0', n=len(D), estimate=float(D.mean()),
                                  p_value=sf['p_value']))
                null_means = eff[nulls].mean(1)
                k_le = int((null_means <= eff[ens].mean()).sum())
                tests.append(dict(family=rank_fam, test=f'{label}: rank among 20 null sets', n=20, estimate=float(eff[ens].mean()), value=k_le,
                                  p_value=(k_le + 1) / 21))
                effects.append(dict(kind=kind, source=src, scene=rec['scene'], histories=eff.shape[1], baseline_solved=base_solved,
                                    ensemble_mean_effect=float(eff[ens].mean()), null_mean_effect=float(null_means.mean()),
                                    null_min=float(null_means.min()), null_max=float(null_means.max()), ensemble_solved=solved[ens],
                                    null_solved_mean=float(np.mean([solved[j] for j in nulls])),
                                    absent_synapses=conds[ens].get('absent'), requested_synapses=conds[ens].get('requested')))
    write(out / 'x6_x7_tests.csv', tests)
    write(out / 'x6_x7_effects.csv', effects)
    for t in tests:
        print(t['family'], '|', t['test'], '|', round(t['estimate'], 4), '|', round(t['p_value'], 5))
    return 0


if __name__ == '__main__':
    sys.exit(main())
