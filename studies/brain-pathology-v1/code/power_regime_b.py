"""Regime B pilot -> replicate number (plan v2, Phase 3 G3d; local, after the 3-replicate pilot of the campaign).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\power_regime_b.py

The campaign ran regime B with 3 replicates per C01 dose and learner (dose 0 = the X1 replicate of the same index). The
per-dose comparisons (F12.regimeB, exact two-sided sign-flip over replicate pairs) cannot reach p < 0.05 with 3 pairs
(smallest attainable p = 0.25). This script fixes n for an extension, as the plan prescribes: 80% power to detect a
drop of 15 percentage points of trained histories solved (2.7 of 18), and, for log(updates), the pilot's own mean
difference. Simulation: n paired differences drawn from a normal with the target mean and the pilot's SD of the paired
differences (pooled over doses and learners; floor 1 history, 0.1 log-update), tested by the exact sign-flip
(studykit.stats_ext.simulate_power; 2,000 data sets per n, n = 3..12). Writes amendments/regime-b-n.json (proposal
with the power curves; the extension runs only after it is committed).
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
sys.path.insert(0, str(ROOT))
from studykit import stats_ext as sx  # noqa: E402

DOSES = (0.5, 1.0, 1.5, 2.0)
LEARNERS = ('M0-head_only', 'M0-joint')
TARGET_SOLVED_DROP = 0.15 * 18


def run(folder):
    r = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    v = json.loads((folder / 'verification.json').read_text(encoding='utf-8'))
    trained = [x for x in v['results']['rollouts'] if (x['delay_ms'], x['prefix_windows']) in {(d, p) for d in (0, 40, 160) for p in (0, 1, 4)}]
    return dict(updates=r['updates'], solved=sum(x['status'] == 'solved' for x in trained), status=r['status'])


def main():
    diffs = dict(solved=[], log_updates=[])
    table = []
    for learner in LEARNERS:
        for rep in (1, 2, 3):
            h = SUPP / 'runs' / f'x1-{learner}-r{rep}'
            if not (h / 'verification.json').exists():
                continue
            base = run(h)
            for dose in DOSES:
                f = STUDY / 'runs' / f'b-C01-dose{dose:g}-{learner}-r{rep}'
                if not (f / 'verification.json').exists():
                    continue
                x = run(f)
                diffs['solved'].append(x['solved'] - base['solved'])
                diffs['log_updates'].append(math.log(x['updates']) - math.log(base['updates']))
                table.append(dict(learner=learner, replicate=rep, dose=dose, status=x['status'], solved=x['solved'], updates=x['updates'],
                                  healthy_solved=base['solved'], healthy_updates=base['updates']))
    if len(table) < len(LEARNERS) * len(DOSES) * 3:
        print(f'{len(table)} of {len(LEARNERS) * len(DOSES) * 3} pilot runs available; power analysis waits for the full pilot')
        for t in table:
            print(t)
        return 0
    sd_solved = max(float(np.std(diffs['solved'], ddof=1)), 1.0)
    sd_log = max(float(np.std(diffs['log_updates'], ddof=1)), 0.1)
    mean_log = float(np.mean(diffs['log_updates']))
    out = dict(created_unix=time.time(), pilot_runs=table, sd_solved_difference=sd_solved, sd_log_updates_difference=sd_log,
               pilot_mean_log_updates_difference=mean_log, target_solved_drop=TARGET_SOLVED_DROP, curves={})
    for name, effect, sd in (('solved', -TARGET_SOLVED_DROP, sd_solved), ('log_updates', mean_log, sd_log)):
        def simulate(rng, n, effect=effect, sd=sd):
            return rng.normal(effect, sd, n)

        def test(data):
            return sx.sign_flip(data, alternative='two-sided', ci=False)['p_value']

        out['curves'][name] = sx.sample_size(simulate, test, list(range(3, 13)), target=0.8, n_sim=2000, test_id=f'regimeB.power.{name}')
    chosen = [c['n'] for c in out['curves'].values() if c['n'] is not None]
    out['proposed_replicates'] = max(chosen) if chosen else None
    out['note'] = ('Proposal: extend regime B to this many replicates per dose and learner (replicates 4..n), healthy partners = X1 '
                   'replicates of the same index. Committed before any extension run.')
    target = STUDY / 'amendments' / 'regime-b-n.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1, default=float), encoding='utf-8', newline='\n')
    print(json.dumps({k: out['curves'][k]['n'] for k in out['curves']}), 'proposed', out['proposed_replicates'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
