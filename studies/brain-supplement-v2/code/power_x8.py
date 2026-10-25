"""X8 pilot -> replicates per arm and map (plan v2: "n then fixed by simulated power"; local, after the 3-replicate pilot,
before any X8 test is computed).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-supplement-v2\\code\\power_x8.py

Follows the regime B power step (studies/brain-pathology-v1/code/power_regime_b.py): 80% power at two-sided alpha 0.05
for the declared primary test (F13.X8.primary: stratified log-rank over the four maps, arm labels permuted within maps;
analyze_x8.logrank_stat), with the pilot's own effect. Simulation: per map and arm, log(updates) ~ normal with the
pilot's cell mean and the pilot's pooled within-cell SD (floor 0.05); every run solves, as all 24 pilot runs did. For
n = 3, 4, ... (up to 10) per arm and map: 500 simulated data sets, each tested with 499 within-map permutations
(p = (k + 1) / 500). The smallest n with power >= 0.8 is proposed; if it is 3, the pilot is the final design.
Writes amendments/x8-n.json (replicates_per_arm_and_map, the power curve, the pilot summary).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
MAPS = ('D1-T0', 'D3-T0', 'D5-T0', 'D7-T0')
ARMS = ('head_only', 'joint')
SETS, PERMS, ALPHA, TARGET = 500, 499, 0.05, 0.8


def main():
    sys.path.insert(0, str(SUPP / 'code'))
    from analyze_x8 import logrank_stat
    pilot = {}
    for m in MAPS:
        for a in ARMS:
            vals = []
            for r in range(3):
                d = SUPP / 'runs' / f'x8-{m}-{a}-r{r}'
                res = json.loads((d / 'result.json').read_text(encoding='utf-8'))
                if res.get('status') not in ('learned_solved', 'already_solved') or not (d / 'verification.json').exists():
                    raise SystemExit(f'{d.name}: the pilot is not complete and solved ({res.get("status")}); the simulation assumes it')
                vals.append(np.log(float(res['updates'])))
            pilot[(m, a)] = vals
    means = {k: float(np.mean(v)) for k, v in pilot.items()}
    sd = max(0.05, float(np.sqrt(np.mean([np.var(v, ddof=1) for v in pilot.values()]))))
    rng = np.random.default_rng(20261020)
    curve = []
    proposed = None
    for n in range(3, 11):
        strata = np.repeat(np.arange(len(MAPS)), 2 * n)
        group = np.tile(np.r_[np.ones(n, bool), np.zeros(n, bool)], len(MAPS))
        events = np.ones(len(strata), bool)
        hits = 0
        for _ in range(SETS):
            times = np.concatenate([np.r_[rng.normal(means[(m, 'head_only')], sd, n), rng.normal(means[(m, 'joint')], sd, n)]
                                    for m in MAPS])
            obs = abs(logrank_stat(times, events, group, strata))
            k = 0
            for _ in range(PERMS):
                g = group.copy()
                for s in range(len(MAPS)):
                    idx = np.flatnonzero(strata == s)
                    g[idx] = rng.permutation(g[idx])
                k += abs(logrank_stat(times, events, g, strata)) >= obs - 1e-12
            hits += (k + 1) / (PERMS + 1) <= ALPHA
        power = hits / SETS
        curve.append(dict(replicates_per_arm_and_map=n, power=power))
        print(f'n = {n}: power {power:.3f}', flush=True)
        if power >= TARGET:
            proposed = n
            break
    if proposed is None:
        proposed = 10
    out = dict(created_unix=time.time(), replicates_per_arm_and_map=proposed, target_power=TARGET, alpha=ALPHA,
               simulated_sets=SETS, permutations=PERMS, sd_log_updates=sd,
               pilot_mean_log_updates={f'{m}/{a}': means[(m, a)] for m in MAPS for a in ARMS},
               pilot_updates={f'{m}/{a}': [round(float(np.exp(x))) for x in pilot[(m, a)]] for m in MAPS for a in ARMS},
               curve=curve,
               note=('Proposal written after the 3-replicate pilot, before any X8 test. n = 3 means the pilot is the final design; '
                     'otherwise replicates 3..n-1 per arm and map extend it. Committed before any test or extension run.'))
    target = SUPP / 'amendments' / 'x8-n.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(out, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps(dict(proposed=proposed, sd=sd)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
