"""Analysis of the training campaign (local, after the pull): X1 replicates and Phase 3 regime B (learning under lesion).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-supplement-v2\\code\\analyze_campaign.py

Per run (result.json, verification.json of the frozen trainer): status, updates, training seconds, histories solved and
cue-pair classes at the 9 trained and the 26 untrained timings (verification: all 35 timings, reloaded model in a fresh
process); net memory use = (both - neither) / pairs.
Replicate 0 check (F13.X1.descriptive): replicate 0 runs the frozen protocol, so its initial, mid and final checkpoints
must hold the frozen model's tensors bit for bit. The files' SHA-256 cannot match: each checkpoint embeds its run id
(config.id, config.replicate, config.replicate_definition) and torch writes a fresh .data/serialization_id per save.
Criterion: every tensor and every value outside config identical; the config keys that differ are listed.
Tests (families declared in code/testplan.py and studies/brain-pathology-v1/code/testplan.py before any campaign result),
computed only once every planned run of the test is verified (a partial set is reported as pending, never tested):
- F13.X1.trained / .untrained / .lock: exact sign-flip tests over replicates, per learner (10 replicates of M0
  readout-only and joint, 7 of S1 and S2);
- F12.C01.primary (2): exact Page trend over doses 0, 0.5, 1, 1.5, 2 of trained histories solved after training under the
  C01 lesion, replicates 1-3 as blocks (dose 0 = healthy X1 replicate of the same index), per learner;
- F12.regimeB (24): each C01 dose against the healthy replicate of the same index, per learner, on log updates and on
  trained and untrained histories solved (exact two-sided sign-flip over the 3 replicate pairs). With 3 pairs the
  smallest attainable p is 0.25, so this family is descriptive in practice. A run that did not learn contributes its
  update count at the stop (a lower bound). The C04, C13 and C16 controls have one replicate each and are reported per
  run only.
Corrections are applied once per complete family by studies/brain-pathology-v1/code/collect_tests.py, not here.
Writes studies/brain-supplement-v2/results/campaign_runs.csv, campaign_tests.csv and campaign_x1_summary.csv.
"""
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
PATHO = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
TRAINED = {(d, p) for d in (0, 40, 160) for p in (0, 1, 4)}
LEARNERS = {'M0-head_only': 'remote-dynamic-M0-head_only', 'M0-joint': 'remote-dynamic-M0-joint',
            'S1-head_only': 'supp-S1-ring-brain', 'S2-head_only': 'supp-S2-context-brain'}
PLANNED = {'M0-head_only': 10, 'M0-joint': 10, 'S1-head_only': 7, 'S2-head_only': 7}
DOSES = (0.0, 0.5, 1.0, 1.5, 2.0)
MISSING = object()
REGIME_B_AMENDMENT = ROOT / 'studies' / 'brain-pathology-v1' / 'amendments' / 'regime-b-n.json'


def regime_b_replicates():
    """Regime B replicates 1..n: n = 3 (the pilot) unless the committed power amendment (plan G3d, power_regime_b.py) sets
    more; the healthy partner of replicate r is the X1 replicate r."""
    n = 3
    if REGIME_B_AMENDMENT.exists():
        n = max(n, int(json.loads(REGIME_B_AMENDMENT.read_text(encoding='utf-8'))['proposed_replicates']))
    return tuple(range(1, n + 1))


def classes(rollouts):
    """Cue-pair classes from rollouts at given timings; a pair = all histories with the same (context, visibility,
    delay, prefix); both = all solved."""
    groups = {}
    for r in rollouts:
        case = str(r.get('case', ''))                    # supplement tasks: c<context>-cue<k>[-visible|-hidden]
        ctx = int(case[1:case.index('-')]) if case.startswith('c') and '-' in case else r.get('context', 0)
        vis = case.endswith('-visible') if case else bool(r.get('visible', False))
        key = (ctx, vis, r['delay_ms'], r['prefix_windows'])
        groups.setdefault(key, []).append(r['status'] == 'solved')
    c = dict(both=0, one=0, none=0)
    for v in groups.values():
        k = sum(v)
        c['both' if k == len(v) else 'none' if k == 0 else 'one'] += 1
    n = len(groups)
    return c, n, ((c['both'] - c['none']) / n if n else float('nan'))


def run_row(folder, jid):
    res = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    ver = json.loads((folder / 'verification.json').read_text(encoding='utf-8')) if (folder / 'verification.json').exists() else None
    row = dict(job=jid, status=res.get('status'), updates=res.get('updates'), training_seconds=res.get('training_seconds'),
               final_solved=(res.get('final') or {}).get('solved'), final_total=(res.get('final') or {}).get('total'),
               model_sha256=res.get('model_sha256'), verified=(ver or {}).get('status'))
    if ver:
        rolls = ver['results']['rollouts']
        tr = [r for r in rolls if (r['delay_ms'], r['prefix_windows']) in TRAINED]
        un = [r for r in rolls if (r['delay_ms'], r['prefix_windows']) not in TRAINED]
        for name, sel in (('trained', tr), ('untrained', un)):
            c, n, use = classes(sel)
            row.update({f'{name}_solved': sum(r['status'] == 'solved' for r in sel), f'{name}_histories': len(sel), f'{name}_pairs': n,
                        f'{name}_both': c['both'], f'{name}_one': c['one'], f'{name}_none': c['none'], f'{name}_net_use': use})
    return row


def flat(obj, prefix=''):
    if isinstance(obj, dict):
        return {k2: v2 for k, v in obj.items() for k2, v2 in flat(v, f'{prefix}{k}.').items()}
    if isinstance(obj, (list, tuple)):
        return {k2: v2 for i, v in enumerate(obj) for k2, v2 in flat(v, f'{prefix}{i}.').items()}
    return {prefix.rstrip('.'): obj}


def checkpoint_match(rep, frozen):
    """Bitwise comparison of replicate 0's checkpoints with the frozen model's (module docstring); None if one is missing."""
    import torch
    names = ('initial.pt', 'mid.pt', 'model.pt')
    if not all((rep / n).exists() and (frozen / n).exists() for n in names):
        return None
    compared, bad, config = 0, [], set()
    for n in names:
        a = flat(torch.load(frozen / n, map_location='cpu', weights_only=True))
        b = flat(torch.load(rep / n, map_location='cpu', weights_only=True))
        for k in sorted(set(a) | set(b)):
            x, y = a.get(k, MISSING), b.get(k, MISSING)
            if torch.is_tensor(x) and torch.is_tensor(y):
                same = x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
            elif torch.is_tensor(x) or torch.is_tensor(y):
                same = False
            else:
                same = x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))
            if k.startswith('config.'):
                if not same:
                    config.add(k.split('.')[1])
            else:
                compared += 1
                if not same:
                    bad.append(f'{n}:{k}')
    return dict(reproduces_frozen_model=not bad, frozen_values_compared=compared, frozen_value_mismatches=len(bad),
                frozen_config_differences=';'.join(sorted(config)))


def signflip(x, mu=0.0, alternative='greater'):
    """Exact sign-flip test of mean(x - mu) > 0, < 0 ('less') or != 0 ('two-sided'); all 2^n flips for n <= 20."""
    d = np.asarray(x, float) - mu
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return float('nan'), 0
    obs = d.mean()
    flips = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    null = (flips * np.abs(d)).mean(1)
    if alternative == 'greater':
        p = np.mean(null >= obs - 1e-12)
    elif alternative == 'less':
        p = np.mean(null <= obs + 1e-12)
    else:
        p = np.mean(np.abs(null) >= abs(obs) - 1e-12)
    return float(p), n


def page_exact(data):
    data = np.asarray(data, float)
    n, k = data.shape
    w = np.arange(1, k + 1)
    ranks = np.vstack([st.rankdata(r) for r in data])
    L = float(np.sum(ranks * w))
    dist = {0.0: 1}
    for r in ranks:
        block = {}
        for perm in itertools.permutations(range(k)):
            v = round(float(np.dot(r[list(perm)], w)), 6)
            block[v] = block.get(v, 0) + 1
        new = {}
        for x, cx in dist.items():
            for v, cv in block.items():
                new[round(x + v, 6)] = new.get(round(x + v, 6), 0) + cx * cv
        dist = new
    total = sum(dist.values())
    return L, float(sum(c for v, c in dist.items() if v >= L - 1e-6) / total)


def verified(row):
    return bool(row) and row.get('verified') == 'verified'


def measure(row, name):
    return math.log(row['updates']) if name == 'log_updates' else row[name]


def spread(values):
    v = np.asarray([x for x in values if x is not None], float)
    return (float(np.mean(v)), float(np.min(v)), float(np.max(v))) if len(v) else (None, None, None)


def write_csv(path, data, header):
    keys = list(dict.fromkeys(k for r in data for k in r)) if data else header
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(data)


def main():
    rows = []
    for base in (SUPP / 'runs', PATHO / 'runs'):
        for folder in sorted(base.glob('x1-*')) + sorted(base.glob('b-*')):
            if (folder / 'result.json').exists():
                rows.append(run_row(folder, folder.name))
    if not rows:
        print('no runs yet')
        return 0
    by = {r['job']: r for r in rows}
    tests, summary, pending = [], [], []
    for short, mid in LEARNERS.items():
        r0 = by.get(f'x1-{short}-r0')
        if verified(r0):
            match = checkpoint_match(SUPP / 'runs' / r0['job'], MECH / 'inputs' / 'models' / mid)
            if match:
                r0.update(match)
        reps = sorted((by[k] for k in by if k.startswith(f'x1-{short}-r') and verified(by[k])), key=lambda r: int(r['job'].rsplit('-r', 1)[1]))
        if not reps:
            continue
        upd = sorted(r['updates'] for r in reps)
        tr_m, tr_lo, tr_hi = spread(r['trained_net_use'] for r in reps)
        un_m, un_lo, un_hi = spread(r['untrained_net_use'] for r in reps)
        summary.append(dict(learner=short, planned=PLANNED[short], verified=len(reps),
                            learned_solved=sum(r['status'] == 'learned_solved' for r in reps),
                            updates_median=float(np.median(upd)), updates_min=upd[0], updates_max=upd[-1],
                            trained_net_use_mean=tr_m, trained_net_use_min=tr_lo, trained_net_use_max=tr_hi,
                            untrained_net_use_mean=un_m, untrained_net_use_min=un_lo, untrained_net_use_max=un_hi,
                            replicate0_reproduces_frozen=(r0 or {}).get('reproduces_frozen_model')))
        if len(reps) < PLANNED[short]:
            pending.append(f'X1 {short}: {len(reps)} of {PLANNED[short]} replicates verified')
            continue
        tr = [r['trained_net_use'] for r in reps]
        un = [r['untrained_net_use'] for r in reps]
        p1, n = signflip(tr, 0.0, 'greater')
        p2, _ = signflip(un, 0.10, 'less')
        p3, _ = signflip(np.subtract(tr, un), 0.0, 'greater')
        tests += [dict(family='F13.X1.trained', test=f'{short}', n=n, mean=float(np.mean(tr)), p_value=p1),
                  dict(family='F13.X1.untrained', test=f'{short}', n=n, mean=float(np.mean(un)), p_value=p2),
                  dict(family='F13.X1.lock', test=f'{short}', n=n, mean=float(np.mean(np.subtract(tr, un))), p_value=p3)]
    for learner in ('M0-head_only', 'M0-joint'):
        reps = regime_b_replicates()
        runs = {d: [by.get(f'x1-{learner}-r{r}' if d == 0 else f'b-C01-dose{d:g}-{learner}-r{r}') for r in reps] for d in DOSES}
        done = {d: all(verified(x) for x in v) for d, v in runs.items()}
        if all(done.values()):
            mat = [[runs[d][i]['trained_solved'] for d in DOSES] for i in range(len(reps))]
            L, p = page_exact(np.asarray(mat)[:, ::-1])          # decreasing in dose
            tests.append(dict(family='F12.C01.primary', test=f'learning under C01 lesion ({learner}): trained solved decreasing in dose',
                              n=len(reps), mean=None, p_value=p, value=L, matrix=json.dumps(mat)))
        else:
            pending.append(f'C01 learning trend {learner}: doses complete {[d for d in DOSES if done[d]]}')
        for d in DOSES[1:]:
            if not (done[0.0] and done[d]):
                continue
            for name in ('log_updates', 'trained_solved', 'untrained_solved'):
                diff = [measure(les, name) - measure(hea, name) for les, hea in zip(runs[d], runs[0.0])]
                p, n = signflip(diff, 0.0, 'two-sided')
                tests.append(dict(family='F12.regimeB', test=f'C01 dose {d:g} vs healthy ({learner}): {name}', n=n,
                                  mean=float(np.mean(diff)), p_value=p))
    out = SUPP / 'results'
    out.mkdir(exist_ok=True)
    write_csv(out / 'campaign_runs.csv', rows, ['job'])
    write_csv(out / 'campaign_tests.csv', tests, ['family', 'test', 'n', 'mean', 'p_value'])
    write_csv(out / 'campaign_x1_summary.csv', summary, ['learner'])
    print(f'{len(rows)} runs, {len(tests)} tests (raw p; corrections in collect_tests.py)')
    for s in summary:
        print('X1', s)
    for t in tests:
        print(t['family'], t['test'], t.get('mean'), round(t['p_value'], 4))
    for msg in pending:
        print('pending:', msg)
    return 0


if __name__ == '__main__':
    sys.exit(main())
