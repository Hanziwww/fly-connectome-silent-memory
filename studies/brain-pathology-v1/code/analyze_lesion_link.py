"""Lesion link analysis (local, after the hash-checked pull of runs/lesion-link). Written before any lesion-link result
existed; families F15.link, F15.link.secondary, F15.link.descriptive (tasks/lesion-link-design.json, amendment 0015).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\analyze_lesion_link.py

Brain conditions W-R (lesion_link.py): the cue trace written and held in state W, read in state R; H healthy, L<d> the C01
lesion at dose d. Checks against the earlier records, before any test (a failed check stops the analysis):
- each healthy-trained readout in H-H and each lesion-trained readout in L<d>-L<d> equals its own verification at the 18
  trained histories: status and steps exactly, first action exactly, first-decision logits and features within 1e-4;
- lesion-trained replicates 1-3 in H-H equal the late restoration (runs/rescue/<run>.json) likewise;
- healthy-trained replicates 1-3 in L<d>-L<d> equal regime C after 0 updates, and the relearned readouts in L<d>-L<d> equal
  it after 100 updates (status and steps exactly);
- the replays reproduce regime C, the host checks of the states job passed, every readout job's features equal the states
  job's, the logits recomputed from the features and each readout equal the recorded ones within 1e-4, and the video
  combinations equal their readout jobs.
Outcome per readout and brain condition: trained-timing histories solved (of 18).
Tests (exact one-sided sign-flip over readouts, zero differences dropped; studykit.stats_ext.sign_flip):
- F15.link (Holm, 3): access follows the state that reads the trace rather than the state that wrote it:
  (1) healthy-trained readouts 0-9 at dose 1: solved(L1-H) - solved(H-L1) > 0;
  (2) lesion-trained readouts (regime B replicates 1-6 at the four doses, 24): solved(H-L<d>) - solved(L<d>-H) > 0;
  (3) relearned readouts (regime C replays, replicates 1-3 at the four doses, 12): solved(H-L<d>) - solved(L<d>-H) > 0.
- F15.link.secondary (BH, 9): test (1) at doses 0.5, 1.5 and 2 (3); test (2) per dose, 6 readouts each (4); healthy-trained
  readouts at dose 1, two-sided: solved(H-L1) - solved(L1-L1) and solved(L1-H) - solved(H-H) (2).
Descriptive (F15.link.descriptive): per readout and condition, the first-decision push P (mean over the two cues of the
logit of the cued choice minus the other choice; memory_tasks choice_targets) and the alignment of the readout-feature cue
difference with the readout's choice axis (cosine of z1 - z0 with w[2] - w[3]); per brain condition and timing (states job):
norms and cells of the cue difference of the stored trace, the decision-window membrane potential and the readout features,
cosines between conditions, differential spikes in the second half of the delay and the delay firing rate.
Outputs (results/): lesion_link_tests.csv, lesion_link_outcomes.csv, lesion_link_states.csv, lesion_link_pairs.csv.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
RUNS = STUDY / 'runs'
LINK = RUNS / 'lesion-link'
RESULTS = STUDY / 'results'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STUDY / 'code'))
from studykit import stats_ext as sx  # noqa: E402
import lesion_link as ll  # noqa: E402

TRAINED = {(d, n) for d in (0, 40, 160) for n in (0, 1, 4)}
TOL = 1e-4
CHOICE = (3, 2)                              # memory_tasks.prepare_memory: choice_targets of cue 0 and cue 1


def key(r):
    return int(r['cue']), int(r['delay_ms']), int(r['prefix_windows'])


def trained_rows(rollouts):
    return {key(r): r for r in rollouts if (int(r['delay_ms']), int(r['prefix_windows'])) in TRAINED}


def check(ok, what):
    if not ok:
        raise SystemExit(f'check failed: {what}')


def compare(mine, ref, what, logits=True, features=None):
    """mine: history rows of a readout job; ref: rollouts of an earlier record (with decisions or first_logits)."""
    ref = trained_rows(ref)
    check(len(ref) == 18 and len(mine) == 18, f'{what}: 18 trained histories')
    worst = 0.0
    for r in mine:
        q = ref[key(r)]
        check(q['status'] == r['status'] and int(q['steps']) == int(r['steps']), f'{what} {key(r)}: status or steps differ')
        if logits:
            if 'decisions' in q:
                first = q['decisions'][0]
                check(int(first['action']) == int(r['first_action']), f'{what} {key(r)}: first action differs')
                want = np.asarray(first['logits'], float)
            else:
                want = np.asarray(q['first_logits'], float)
            worst = max(worst, float(np.max(np.abs(want - np.asarray(r['first_logits'], float)))))
            if features is not None and 'decisions' in q:
                worst = max(worst, float(np.max(np.abs(np.asarray(q['decisions'][0]['features'], float) - features[key(r)]))))
    check(worst <= TOL, f'{what}: first-decision logits or features differ by {worst:g}')
    return worst


def push(logits, cases):
    """P: mean over the cues of the cued choice's logit minus the other choice's, at the first decision."""
    out = {}
    for t in sorted({(d, n) for _, d, n in cases}):
        i0, i1 = cases.index((0, *t)), cases.index((1, *t))
        out[t] = ((logits[i0][CHOICE[0]] - logits[i0][CHOICE[1]]) + (logits[i1][CHOICE[1]] - logits[i1][CHOICE[0]])) / 2
    return out


def alignment(features, head, cases):
    w = head['proposal.weight'].astype(np.float64)
    axis = w[CHOICE[1]] - w[CHOICE[0]]
    z = np.clip((features.astype(np.float64) - head['mean']) / head['scale'], -10, 10)
    out = {}
    for t in sorted({(d, n) for _, d, n in cases}):
        dz = z[cases.index((1, *t))] - z[cases.index((0, *t))]
        nz = np.linalg.norm(dz)
        out[t] = float(dz @ axis / (nz * np.linalg.norm(axis))) if nz > 0 else np.nan
    return out


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    if not (LINK / 'done.json').exists():
        raise SystemExit('runs/lesion-link is not complete (pull it hash-checked first)')
    design = json.loads(ll.DESIGN.read_text(encoding='utf-8'))
    states = json.loads((LINK / 'states.json').read_text(encoding='utf-8'))
    check(all(states['checks'].values()), 'host checks of the states job')
    cases = [tuple(c) for c in states['cases']]
    sfeat = np.load(LINK / 'states-features.npz')
    gate = dict(max_logit_or_feature_difference=0.0)
    # relearning replays
    for d in ll.DOSES:
        for r in (1, 2, 3):
            rec = json.loads((LINK / f'relearn-d{d:g}-r{r}.json').read_text(encoding='utf-8'))
            check(rec['reproduces_regime_c'] and all(rec['checks'].values()), f'relearn d{d:g} r{r}')
    outcomes, solved = [], {}
    for ro in ll.readouts():
        job = json.loads((LINK / f"readout-{ro['name']}.json").read_text(encoding='utf-8'))
        z = np.load(LINK / f"readout-{ro['name']}.npz")
        head = {k.split('|', 1)[1]: z[k].astype(np.float64) for k in z.files if k.startswith('head|')}
        check(job['checked_against_states'], f"{ro['name']}: features checked against the states job on the host")
        for c in ll.conditions(ro):
            cn = ll.name(c)
            rows = job['histories'][cn]
            check([key(r) for r in rows] == cases, f"{ro['name']} {cn}: history order")
            feats = z[f'{cn}|features']
            check(np.array_equal(feats, sfeat[cn]), f"{ro['name']} {cn}: features equal the states job")
            lg = np.asarray([r['first_logits'] for r in rows], float)
            zz = np.clip((feats.astype(np.float64) - head['mean']) / head['scale'], -10, 10)
            again = zz @ head['proposal.weight'].T + head['proposal.bias']
            check(np.max(np.abs(again - lg)) <= TOL, f"{ro['name']} {cn}: recomputed logits")
            fmap = {key(r): feats[i].astype(np.float64) for i, r in enumerate(rows)}
            # the earlier records
            src = ro['source']
            if ro['kind'] == 'healthy-trained' and cn == 'H-H':
                ver = json.loads((SUPP / 'runs' / src / 'verification.json').read_text(encoding='utf-8'))
                gate['max_logit_or_feature_difference'] = max(gate['max_logit_or_feature_difference'],
                                                              compare(rows, ver['results']['rollouts'], f"{ro['name']} verification", features=fmap))
            if ro['kind'] == 'lesion-trained' and cn == f"{ll.L(ro['dose'])}-{ll.L(ro['dose'])}":
                ver = json.loads((RUNS / src / 'verification.json').read_text(encoding='utf-8'))
                gate['max_logit_or_feature_difference'] = max(gate['max_logit_or_feature_difference'],
                                                              compare(rows, ver['results']['rollouts'], f"{ro['name']} verification", features=fmap))
            if ro['kind'] == 'lesion-trained' and cn == 'H-H' and int(src.rsplit('-r', 1)[1]) <= 3:
                res = json.loads((RUNS / 'rescue' / f'{src}.json').read_text(encoding='utf-8'))
                gate['max_logit_or_feature_difference'] = max(gate['max_logit_or_feature_difference'],
                                                              compare(rows, res['results']['rollouts'], f"{ro['name']} late restoration"))
            if ro['kind'] in ('healthy-trained', 'relearned') and c[0] == c[1] and c[0] != 'H':
                rep = int(src.rsplit('-r', 1)[1])
                if 1 <= rep <= 3 and (ro['kind'] == 'healthy-trained' or c[0] == ll.L(ro['dose'])):
                    dose = float(c[0][1:])
                    rc = json.loads((RUNS / 'regime-c' / f'{ll.regime_c(dose, rep)}.json').read_text(encoding='utf-8'))
                    at = '0' if ro['kind'] == 'healthy-trained' else '100'
                    compare(rows, rc['assessments'][at]['rollouts'], f"{ro['name']} {cn} regime C after {at} updates", logits=False)
            p, al = push(lg, cases), alignment(feats, head, cases)
            n_solved = sum(r['status'] == 'solved' for r in rows)
            solved[(ro['name'], cn)] = n_solved
            outcomes.append(dict(readout=ro['name'], kind=ro['kind'], learned_in=ro['learned'], dose=ro['dose'],
                                 replicate=int(src.rsplit('-r', 1)[1]), condition=cn, written_in=c[0], read_in=c[1],
                                 solved=n_solved, histories=len(rows), push_mean=float(np.mean(list(p.values()))),
                                 alignment_mean=float(np.nanmean(list(al.values()))),
                                 **{f'push_d{t[0]}_p{t[1]}': v for t, v in p.items()},
                                 **{f'alignment_d{t[0]}_p{t[1]}': v for t, v in al.items()},
                                 statuses=';'.join(f"{r['cue']}/{r['delay_ms']}/{r['prefix_windows']}:{r['status']}" for r in rows)))
    video = json.loads((LINK / 'video.json').read_text(encoding='utf-8'))
    check(all(c['equals_readout_job'] for c in video['combinations']), 'video combinations equal their readout jobs')

    # tests
    def diffs(names, a, b):
        return np.array([solved[(n, a(n))] - solved[(n, b(n))] for n in names], float)

    x1 = [f'x1-r{r}' for r in range(10)]
    bro = {d: [f'b-d{d:g}-r{r}' for r in range(1, 7)] for d in ll.DOSES}
    cro = {d: [f'c-d{d:g}-r{r}' for r in (1, 2, 3)] for d in ll.DOSES}
    dose_of = {n: d for d in ll.DOSES for n in bro[d] + cro[d]}

    def matched(n):              # the reading state matches the learning state; the trace was written in the other
        return f"H-{ll.L(dose_of[n])}"

    def written(n):
        return f"{ll.L(dose_of[n])}-H"

    tests = []

    def add(family, test, d, alternative='greater'):
        sf = sx.sign_flip(d, alternative=alternative, ci=False)
        tests.append(dict(family=family, test=test, n=len(d), n_used=sf['n_used'], estimate=float(np.mean(d)),
                          alternative=alternative, p_value=sf['p_value'], min_attainable_p=sf['min_attainable_p']))

    add('F15.link', 'healthy-trained readouts, dose 1: solved(lesion-written, read healthy) - solved(healthy-written, read under the lesion) > 0',
        diffs(x1, lambda n: 'L1-H', lambda n: 'H-L1'))
    allb = [n for d in ll.DOSES for n in bro[d]]
    add('F15.link', 'lesion-trained readouts, 4 doses: solved(healthy-written, read under the lesion) - solved(lesion-written, read healthy) > 0',
        diffs(allb, matched, written))
    allc = [n for d in ll.DOSES for n in cro[d]]
    add('F15.link', 'relearned readouts, 4 doses: solved(healthy-written, read under the lesion) - solved(lesion-written, read healthy) > 0',
        diffs(allc, matched, written))
    for d in (0.5, 1.5, 2.0):
        add('F15.link.secondary', f'healthy-trained readouts, dose {d:g}: solved(L{d:g}-H) - solved(H-L{d:g}) > 0',
            diffs(x1, lambda n, d=d: f'{ll.L(d)}-H', lambda n, d=d: f'H-{ll.L(d)}'))
    for d in ll.DOSES:
        add('F15.link.secondary', f'lesion-trained readouts, dose {d:g}: solved(H-L{d:g}) - solved(L{d:g}-H) > 0', diffs(bro[d], matched, written))
    add('F15.link.secondary', 'healthy-trained readouts, dose 1: solved(H-L1) - solved(L1-L1), two-sided',
        diffs(x1, lambda n: 'H-L1', lambda n: 'L1-L1'), 'two-sided')
    add('F15.link.secondary', 'healthy-trained readouts, dose 1: solved(L1-H) - solved(H-H), two-sided',
        diffs(x1, lambda n: 'L1-H', lambda n: 'H-H'), 'two-sided')

    RESULTS.mkdir(exist_ok=True)
    write(RESULTS / 'lesion_link_tests.csv', tests)
    write(RESULTS / 'lesion_link_outcomes.csv', outcomes)
    write(RESULTS / 'lesion_link_states.csv', states['per_condition'])
    write(RESULTS / 'lesion_link_pairs.csv', states['pairs'])
    (RESULTS / 'lesion_link_checks.json').write_text(json.dumps(dict(gate, design_sha256=ll.sha(ll.DESIGN), passed=True), indent=1),
                                                     encoding='utf-8', newline='\n')
    for t in tests:
        print(f"{t['family']:20s} n={t['n']:2d} mean {t['estimate']:+.2f} p {t['p_value']:.3g}  {t['test']}")
    print(f"checks passed; largest first-decision difference from the earlier records {gate['max_logit_or_feature_difference']:.2g}")


if __name__ == '__main__':
    main()
