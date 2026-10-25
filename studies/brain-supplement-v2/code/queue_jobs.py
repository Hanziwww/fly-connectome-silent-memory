"""Job lists for the GPU work queues once a second host joined (plan v2; 2026-10-19). Run on the host it is for.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/queue_jobs.py --host x6 > runtime/queue/jobs.tsv
    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/queue_jobs.py --host first > runtime/queue/jobs.tsv

One line per job: id, done marker (the job is skipped once it exists), estimated minutes, required markers (comma
separated or '-'; the job waits until they exist), shell command (run from the project root by queue.sh with its GPU
in CUDA_VISIBLE_DEVICES). Jobs are written longest first (except those with requirements, which come last), so the
GPUs that pull from the queue finish together.
- x6 (six RTX 5090s): the training campaign's remaining jobs (X1 M0 joint, regime B joint, the C04/C13/C16 controls,
  X1 S1 and S2; runtime/campaign/jobs.txt from campaign_jobs.py), then every second-and-later batch (G3a, X5, X4,
  X6/X7, X3, the Phase 4 dose series and memory curve, the X8 pilot, X9, X2, X10 replicates 1-3) and the rescue of
  the regime B joint models.
- first (one RTX 5090): the rest of regime B readout-only, the rescue of the regime B readout-only models, regime C,
  and X10 replicates 4-5.
Training jobs train, then verify in a separate process, as campaign.sh does.
Jobs whose id starts with PRIORITY are written first, in list order (a worker skips a job whose requirements are not met
yet, so they cannot block the rest): the structural-variant runs (O12, O14) are needed before the write-up, while the
first host's X10 replicates only complete X10, whose replicates 1-3 finish later on the six-GPU host anyway.
T-DY2 (robust readout training, added 2026-10-20; tasks/dy2-design.json): the 45 healthy runs and their evaluations on
x6; the evaluations of the X1 readout-only and regime B C01 dose 0.5 references, then the 5 C01-trained runs and their
evaluations, on the first host, where those reference runs are. Each evaluation waits for its run's verification.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = '.venv-linux/bin/python -B -X utf8'
S = 'studies/brain-supplement-v2'
P = 'studies/brain-pathology-v1'
PRIORITY = ('verify-', 'phase2-variants', 'o14-', 'phase4-dose-low', 'dy1-', 'phase4-dose-micro', 'dy2-')
# Moved from x6 to the first host on 2026-10-20 06:58 PDT, when the first host's queue drained: the X10 replicate-3 jobs
# no x6 worker had claimed (the other four r3 maps stay on x6).
MOVED_TO_FIRST = frozenset(f'difficulty-{m}-joint-r3' for m in ('D6-T0', 'D6-T1', 'D2-T1', 'D5-T1', 'D5-T2', 'D4-T0', 'D5-T0', 'D3-T0', 'D3-T1', 'D3-T2', 'D4-T1', 'D4-T2', 'D2-T0', 'D2-T2', 'D0-T0', 'D0-T1', 'D0-T2', 'D1-T0', 'D1-T1', 'D1-T2'))


def train(jid, folder, script, args, minutes, requires='-'):
    cmd = (f'mkdir -p {folder} && {PY} {script} {args} --out {folder} > {folder}/train.log 2>&1 && '
           f'{PY} {script} {args} --out {folder} --verify > {folder}/verify.log 2>&1')
    return (jid, f'{folder}/verification.json', minutes, requires, cmd)


def script(jid, path, marker, minutes, args='', requires='-'):
    return (jid, marker, minutes, requires, f'{PY} {path} {args} > runtime/queue/logs/{jid}.log 2>&1')


def campaign(ids):
    """Training jobs of runtime/campaign/jobs.txt (id, folder, model, replicate, lesion) with the given ids."""
    rows = {}
    for line in (ROOT / 'runtime' / 'campaign' / 'jobs.txt').read_text(encoding='utf-8').splitlines():
        if line.strip():
            jid, folder, model, rep, lesion = line.split('\t')
            rows[jid] = (folder, model, rep, lesion)
    out = []
    for jid in ids:
        folder, model, rep, lesion = rows[jid]
        les = '' if lesion == '-' else f' --lesion {lesion}'
        minutes = 65 if 'S2' in jid else 55 if 'S1' in jid else 35 if 'joint' in jid else 6
        out.append(train(jid, folder, f'{S}/code/replicate_training.py', f'--model {model} --replicate {rep}{les}', minutes))
    return out


# Training seconds of the frozen Fig. 2 runs (studies/brain-sokoban-v1/runs/difficulty-<map>/result.json, which stay on the
# local machine): the queue's duration estimates for X10.
FROZEN_SECONDS = {'D0-T0': 89, 'D0-T1': 87, 'D0-T2': 93, 'D1-T0': 84, 'D1-T1': 81, 'D1-T2': 85, 'D2-T0': 182, 'D2-T1': 867,
                  'D2-T2': 233, 'D3-T0': 438, 'D3-T1': 427, 'D3-T2': 431, 'D4-T0': 584, 'D4-T1': 447, 'D4-T2': 467,
                  'D5-T0': 537, 'D5-T1': 810, 'D5-T2': 735, 'D6-T0': 1710, 'D6-T1': 1257, 'D6-T2': 2295, 'D7-T0': 4595,
                  'D7-T1': 2842, 'D7-T2': 3096}


def x10(replicates):
    out = []
    for r in replicates:
        for t in range(8):
            for k in range(3):
                m = f'D{t}-T{k}'
                minutes = int(FROZEN_SECONDS[m] / 60) + 3
                out.append(train(f'difficulty-{m}-joint-r{r}', f'{S}/runs/difficulty-{m}-joint-r{r}', f'{S}/code/sokoban_replicate.py',
                                 f'--map {m} --condition joint --replicate {r} --module difficulty', minutes))
    return out


# Re-verification (added 2026-10-19): X1 S2 replicates that trained, then failed verification before fc0cce6 (the
# frozen supplement's verification adapter). Each job waits for <run>/reverify.requested, written by hand once the failed
# verification files are moved aside (logged in queue.log); its done marker is the new verification.json.
REVERIFY = {f'x1-S2-head_only-r{r}': ('supp-S2-context-brain', r) for r in (0, 1, 2, 3)}


# T-DY2 stage 1 (F14.DY.robust; amendment 0003 of brain-pathology-v1): training conditions, the expected slowest first.
DY2 = ('in5', 'mem1', 'in1-th0.1', 'in1', 'th0.5', 'mem0.1', 'th0.1', 'in0.2', 'th0.001')
DY2_LESION = f'{P}/runs/regime-b-lesions/C01-dose0.5.json'


def dy2_train(name, replicate, extra=''):
    return train(f'dy2-{name}-r{replicate}', f'{P}/runs/dy2-{name}-r{replicate}', f'{S}/code/replicate_training.py',
                 f'--model remote-dynamic-M0-head_only --replicate {replicate}{extra}', 60)


def dy2_eval(folder, minutes, lesion=None):
    run = folder.split('/')[-1]
    return script(f'dy2-eval-{run}', f'{P}/code/dy2_evaluate.py', f'{P}/runs/dy2-eval/{run}.json', minutes,
                  args=f'--run {folder}' + (f' --lesion {lesion}' if lesion else ''), requires=f'{folder}/verification.json')


def reverify_jobs():
    out = []
    for run, (model, rep) in REVERIFY.items():
        folder = f'{S}/runs/{run}'
        out.append((f'verify-{run}', f'{folder}/verification.json', 20, f'{folder}/reverify.requested',
                    f'{PY} {S}/code/replicate_training.py --model {model} --replicate {rep} --out {folder} --verify > {folder}/verify.log 2>&1'))
    return out


def x6_jobs():
    jobs = campaign([f'x1-M0-joint-r{r}' for r in range(1, 10)]          # r0 ran on the first host (campaign.log)
                    + [f'b-C01-dose{d:g}-M0-joint-r{r}' for d in (0.5, 1.0, 1.5, 2.0) for r in (1, 2, 3)]
                    + [f'b-{c}-dose1-M0-head_only-r1' for c in ('C04', 'C13', 'C16')]
                    + [f'x1-{f}-head_only-r{r}' for f in ('S1', 'S2') for r in range(7)])
    jobs += [script('g3a-dependency', f'{P}/code/phase3_dependency.py', f'{P}/runs/phase3-dependency/done.json', 180),
             script('x5-swaps', f'{S}/code/x5_swaps.py', f'{S}/runs/x5-swaps/done.json', 40),
             script('x4-scrambles', f'{S}/code/x4_scrambles.py', f'{S}/runs/x4-scrambles/done.json', 300),
             script('x6-x7-ensemble', f'{S}/code/x6_x7_ensemble.py', f'{S}/runs/x6-x7/done.json', 90),
             script('phase4-dose-series', f'{P}/code/phase4_dose_series.py', f'{P}/runs/phase4-dose/done.json', 180),
             script('phase4-memory-curve', f'{P}/code/phase4_memory_curve.py', f'{P}/runs/phase4-memory-curve/done.json', 60),
             script('x9-replays', f'{S}/code/x9_replays.py', f'{S}/runs/x9-replays/done.json', 150)]
    jobs += [train(f'x3-M0-head_only-r{r}', f'{S}/runs/x3-M0-head_only-r{r}', f'{S}/code/x3_jittered_training.py', f'--replicate {r}', 8)
             for r in range(10)]
    x8_minutes = {('head_only', m): 20 for m in ('D1-T0', 'D3-T0', 'D5-T0', 'D7-T0')}
    x8_minutes.update({('joint', 'D1-T0'): 3, ('joint', 'D3-T0'): 8, ('joint', 'D5-T0'): 12, ('joint', 'D7-T0'): 90})
    jobs += [train(f'x8-{m}-{a}-r{r}', f'{S}/runs/x8-{m}-{a}-r{r}', f'{S}/code/x8_neutral_training.py', f'--map {m} --arm {a} --replicate {r}',
                   x8_minutes[(a, m)]) for a in ('head_only', 'joint') for m in ('D1-T0', 'D3-T0', 'D5-T0', 'D7-T0') for r in range(3)]
    d7 = dict(frozen=2, board_mlp=15, entry_ce=60, alternating=60, neutral=90, rewired=80, weight_shuffle=80, no_recovery=65, brain_only=145)
    jobs += [train(f'controls-D7-T0-{a}-r0', f'{S}/runs/controls-D7-T0-{a}-r0', f'{S}/code/sokoban_replicate.py',
                   f'--map D7-T0 --condition {a} --replicate 0 --module controls', m) for a, m in d7.items()]
    x2_minutes = {'M1': 6, 'M2': 6, 'S1-mirror': 55, 'S1-minus45': 55, 'S2-mirror': 65, 'S2-minus48': 65}
    jobs += [train(f'x2-{sc}-r{r}', f'{S}/runs/x2-{sc}-r{r}', f'{S}/code/x2_scene_training.py', f'--scene {sc} --replicate {r}', m)
             for sc, m in x2_minutes.items() for r in range(7)]
    jobs += [j for j in x10((1, 2, 3)) if j[0] not in MOVED_TO_FIRST]
    joint_b = ','.join(f'{P}/runs/b-C01-dose{d:g}-M0-joint-r{r}/verification.json' for d in (0.5, 1.0, 1.5, 2.0) for r in (1, 2, 3))
    jobs.append(script('rescue-M0-joint', f'{P}/code/phase3_rescue.py', f'{P}/runs/rescue/done-M0-joint.json', 30,
                       args='--learners M0-joint', requires=joint_b))
    jobs += reverify_jobs()
    # Regime B extension to 6 replicates (amendments/regime-b-n.json, committed 722a73d): the joint learner here.
    jobs += campaign([f'b-C01-dose{d:g}-M0-joint-r{r}' for d in (0.5, 1.0, 1.5, 2.0) for r in (4, 5, 6)])
    # T-DY1 (dynamics solutions plan; F14.DY.sensitivity, amendment 0002 of brain-pathology-v1; added 2026-10-20)
    jobs.append(script('dy1-sensitivity', f'{P}/code/dy1_sensitivity.py', f'{P}/runs/dy1-sensitivity/done.json', 30))
    jobs.append(script('phase4-dose-micro', f'{P}/code/phase4_dose_series.py', f'{P}/runs/phase4-dose-micro/done.json', 30,
                       args='--set micro'))
    # T-DY2 stage 1 (added 2026-10-20): the healthy runs, then their evaluations.
    jobs += [dy2_train(c, r, f' --perturb {c}') for c in DY2 for r in range(5)]
    jobs += [dy2_eval(f'{P}/runs/dy2-{c}-r{r}', 15) for c in DY2 for r in range(5)]
    return jobs


def first_jobs():
    jobs = campaign([f'b-C01-dose2-M0-head_only-r{r}' for r in (2, 3)])
    head_b = ','.join(f'{P}/runs/b-C01-dose{d:g}-M0-head_only-r{r}/verification.json' for d in (0.5, 1.0, 1.5, 2.0) for r in (1, 2, 3))
    jobs.append(script('rescue-M0-head_only', f'{P}/code/phase3_rescue.py', f'{P}/runs/rescue/done-M0-head_only.json', 30,
                       args='--learners M0-head_only', requires=head_b))
    jobs.append(script('regime-c', f'{P}/code/phase3_regime_c.py', f'{P}/runs/regime-c/done.json', 120))
    # O12, circuit level (added 2026-10-19): C4-C6 under the structural variants; on the host of the frozen Phase 2 run.
    jobs.append(script('phase2-variants', f'{P}/code/phase2_variants.py', f'{P}/runs/phase2-variants/done.json', 60))
    jobs += x10((4, 5)) + [j for j in x10((3,)) if j[0] in MOVED_TO_FIRST]
    # Regime B extension to 6 replicates (amendments/regime-b-n.json): the readout-only learner here, as replicates 1-3.
    jobs += campaign([f'b-C01-dose{d:g}-M0-head_only-r{r}' for d in (0.5, 1.0, 1.5, 2.0) for r in (4, 5, 6)])
    # O14, task level (added 2026-10-19): M0 readout-only replicates 0-2 trained under each variant, after O12; then C01.
    o14 = []
    for v in ('receptor-none', 'same-type-0.05', 'same-type-0.2', 'same-type-off', 'conn-cap-30', 'conn-cap-120', 'conn-cap-off'):
        for r in (0, 1, 2):
            o14.append(train(f'o14-{v}-r{r}', f'{P}/runs/o14-{v}-r{r}', f'{S}/code/replicate_training.py',
                             f'--model remote-dynamic-M0-head_only --replicate {r} --variant {v}', 15,
                             requires=f'{P}/runs/phase2-variants/done.json'))
    jobs += o14
    jobs.append(script('o14-lesion', f'{P}/code/variant_lesion.py', f'{P}/runs/o14-lesion/done.json', 30,
                       requires=','.join(j[1] for j in o14)))
    # Dose series low extension (post hoc, amendment 0001 of brain-pathology-v1; added 2026-10-19), after O14.
    jobs.append(script('phase4-dose-low', f'{P}/code/phase4_dose_series.py', f'{P}/runs/phase4-dose-low/done.json', 75, args='--set low'))
    # T-DY2 stage 1 (added 2026-10-20): the reference evaluations first, then the C01-trained runs and their evaluations.
    jobs += [dy2_eval(f'{S}/runs/x1-M0-head_only-r{r}', 15) for r in range(10)]
    jobs += [dy2_eval(f'{P}/runs/b-C01-dose0.5-M0-head_only-r{r}', 5, DY2_LESION) for r in range(1, 7)]
    jobs += [dy2_train('C01-dose0.5-th0.5', r, f' --lesion {DY2_LESION} --perturb th0.5') for r in range(1, 6)]
    jobs += [dy2_eval(f'{P}/runs/dy2-C01-dose0.5-th0.5-r{r}', 5, DY2_LESION) for r in range(1, 6)]
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', required=True, choices=('x6', 'first'))
    jobs = x6_jobs() if ap.parse_args().host == 'x6' else first_jobs()
    ids = [j[0] for j in jobs]
    if len(ids) != len(set(ids)):
        raise SystemExit('duplicate job ids')
    front = [j for j in jobs if j[0].startswith(PRIORITY)]
    rest = [j for j in jobs if not j[0].startswith(PRIORITY)]
    free = sorted((j for j in rest if j[3] == '-'), key=lambda j: -j[2])
    waiting = [j for j in rest if j[3] != '-']
    for jid, marker, minutes, requires, cmd in front + free + waiting:
        sys.stdout.write(f'{jid}\t{marker}\t{minutes}\t{requires}\t{cmd}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
