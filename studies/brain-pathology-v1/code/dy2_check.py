"""T-DY2 wrapper check (GPU host, on an idle GPU; before any T-DY2 run). Writes runtime/dy2-check/report.json and exits 1
if any check fails.

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/dy2_check.py [--model-pt PATH] [--train-seconds 420]

1. Pure parts: every condition's acceptance and held-out draws are reproducible and distinct from each other; merge()
   on synthetic assessments; NoCache.
2. Off is frozen: with the patches installed and no draw active, the frozen assessment (trained timings, then all 35)
   equals the unpatched one exactly (statuses, steps, actions, probe logits, margin), also after draws were active.
3. Draws act and repeat: input noise, membrane noise and a threshold draw each give the same assessment twice and
   differ from the unperturbed one in at least one probe logit.
4. Threshold arithmetic: a draw with theta -1e-3 mV equals operators.apply(threshold_offset, every cell, -1e-3).
5. Training episodes: the same update and episode give identical features, another update different ones, and ending
   the update restores the unperturbed features.
6. End to end: replicate_training.py --perturb in1-th0.1 for --train-seconds, stopped through the run's STOP file, then
   resumed for as long again (in runtime/dy2-check/train-in1-th0.1, cleared first): config record, merged assessments
   (3 draws each), episode count, continued history, and the worker's clock at a third of real time.
Timings (seconds per assessment unperturbed and per draw, per update) go into the report for the queue estimates.
The policy is the M0 readout-only model (remote-dynamic-M0-head_only) with the P05 readout, or --model-pt (a trained pair).
"""
import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
OUT = ROOT / 'runtime' / 'dy2-check'
MODEL = 'remote-dynamic-M0-head_only'


def outcome(res):
    rolls = [(r['cue'], r['delay_ms'], r['prefix_windows'], r['status'], r['steps'], tuple(d['action'] for d in r['decisions']))
             for r in res['rollouts']]
    probes = [(p['delay_ms'], p['prefix_windows'], tuple((x['choice'], tuple(x['logits'])) for x in p['records'])) for p in res['probes']]
    return rolls, probes, res['min_choice_margin']


def pure_checks(dy2):
    out = {}
    keys = set()
    for name in dy2.CONDITIONS:
        a, b = dy2.accept_draws(name), dy2.accept_draws(name)
        h, h2 = dy2.heldout_draws(name), dy2.heldout_draws(name)
        assert a == b and h == h2, name
        keys.update(d['key'] for d in a + h)
        sd = dy2.CONDITIONS[name].get('threshold_sd_mv', 0.0)
        assert [d['theta_mv'] for d in a] == [-sd, sd], name
        assert json.loads(json.dumps(dy2.describe(name))) == dy2.describe(name)
    assert len(keys) == 4 * len(dy2.CONDITIONS), 'draw keys are not distinct'
    out['draw_keys'] = len(keys)

    def fake(statuses, corrects, margin):
        rolls = [dict(cue=i % 2, delay_ms=0, prefix_windows=i // 2, status=s, decisions=[]) for i, s in enumerate(statuses)]
        probes = [dict(delay_ms=0, prefix_windows=i, records=[dict(correct=c0), dict(correct=c1)], correct=c0 + c1)
                  for i, (c0, c1) in enumerate(corrects)]
        return dict(probes=probes, rollouts=rolls, probe_correct=sum(p['correct'] for p in probes), probe_total=2 * len(probes),
                    solved=sum(s == 'solved' for s in statuses), total=len(statuses), min_choice_margin=margin)
    base = fake(['solved', 'solved', 'blocked', 'solved'], [(True, True), (True, False)], 0.5)
    d1 = fake(['solved', 'action_limit', 'solved', 'solved'], [(True, True), (True, True)], 0.2)
    m = dy2.merge(base, [d1])
    assert m['solved'] == 2 and m['total'] == 4 and m['probe_correct'] == 3 and m['min_choice_margin'] == 0.2
    assert [r['status'] for r in m['rollouts']] == ['solved', 'action_limit', 'blocked', 'solved']
    assert m['rollouts'][1]['draw_status'] == ['solved', 'action_limit'] and len(m['draws']) == 2
    c = dy2.NoCache()
    c['k'] = 1
    assert c.get('k') is None and len(c) == 0
    out['merge'] = 'ok'
    return out


def timed(fn, *a, **k):
    t0 = time.time()
    r = fn(*a, **k)
    return r, time.time() - t0


def gpu_checks(dy2, legacy, operators, torch, model_pt, report):
    config = json.loads((MECH / 'inputs' / 'models' / MODEL / 'source-config.json').read_text(encoding='utf-8'))
    task = next(t for t in legacy.prepare_memory() if t['id'] == config['task'])
    policy = legacy.MemoryPolicy(config, OUT / 'meta')
    if model_pt:
        saved = torch.load(model_pt, map_location='cpu', weights_only=True)
        policy.load(saved['pair'])
    t = report['timing'] = {}
    r0, t['assess_trained_unpatched'] = timed(legacy.assess, policy, task, False)
    r0full, t['assess_full_unpatched'] = timed(legacy.assess, policy, task, True)
    every = np.arange(policy.encoder.brain.n)
    operators.apply(policy.encoder, [operators.spec('threshold_offset', every, -1e-3, 'dy2 check')])
    r_op = legacy.assess(policy, task, False)
    operators.apply(policy.encoder, [])                 # identity: back to the unbound frozen brain
    assert outcome(legacy.assess(policy, task, False)) == outcome(r0), 'operators at zero dose differ from the frozen brain'
    pert = dy2.Perturbation(policy.encoder)
    checks = report['checks'] = {}
    r1, t['assess_trained_off'] = timed(legacy.assess, policy, task, False)
    checks['off_trained_equals_frozen'] = outcome(r1) == outcome(r0)
    checks['off_full_equals_frozen'] = outcome(legacy.assess(policy, task, True)) == outcome(r0full)
    pert.set_draw(dict(input_hz=0.0, membrane_sd_mv=0.0, theta_mv=-1e-3, key='dy2-check|theta'))
    checks['theta_equals_operator'] = outcome(legacy.assess(policy, task, False)) == outcome(r_op)
    pert.set_off()
    for name, draw in (('input', dy2.accept_draws('in1')[0]), ('membrane', dy2.accept_draws('mem0.1')[0]),
                       ('threshold', dy2.accept_draws('th0.1')[0]), ('combined', dy2.accept_draws('in1-th0.1')[1])):
        pert.set_draw(draw)
        a, t[f'assess_trained_{name}'] = timed(legacy.assess, policy, task, False)
        pert.set_draw(draw)
        b = legacy.assess(policy, task, False)
        pert.set_off()
        checks[f'{name}_repeats'] = outcome(a) == outcome(b)
        checks[f'{name}_acts'] = outcome(a)[1] != outcome(r0)[1]
        report.setdefault('draw_effects', {})[name] = dict(solved=a['solved'], total=a['total'], probe_correct=a['probe_correct'],
                                                         unperturbed_solved=r0['solved'], unperturbed_probe_correct=r0['probe_correct'])
    checks['off_after_draws_equals_frozen'] = outcome(legacy.assess(policy, task, False)) == outcome(r0)
    merged, t['merged_assess_in1-th0.1'] = timed(dy2.merged_assess(legacy.assess, pert, dy2.accept_draws('in1-th0.1')), policy, task, False)
    checks['merged_has_three_draws'] = len(merged['draws']) == 3 and merged['draws'][0]['solved'] == r0['solved']
    # training episodes (the feature pass of the readout-only training step)
    training = dy2.Training(pert, 'in1-th0.1', 1)
    frames = legacy.frames(task, 0, 160, 4, True)

    def features():
        policy.reset()
        with torch.no_grad():
            return torch.cat([policy.advance(f['board'], f['ms'])[1] for f in frames]).cpu()

    unperturbed = features()
    training.begin(7)
    training.episode()
    f1 = features()
    training.begin(7)
    training.episode()
    f2 = features()
    training.begin(8)
    training.episode()
    f3 = features()
    training.end()
    f4 = features()
    checks['episode_repeats'] = bool(torch.equal(f1, f2))
    checks['episode_differs_by_update'] = not bool(torch.equal(f1, f3))
    checks['episode_perturbs'] = not bool(torch.equal(f1, unperturbed))
    checks['end_restores'] = bool(torch.equal(f4, unperturbed))
    report['membrane'] = dict(a_m=pert.a_m, step_factor=pert.step_factor)
    del policy, pert, training
    gc.collect()
    torch.cuda.empty_cache()


def end_to_end(seconds, report, torch):
    run = OUT / 'train-in1-th0.1'
    if run.exists():
        shutil.rmtree(run)                              # this check's own scratch run from an earlier check
    py = [sys.executable, '-B', '-X', 'utf8', str(ROOT / 'studies' / 'brain-supplement-v2' / 'code' / 'replicate_training.py'),
          '--model', MODEL, '--replicate', '1', '--out', str(run), '--perturb', 'in1-th0.1']
    def lines(name):
        p = run / name
        return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []

    sessions, before, clock = [], 0, 0.0
    for k in range(2):
        (run / 'STOP').unlink(missing_ok=True)
        run.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(run / f'train-{k}.log', 'w', encoding='utf-8') as log:
            proc = subprocess.Popen(py, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            time.sleep(seconds)
            (run / 'STOP').write_text('dy2 check', encoding='utf-8')
            code = proc.wait(timeout=1800)
        wall = time.time() - t0
        grads, hist = lines('gradients.jsonl'), lines('history.jsonl')
        rep = json.loads((run / 'replicate.json').read_text(encoding='utf-8'))
        now = float(torch.load(run / 'resume.pt', map_location='cpu', weights_only=True)['state']['elapsed'])
        sessions.append(dict(exit_code=code, wall_seconds=wall, updates=len(grads) - before, assessments=len(hist),
                             clock_seconds=now - clock, real_seconds=rep.get('real_seconds'),
                             episodes=rep['applied']['perturbation']['episodes'],
                             status=json.loads((run / 'result.json').read_text(encoding='utf-8'))['status']))
        before, clock = len(grads), now
    (run / 'STOP').unlink(missing_ok=True)
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    last = lines('behavior.jsonl')[-1]
    grads = lines('gradients.jsonl')
    report['end_to_end'] = dict(sessions=sessions, gradient_lines=len(grads), last_draws=last.get('draws'),
                                config_perturbation=config.get('training_perturbation', {}).get('name'))
    s0, s1 = sessions
    c = report['checks']
    c['e2e_config_records_condition'] = config.get('id', '').endswith('-in1-th0.1') and config['training_perturbation']['name'] == 'in1-th0.1'
    c['e2e_merged_assessments'] = len(last.get('draws') or []) == 3
    c['e2e_stopped_and_resumed'] = s0['status'] == 'stopped' and s1['status'] == 'stopped' and s0['updates'] > 0 and s1['updates'] > 0
    c['e2e_episodes_two_per_update'] = s0['episodes'] == 2 * s0['updates'] and s1['episodes'] == 2 * s1['updates']
    c['e2e_updates_logged_once'] = [g['update'] for g in grads] == list(range(1, len(grads) + 1))
    c['e2e_clock_a_third'] = all(0.5 * x['real_seconds'] / 3 <= x['clock_seconds'] <= x['real_seconds'] / 3 + 1 for x in sessions)
    c['e2e_config_clock_factor'] = config['training_perturbation'].get('clock_factor') == 3.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-pt', default=None)
    ap.add_argument('--train-seconds', type=float, default=420)
    args = ap.parse_args()
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    import mechanism_base  # noqa: F401  (path setup: flybrain, sokoban, frozen study code)
    import torch
    import memory as legacy
    import operators
    import dy2_perturbation as dy2
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(started_unix=time.time(), model_pt=args.model_pt, gpu=torch.cuda.get_device_name())
    report['pure'] = pure_checks(dy2)
    gpu_checks(dy2, legacy, operators, torch, args.model_pt, report)
    end_to_end(args.train_seconds, report, torch)
    report['ok'] = all(report['checks'].values())
    report['finished_unix'] = time.time()
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8', newline='\n')
    print(json.dumps(dict(ok=report['ok'], checks=report['checks'], timing=report['timing']), indent=1))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
