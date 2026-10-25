"""Phase 4: linear response of the decision-time state to the cue's adaptation trace (the timing lock as physics).

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/phase4_linear_response.py

Phase 1B (fu4) found that at the end of the delay the cue difference lives almost entirely in spike-frequency
adaptation (99.9% of the voltage difference; no differential spiking), and that input after the delay turns it back into
synaptic differences at the first decision. If so, the decision-time cue code is the network's response to an
adaptation perturbation:   dF(d) ~= L(d) da(d),   da(d) = a1(d) - a0(d) ~ exp(-d / tau_a) da_0,
where L(d) is the linear response of the first-decision features to an adaptation perturbation at the end of a delay d,
evaluated around the cue-0 history. Then
1. (sufficiency) adding da(d) (and nothing else) to the cue-0 state at the end of the delay reproduces the cue-1 decision
   features: cos(dF_pred, dF_actual) ~ 1 and the readout's differential margin is reproduced;
2. (linearity) responses to eps x da(d) scale with eps;
3. (operator drift) applying the same unit-norm trace pattern at different delays gives decision-feature responses whose
   direction changes with the delay: the code drifts because the network's operating point at decision time (the
   cue-independent, decaying adaptation of all cells) changes, not because the trace pattern changes.
Model: M0 readout-only (P05 brain, M0 task), prefix 4 and 0, all 7 delays. Writes runs/g4-linear-response/result.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
OUT = ROOT / 'studies' / 'brain-supplement-v2' / 'runs' / 'g4-linear-response'
DELAYS = (0, 20, 40, 80, 160, 320, 640)
EPS = (0.25, 0.5, 1.0)
MODEL = 'remote-dynamic-M0-head_only'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(MECH / 'code'))
    import mechanism_base as mb
    import torch
    from mechanism_model import Frozen, tasklib
    from game import state_from
    lib = tasklib()
    model = mb.model_by_id(MODEL)
    tasks = json.loads((MECH / 'inputs' / 'memory-tasks.json').read_text(encoding='utf-8'))
    task = next(x for x in tasks if x['id'] == model['task'])
    policy = Frozen(model, OUT / 'meta', batch=1)
    policy.load(MECH / model['checkpoints']['after']['path'])
    head = mb.head_arrays(mb.load_pair(MECH / model['checkpoints']['after']['path']))
    e = policy.encoder
    cases = task['cases']

    def to_delay_end(case, d, p):
        xs = lib.frames(case, d, p, False, 0)[:-1]
        policy.reset(1)
        for f in xs:
            if f['kind'] in ('cue', 'delay'):
                policy.advance(f['board'], f['ms'])
        rest = [f for f in xs if f['kind'] not in ('cue', 'delay')]
        return e.capture_state(detach=True), rest

    def from_state(case, snapshot, rest):
        e.restore_state(snapshot)
        for f in rest:
            policy.advance(f['board'], f['ms'])
        q, f = policy.advance(lib.board(case, state_from(case['level']['start'])), 40)
        return np.asarray(q[0], np.float64), np.asarray(f[0], np.float64)

    def perturbed(snapshot, field, delta, eps):
        s = dict(snapshot, tensors=dict(snapshot['tensors']))
        s['tensors'][field] = snapshot['tensors'][field] + eps * delta
        return s

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float('nan')

    W = head['proposal.weight'].astype(np.float64)
    bias = head['proposal.bias'].astype(np.float64)
    mean, scale = head['mean'].astype(np.float64), head['scale'].astype(np.float64)
    from game import legal, restore

    def logits(f):
        return W @ np.clip((f - mean) / scale, -10, 10) + bias

    ok = legal(restore(cases[0]['level']), state_from(cases[0]['level']['start']))
    res = dict(model=MODEL, rows=[], transfer={})
    t0 = time.time()
    s0, rest = to_delay_end(cases[0], 0, 0)
    s1, _ = to_delay_end(cases[1], 0, 0)
    a0 = int(np.argmax(np.where(ok, from_state(cases[0], s0, rest)[0], -np.inf)))
    a1 = int(np.argmax(np.where(ok, from_state(cases[1], s1, rest)[0], -np.inf)))
    res['cue_pushes'] = [a0, a1]

    def margin(f):
        q = logits(f)
        return float(q[a1] - q[a0])

    for p in (4, 0):
        traces, base = {}, {}
        for d in DELAYS:
            s0, rest = to_delay_end(cases[0], d, p)
            s1, _ = to_delay_end(cases[1], d, p)
            q0, f0 = from_state(cases[0], s0, rest)
            q1, f1 = from_state(cases[1], s1, rest)
            dF = f1 - f0
            row = dict(prefix=p, delay=d, dF_norm=float(np.linalg.norm(dF)), margin_cue0=margin(f0), margin_cue1=margin(f1))
            for field in ('adapt', 'v', 'g'):
                delta = s1['tensors'][field] - s0['tensors'][field]
                row[f'd{field}_norm'] = float(delta.norm())
                for eps in EPS:
                    _, fe = from_state(cases[0], perturbed(s0, field, delta, eps), rest)
                    resp = (fe - f0) / eps
                    row[f'{field}_eps{eps:g}_cos'] = cos(resp, dF)
                    row[f'{field}_eps{eps:g}_gain'] = float(np.linalg.norm(resp) / np.linalg.norm(dF)) if np.linalg.norm(dF) > 0 else None
                    if eps == 1.0:
                        row[f'{field}_margin'] = margin(fe)
            swapped = dict(s0, tensors={k: (s1['tensors'][k] if k in ('adapt', 'v', 'g') else v) for k, v in s0['tensors'].items()})
            _, fall = from_state(cases[0], swapped, rest)
            row['all3_cos'] = cos(fall - f0, dF)
            row['all3_margin'] = margin(fall)
            res['rows'].append(row)
            da = s1['tensors']['adapt'] - s0['tensors']['adapt']
            traces[d] = da / da.norm() if float(da.norm()) > 0 else da
            base[d] = (s0, rest, f0)
            print(f'p{p} d{d}: |dF| {row["dF_norm"]:.3f}, adapt-only cos {row["adapt_eps1_cos"]:.3f}, margins cue0 {row["margin_cue0"]:.2f} '
                  f'cue1 {row["margin_cue1"]:.2f} adapt-only {row["adapt_margin"]:.2f}', flush=True)
        # operator drift: one unit trace pattern (the one at 160 ms) applied at every delay with that delay's trace norm
        ref = traces[160]
        resp, marg = {}, {}
        for d in DELAYS:
            s0, rest, f0 = base[d]
            scale_d = next(r['dadapt_norm'] for r in res['rows'] if r['prefix'] == p and r['delay'] == d)
            _, fe = from_state(cases[0], perturbed(s0, 'adapt', ref * scale_d, 1.0), rest)
            resp[d] = fe - f0
            marg[str(d)] = margin(fe) - margin(f0)
        res['transfer'][f'p{p}_cosine'] = {f'{a}-{b}': cos(resp[a], resp[b]) for a in DELAYS for b in DELAYS if a < b}
        res['transfer'][f'p{p}_margin_shift'] = marg
    res['seconds'] = time.time() - t0
    (OUT / 'result.json').write_text(json.dumps(res, indent=1), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
