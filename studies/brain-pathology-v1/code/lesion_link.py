"""Lesion link (amendment 0015, declared before any run; F15.link*): where the cue trace is written and where it is read.

    local, once:   .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\lesion_link.py --prepare
    host, once:    CUDA_VISIBLE_DEVICES=<g> PATH=/usr/local/cuda/bin:$PATH .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/lesion_link.py --smoke <g>
    host, per GPU: CUDA_VISIBLE_DEVICES=<g> PATH=/usr/local/cuda/bin:$PATH .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/lesion_link.py --worker <g>

Question. The C01 lesion (Abeta42-like Kv4 loss: spike threshold -3.22 mV x dose on every cell) after learning removes
access to the cue memory, learning under it is spared, relearning in it restores access in part, and restoring
excitability late removes access again (regimes A-C and the late restoration). Is access lost because the trace is no
longer stored, or because the network that reads it has changed?

The learner is the scene-1 readout-trained one: its brain is the planning-trained connectome, fixed, so the brain states
depend only on the input and the lesion, and the trained readouts differ. Brain conditions W-R, at the 9 trained timings
and both cues (18 histories, batched exactly as the frozen assessment, memory.batched_assessment):
- H-H healthy; L<d>-L<d> the C01 lesion at dose d (0.5, 1, 1.5, 2);
- H-L<d>: the trace is written and held in the healthy brain and read under the lesion: at the end of each history's delay
  (the start of its prefix input; for 0 prefix windows, the start of the first decision) its full simulator state is
  replaced by the state the healthy brain reaches there on the same input, and the run continues under the lesion;
- L<d>-H: written and held under the lesion, read healthy, likewise.
Readouts: healthy-trained (studies/brain-supplement-v2/runs/x1-M0-head_only-r0..9, learned in H); lesion-trained (regime
B, studies/brain-pathology-v1/runs/b-C01-dose<d>-M0-head_only-r1..6, learned in L<d>); relearned (regime C, replicates
1-3 at each dose, learned in H and then relearned in L<d>), rebuilt by replaying the 100 readout-only updates of
phase3_regime_c.py exactly. Inputs (copied by --prepare): inputs/lesion-link/<run>/{config.json, model.pt} and the regime C
records inputs/lesion-link/regime-c/<job>.json; the design file tasks/lesion-link-design.json.

Jobs (outputs runs/lesion-link; workers claim them in order through runs/lesion-link/claims):
- relearn-d<d>-r<r>: the regime C replay; relearn-d<d>-r<r>.pt (the readout after 100 updates) and .json. Check: the audit,
  the three assessments and every update's loss and gradient norm equal the regime C record exactly.
- states: every brain condition's input phase and first decision, read by the healthy-trained readout 0. Per history: the
  adaptation at the end of the delay after any transplant (the stored trace that is then read), the membrane potential
  and adaptation over the first decision window (means), the 512 readout features; differential spikes in the second
  half of the delay; mean firing in the delay. Writes states.json (per timing: cue-difference norms, cells, cosines between
  conditions) and states.npz (per-cell cue differences at the 160-ms delay with 4 prefix windows). Checks: the replica
  equals memory.assess in every decision (H-H and L1-L1); sham transplants (W = R) leave every decision bit-identical;
  the transplanted trace equals the donor's.
- readout-<name>: the rollouts (512 decisions) of one readout in each of its brain conditions (healthy-trained: all 13;
  lesion-trained and relearned: H-H, L<d>-L<d>, H-L<d>, L<d>-H at their dose). Per history: status, steps, first action,
  first-decision logits and features (readout-<name>.json, .npz). Check: the first-decision features of each condition
  equal those of the states job.
- video: per-ms spikes of every cell, and the decisions, for the 160-ms delay with 4 prefix windows, both cues, at dose 1:
  healthy-trained readout 1 in H-H, L1-L1, H-L1 and L1-H; relearned readout d1-r1 in L1-L1; lesion-trained readout d1-r1
  in L1-L1 and H-H (video.npz, video.json). Check: each combination's decisions equal its readout job's.
The comparisons with the earlier records (verification, regime C, late restoration) run locally in analyze_lesion_link.py.
"""
import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
INPUTS = STUDY / 'inputs' / 'lesion-link'
DESIGN = STUDY / 'tasks' / 'lesion-link-design.json'
OUT = STUDY / 'runs' / 'lesion-link'
DOSES = (0.5, 1.0, 1.5, 2.0)
SHOWCASE = (160, 4)                      # delay (ms), prefix windows: the timing of the per-cell maps and the video
UPDATES, ASSESS_AT = 100, (0, 50, 100)   # phase3_regime_c.py
VIDEO_DECISIONS = 30                     # spikes are kept for the input phase and the first 30 decisions (1.2 s)
AXIS1 = ('g_slow_cls', 'spike_buf')      # state tensors with the batch on axis 1 (backend.StudyBrain.reset_batch)


def x1(r):
    return f'x1-M0-head_only-r{r}'


def regime_b(d, r):
    return f'b-C01-dose{d:g}-M0-head_only-r{r}'


def regime_c(d, r):
    return f'C01-dose{d:g}-head_only-r{r}'


def L(d):
    return f'L{d:g}'


def readouts():
    return ([dict(name=f'x1-r{r}', kind='healthy-trained', source=x1(r), learned='H', dose=None) for r in range(10)]
            + [dict(name=f'b-d{d:g}-r{r}', kind='lesion-trained', source=regime_b(d, r), learned=L(d), dose=d)
               for d in DOSES for r in range(1, 7)]
            + [dict(name=f'c-d{d:g}-r{r}', kind='relearned', source=x1(r), learned=L(d), dose=d) for d in DOSES for r in (1, 2, 3)])


def conditions(ro):
    """Brain conditions (writing state, reading state) of a readout."""
    if ro['kind'] == 'healthy-trained':
        return [('H', 'H')] + [c for d in DOSES for c in ((L(d), L(d)), ('H', L(d)), (L(d), 'H'))]
    d = L(ro['dose'])
    return [('H', 'H'), (d, d), ('H', d), (d, 'H')]


def name(c):
    return f'{c[0]}-{c[1]}'


VIDEO = [('x1-r1', ('H', 'H')), ('x1-r1', ('L1', 'L1')), ('x1-r1', ('H', 'L1')), ('x1-r1', ('L1', 'H')),
         ('c-d1-r1', ('L1', 'L1')), ('b-d1-r1', ('L1', 'L1')), ('b-d1-r1', ('H', 'H'))]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ local preparation

def prepare():
    if DESIGN.exists():
        raise SystemExit(f'{DESIGN.name} exists; task files are written once, before their runs')
    copies = []
    for r in range(10):
        copies.append((SUPP / 'runs' / x1(r), x1(r)))
    for d in DOSES:
        for r in range(1, 7):
            copies.append((STUDY / 'runs' / regime_b(d, r), regime_b(d, r)))
    models = {}
    for src, dst in copies:
        res = json.loads((src / 'result.json').read_text(encoding='utf-8'))
        if not (src / 'verification.json').exists() or sha(src / 'model.pt') != res['model_sha256']:
            raise SystemExit(f'{src.name}: not verified, or model.pt differs from result.json')
        (INPUTS / dst).mkdir(parents=True, exist_ok=True)
        for f in ('config.json', 'model.pt'):
            shutil.copyfile(src / f, INPUTS / dst / f)
        models[dst] = sha(INPUTS / dst / 'model.pt')
    (INPUTS / 'regime-c').mkdir(parents=True, exist_ok=True)
    records = {}
    for d in DOSES:
        for r in (1, 2, 3):
            src = STUDY / 'runs' / 'regime-c' / f'{regime_c(d, r)}.json'
            shutil.copyfile(src, INPUTS / 'regime-c' / src.name)
            records[src.stem] = sha(src)
    lesions = {}
    for d in DOSES:
        found = {json.dumps(json.loads((INPUTS / regime_b(d, r) / 'config.json').read_text(encoding='utf-8'))['lesion'])
                 for r in range(1, 7)}
        if len(found) != 1:
            raise SystemExit(f'dose {d:g}: the regime B runs differ in their lesion')
        (lesion,) = [json.loads(f) for f in found]
        if len(lesion) != 1 or lesion[0]['kind'] != 'threshold_offset' or lesion[0]['cells'] != 167106:
            raise SystemExit(f'dose {d:g}: unexpected lesion {lesion}')
        lesions[L(d)] = lesion[0]
    design = dict(
        amendment=15, written_unix=time.time(), learner='scene 1 (M0), readout trained (head_only); brain fixed',
        timings='the 9 trained timings (delays 0, 40, 160 ms x prefix windows 0, 1, 4), both cues, hidden goal',
        lesions=lesions, doses=list(DOSES), showcase=dict(delay_ms=SHOWCASE[0], prefix_windows=SHOWCASE[1]),
        readouts=[dict(ro, conditions=[name(c) for c in conditions(ro)]) for ro in readouts()],
        transplant='full simulator state (backend.STATE_FIELDS and the clock accumulators) of each history at the end of '
                   'its delay, from the writing state, into the reading state',
        video=[dict(readout=r, condition=name(c)) for r, c in VIDEO],
        inputs=dict(models=models, regime_c=records))
    DESIGN.write_text(json.dumps(design, indent=1), encoding='utf-8', newline='\n')
    print(f'{DESIGN.name}: {len(design["readouts"])} readouts, {len(models)} models copied to {INPUTS}')


# ------------------------------------------------------------------------------------------------ simulation (host)

def take_rows(b, rows):
    from backend import STATE_FIELDS
    s = {n: (getattr(b, n)[:, rows] if n in AXIS1 else getattr(b, n)[rows]).clone() for n in STATE_FIELDS}
    s['_acc'] = {k: v[rows].clone() for k, v in b._acc.items()}
    return s


def put_rows(b, rows, s):
    from backend import STATE_FIELDS
    for n in STATE_FIELDS:
        t = getattr(b, n)
        if n in AXIS1:
            t[:, rows] = s[n]
        else:
            t[rows] = s[n]
    for k, v in b._acc.items():
        v[rows] = s['_acc'][k]
    b._rate_np_key = None


def replica(policy, task, legacy, torch, donor=None, take=False, window=False, spike_rows=(), max_steps=512):
    """memory.batched_assessment at the trained timings (no intervention), with additions that leave it unchanged when
    unused: donor states put in at each history's end of delay; take=True keeps the states there (a donor); window=True
    keeps the first decision window's mean membrane potential and adaptation; spike_rows: per-ms spikes of those rows.
    max_steps=0 stops after the input phase."""
    from game import legal, restore, solved, state_from, step
    cfg = legacy.read(legacy.STUDY / 'protocol.json')
    cases = [(cue, d, n) for d in cfg['trained_delay_ms'] for n in cfg['trained_prefix_windows'] for cue in (0, 1)]
    batch = len(cases)
    seqs = []
    for cue, d, n in cases:
        fs = legacy.frames(task, cue, d, n, False)[:-1]
        seqs.append(np.concatenate([np.repeat(f['board'][None], f['ms'], axis=0) for f in fs]))
    length = max(map(len, seqs))
    data = np.zeros((length, batch, 4, 10, 10), np.float32)
    resets, marks, half, full = {}, {}, {}, {}
    for row, ((cue, d, n), s) in enumerate(zip(cases, seqs)):
        offset = length - len(s)
        data[offset:, row] = s
        resets.setdefault(offset, []).append(row)
        marks.setdefault(offset + 40 + d, []).append(row)
        half[row] = (offset + 40 + d // 2, offset + 40 + d)
        full[row] = (offset + 40, offset + 40 + d)
    pairs = [(2 * k, 2 * k + 1) for k in range(batch // 2)]          # the cue-0 and cue-1 rows of one timing are adjacent
    if any(cases[i][1:] != cases[j][1:] or cases[i][0] != 0 or cases[j][0] != 1 for i, j in pairs):
        raise RuntimeError('unexpected case order')
    e = policy.encoder
    e.reset_batch(batch)
    e.bind()
    b = e.brain
    inputs = torch.as_tensor(data, device='cuda')
    out = dict(cases=cases, taken={}, mark={}, spikes=[])
    sel = torch.as_tensor(list(spike_rows), device='cuda', dtype=torch.long) if spike_rows else None
    with torch.no_grad():
        differ = torch.zeros((len(pairs), b.n), dtype=torch.bool, device='cuda')
        delay_spikes = torch.zeros(batch, dtype=torch.float64, device='cuda')

        def at_mark(tick):
            rows = marks.get(tick)
            if not rows:
                return
            idx = torch.as_tensor(rows, device='cuda')
            if donor is not None:
                put_rows(b, idx, donor[tick])
            if take:
                out['taken'][tick] = take_rows(b, idx)
            for r in rows:
                out['mark'][r] = b.adapt[r].float().cpu().numpy()

        def spikes(t):
            if sel is not None:
                nz = torch.nonzero(b.spikes[sel] > 0).cpu().numpy().astype(np.int32)
                out['spikes'].append(np.column_stack([np.full(len(nz), t, np.int32), nz]))

        for tick in range(length):
            if (OUT / 'STOP').exists():
                raise KeyboardInterrupt('STOP requested')
            if tick in resets:
                b.reset(rows=resets[tick])
            at_mark(tick)
            if tick == 0 or tick % 20 == 0:
                b.set_drive(e.input_index, inputs[tick].reshape(batch, 400).repeat_interleave(8, 1) * 18)
            b.step()
            for k, (r0, r1) in enumerate(pairs):
                if half[r0][0] <= tick < half[r0][1]:
                    differ[k] |= b.spikes[r0] != b.spikes[r1]
            for r in range(batch):
                if full[r][0] <= tick < full[r][1]:
                    delay_spikes[r] += b.spikes[r].double().sum()
            spikes(tick)
        at_mark(length)
        out['delay_differ_cells'] = differ.sum(1).cpu().numpy().astype(int).tolist()
        out['delay_rate_hz'] = [float(delay_spikes[r]) / b.n / ((cases[r][1] or np.nan) / 1000.0) for r in range(batch)]
        if max_steps == 0:
            e.reset_batch(1)
            return out
        states = [state_from(task['query_state']) for _ in cases]
        maps = [restore(task['levels'][cue]) for cue, _, _ in cases]
        records = [dict(cue=cue, delay_ms=d, prefix_windows=n, status='running', decisions=[]) for cue, d, n in cases]
        clock = [length]
        for index in range(max_steps):
            if (OUT / 'STOP').exists():
                raise KeyboardInterrupt('STOP requested')
            boards = np.stack([legacy.hidden_board(task['levels'][cue], s) for (cue, _, _), s in zip(cases, states)])
            hook = None
            if (window and index == 0) or sel is not None:
                sv = torch.zeros((batch, b.n), dtype=torch.float64, device='cuda') if window and index == 0 else None
                sa = torch.zeros_like(sv) if sv is not None else None

                def hook(brain, t, sv=sv, sa=sa):
                    if sv is not None:
                        sv.add_(brain.v)
                        sa.add_(brain.adapt)
                    if index < VIDEO_DECISIONS:
                        spikes(clock[0] + t - 1)
            f = e.advance(torch.as_tensor(boards, device='cuda'), reset=False, hook=hook)
            clock[0] += 40
            q = policy.head(f)['final_logits'].cpu().numpy()
            features = f.cpu().numpy()
            if not np.isfinite(q).all():
                raise FloatingPointError('Nonfinite memory policy logits')
            if index == 0:
                out['first_features'] = features.copy()
                if window:
                    out['window_v'] = (sv / 40).float().cpu().numpy()
                    out['window_adapt'] = (sa / 40).float().cpu().numpy()
            for row, ((cue, d, n), s, m, r) in enumerate(zip(cases, states, maps, records)):
                if r['status'] != 'running':
                    continue
                mask = legal(m, s)
                if not mask.any():
                    r['status'] = 'blocked'
                    continue
                a = int(np.where(mask, q[row], -np.inf).argmax())
                nxt = step(m, s, a)
                r['decisions'].append(dict(state=[s[0], list(s[1])], action=a, logits=q[row].tolist()))
                if solved(m, nxt):
                    r['status'] = 'solved'
                elif nxt[1] != s[1] and ((cue == 0 and nxt[1][0] > 34) or (cue == 1 and nxt[1][0] < 34)):
                    r['status'] = 'wrong_irreversible_push'
                states[row] = nxt
            if all(r['status'] != 'running' for r in records):
                break
        for r in records:
            if r['status'] == 'running':
                r['status'] = 'action_limit'
            r['steps'] = len(r['decisions'])
        out['records'] = records
    e.reset_batch(1)
    return out


class Session:
    """One frozen memory policy per job (as phase3_rescue.py), the readout swapped in, the lesion set per run."""

    def __init__(self, meta):
        import torch
        import memory as legacy
        import operators as op
        self.torch, self.legacy, self.op = torch, legacy, op
        self.design = json.loads(DESIGN.read_text(encoding='utf-8'))
        config = json.loads((INPUTS / x1(0) / 'config.json').read_text(encoding='utf-8'))
        meta.mkdir(parents=True, exist_ok=True)
        self.policy = legacy.MemoryPolicy(config, meta)
        self.task = next(t for t in legacy.prepare_memory() if t['id'] == config['task'])
        self.core = None
        self.n = self.policy.encoder.brain.n

    def readout(self, ro):
        import mechanism_base as mb
        torch = self.torch
        if ro['kind'] == 'relearned':
            folder = INPUTS / ro['source']
            if sha(folder / 'model.pt') != self.design['inputs']['models'][ro['source']]:
                raise RuntimeError(f"{ro['name']}: input model differs from the design file")
            saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
            self.policy.load(saved['pair'])
            head = torch.load(OUT / f"relearn-d{ro['dose']:g}-r{ro['source'].rsplit('-r', 1)[1]}.pt", map_location='cpu', weights_only=True)
            self.policy.head.load_state_dict(head)
        else:
            folder = INPUTS / ro['source']
            config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
            if sha(folder / 'model.pt') != self.design['inputs']['models'][ro['source']] or config['condition'] != 'head_only':
                raise RuntimeError(f"{ro['name']}: input model differs from the design file")
            saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
            if saved['config'] != config:
                raise RuntimeError(f"{ro['name']}: model.pt was written with another config")
            self.policy.load(saved['pair'])
        core = mb.tensors_hash(self.policy.encoder.core.state_dict())
        if self.core is None:
            self.core = core
        if core != self.core:
            raise RuntimeError(f"{ro['name']}: the brain differs from the other readouts' (readout-only learners share it)")

    def state(self, s):
        """Set the brain state H (healthy) or L<d> (the regime B lesion of dose d)."""
        if s == 'H':
            specs = []
        else:
            les = self.design['lesions'][s]
            specs = [self.op.spec('threshold_offset', np.arange(self.n), les['value'], les['label'])]
        return self.op.apply(self.policy.encoder, specs)

    def run(self, cond, donors=None, **kw):
        w, r = cond
        donor = None
        if w != r:
            if donors is None or w not in donors:
                self.state(w)
                d = replica(self.policy, self.task, self.legacy, self.torch, take=True, max_steps=0)['taken']
                if donors is None:
                    donor = d
                else:
                    donors[w] = d
            donor = donors[w] if donors is not None else donor
        self.state(r)
        return replica(self.policy, self.task, self.legacy, self.torch, donor=donor, **kw)

    def close(self):
        del self.policy
        gc.collect()                   # operators.apply leaves a reference cycle (enc.bind); free the GPU memory now
        self.torch.cuda.empty_cache()


def decisions(out):
    return [[(x['action'], x['logits']) for x in r['decisions']] + [r['status']] for r in out['records']]


def history_rows(out):
    return [dict(cue=r['cue'], delay_ms=r['delay_ms'], prefix_windows=r['prefix_windows'], status=r['status'], steps=r['steps'],
                 first_action=r['decisions'][0]['action'] if r['decisions'] else None,
                 first_logits=r['decisions'][0]['logits'] if r['decisions'] else None) for r in out['records']]


def job_relearn(d, rep, gpu):
    """phase3_regime_c.run_job for one dose and replicate, readout-only, with the readout kept."""
    import torch
    import memory as legacy
    import operators
    from replicate_training import timing_iteration
    import phase3_regime_c as rc
    start = INPUTS / x1(rep)
    base = json.loads((start / 'config.json').read_text(encoding='utf-8'))
    saved = torch.load(start / 'model.pt', map_location='cpu', weights_only=True)
    if saved['config'] != base:
        raise RuntimeError(f'{start.name}: model.pt was written with another config')
    design = json.loads(DESIGN.read_text(encoding='utf-8'))
    les = design['lesions'][L(d)]
    config = dict(base, condition='head_only', id=f'regime-c-C01-dose{d:g}-head_only-r{rep}')
    meta = OUT / 'meta' / f'relearn-d{d:g}-r{rep}'
    meta.mkdir(parents=True, exist_ok=True)
    policy = legacy.MemoryPolicy(config, meta)
    policy.load(saved['pair'])
    audit = operators.apply(policy.encoder, [operators.spec('threshold_offset', np.arange(policy.encoder.brain.n), les['value'], les['label'])])
    task = next(t for t in legacy.prepare_memory() if t['id'] == config['task'])
    optimizer = torch.optim.Adam([dict(params=policy.head.parameters(), lr=.0003)])
    assessments, steps = {}, []
    t0 = time.time()
    for u in range(UPDATES + 1):
        if u in ASSESS_AT:
            assessments[str(u)] = rc.summary(legacy.assess(policy, task))
        if u == UPDATES:
            break
        g = legacy.training_step(policy, task, optimizer, timing_iteration(u, rep))
        steps.append(dict(update=u + 1, loss=g['loss'], gradient_norm=g['gradient_norm'], delay_ms=g['delay_ms'], prefix_windows=g['prefix_windows']))
    want = json.loads((INPUTS / 'regime-c' / f'{regime_c(d, rep)}.json').read_text(encoding='utf-8'))
    same = dict(audit=audit == want['audit'], assessments=assessments == want['assessments'], steps=steps == want['steps'],
                start=sha(start / 'model.pt') == want['start_model_sha256'])
    rec = dict(job=f'relearn-d{d:g}-r{rep}', dose=d, replicate=rep, audit=audit, assessments=assessments, steps=steps,
               reproduces_regime_c=all(same.values()), checks=same, seconds=time.time() - t0, gpu=gpu)
    if not rec['reproduces_regime_c']:
        (OUT / f"relearn-d{d:g}-r{rep}.failed.json").write_text(json.dumps(rec, default=str), encoding='utf-8', newline='\n')
        raise RuntimeError(f'relearn d{d:g} r{rep}: the replay does not reproduce regime C ({same})')
    torch.save({k: v.detach().cpu() for k, v in policy.head.state_dict().items()}, OUT / f'relearn-d{d:g}-r{rep}.pt')
    (OUT / f'relearn-d{d:g}-r{rep}.json').write_text(json.dumps(rec, default=str), encoding='utf-8', newline='\n')
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return f"solved {assessments['0']['solved']} -> {assessments['100']['solved']}, reproduces regime C"


def job_states(s, gpu):
    s.readout(next(ro for ro in readouts() if ro['name'] == 'x1-r0'))
    legacy, torch = s.legacy, s.torch
    checks = {}
    # the replica is the frozen assessment, and sham transplants change nothing
    for st in ('H', 'L1'):
        s.state(st)
        official = legacy.assess(s.policy, s.task)
        plain = replica(s.policy, s.task, legacy, torch)
        got = [[(x['action'], x['logits']) for x in r['decisions']] + [r['status']] for r in official['rollouts']]
        checks[f'replica-equals-assess-{st}'] = got == decisions(plain)
        donor = replica(s.policy, s.task, legacy, torch, take=True, max_steps=0)['taken']
        sham = replica(s.policy, s.task, legacy, torch, donor=donor)
        checks[f'sham-transplant-{st}'] = decisions(sham) == decisions(plain)
    if not all(checks.values()):
        (OUT / 'states.failed.json').write_text(json.dumps(checks), encoding='utf-8', newline='\n')
        raise RuntimeError(f'states: host checks failed {checks}')
    conds = [('H', 'H')] + [c for d in DOSES for c in ((L(d), L(d)), ('H', L(d)), (L(d), 'H'))]
    donors, got = {}, {}
    for c in conds:
        o = s.run(c, donors, window=True, max_steps=1)
        if c[0] != c[1]:
            ref = got[(c[0], c[0])]['mark']
            checks[f'transplanted-trace-{name(c)}'] = all(np.array_equal(o['mark'][r], ref[r]) for r in ref)
        got[c] = o
    if not all(checks.values()):
        (OUT / 'states.failed.json').write_text(json.dumps(checks), encoding='utf-8', newline='\n')
        raise RuntimeError(f'states: host checks failed {checks}')
    cases = got[('H', 'H')]['cases']
    timings = sorted({(d, n) for _, d, n in cases})
    rows_of = {t: (cases.index((0, *t)), cases.index((1, *t))) for t in timings}

    def diff(o, q, t):
        i0, i1 = rows_of[t]
        if q == 'trace':
            a, b = o['mark'][i0], o['mark'][i1]
        elif q == 'features':
            a, b = o['first_features'][i0], o['first_features'][i1]
        else:
            a, b = o[q][i0], o[q][i1]
        return b.astype(np.float64) - a.astype(np.float64)

    quantities = ('trace', 'window_v', 'window_adapt', 'features')
    per, pairs = [], []
    for t in timings:
        D = {(c, q): diff(got[c], q, t) for c in conds for q in quantities}
        for c in conds:
            i0, i1 = rows_of[t]
            o = got[c]
            own = c[0] == c[1]             # after a transplant the delay that was held is the donor's (see its W-W row)
            per.append(dict(condition=name(c), delay_ms=t[0], prefix_windows=t[1],
                            trace_norm=float(np.linalg.norm(D[(c, 'trace')])), trace_cells=int(np.count_nonzero(D[(c, 'trace')])),
                            decision_v_norm=float(np.linalg.norm(D[(c, 'window_v')])),
                            decision_v_cells_over_0p01=int(np.sum(np.abs(D[(c, 'window_v')]) > 0.01)),
                            decision_adapt_norm=float(np.linalg.norm(D[(c, 'window_adapt')])),
                            features_norm=float(np.linalg.norm(D[(c, 'features')])),
                            delay_differ_cells=int(o['delay_differ_cells'][i0 // 2]) if own else None,
                            delay_rate_hz_cue0=o['delay_rate_hz'][i0] if own else None,
                            delay_rate_hz_cue1=o['delay_rate_hz'][i1] if own else None))
        for q in quantities:
            for i, a in enumerate(conds):
                for bb in conds[i + 1:]:
                    x, y = D[(a, q)], D[(bb, q)]
                    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
                    pairs.append(dict(quantity=q, delay_ms=t[0], prefix_windows=t[1], condition_a=name(a), condition_b=name(bb),
                                      cosine=float(x @ y / (nx * ny)) if nx > 0 and ny > 0 else None))
    show = {}
    for c in conds:
        for q in ('trace', 'window_v', 'window_adapt'):
            show[f'{name(c)}|{q}'] = diff(got[c], q, SHOWCASE).astype(np.float32)
    np.savez_compressed(OUT / 'states.npz', **show)
    features = {name(c): np.asarray(got[c]['first_features'], np.float32) for c in conds}
    np.savez_compressed(OUT / 'states-features.npz', **features)
    first = {name(c): [dict(cue=r['cue'], delay_ms=r['delay_ms'], prefix_windows=r['prefix_windows'], first_action=r['decisions'][0]['action'],
                            first_logits=r['decisions'][0]['logits']) for r in got[c]['records']] for c in conds}
    rec = dict(job='states', readout='x1-r0', checks=checks, cases=[list(c) for c in cases], per_condition=per, pairs=pairs,
               showcase=dict(delay_ms=SHOWCASE[0], prefix_windows=SHOWCASE[1]), gpu=gpu, first_decisions=first)
    (OUT / 'states.json').write_text(json.dumps(rec, default=str), encoding='utf-8', newline='\n')
    return f'{len(conds)} conditions; host checks pass'


def job_readout(s, ro, gpu):
    s.readout(ro)
    states = np.load(OUT / 'states-features.npz') if (OUT / 'states-features.npz').exists() else None
    donors, recs, feats, logits = {}, {}, {}, {}
    for c in conditions(ro):
        o = s.run(c, donors)
        recs[name(c)] = history_rows(o)
        feats[name(c)] = np.asarray(o['first_features'], np.float32)
        logits[name(c)] = np.asarray([r['first_logits'] for r in recs[name(c)]], np.float32)
        if states is not None and not np.array_equal(states[name(c)], feats[name(c)]):
            raise RuntimeError(f"{ro['name']} {name(c)}: first-decision features differ from the states job")
    head = {f'head|{k}': v.detach().cpu().numpy() for k, v in s.policy.head.state_dict().items()}
    np.savez_compressed(OUT / f"readout-{ro['name']}.npz", **{f'{k}|features': v for k, v in feats.items()},
                        **{f'{k}|logits': v for k, v in logits.items()}, **head)
    rec = dict(job=f"readout-{ro['name']}", readout=ro, histories=recs, checked_against_states=states is not None, gpu=gpu,
               solved={k: sum(r['status'] == 'solved' for r in v) for k, v in recs.items()})
    (OUT / f"readout-{ro['name']}.json").write_text(json.dumps(rec, default=str), encoding='utf-8', newline='\n')
    return ', '.join(f'{k} {v}' for k, v in rec['solved'].items())


def job_video(s, gpu):
    every = {ro['name']: ro for ro in readouts()}
    cases = None
    arrays, rec = {}, dict(job='video', combinations=[], gpu=gpu)
    for ro_name, c in VIDEO:
        s.readout(every[ro_name])
        if cases is None:
            cases = s.legacy.read(s.legacy.STUDY / 'protocol.json')
            order = [(cue, d, n) for d in cases['trained_delay_ms'] for n in cases['trained_prefix_windows'] for cue in (0, 1)]
            rows = [order.index((cue, *SHOWCASE)) for cue in (0, 1)]
        o = s.run(c, spike_rows=rows)
        ev = np.concatenate(o['spikes']) if o['spikes'] else np.zeros((0, 3), np.int32)
        arrays[f'{ro_name}|{name(c)}'] = ev
        mine = json.loads((OUT / f'readout-{ro_name}.json').read_text(encoding='utf-8'))['histories'][name(c)] \
            if (OUT / f'readout-{ro_name}.json').exists() else None
        rows_out = history_rows(o)
        same = None if mine is None else [{k: v for k, v in r.items()} for r in rows_out] == mine
        rec['combinations'].append(dict(readout=ro_name, condition=name(c), rows=rows, events=int(len(ev)),
                                        decisions=[o['records'][i]['decisions'] for i in rows],
                                        status=[o['records'][i]['status'] for i in rows], equals_readout_job=same))
    np.savez_compressed(OUT / 'video.npz', **arrays)
    (OUT / 'video.json').write_text(json.dumps(rec, default=str), encoding='utf-8', newline='\n')
    return f"{len(VIDEO)} combinations; " + ', '.join(f"{c['readout']} {c['condition']}: {c['status']}" for c in rec['combinations'])


def jobs():
    out = [f'relearn-d{d:g}-r{r}' for d in DOSES for r in (1, 2, 3)] + ['states']
    out += [f"readout-{ro['name']}" for ro in readouts()] + ['video']
    return out


def finished(job):
    return (OUT / f'{job}.json').exists()


def ready(job):
    """Dependencies: every readout and the video job after the states job; relearned readouts after their replay."""
    if job.startswith('relearn') or job == 'states':
        return True
    if not (OUT / 'states.json').exists():
        return False
    if job.startswith('readout-c-') or job == 'video':
        needs = [job[len('readout-c-'):]] if job.startswith('readout-c-') else ['d1-r1']
        if job == 'video':
            if not all((OUT / f'readout-{r}.json').exists() for r in {v[0] for v in VIDEO}):
                return False
        return all((OUT / f'relearn-{n}.pt').exists() for n in needs)
    return True


def claim(job):
    (OUT / 'claims').mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(OUT / 'claims' / job, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def worker(gpu):
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    sys.path.insert(0, str(SUPP / 'code'))
    import mechanism_base  # noqa: F401  (path setup: flybrain, sokoban, frozen study code)
    OUT.mkdir(parents=True, exist_ok=True)
    every = {ro['name']: ro for ro in readouts()}
    log = OUT / f'worker-{gpu}.log'
    session = None                 # one brain per worker for every job but the replays (a build takes minutes on this host)
    while True:
        if any((OUT / 'claims').glob('*.failed')):
            raise SystemExit('a job failed (runs/lesion-link/claims/*.failed); this worker stops')
        pending = [j for j in jobs() if not (OUT / 'claims' / j).exists()]
        if not pending:
            break
        todo = next((j for j in pending if ready(j)), None)
        if todo is None:
            if session is None and not any(j.startswith('relearn') for j in pending):
                session = Session(OUT / 'meta' / f'worker-{gpu}')   # build the brain while the states job runs
            time.sleep(10)
            continue
        if not claim(todo):
            continue
        t0 = time.time()
        try:
            if todo.startswith('relearn'):
                d, r = todo[len('relearn-d'):].split('-r')
                msg = job_relearn(float(d), int(r), gpu)
            else:
                if session is None:
                    session = Session(OUT / 'meta' / f'worker-{gpu}')
                if todo == 'states':
                    msg = job_states(session, gpu)
                elif todo == 'video':
                    msg = job_video(session, gpu)
                else:
                    msg = job_readout(session, every[todo[len('readout-'):]], gpu)
        except Exception as exc:
            (OUT / 'claims' / f'{todo}.failed').write_text(repr(exc), encoding='utf-8')
            with open(log, 'a', encoding='utf-8') as f:
                f.write(f'{time.strftime("%H:%M:%S")} {todo} FAILED {exc!r}\n')
            raise
        with open(log, 'a', encoding='utf-8') as f:
            f.write(f'{time.strftime("%H:%M:%S")} {todo} {time.time() - t0:.0f} s: {msg}\n')
    if all(finished(j) for j in jobs()):
        (OUT / 'done.json').write_text(json.dumps(dict(finished_unix=time.time(), jobs=len(jobs()))), encoding='utf-8')


def smoke(gpu):
    """Engineering check before the declaration, on recorded conditions only (no new result): the replica equals
    memory.assess for the healthy-trained readout 0 in H and L1, a sham transplant and the decision-window hook change no
    decision, and the regime C replay of dose 1, replicate 1 reproduces its record. Writes runs/lesion-link-smoke/smoke.json."""
    global OUT
    OUT = STUDY / 'runs' / 'lesion-link-smoke'
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    sys.path.insert(0, str(SUPP / 'code'))
    import mechanism_base  # noqa: F401
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s = Session(OUT / 'meta')
    s.readout(next(ro for ro in readouts() if ro['name'] == 'x1-r0'))
    checks, solved = {}, {}
    for st in ('H', 'L1'):
        s.state(st)
        official = s.legacy.assess(s.policy, s.task)
        plain = replica(s.policy, s.task, s.legacy, s.torch)
        got = [[(x['action'], x['logits']) for x in r['decisions']] + [r['status']] for r in official['rollouts']]
        checks[f'replica-equals-assess-{st}'] = got == decisions(plain)
        donor = replica(s.policy, s.task, s.legacy, s.torch, take=True, max_steps=0)['taken']
        checks[f'sham-transplant-{st}'] = decisions(replica(s.policy, s.task, s.legacy, s.torch, donor=donor)) == decisions(plain)
        checks[f'window-hook-{st}'] = decisions(replica(s.policy, s.task, s.legacy, s.torch, window=True)) == decisions(plain)
        solved[st] = official['solved']
    s.close()
    msg = job_relearn(1.0, 1, gpu)
    rec = dict(checks=checks, solved_recorded_conditions=solved, relearn=msg, seconds=time.time() - t0, gpu=gpu)
    (OUT / 'smoke.json').write_text(json.dumps(rec, default=str), encoding='utf-8', newline='\n')
    print(json.dumps(rec, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--worker', type=int)
    ap.add_argument('--smoke', type=int, metavar='GPU')
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.smoke is not None:
        smoke(a.smoke)
    elif a.worker is not None:
        worker(a.worker)
    else:
        ap.error('--prepare, --smoke <gpu> or --worker <gpu>')


if __name__ == '__main__':
    main()
