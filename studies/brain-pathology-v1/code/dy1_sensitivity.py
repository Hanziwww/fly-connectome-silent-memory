"""T-DY1, sensitivity diagnosis (manuscript/DYNAMICS_SOLUTIONS_PLAN.md, DY1; GPU host): is the network chaotic, or only
sensitive? Family F14.DY.sensitivity (descriptive), declared before any run.

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/dy1_sensitivity.py

Network: the healthy P05 brain (credibility.Rig; the M0 readout-only model reads this brain unchanged), batch 1, driven by
the M0 task input (memory_tasks.frames: the cue frame, 40 ms; the blank delay; the common prefix windows; the decision
frame), at two trained timings, (160 ms, 4 prefix windows) and (40 ms, 0), and both cues. Per timing and cue there is one
unperturbed reference run, then perturbed runs identical to it up to cue offset, where the perturbation starts:
- threshold: a uniform threshold offset -d on every cell from cue offset on (operators.py 'threshold_offset', the C01
  operator), d = 1e-4 to 1e-1 mV;
- spike: one extra spike in one cell at cue offset (its voltage set 1 mV above its threshold for one step); 5 cells drawn
  (seed 20261020) from the cells that are neither input nor readout cells;
- noise: independent Poisson spikes at 5 Hz on every visual input cell from cue offset on (the memory curve's noise), 5
  seeds.
Recorded from cue offset to the end of the decision frame: the spike Hamming distance to the reference every ms (cells
whose spike differs), and the L2 distances of voltage v, adaptation and synaptic input g every 5 ms. The drive is
deterministic otherwise, so the reference is exact. Outputs runs/dy1-sensitivity/d<delay>-p<prefix>-cue<c>.json;
resumable. The memory part (trained histories solved under static shifts of -1e-4 to -3e-3 mV) is
phase4_dose_series.py --set micro; the analysis is analyze_dy1.py (local).
"""
import gc
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
OUT = STUDY / 'runs' / 'dy1-sensitivity'
TIMINGS = ((160, 4), (40, 0))
DELTAS_MV = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
SPIKE_CELLS, NOISE_SEEDS, NOISE_HZ, EVERY, SEED = 5, 5, 5.0, 5, 20261020


def main():
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    import mechanism_base  # noqa: F401  (path setup: flybrain, sokoban, frozen study code)
    import pandas as pd
    import torch
    import operators as op
    from credibility import Rig
    OUT.mkdir(parents=True, exist_ok=True)
    rig = Rig(ROOT / 'runs' / 'linear-joint' / '20261002-p05-01' / 'model.pt', OUT / 'meta-rig')
    e, b = rig.enc, rig.enc.brain
    task = next(t for t in rig.memory_tasks.prepare_memory() if t['id'] == 'M0')
    ann = pd.read_parquet(MECH / 'neurons' / 'annotations.parquet', columns=['direct_input', 'observed_by_policy'])
    pool = np.flatnonzero(~ann['direct_input'].to_numpy(bool) & ~ann['observed_by_policy'].to_numpy(bool))
    spike_cells = np.sort(np.random.default_rng(SEED).choice(pool, SPIKE_CELLS, replace=False))
    inputs = e.input_index
    n_in = int(inputs.numel())
    every_cell = np.arange(rig.n)
    perturbations = ([('threshold', d) for d in DELTAS_MV] + [('spike', int(c)) for c in spike_cells]
                     + [('noise', k) for k in range(NOISE_SEEDS)])

    def run(frames, kind=None, arg=None, ref=None):
        """One trial. Without ref: the reference (spikes every ms, states every EVERY ms from cue offset). With ref: the
        distances to it."""
        rec = dict(spikes=[], snaps=[], hamming=[], dv=[], dadapt=[], dg=[], n_spikes=0)
        with torch.no_grad():
            rig.lesion([])
            e.reset()
            e.bind()
            b.set_poisson(inputs, torch.zeros((1, n_in), device='cuda'))
            t, onset = 0, None
            for i, f in enumerate(frames):
                x = torch.as_tensor(f['board'], dtype=torch.float32, device='cuda').reshape(1, 400).repeat_interleave(8, 1) * 18
                b.set_drive(inputs, x)
                if i == 1:                                         # cue offset: the frame after the cue frame
                    onset = t
                    if kind == 'threshold':
                        rig.lesion([op.spec('threshold_offset', every_cell, -arg, f'dy1 -{arg:g} mV')])
                        e.bind()
                    elif kind == 'noise':
                        b.gen.manual_seed(zlib.crc32(f'dy1-noise-{arg}'.encode()))
                        b.set_poisson(inputs, torch.full((1, n_in), NOISE_HZ, device='cuda'))
                for _ in range(f['ms']):
                    if kind == 'spike' and t == onset:
                        th = b.p.v_th[arg] if torch.is_tensor(b.p.v_th) else b.p.v_th
                        rec['refractory_at_onset'] = bool(b.refrac[0, arg] > 0)
                        b.v[0, arg] = th + 1.0
                    b.step()
                    if onset is not None:
                        k = t - onset
                        s = b.spikes[0].to(torch.uint8)
                        rec['n_spikes'] += int(s.sum())
                        if ref is None:
                            rec['spikes'].append(s)
                            if k % EVERY == 0:
                                rec['snaps'].append((b.v[0].clone(), b.adapt[0].clone(), b.g[0].clone()))
                        else:
                            rec['hamming'].append(int((s != ref['spikes'][k]).sum()))
                            if k % EVERY == 0:
                                v0, a0, g0 = ref['snaps'][k // EVERY]
                                rec['dv'].append(float(torch.linalg.vector_norm(b.v[0] - v0)))
                                rec['dadapt'].append(float(torch.linalg.vector_norm(b.adapt[0] - a0)))
                                rec['dg'].append(float(torch.linalg.vector_norm(b.g[0] - g0)))
                    t += 1
            rec['onset_ms'], rec['total_ms'] = onset, t
            rec['decision_ms'] = frames[-1]['ms']
            rig.lesion([])
            e.bind()
        return rec

    for delay, prefix in TIMINGS:
        for cue in (0, 1):
            target = OUT / f'd{delay}-p{prefix}-cue{cue}.json'
            if target.exists():
                continue
            t0 = time.time()
            frames = rig.memory_tasks.frames(task, cue, delay, prefix, False)
            ref = run(frames)
            again = run(frames, ref=ref)                           # the reference must reproduce itself exactly
            if any(again['hamming']) or any(again['dv']):
                raise RuntimeError('the unperturbed run is not deterministic')
            out = dict(delay_ms=delay, prefix_windows=prefix, cue=cue, onset_ms=ref['onset_ms'], total_ms=ref['total_ms'],
                       decision_ms=ref['decision_ms'], every_ms=EVERY, reference_spikes_after_onset=ref['n_spikes'],
                       reference_cells=int(rig.n), spike_cells=[int(c) for c in spike_cells], noise_hz=NOISE_HZ, runs=[])
            for kind, arg in perturbations:
                r = run(frames, kind, arg, ref)
                out['runs'].append(dict(kind=kind, arg=arg, hamming=r['hamming'], dv=r['dv'], dadapt=r['dadapt'], dg=r['dg'],
                                        n_spikes=r['n_spikes'], refractory_at_onset=r.get('refractory_at_onset')))
            out['seconds'] = time.time() - t0
            target.write_text(json.dumps(out), encoding='utf-8', newline='\n')
            last = {f"{x['kind']}:{x['arg']:g}": x['hamming'][-1] for x in out['runs'] if x['kind'] == 'threshold'}
            print(f'{target.stem}: {len(out["runs"])} perturbations, final spike Hamming by threshold shift {last}, '
                  f'{out["seconds"]:.0f} s', flush=True)
            del ref, again
            gc.collect()
            torch.cuda.empty_cache()
    (OUT / 'done.json').write_text(json.dumps(dict(finished_unix=time.time())), encoding='utf-8')


if __name__ == '__main__':
    main()
