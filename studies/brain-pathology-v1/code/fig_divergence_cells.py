"""Phase 4 figure data (descriptive; amendment 0014; GPU host): per-cell time of spike divergence under the T-DY1
threshold shift, for the dark band of Fig. 6 ("a 0.3-uV shift spreads through the brain").

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/fig_divergence_cells.py

Exactly the T-DY1 protocol (dy1_sensitivity.py): the healthy P05 brain read by the M0 readout-only model, batch 1, the M0
task input at the trained timing (160 ms, 4 prefix windows), both cues; one unperturbed reference run, then the run with a
uniform threshold offset of -3e-4 mV on every cell from cue offset on. New here: per cell, the first ms after cue offset at
which its spike train differs from the reference (none if it never differs before the end of the decision frame).
Check: the per-ms count of differing cells must equal T-DY1's recorded spike Hamming trajectory for the same condition
(runs/dy1-sensitivity/d160-p4-cue<c>.json, kind 'threshold', arg 3e-4) at every ms, or nothing is written.
Outputs runs/fig-divergence-cells/d160-p4-cue<c>.parquet (cell, first_divergent_ms) and .json (summary and check).
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
DY1 = STUDY / 'runs' / 'dy1-sensitivity'
OUT = STUDY / 'runs' / 'fig-divergence-cells'
DELAY, PREFIX, SHIFT_MV = 160, 4, 3e-4


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
    inputs = e.input_index
    n_in = int(inputs.numel())
    every_cell = np.arange(rig.n)

    def run(frames, shift=None, ref=None):
        """Without ref: the reference spikes every ms from cue offset. With ref: per-ms Hamming distance and the first
        divergent ms per cell."""
        spikes, hamming = [], []
        first = torch.full((rig.n,), -1, dtype=torch.int32, device='cuda')
        with torch.no_grad():
            rig.lesion([])
            e.reset()
            e.bind()
            b.set_poisson(inputs, torch.zeros((1, n_in), device='cuda'))
            t, onset = 0, None
            for i, f in enumerate(frames):
                x = torch.as_tensor(f['board'], dtype=torch.float32, device='cuda').reshape(1, 400).repeat_interleave(8, 1) * 18
                b.set_drive(inputs, x)
                if i == 1:                                     # cue offset: the frame after the cue frame
                    onset = t
                    if shift is not None:
                        rig.lesion([op.spec('threshold_offset', every_cell, -shift, f'fig6 -{shift:g} mV')])
                        e.bind()
                for _ in range(f['ms']):
                    b.step()
                    if onset is not None:
                        k = t - onset
                        s = b.spikes[0].to(torch.uint8)
                        if ref is None:
                            spikes.append(s)
                        else:
                            d = s != ref[k]
                            hamming.append(int(d.sum()))
                            new = d & (first < 0)
                            first[new] = k
                    t += 1
            rig.lesion([])
            e.bind()
        return spikes, hamming, first.cpu().numpy(), onset, t

    for cue in (0, 1):
        stem = f'd{DELAY}-p{PREFIX}-cue{cue}'
        if (OUT / f'{stem}.json').exists():
            continue
        t0 = time.time()
        frames = rig.memory_tasks.frames(task, cue, DELAY, PREFIX, False)
        ref, _, _, onset, total = run(frames)
        _, again, _, _, _ = run(frames, ref=ref)
        if any(again):
            raise RuntimeError('the unperturbed run is not deterministic')
        _, ham, first, _, _ = run(frames, SHIFT_MV, ref)
        rec = json.loads((DY1 / f'{stem}.json').read_text(encoding='utf-8'))
        dy1 = next(r for r in rec['runs'] if r['kind'] == 'threshold' and abs(float(r['arg']) - SHIFT_MV) < 1e-12)
        if list(dy1['hamming']) != ham:
            raise RuntimeError(f'{stem}: the per-ms Hamming trajectory does not reproduce T-DY1')
        cells = pd.DataFrame(dict(cell=np.arange(rig.n, dtype=np.int64),
                                  first_divergent_ms=np.where(first >= 0, first, np.nan)))
        cells.to_parquet(OUT / f'{stem}.parquet', index=False)
        summary = dict(delay_ms=DELAY, prefix_windows=PREFIX, cue=cue, shift_mv=-SHIFT_MV, onset_ms=onset, total_ms=total,
                       ms_after_onset=len(ham), cells=int(rig.n), cells_diverged=int((first >= 0).sum()),
                       hamming=ham, reproduces_dy1_hamming=True, seconds=time.time() - t0)
        (OUT / f'{stem}.json').write_text(json.dumps(summary), encoding='utf-8', newline='\n')
        print(f'{stem}: {summary["cells_diverged"]:,} of {rig.n:,} cells diverge; reproduces T-DY1; {summary["seconds"]:.0f} s',
              flush=True)
        del ref
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
