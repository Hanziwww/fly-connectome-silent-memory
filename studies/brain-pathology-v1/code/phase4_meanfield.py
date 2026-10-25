"""Phase 4: mean-field stability of the healthy and the C01 (Abeta42 Kv4-loss) network. GPU host (rates), host CPU (eigs).

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/phase4_meanfield.py

Rate picture of the LIF network around a steady operating point (a constant M0 decision board):
    r_i = phi_i(I_i),  I_i = tau_syn sum_j W_ij r_j + drive_i - jump tau_a r_i
(each presynaptic spike adds W_ij mV to g_i, which decays with tau_syn; each own spike adds `jump` mV of adaptation
that decays with tau_a). Linearising, a perturbation obeys  dr = M dr  with
    M = (I + D jump tau_a)^-1 D tau_syn W_eff,   D = diag(phi_i'),
and the steady state loses stability when the spectral radius rho(M) reaches 1. phi_i' is measured, not assumed: a
threshold offset of -delta is equivalent to an input of +delta, so phi_i' = (r_i(theta - delta) - r_i(theta)) / delta.
For each uniform threshold offset (0 ... -7 mV, the C01 calibrated dose is -3.22) the job records the firing rates,
estimates D, and computes rho(M) restricted to the cells with phi' > 0 (scipy ARPACK, largest magnitude). W_eff is the
matrix the simulation uses (brain._W_cpu with the P05 pre and post gains). Output: runs/g4-meanfield/result.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
OUT = STUDY / 'runs' / 'g4-meanfield'
OFFSETS = [round(-0.5 * k, 2) for k in range(15)]          # 0 ... -7 mV
DELTA, SETTLE, WINDOW = 0.5, 200, 800


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    import mechanism_base  # noqa: F401
    import torch
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigs
    import operators as op
    from credibility import Rig
    import memory_tasks
    from game import state_from
    rig = Rig(ROOT / 'runs' / 'linear-joint' / '20261002-p05-01' / 'model.pt', OUT / 'meta-rig')
    e, b = rig.enc, rig.enc.brain
    task = next(t for t in memory_tasks.prepare_memory() if t['id'] == 'M0')
    board = memory_tasks.hidden_board(task['levels'][0], state_from(task['query_state']))
    x = torch.as_tensor(board, dtype=torch.float32, device='cuda').reshape(1, 400).repeat_interleave(8, 1) * 18
    every = np.arange(rig.n)
    rates = {}
    t0 = time.time()
    for off in sorted(set(OFFSETS + [round(o - DELTA, 2) for o in OFFSETS])):
        rig.lesion([op.spec('threshold_offset', every, off)])
        with torch.no_grad():
            e.reset()
            e.bind()
            b.set_drive(e.input_index, x)
            for _ in range(SETTLE):
                b.step()
            c = torch.zeros(rig.n, device='cuda')
            for _ in range(WINDOW):
                b.step()
                c += b.spikes[0]
        rates[off] = (c / WINDOW).cpu().numpy().astype(np.float64)          # spikes per ms
    rig.lesion([])
    # effective weights the simulation uses: post_i x W_ij x pre_j (mV per presynaptic spike)
    e.bind()
    W = e.brain._W_cpu.tocsr().astype(np.float64)
    pre = e.matrix.pre.double().cpu().numpy()
    post = e.matrix.post.double().cpu().numpy()
    Weff = sp.diags(post) @ W @ sp.diags(pre)
    p = e.params
    tau_syn, tau_a, jump = float(p.tau_syn), float(p.adapt_tau), float(p.adapt_jump)
    rows = []
    for off in OFFSETS:
        r0, r1 = rates[off], rates[round(off - DELTA, 2)]
        d = np.clip((r1 - r0) / DELTA, 0, None)                              # phi' (spikes / ms / mV)
        act = np.flatnonzero(d > 0)
        row = dict(offset_mv=off, mean_rate_hz=float(r0.mean() * 1000), fraction_firing=float((r0 > 0).mean()),
                   fraction_over_100hz=float((r0 > 0.1).mean()), cells_with_gain=int(len(act)))
        if len(act) >= 3:
            Dg = d[act]
            sub = Weff[act][:, act]
            M = sp.diags(1.0 / (1.0 + Dg * jump * tau_a)) @ sp.diags(Dg * tau_syn) @ sub
            try:
                val = eigs(M, k=1, which='LM', return_eigenvectors=False, maxiter=5000, tol=1e-6)
                row['spectral_radius'] = float(np.abs(val[0]))
                row['leading_eigenvalue_real'] = float(np.real(val[0]))
            except Exception as exc:                                         # report, never hide
                row['spectral_radius'] = None
                row['eigs_error'] = repr(exc)[:200]
        rows.append(row)
        print(f'offset {off:5.1f} mV: rate {row["mean_rate_hz"]:.2f} Hz, cells with gain {row["cells_with_gain"]}, '
              f'rho {row.get("spectral_radius")}', flush=True)
    res = dict(rows=rows, tau_syn=tau_syn, tau_adapt=tau_a, adapt_jump=jump, delta_mv=DELTA, settle_ms=SETTLE, window_ms=WINDOW,
               c01_calibrated_offset_mv=-3.22, seconds=time.time() - t0)
    (OUT / 'result.json').write_text(json.dumps(res, indent=1), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
