"""Phase 3: olfactory cue-memory curves for the G2 shortlist (the assay the plan adds because the retained MB and AL
cases act on circuits the visual tasks do not recruit, and C4 confirmed healthy MB function). GPU host.

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/phase3_olfactory.py

An empirical memory curve, not a trained policy: how well a cross-validated linear decoder reads which of two odours
was presented from the brain state after a delay.
- Trials: 48 per odour, in one batch of 96 columns. ORNs fire as Poisson processes (flybrain.olfaction rates: 5 Hz
  baseline on every ORN, baseline + 150 Hz x saturating response for the odour's glomeruli), so trials differ only by
  input noise (seeded generator).
- Timeline per trial: 100 ms baseline, 200 ms odour, delay d (baseline), 40-ms probe window (baseline); the state is the
  probe-window mean of v (and g) of each read-out cell set.
- Odour pairs: dissimilar (apple vs banana) and similar (apple vs grape; they share DM1, DM4 and VA2), the fine
  discrimination that APL feedback protects in flies (Lin 2014, doi:10.1038/nn.3660).
- Read-out sets: the study's 512 readout features (observed cells, pooled), the 97 MBONs (g, v), Kenyon cells (v),
  uniglomerular PNs (v).
- Decoder: shrinkage LDA (Ledoit-Wolf), 6-fold cross-validation over trials; accuracy and d' of the held-out decision
  variable.
Conditions: runs/phase3-regime-a/conditions.json (healthy, C01 doses, C04, C13, C16 with doses and random controls).
Output: runs/phase3-olfactory/<condition>.json.
"""
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
OUT = STUDY / 'runs' / 'phase3-olfactory'
TRIALS, BASE_HZ, MAX_HZ = 48, 5.0, 150.0
SETTLE, ODOUR, PROBE = 100, 200, 40
DELAYS = (0, 40, 80, 160, 320, 640)
PAIRS = (('apple', 'banana'), ('apple', 'grape'))
FOLDS = 6


def lw_lda_scores(Xtr, ytr, Xte):
    """Shrinkage-LDA decision scores (Ledoit-Wolf covariance of the class-centred training rows)."""
    m0, m1 = Xtr[ytr == 0].mean(0), Xtr[ytr == 1].mean(0)
    Xc = np.vstack([Xtr[ytr == 0] - m0, Xtr[ytr == 1] - m1])
    n, p = Xc.shape
    G = Xc @ Xc.T
    s2 = np.sum(G ** 2) / n ** 2
    mu = np.trace(G) / (n * p)
    d2 = (s2 - p * mu ** 2) / p
    b2 = (np.sum(np.einsum('ij,ij->i', Xc, Xc) ** 2) - n * s2) / n ** 2 / p
    a = min(b2, d2) / d2 if d2 > 0 else 1.0
    c, k = a * mu, (1 - a) / n
    v = m1 - m0
    if c <= 0:
        return np.zeros(len(Xte))
    w = (v - k * Xc.T @ np.linalg.solve(c * np.eye(n) + k * G, Xc @ v)) / c
    return (Xte - (m0 + m1) / 2) @ w


def decode(X, y, seed):
    """6-fold cross-validated accuracy and d' (held-out scores)."""
    X = X - X.mean(0)
    sd = X.std(0)
    keep = sd > 1e-9
    if not keep.any():
        return dict(accuracy=0.5, dprime=0.0, informative_features=0, correct=None)
    X = X[:, keep] / sd[keep]
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(y), int)
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        folds[idx] = np.arange(len(idx)) % FOLDS
    scores = np.zeros(len(y))
    for f in range(FOLDS):
        te = folds == f
        scores[te] = lw_lda_scores(X[~te], y[~te], X[te])
    correct = (scores > 0).astype(int) == y
    acc = float(np.mean(correct))
    s0, s1 = scores[y == 0], scores[y == 1]
    pooled = np.sqrt((s0.var() + s1.var()) / 2)
    return dict(accuracy=acc, dprime=float((s1.mean() - s0.mean()) / pooled) if pooled > 0 else 0.0, informative_features=int(keep.sum()),
                correct=correct.astype(int).tolist())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    import mechanism_base  # noqa: F401
    import torch
    import operators as op
    from credibility import Rig
    rig = Rig(ROOT / 'runs' / 'linear-joint' / '20261002-p05-01' / 'model.pt', OUT / 'meta-rig')
    e, b, s = rig.enc, rig.enc.brain, rig.sets
    B = 2 * TRIALS
    e.reset_batch(B)
    blank = torch.zeros((B, 3200), device='cuda')
    orn = s['orn']
    obs = e.observed.cpu().numpy() if hasattr(e.observed, 'cpu') else np.asarray(e.observed)
    readsets = dict(mbon=s['mbon'], kc=s['kc'], pn=s['alpn'])
    conds = json.loads((STUDY / 'runs' / 'phase3-regime-a' / 'conditions.json').read_text(encoding='utf-8'))

    def rates(name):
        w = np.array([rig.odours[name].get(g, 0.0) for g in rig.orn_glom])
        return BASE_HZ + MAX_HZ * w / (w + 0.5)

    for cond in conds:
        target = OUT / f'{cond["name"]}.json'
        if target.exists():
            continue
        t0 = time.time()
        specs = [op.spec(x['kind'], np.asarray(x['cells'], dtype=np.int64), x['value'], x['label']) for x in cond['operators']]
        rig.lesion(specs)
        b.gen.manual_seed(10)
        res = dict(condition=cond['name'], case=cond['case'], dose=cond['dose'], trials=TRIALS, pairs={})
        for pair in PAIRS:
            y = np.r_[np.zeros(TRIALS, int), np.ones(TRIALS, int)]
            odour_rates = np.stack([rates(pair[k]) for k in y])               # (B, n_orn)
            rows = []
            for d in DELAYS:
                with torch.no_grad():
                    e.reset()
                    e.bind()
                    b.set_drive(e.input_index, blank)
                    b.set_poisson(orn, torch.full((B, len(orn)), BASE_HZ, device='cuda'))
                    for _ in range(SETTLE):
                        b.step()
                    b.set_poisson(orn, torch.as_tensor(odour_rates, dtype=torch.float32, device='cuda'))
                    for _ in range(ODOUR):
                        b.step()
                    b.set_poisson(orn, torch.full((B, len(orn)), BASE_HZ, device='cuda'))
                    for _ in range(d):
                        b.step()
                    acc_v = torch.zeros((B, b.n), device='cuda')
                    acc_g = torch.zeros((B, b.n), device='cuda')
                    for _ in range(PROBE):
                        b.step()
                        acc_v += b.v
                        acc_g += b.g
                    v = (acc_v / PROBE).cpu().numpy()
                    g = (acc_g / PROBE).cpu().numpy()
                feats = dict(readout512=np.c_[g[:, obs].reshape(B, 256, 16).mean(2), v[:, obs].reshape(B, 256, 16).mean(2)],
                             mbon=np.c_[g[:, readsets['mbon']], v[:, readsets['mbon']]], kc=v[:, readsets['kc']], pn=v[:, readsets['pn']])
                row = dict(delay_ms=d)
                for k, X in feats.items():
                    row[k] = decode(X.astype(np.float64), y, seed=zlib.crc32(f'{cond["name"]}|{pair}|{d}|{k}'.encode()))
                rows.append(row)
            res['pairs'][f'{pair[0]}-{pair[1]}'] = rows
        rig.lesion([])
        res['seconds'] = time.time() - t0
        target.write_text(json.dumps(res), encoding='utf-8', newline='\n')
        summary = {p: [round(r['mbon']['accuracy'], 2) for r in rows] for p, rows in res['pairs'].items()}
        print(f'{cond["name"]}: {res["seconds"]:.0f} s, MBON accuracy by delay {summary}', flush=True)
    (OUT / 'done.json').write_text(json.dumps(dict(finished_unix=time.time())), encoding='utf-8')


if __name__ == '__main__':
    main()
