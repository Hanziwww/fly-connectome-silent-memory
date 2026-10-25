"""Structural variants of the healthy brain (plan v2, Phase 4 'biological vs model-specific'; open item O12).

A variant changes one of flybrain's connection rules (flybrain/flybrain/brain.py LIFParams) and rebuilds the effective
weight matrix through flybrain's own pipeline: _shaped_weights (receptor signs, connection cap, path gains, same-type
gain), fan-in normalisation, w_syn. Nothing else changes: the trained type-level parameters (core) still act through
the matrix's pre/post gains and thresholds, and the lesion operators (operators.py) act on whatever matrix is bound.

Variants (declared 2026-10-19, before any variant was run): the defaults are receptor_model 'sign', same_type_gain
0.1 and conn_cap 60 synapses; each variant changes one rule.
- receptor-none: receptor_model None (the presynaptic-sign rule used before the receptor table);
- same-type-0.05 / same-type-0.2: same-type synapses x0.05 / x0.2 (half / twice the default damping);
- same-type-off: same_type_gain 1 (no damping of within-type synapses);
- conn-cap-30 / conn-cap-120: connection cap 30 / 120 synapse-equivalents (half / twice);
- conn-cap-off: conn_cap 0 (linear in synapse count, as in Shiu et al.).
'default' rebuilds the healthy matrix and must reproduce it bit for bit (base_graph_sha256 of the frozen brain.json).
"""
import dataclasses
import hashlib

import numpy as np

VARIANTS = {
    'default': {},
    'receptor-none': dict(receptor_model=None),
    'same-type-0.05': dict(same_type_gain=0.05),
    'same-type-0.2': dict(same_type_gain=0.2),
    'same-type-off': dict(same_type_gain=1.0),
    'conn-cap-30': dict(conn_cap=30.0),
    'conn-cap-120': dict(conn_cap=120.0),
    'conn-cap-off': dict(conn_cap=0.0),
}
HEALTHY_SHA256 = 'c430ca940c0e889d276ed529eb87c2f2853c5ba0833f7588a35f987f972a4545'   # archive/experiments/campaign/run/brain.json


def params(overrides):
    """The study brain's LIFParams (sokoban/lib/brain_model.py FullBrain) with one rule changed."""
    from flybrain.brain import LIFParams
    return LIFParams(dt=1., event_driven=False, surrogate_grad=True, **overrides)


def effective_matrix(c, overrides):
    """The effective weight matrix (csr, float32, mV per presynaptic spike) as flybrain.brain.Brain builds it."""
    import scipy.sparse as sp
    from flybrain.brain import _receptor, _shaped_weights
    p = params(overrides)
    w = _shaped_weights(c, p, _receptor(c, p))
    if p.input_norm_alpha > 0:
        if c.reference is not c:
            raise ValueError('the variant builder assumes the full connectome (its own fan-in reference)')
        tot = np.asarray(abs(w).sum(axis=1)).ravel()
        scale = np.clip((p.input_norm_ref / np.maximum(tot, 1.0)) ** p.input_norm_alpha, 0.02, 1.0).astype(np.float32)
        w = (sp.diags(scale) @ w).tocsr()
    w = w.copy()
    w.data = w.data * np.float32(p.w_syn)
    return w


def digest(w):
    """FullBrain.graph_digest of a csr matrix."""
    h = hashlib.sha256()
    for a in (w.indptr, w.indices, w.data):
        h.update(a.tobytes())
    return h.hexdigest()


def audit(w, healthy):
    """How a variant matrix differs from the healthy one (same sparsity pattern expected)."""
    same_pattern = bool(np.array_equal(w.indptr, healthy.indptr) and np.array_equal(w.indices, healthy.indices))
    out = dict(sha256=digest(w), nnz=int(w.nnz), same_pattern=same_pattern, abs_weight_sum=float(np.abs(w.data).sum()),
               abs_weight_sum_healthy=float(np.abs(healthy.data).sum()))
    if same_pattern:
        d = w.data != healthy.data
        out.update(entries_changed=int(d.sum()), sign_flips=int((np.sign(w.data) != np.sign(healthy.data)).sum()),
                   max_abs_change=float(np.abs(w.data - healthy.data).max()) if d.any() else 0.0)
    return out


_BUILT = {}                                            # per process: name -> (matrix, audit); trainers build several policies


def apply(enc, name):
    """Rebuild `enc` (a StudyBrain) with variant `name`; returns the audit. The healthy rebuild is checked first."""
    import torch
    from backend import Matrix, MixedMatrix, SegmentedMatrix
    if digest(enc.brain._W_cpu) != HEALTHY_SHA256:
        raise RuntimeError('the brain is not the frozen healthy one')
    if name not in _BUILT:
        healthy = effective_matrix(enc.c, {})
        if digest(healthy) != HEALTHY_SHA256:
            raise RuntimeError('the healthy rebuild does not reproduce the frozen effective matrix')
        w = healthy if not VARIANTS[name] else effective_matrix(enc.c, VARIANTS[name])
        _BUILT[name] = (w, audit(w, healthy))
    w, aud = _BUILT[name]
    overrides = VARIANTS[name]
    rec = dict(variant=name, overrides=overrides, **aud)
    if not overrides:
        return rec                                     # the brain is already the healthy one
    base = enc.brain._sparse(w, torch.float32)
    if enc.precision == 'float32_segmented':
        enc.matrix = SegmentedMatrix(base)
    elif enc.precision == 'mixed':
        enc.matrix = MixedMatrix(base)
    else:
        enc.matrix = Matrix(base, {'float32': torch.float32, 'float64': torch.float64}[enc.precision])
    enc.brain.W = enc.matrix
    enc.brain._W_cpu = w
    for k, v in overrides.items():                     # enc.params is brain.p (one object); record the rule in force
        setattr(enc.params, k, v)
    rec['params'] = {k: getattr(enc.params, k) for k in ('receptor_model', 'same_type_gain', 'conn_cap')}
    return rec


def lif_dict(overrides):
    return dataclasses.asdict(params(overrides))


def main():
    """CPU check and record (local): the default rebuild equals the frozen healthy matrix; how each variant changes it.

        .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\variants.py
    """
    import csv
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / 'flybrain'))
    from flybrain.connectome import load
    c = load(cache_dir=root / 'flybrain' / 'cache', verbose=False)
    healthy = effective_matrix(c, {})
    if digest(healthy) != HEALTHY_SHA256:
        raise SystemExit('the healthy rebuild does not reproduce the frozen effective matrix')
    rows = []
    for name, ov in VARIANTS.items():
        a = audit(healthy if not ov else effective_matrix(c, ov), healthy)
        rows.append(dict(variant=name, overrides=repr(ov), **a))
        print(name, {k: v for k, v in a.items() if k != 'sha256'}, flush=True)
    path = root / 'studies' / 'brain-pathology-v1' / 'results' / 'phase2_variant_matrices.csv'
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
