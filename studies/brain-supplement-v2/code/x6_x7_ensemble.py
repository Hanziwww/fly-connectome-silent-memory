"""X6 and X7 (plan v2, Phase 1C): causal tests of the cue ensemble of Fig. 6. GPU host.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x6_x7_ensemble.py

Sets: tasks/x6-x7-sets.json (x6_x7_prepare.py, built locally before any run). The cell sets are read from it; the edge
sets are regenerated here with the same code and seed (x6_x7_prepare.build) and must match the stored SHA-256.
Scenes: the M0 readout-only and S1 memory tasks (both on the P05 brain) at the 9 trained timings, replayed with the
frozen mechanism protocol (mechanism_model.memory_rollout). Each ensemble (derived on M0 or on S1) and its 20 null sets
are tested in both scenes:
- X6: outgoing gate of the set at strength 1 during the first decision (memory_rollout gate_spec, window 'read'; the
  paper's group-test protocol);
- X7: the synapses within the set (positions in the stored connectome) set to zero in the brain's weight matrix for
  the whole history, then restored. Synapses the brain's matrix does not hold are counted (already absent).
Outputs runs/x6-x7/<scene>.json: per timing the baseline and every condition (first-decision action, logits and legal
mask, status and steps per history); resumable per scene.
"""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
OUT = SUPP / 'runs' / 'x6-x7'
SETS = SUPP / 'tasks' / 'x6-x7-sets.json'
TRAINED = [(d, p) for d in (0, 40, 160) for p in (0, 1, 4)]
SCENES = {'m0': 'remote-dynamic-M0-head_only', 's1': 'supp-S1-ring-brain'}


def compact(rows):
    out = []
    for r in rows:
        d0 = r['decisions'][0] if r['decisions'] else None
        out.append(dict(case=r['case'], cue=r['cue'], context=r['context'], visible=r['visible'], status=r['status'], steps=len(r['decisions']),
                        first_action=d0['action'] if d0 else None, first_logits=d0['logits'] if d0 else None, first_legal=d0['legal'] if d0 else None))
    return out


def positions(base_keys, base_order, n, post, pre):
    """Positions in the brain's CSR values of the pairs (post, pre); -1 where the brain's matrix holds no entry."""
    q = post.astype(np.int64) * n + pre
    i = np.searchsorted(base_keys, q)
    i = np.minimum(i, len(base_keys) - 1)
    hit = base_keys[i] == q
    return np.where(hit, base_order[i], -1)


def main():
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(SUPP / 'code'))
    import mechanism_base as mb
    import torch
    from mechanism_model import Frozen, memory_rollout
    from x6_x7_prepare import build, edge_hash, edge_arrays
    spec = json.loads(SETS.read_text(encoding='utf-8'))
    W, sets = build()
    for key, s in sets.items():
        d = spec['sets'][key]
        if s['ensemble'].tolist() != d['cells'] or [c.tolist() for c, _ in s['nulls']] != [x['cells'] for x in d['nulls']]:
            raise RuntimeError(f'{key}: cell sets differ from the tracked file')
        if edge_hash([s['ensemble_edges']]) != d['ensemble_edges_sha256'] or edge_hash(s['edge_sets']) != d['edge_sets_sha256']:
            raise RuntimeError(f'{key}: regenerated edge sets differ from the tracked hashes')
    post, pre, _ = edge_arrays(W)
    n = W.shape[0]
    tasks = json.loads((MECH / 'inputs' / 'memory-tasks.json').read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    for scene, mid in SCENES.items():
        target = OUT / f'{scene}.json'
        if target.exists():
            continue
        t0 = time.time()
        model = mb.model_by_id(mid)
        task = next(x for x in tasks if x['id'] == model['task'])
        policy = Frozen(model, OUT / 'meta' / scene, batch=len(task['cases']))
        policy.load(MECH / model['checkpoints']['after']['path'])
        order_ids = policy.encoder.c.neurons['bodyId'].to_numpy(np.int64)
        import pandas as pd
        cached = pd.read_parquet(ROOT / 'flybrain' / 'cache' / 'neurons.parquet', columns=['bodyId'])['bodyId'].to_numpy(np.int64)
        if not np.array_equal(order_ids, cached):
            raise RuntimeError('the brain cell order differs from the cached annotations')
        matrix = policy.encoder.matrix
        base0 = matrix.base
        crow = base0.crow_indices().cpu().numpy().astype(np.int64)
        col = base0.col_indices().cpu().numpy().astype(np.int64)
        rows = np.repeat(np.arange(len(crow) - 1), np.diff(crow))
        keys = rows * n + col
        border = np.argsort(keys, kind='mergesort')
        bkeys = keys[border]
        conditions = []
        for src, s in sets.items():
            conditions.append(dict(kind='x6', source=src, set='ensemble', gate=s['ensemble'].tolist()))
            conditions += [dict(kind='x6', source=src, set=f'null{i}', gate=c.tolist()) for i, (c, _) in enumerate(s['nulls'])]
            for name, e in [('ensemble', s['ensemble_edges'])] + [(f'null{i}', e) for i, e in enumerate(s['edge_sets'])]:
                pos = positions(bkeys, border, n, post[e], pre[e])
                conditions.append(dict(kind='x7', source=src, set=name, positions=pos[pos >= 0], absent=int((pos < 0).sum()), requested=int(len(e))))
        record = dict(scene=scene, model=mid, task=model['task'], timings=TRAINED, sets_sha256=hashlib.sha256(SETS.read_bytes()).hexdigest(),
                      conditions=[{k: v for k, v in c.items() if k not in ('gate', 'positions')} for c in conditions], results=[])
        for d, p in TRAINED:
            entry = dict(delay=d, prefix=p, baseline=compact(memory_rollout(policy, task, d, p)), outcomes=[])
            for c in conditions:
                if c['kind'] == 'x6':
                    rows_ = memory_rollout(policy, task, d, p, gate_spec=dict(indices=c['gate'], strength=1.0, window='read'))
                else:
                    values = base0.values().clone()
                    values[torch.as_tensor(c['positions'], device=values.device)] = 0
                    matrix.base = torch.sparse_csr_tensor(base0.crow_indices(), base0.col_indices(), values, size=base0.shape)
                    try:
                        rows_ = memory_rollout(policy, task, d, p)
                    finally:
                        matrix.base = base0
                entry['outcomes'].append(compact(rows_))
            again = compact(memory_rollout(policy, task, d, p))
            if again != entry['baseline']:
                raise RuntimeError(f'{scene} d{d} p{p}: the baseline changed after the conditions (restoration failed)')
            record['results'].append(entry)
            print(f'{scene} d{d} p{p}: {len(conditions)} conditions, {time.time() - t0:.0f} s', flush=True)
        record['seconds'] = time.time() - t0
        target.write_text(json.dumps(record), encoding='utf-8', newline='\n')
        del policy, matrix, base0
        gc.collect()                   # operators.apply leaves a reference cycle (enc.bind); free the GPU memory now
        torch.cuda.empty_cache()
    (OUT / 'done.json').write_text(json.dumps(dict(finished_unix=time.time())), encoding='utf-8')


if __name__ == '__main__':
    main()
