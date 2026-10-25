"""T-SC6, intrinsic variants (F14.SC.variant, .secondary, .descriptive; design declared in tasks/sc-panels-v1.json 'variant',
amendment 0005). Run because Gate SC-B passed: the F14.SC.kv4 tests at doses 1.5 and 2 are significant (Holm).

    local, once:  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\sc_variant.py --prepare
    host:         sh studies/brain-pathology-v1/code/sc_variant_queue.sh <gpu>   (one worker per GPU)

--prepare writes, once and before any T-SC6 run (amendment 0012):
- tasks/sc-variant-lesion.json: adaptation jump x 0.5 / x 1.5 for the mapped MaleCNS types (mapping v1, any tier) in the
  low / high tertile of the adaptation index A (tertiles over the mapped types, each type once); x 1 for the middle
  tertile and every unmapped cell (no spec);
- tasks/sc-variant-null-lesion.json: the same per-type multipliers permuted across the mapped types
  (numpy default_rng(20261022));
- tasks/sc-variant-design.json: thresholds, counts, the run list (M0 readout-only r0-9 and S1 readout-only r0-4 per brain,
  replicate_training.py --lesion <file>, X1 protocol, then --verify; sc_variant_evaluate.py per run).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
TASKS = STUDY / 'tasks'
SEED = 20261022
BRAINS = {'variant': TASKS / 'sc-variant-lesion.json', 'null': TASKS / 'sc-variant-null-lesion.json'}
DESIGN = TASKS / 'sc-variant-design.json'
MODELS = {'M0': ('remote-dynamic-M0-head_only', range(10)), 'S1': ('supp-S1-ring-brain', range(5))}


def runs():
    """(brain, family, model id, replicate, run folder) for every T-SC6 training run."""
    return [(b, fam, model, r, f'studies/brain-pathology-v1/runs/sc-variant-{b}-{fam}-r{r}')
            for b in BRAINS for fam, (model, reps) in MODELS.items() for r in reps]


def prepare():
    sys.path.insert(0, str(STUDY / 'code'))
    from sc_index import cell_map
    for p in (*BRAINS.values(), DESIGN):
        if p.exists():
            raise SystemExit(f'{p.name} exists; task files are written once, before their runs')
    cells = cell_map()
    vals = pd.read_csv(STUDY / 'results' / 'sc_unit_values.csv', dtype={'unit': str})
    cv = cells.merge(vals[['source', 'unit', 'A']], on=['source', 'unit'], how='left')
    mapped = cv[cv['tier'] != 'unmapped']
    per_type = mapped.groupby('type')['A'].agg(['first', 'nunique'])
    if (per_type['nunique'] > 1).any() or per_type['first'].isna().any():
        raise SystemExit('a mapped type has no single adaptation index')
    a = per_type['first']
    q1, q2 = np.quantile(a.to_numpy(float), [1 / 3, 2 / 3])
    mult = pd.Series(np.where(a <= q1, 0.5, np.where(a > q2, 1.5, 1.0)), index=a.index)
    null = pd.Series(np.random.default_rng(SEED).permutation(mult.to_numpy()), index=mult.index)
    types = cells['type'].astype('object').to_numpy()

    def specs(m, label):
        out = []
        for value in (0.5, 1.5):
            chosen = set(m.index[m == value])
            idx = np.flatnonzero([t in chosen for t in types])
            out.append(dict(kind='adaptation_scale', cells=idx.tolist(), value=value,
                            label=f'T-SC6 {label}: adaptation jump x{value:g}'))
        return out
    texts = {}
    for brain, m in (('variant', mult), ('null', null)):
        text = json.dumps(specs(m, brain), separators=(',', ':'))
        BRAINS[brain].write_text(text, encoding='utf-8', newline='\n')
        texts[brain] = hashlib.sha256(text.encode()).hexdigest()
    design = dict(
        task='T-SC6 intrinsic variants (tasks/sc-panels-v1.json variant; Gate SC-B passed through F14.SC.kv4 at doses 1.5-2)',
        declared='2026-10-22, before any T-SC6 run',
        mapping='mapping v1 (amendment 0004), any tier; per type the adaptation index A of its best unit (results/sc_unit_values.csv)',
        tertiles=dict(q1=float(q1), q2=float(q2), types=int(len(a)), low=int((mult == 0.5).sum()), middle=int((mult == 1.0).sum()),
                      high=int((mult == 1.5).sum())),
        cells={b: {f'x{v:g}': int(sum(1 for t in types if t in set(m.index[m == v]))) for v in (0.5, 1.0, 1.5)}
               for b, m in (('variant', mult), ('null', null))},
        null_seed=SEED, lesion_files={b: dict(file=f'tasks/{p.name}', sha256=texts[b]) for b, p in BRAINS.items()},
        runs=[dict(brain=b, family=f, model=m, replicate=r, out=o) for b, f, m, r, o in runs()],
        training='replicate_training.py --model <model> --replicate r --out <run> --lesion <brain file>, then --verify (the X1 '
                 'protocol: timing order seed 1000 + r, readout jitter seed 2000 + r; the frozen trainers and stopping rules)',
        evaluation='sc_variant_evaluate.py: as-trained (all 35 timings; must reproduce the run verification), adaptation clear '
                   'and swap at decision onset (prefix-0 trained timings), and for M0 the C01 shifts of dy2_evaluate.SHIFTS_MV on '
                   'top of the variant (trained timings); net use, lock and C01 ED50 as in tasks/dy2-design.json',
        references='X1 M0 readout-only r0-r9 (results/dy2_models.csv); S1 tests are within the variant brain')
    text = json.dumps(design, indent=1) + '\n'
    DESIGN.write_text(text, encoding='utf-8', newline='\n')
    sys.path.insert(0, str(ROOT))
    from studykit.registry import Study
    study = Study.named('brain-pathology-v1')
    study.lock_tasks(status='draft', note='inputs written before their runs (plan v2 batches); re-locked for amendment 0012 '
                                          '(T-SC6 intrinsic variants)')
    path = study.amend('T-SC6 intrinsic variant brains', dict(
        trigger='Gate SC-B (tasks/sc-panels-v1.json): H1 v1 null, but F14.SC.kv4 significant at doses 1.5 and 2 (Holm), so '
                'T-SC6 runs as declared',
        added_task_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (*BRAINS.values(), DESIGN)},
        design=dict(tertiles=design['tertiles'], cells=design['cells'], runs=len(design['runs']))))
    print(json.dumps(design['tertiles']), json.dumps(design['cells']), path.name)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', action='store_true')
    a = ap.parse_args()
    prepare() if a.prepare else print('\n'.join(' '.join(map(str, r)) for r in runs()))
