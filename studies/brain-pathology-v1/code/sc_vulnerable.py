"""T-SC3 annotation input (GPU host, .venv-linux): one per-cell activity recording on the P05 rig under the M0 memory input
(credibility.Rig.run_sequence, as the activity check of phase3_regime_a.py) for the healthy brain, uniform C01 at doses 0.5
and 1, and the class-weighted Kv4 C01 at dose 1 (tasks/sc-kv4-conditions.json). Descriptive (F14.SC.annotation): the
C01-vulnerable cells are the memory-relevant cells with the largest rate change (sc_annotation.py).

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/sc_vulnerable.py

Output runs/sc-vulnerable/rates.parquet (bodyId and one rate column in Hz per condition) and provenance.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
OUT = STUDY / 'runs' / 'sc-vulnerable'


def main():
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    import mechanism_base  # noqa: F401  (path setup)
    import pandas as pd
    import operators as op
    import sc_kv4
    from credibility import Rig
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(sc_kv4.CONDITIONS.read_bytes())
    rig = Rig(ROOT / 'runs' / 'linear-joint' / '20261002-p05-01' / 'model.pt', OUT / 'meta-rig')
    n = rig.n
    seq = rig.memory_input()
    every = np.arange(n)
    conditions = {'healthy': [], 'c01_dose0.5': [op.spec('threshold_offset', every, sc_kv4.C01_MV * 0.5, 'C01 dose 0.5')],
                  'c01_dose1': [op.spec('threshold_offset', every, sc_kv4.C01_MV * 1.0, 'C01 dose 1')],
                  'kv4_class_dose1': sc_kv4.offset_specs(op, sc_kv4.weight_vector(spec, dict(weights='class'), n), 1.0,
                                                         'Kv4-weighted C01 dose 1')}
    rates, seconds = {}, {}
    for name, specs in conditions.items():
        t0 = time.time()
        rig.lesion(specs)
        counts, total, _ = rig.run_sequence(seq)
        rates[name] = np.asarray(counts, float) / (total / 1000)
        seconds[name] = time.time() - t0
    rig.lesion([])
    bodies = pd.read_parquet(ROOT / 'flybrain' / 'cache' / 'neurons.parquet', columns=['bodyId'])['bodyId'].to_numpy()
    pd.DataFrame(dict(bodyId=bodies, **{f'rate_{k}': v for k, v in rates.items()})).to_parquet(OUT / 'rates.parquet', index=False)
    (OUT / 'provenance.json').write_text(json.dumps(dict(conditions=list(conditions), seconds=seconds, sequence_ms=float(total),
                                                         finished_unix=time.time()), indent=1), encoding='utf-8')
    print({k: float(v.mean()) for k, v in rates.items()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
