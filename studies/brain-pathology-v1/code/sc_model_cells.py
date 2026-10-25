"""Single-cell study, model side (local): one compact table of the model's 167,106 cells for the mapping (T-SC1) and the
annotation tests (T-SC3), so the GPU host needs no frozen-study files beyond it.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\sc_model_cells.py

Columns: bodyId, type, superclass, class, nt (EM-predicted), observed_by_policy (readout cells), direct_input (visual
input cells), the MaleCNS hemilineages itoleeHl and trumanHl, and per memory model the cue difference at the decision of
delay 160 ms, prefix 4 (mechanisms-v1 interpretation tables, cue 0 minus cue 1, hidden goal): delta_adapt_<m>,
delta_v_<m> in mV. Writes studies/brain-pathology-v1/inputs/sc_model_cells.parquet and prints its SHA-256.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_common import MODEL_CELLS, ROOT  # noqa: E402

M1 = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
MODELS = {'m0': 'remote-dynamic-M0-head_only', 'm0_joint': 'remote-dynamic-M0-joint', 'm0vis': 'supp-M0-visibility',
          's1': 'supp-S1-ring-brain', 's2': 'supp-S2-context-brain'}


def main():
    a = pd.read_parquet(M1 / 'neurons' / 'annotations.parquet',
                        columns=['bodyId', 'type', 'superclass', 'class', 'nt', 'observed_by_policy', 'direct_input'])
    hl = pd.read_feather(ROOT / 'flybrain' / 'data' / 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
                         columns=['bodyId', 'itoleeHl', 'trumanHl'])
    t = a.merge(hl.drop_duplicates('bodyId'), on='bodyId', how='left')
    for key, model in MODELS.items():
        m = pd.read_parquet(M1 / 'interpretation' / f'{model}-memory-cells-d160-p4.parquet',
                            columns=['bodyId', 'c0-visible0-delta-adapt', 'c0-visible0-delta-v'])
        m = m.rename(columns={'c0-visible0-delta-adapt': f'delta_adapt_{key}', 'c0-visible0-delta-v': f'delta_v_{key}'})
        t = t.merge(m, on='bodyId', how='left')
    if len(t) != len(a) or t['bodyId'].duplicated().any():
        raise RuntimeError('model cell table does not match the annotations')
    MODEL_CELLS.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(MODEL_CELLS, index=False)
    print(MODEL_CELLS.name, t.shape, hashlib.sha256(MODEL_CELLS.read_bytes()).hexdigest())
    return 0


if __name__ == '__main__':
    sys.exit(main())
