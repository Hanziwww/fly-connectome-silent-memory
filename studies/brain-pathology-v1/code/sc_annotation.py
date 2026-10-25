"""T-SC3 annotation (local; descriptive, F14.SC.annotation; tasks/sc-panels-v1.json): the molecular profile and genetic
access of the cell sets our analyses name.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\sc_annotation.py

Cell sets (indices into the model's 167,106 cells):
- trace carriers: nonzero decision-time cue difference in adaptation (delay 160 ms, prefix 4), M0 readout-only;
- cue ensembles of Fig. 6 (decision units), derived on M0 and on S1 (studies/brain-supplement-v2/tasks/x6-x7-sets.json);
- G3a populations: Kenyon cells and GABAergic antennal-lobe local neurons (credibility.cell_sets);
- readout cells (observed by the policy) and ellipsoid-body ring neurons (ER*);
- C01-vulnerable cells: among the memory-relevant cells (the union of the sets above except the G3a populations), the 1%
  with the largest absolute rate change under uniform C01 dose 1 (runs/sc-vulnerable, pulled), if recorded.
Per set: size, tier shares, superclass composition; per mapped MaleCNS type in a set: cells, source, unit, tier, the
adaptation index and the panel z-scores, Shal's relative expression, the EM transmitter and Davis 2020 driver lines.
Outputs: results/sc_annotation_sets.csv, results/sc_annotation_types.csv.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_common import FLYDATA, INDEX_GENES, KV4_GENE, RESULTS, ROOT, RUNS  # noqa: E402
from sc_index import cell_map, cell_values  # noqa: E402

SUPP = ROOT / 'studies' / 'brain-supplement-v2'
VULNERABLE_SHARE = 0.01


def driver_lines():
    m = pd.read_csv(FLYDATA / 'type_map_davis2020.csv', comment='#', dtype=str)
    out = {}
    for r in m.itertuples():
        lines = re.findall(r'driver ([A-Z]{1,3}\d{3,}[A-Z]?)', str(r.evidence))
        if lines and r.tier in ('exact', 'alias', 'fuzzy'):
            out.setdefault(r.malecns_type, set()).update(lines)
    return {k: ';'.join(sorted(v)) for k, v in out.items()}


def main():
    cells = cell_map()
    table = pd.read_csv(RESULTS / 'sc_unit_values.csv', dtype={'unit': str})
    cv = cell_values(cells, table)
    typ = cv['type'].fillna('').astype(str)
    cls = cv['class'].fillna('').astype(str)
    nt = cv['nt'].fillna('').astype(str)
    ens = json.loads((SUPP / 'tasks' / 'x6-x7-sets.json').read_text(encoding='utf-8'))['sets']
    n = len(cv)

    def mask(idx):
        m = np.zeros(n, bool)
        m[np.asarray(idx, int)] = True
        return m
    sets = {'trace carriers (M0)': cv['delta_adapt_m0'].fillna(0).ne(0).to_numpy(),
            'cue ensemble of Fig. 6 (M0)': mask(ens['m0']['cells']),
            'cue ensemble of Fig. 6 (S1)': mask(ens['s1']['cells']),
            'Kenyon cells (G3a)': typ.str.startswith('KC').to_numpy(),
            'GABAergic AL local neurons (G3a)': ((cls == 'ALLN') & (nt == 'gaba')).to_numpy(),
            'readout cells': cv['observed_by_policy'].to_numpy(bool),
            'ring neurons (ER)': typ.str.match(r'ER\d').to_numpy()}
    rates = RUNS / 'sc-vulnerable' / 'rates.parquet'
    if rates.exists():
        r = pd.read_parquet(rates)
        if not (r['bodyId'].to_numpy() == cv['bodyId'].to_numpy()).all():
            raise SystemExit('rate table order differs')
        relevant = sets['trace carriers (M0)'] | sets['cue ensemble of Fig. 6 (M0)'] | sets['cue ensemble of Fig. 6 (S1)'] | \
            sets['readout cells'] | sets['ring neurons (ER)']
        change = np.abs(r['rate_c01_dose1'].to_numpy() - r['rate_healthy'].to_numpy())
        k = int(round(VULNERABLE_SHARE * relevant.sum()))
        top = np.flatnonzero(relevant)[np.argsort(-change[relevant], kind='mergesort')[:k]]
        sets['C01-vulnerable cells (top 1% of memory-relevant)'] = mask(top)
    drivers = driver_lines()
    set_rows, type_rows = [], []
    for name, m in sets.items():
        sub = cv[m]
        tiers = sub['tier'].value_counts(normalize=True)
        set_rows.append(dict(cell_set=name, cells=int(m.sum()), **{f'tier_{t}': float(tiers.get(t, 0.0)) for t in
                                                                   ('exact', 'alias', 'fuzzy', 'class', 'unmapped')},
                             mapped_types=int(sub.loc[sub['tier'] != 'unmapped', 'type'].nunique()),
                             superclasses='; '.join(f'{k} {v}' for k, v in sub['superclass'].fillna('NA').value_counts().head(6).items()),
                             mean_A_mapped=float(sub['A'].mean()) if sub['A'].notna().any() else None))
        mapped = sub[sub['tier'] != 'unmapped']
        for t, g in mapped.groupby('type'):
            first = g.iloc[0]
            row = dict(cell_set=name, type=t, cells=len(g), superclass=first['superclass'], source=first['source'],
                       unit=first['unit'], tier=first['tier'], em_transmitter=g['nt'].astype(str).value_counts().idxmax(),
                       A=first['A'], shal_r=first['shal_r'], driver_lines=drivers.get(t, ''))
            for gene in INDEX_GENES + (KV4_GENE,):
                row[f'z_{gene}'] = first[f'z_{gene}']
                row[f'low_detection_{gene}'] = bool(first[f'low_detection_{gene}']) if pd.notna(first[f'low_detection_{gene}']) else None
            type_rows.append(row)
    pd.DataFrame(set_rows).to_csv(RESULTS / 'sc_annotation_sets.csv', index=False, lineterminator='\n', float_format='%.6g')
    pd.DataFrame(type_rows).sort_values(['cell_set', 'cells'], ascending=[True, False]).to_csv(
        RESULTS / 'sc_annotation_types.csv', index=False, lineterminator='\n', float_format='%.6g')
    print(pd.DataFrame(set_rows)[['cell_set', 'cells', 'tier_exact', 'tier_fuzzy', 'tier_class', 'tier_unmapped', 'mapped_types',
                                  'mean_A_mapped']].to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
