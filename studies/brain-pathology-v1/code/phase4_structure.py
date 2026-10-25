"""Phase 4 (local, CPU): where the retained cases' circuits sit relative to the visual tasks, structurally and functionally.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\phase4_structure.py

For each circuit (KCs, uniglomerular PNs, GABAergic ALLNs, MBONs, APL, ORNs, the visual input cells, the readout cells):
- synaptic distance from the nearest visual input cell and to the nearest readout cell (figure_data/fd_cell_distance.csv,
  P6b: BFS on the stored connectome);
- share of the cue-difference energy (sum of squared membrane-potential differences between cues, M0 readout-only, the
  last recorded window of the 160-ms-delay, 4-prefix history; mechanisms-v1 interpretation table) against the circuit's
  share of cells.
A circuit that is near in synapses but carries almost none of the cue difference is functionally unrecruited, and a null
effect of its lesion on the visual tasks says nothing about its own capacity. Writes results/phase4_structure.csv.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'


def main():
    a = pd.read_parquet(MECH / 'neurons' / 'annotations.parquet')
    d = pd.read_csv(ROOT / 'manuscript' / 'results' / 'figure_data' / 'fd_cell_distance.csv')
    if not np.array_equal(d['body_id'].to_numpy(), a['bodyId'].to_numpy()):
        raise RuntimeError('cell order differs between the distance table and the annotations')
    rows = []
    for model in ('remote-dynamic-M0-head_only', 'supp-S1-ring-brain', 'supp-S2-context-brain'):
        m = pd.read_parquet(MECH / 'interpretation' / f'{model}-memory-cells-d160-p4.parquet')
        dv2 = m['c0-visible0-delta-v'].to_numpy(np.float64) ** 2
        total = dv2.sum()
        t, c, nt = a['type'].fillna('').astype(str), a['class'].fillna('').astype(str), a['nt'].fillna('').astype(str)
        groups = {'Kenyon cells': t.str.startswith('KC'), 'uniglomerular PNs (ACh ALPN)': (c == 'ALPN') & (nt == 'acetylcholine'),
                  'GABAergic ALLNs': (c == 'ALLN') & (nt == 'gaba'), 'MBONs': c == 'MBON', 'APL': t == 'APL', 'ORNs': c == 'olfactory',
                  'visual input cells': a['direct_input'].astype(bool), 'readout cells': a['observed_by_policy'].astype(bool)}
        for name, mask in groups.items():
            mask = mask.to_numpy()
            hi, ho = d['hops_from_input'].to_numpy()[mask], d['hops_to_readout'].to_numpy()[mask]
            rows.append(dict(model=model, circuit=name, cells=int(mask.sum()), share_of_cells=float(mask.mean()),
                             share_of_cue_energy=float(dv2[mask].sum() / total),
                             enrichment=float((dv2[mask].sum() / total) / mask.mean()),
                             median_hops_from_input=float(np.median(hi[hi >= 0])) if (hi >= 0).any() else None,
                             median_hops_to_readout=float(np.median(ho[ho >= 0])) if (ho >= 0).any() else None))
    out = STUDY / 'results'
    out.mkdir(exist_ok=True)
    with open(out / 'phase4_structure.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model'][:22]:22s} {r['circuit']:30s} cells {r['cells']:6d} cue share {r['share_of_cue_energy']:.5f} "
              f"enrichment {r['enrichment']:7.3f} hops in {r['median_hops_from_input']} out {r['median_hops_to_readout']}")


if __name__ == '__main__':
    main()
