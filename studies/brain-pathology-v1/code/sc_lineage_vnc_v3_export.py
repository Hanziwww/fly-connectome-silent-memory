"""Phase 4 figure data (descriptive; amendment 0014; host, .venv-sc): the v3 VNC bridge re-run with a per-cell export,
for the "Rosetta" view of Fig. 4 (atlas UMAP beside the MaleCNS VNC in the same hemilineage colours).

    .venv-sc/bin/python -I studies/brain-pathology-v1/code/sc_lineage_vnc_v3_export.py

Runs sc_lineage_vnc_v3.py's own functions unchanged (seed 0) into runs/sc-lineage-vnc-v3-export. Check: the recovered
cell count of every published cluster, the units and gates V1-V4 must equal runs/sc-lineage-vnc-v3 exactly, or nothing
is exported. Then a 2-D UMAP of the same 30-neighbour graph on 45 principal components (scanpy, random_state 0).
Output per cell: index, replicate, UMAP coordinates, Leiden cluster, assigned published cluster and hemilineage unit
(cells.parquet), plus check.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sc_lineage_vnc as run2  # noqa: E402
import sc_lineage_vnc_v3 as v3  # noqa: E402
from sc_common import RUNS  # noqa: E402

REF = RUNS / 'sc-lineage-vnc-v3'
OUT = RUNS / 'sc-lineage-vnc-v3-export'
GATE_KEYS = ('variable_genes', 'leiden_resolution', 'leiden_clusters', 'assignments_kept', 'cells_assigned',
             'cells_in_units', 'units', 'V1_rho', 'V2_concordance', 'V3_share', 'V4_rho', 'passed')


def main():
    import scanpy as sc
    OUT.mkdir(parents=True, exist_ok=True)
    v3.OUT = OUT                                   # finish() writes its tables here, never over the declared run
    captured = {}
    leiden_at = v3.leiden_at

    def capture(ad, target):
        captured['graph'] = ad
        return leiden_at(ad, target)

    v3.leiden_at = capture
    markers, clusters, hl_markers = run2.published_tables()
    X, genes, reps, kept = run2.read_dge()
    labels, resolution, trace, n_hvg, L = v3.cluster_cells(X, genes, reps, v3.OVERCLUSTER * len(markers))
    S, names = v3.marker_scores(L, genes, markers)
    assigned, table = v3.correspond(S, labels, names)
    extra = dict(variable_genes=n_hvg, leiden_resolution=resolution, leiden_clusters=int(len(set(labels))), leiden_trace=trace,
                 assignments_kept=int(table['kept'].sum()), one_to_one=int((table['step'] == 'one-to-one').sum()))
    gates, _ = v3.finish(assigned, L, genes, clusters, hl_markers, kept, X.shape[0], extra)
    ref_gates = json.loads((REF / 'gates.json').read_text(encoding='utf-8'))
    ref_clusters = pd.read_csv(REF / 'clusters.csv')
    new_clusters = pd.read_csv(OUT / 'clusters.csv')
    same_gates = {k: gates[k] == ref_gates[k] for k in GATE_KEYS}
    same_clusters = bool((ref_clusters['recovered_cells'].to_numpy() == new_clusters['recovered_cells'].to_numpy()).all())
    check = dict(same_gates=same_gates, same_cluster_recovery=same_clusters)
    (OUT / 'check.json').write_text(json.dumps(check, indent=1) + '\n', encoding='utf-8', newline='\n')
    if not (all(same_gates.values()) and same_clusters):
        raise SystemExit(f'the re-run does not reproduce the declared bridge: {check}')
    ad = captured['graph']
    sc.tl.umap(ad, random_state=0)
    unit_of = dict(zip(clusters['cluster'].astype(str), clusters['unit']))
    cells = pd.DataFrame(dict(cell=np.arange(len(labels), dtype=np.int64), replicate=np.asarray(reps).astype(str),
                              umap1=ad.obsm['X_umap'][:, 0], umap2=ad.obsm['X_umap'][:, 1], leiden=np.asarray(labels).astype(str),
                              published=[a if a is not None else None for a in assigned],
                              unit=[unit_of.get(a) if a is not None else None for a in assigned]))
    cells.to_parquet(OUT / 'cells.parquet', index=False)
    print(json.dumps(dict(check=check, cells=len(cells), in_units=int(cells['unit'].notna().sum())), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
