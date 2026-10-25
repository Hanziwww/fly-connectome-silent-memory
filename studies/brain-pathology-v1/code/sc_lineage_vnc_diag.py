"""T-SC1b diagnostic (no gate, no H1; host, .venv-sc -I): why do clusters 109 and 18 absorb a third of the assigned cells in run 2?
Recomputes run 2's scores exactly (sc_lineage_vnc.py) and reports, per cluster: marker detection breadth, the recovery
ratio, and for the sink cells their depth and second-best cluster. Writes runs/sc-lineage-vnc-diag/vnc_sink_diag.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats as st

ROOT = Path('/root/autodl-tmp/flybrain')
sys.path.insert(0, str(ROOT / 'studies/brain-pathology-v1/code'))
import sc_lineage_vnc as b  # noqa: E402

markers, clusters, _ = b.published_tables()
X, genes, reps, kept = b.read_dge()
umi = np.asarray(X.sum(axis=1)).ravel()
ngenes = np.asarray((X > 0).sum(axis=1)).ravel()
L = sp.diags(1e4 / umi) @ X
L.data = np.log1p(L.data)
L = L.tocsr()
index = {g: i for i, g in enumerate(genes)}
used = sorted({g for m in markers.values() for g in m if g in index})
D = L[:, [index[g] for g in used]].toarray()
Z = (D - D.mean(0)) / np.where(D.std(0) > 0, D.std(0), 1)
det = (D > 0).mean(0)
zpos = {g: j for j, g in enumerate(used)}
names = sorted(markers, key=lambda k: int(k) if k.isdigit() else k)
S = np.column_stack([Z[:, [zpos[g] for g in markers[k] if g in zpos]].mean(1) for k in names])
order = np.argsort(-S, axis=1)
best, second = S[np.arange(len(S)), order[:, 0]], S[np.arange(len(S)), order[:, 1]]
ok = (best >= b.MIN_SCORE) & (best - second >= b.MIN_MARGIN)
nm = np.asarray(names, dtype=object)
assigned = np.where(ok, nm[order[:, 0]], None)
rec = pd.Series(assigned).value_counts()
cl = clusters.set_index('cluster')
cl['recovered'] = pd.Series(rec).reindex(cl.index).fillna(0).astype(int)
assert int(cl['recovered'].sum()) == 16796, cl['recovered'].sum()
cl['ratio'] = cl['recovered'] / cl['published_cells']
cl['n_markers'] = [len([g for g in markers.get(k, []) if g in zpos]) for k in cl.index]
cl['marker_detection'] = [float(np.mean([det[zpos[g]] for g in markers.get(k, []) if g in zpos])) for k in cl.index]
cl['score_sd'] = [float(S[:, names.index(k)].std()) for k in cl.index]
cl['score_mean_depth_rho'] = [float(st.spearmanr(S[:, names.index(k)], umi).statistic) for k in cl.index]
out = dict(cells=int(len(S)), assigned=int(ok.sum()))
out['rho_ratio_vs_marker_detection'] = float(st.spearmanr(cl['ratio'], cl['marker_detection']).statistic)
out['rho_ratio_vs_score_depth'] = float(st.spearmanr(cl['ratio'], cl['score_mean_depth_rho']).statistic)
out['rho_ratio_vs_n_markers'] = float(st.spearmanr(cl['ratio'], cl['n_markers']).statistic)
sinks = {}
for k in ('109', '18'):
    m = assigned == k
    sec = pd.Series(nm[order[m, 1]]).value_counts().head(8)
    sinks[k] = dict(markers=markers[k], marker_detection={g: round(float(det[zpos[g]]), 3) for g in markers[k] if g in zpos},
                    cells=int(m.sum()), umi_median=float(np.median(umi[m])), genes_median=float(np.median(ngenes[m])),
                    best_median=float(np.median(best[m])), margin_median=float(np.median((best - second)[m])),
                    second_best=sec.to_dict(),
                    second_best_published=cl.loc[sec.index, 'published_cells'].to_dict(),
                    by_replicate=pd.Series(reps[m]).value_counts().to_dict())
out['sinks'] = sinks
out['umi_median'] = dict(all=float(np.median(umi)), assigned_other=float(np.median(umi[ok & ~np.isin(assigned, ['109', '18'])])),
                         unassigned=float(np.median(umi[~ok])))
out['genes_median'] = dict(all=float(np.median(ngenes)), assigned_other=float(np.median(ngenes[ok & ~np.isin(assigned, ['109', '18'])])),
                           unassigned=float(np.median(ngenes[~ok])))
cols = ['published_cells', 'neuronal', 'unit', 'recovered', 'ratio', 'n_markers', 'marker_detection', 'score_sd', 'score_mean_depth_rho']
out['clusters'] = cl[cols].reset_index().to_dict(orient='records')
OUT = ROOT / 'studies/brain-pathology-v1/runs/sc-lineage-vnc-diag'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'vnc_sink_diag.json').write_text(json.dumps(out, indent=1, default=str) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k != 'clusters'}, indent=1, default=str))
print(cl[cols].sort_values('ratio').to_string())
