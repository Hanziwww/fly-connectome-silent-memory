"""T-SC1b bridge v3 on rtx5090x4 (.venv-sc): the VNC lineage tier by cluster correspondence, after runs 1-2 of
sc_lineage_vnc.py failed gate V1 (amendments 0010-0011). Declared by amendment 0013 (approved by the user 2026-10-22)
before it read any atlas cell; the final attempt.

    .venv-sc/bin/python -I studies/brain-pathology-v1/code/sc_lineage_vnc_v3.py            (the declared run)
    .venv-sc/bin/python -I studies/brain-pathology-v1/code/sc_lineage_vnc_v3.py --selftest (synthetic data only)

Why (runs/sc-lineage-vnc-diag/vnc_sink_diag.json, sc_lineage_vnc_diag.py): in run 2, 35% of the assigned cells went to
two published clusters (109: 2,935 cells against 63 published; 18: 2,982 against 363). Each has a single published
marker (109: br; 18: CG13056), so its score is one gene's z, with 3-4 times the spread of the 12-25-marker means of the
other clusters: any cell that detects that gene wins the per-cell argmax. The other clusters are under-filled (median
recovery 0.42).
Procedure (the authors' Seurat 2.3.4 pipeline, Allen et al. 2020, Methods "Data analysis with seurat", in scanpy):
- the authors' QC and log1p(cp10k), as run 2 (26,775 cells);
- variable genes per sex (female and male replicates pooled): Seurat v2 dispersion method, mean and dispersion
  cut-offs 0.001; their intersection;
- per sex, as the authors: replicate (a factor), UMI and mitochondrial fraction regressed out per gene, then scaled and
  clipped at 10; the two sexes stacked;
- 45 principal components (the authors aligned the sexes by 45 CCA dimensions, not available here); a
  30-nearest-neighbour graph; Leiden over-clustered at twice the published granularity, 240 clusters (bisection on log
  resolution in [0.1, 100], seed 0; else the closest count), so that a published cluster is a union of new ones and
  boundary differences resolve by merging, not splitting;
- correspondence: run 2's cell scores (mean z of each published cluster's 25 strongest markers), each standardised over
  the cells so that clusters with few markers do not dominate; each new cluster's mean standardised score per published
  cluster; every published cluster first takes one new cluster by a one-to-one assignment maximising the summed score
  (Hungarian), then every remaining new cluster joins its best-scoring published cluster; an assignment counts if its
  score is >= 0.25, else those cells stay unassigned.
Units, label map and gates V1-V3 as run 2, thresholds unchanged. Final attempt: if V1 fails the VNC stays unmapped.
Outputs (runs/sc-lineage-vnc-v3/): as run 2 (units_allen2020vnc_mean/frac.parquet, label_map.csv, clusters.csv,
units.csv, gates.json, provenance.json) plus correspondence.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sc_lineage_vnc as run2  # noqa: E402
from sc_common import MODEL_CELLS, RUNS, write_provenance  # noqa: E402

OUT = RUNS / 'sc-lineage-vnc-v3'
N_PCS, N_NEIGHBORS, SCALE_MAX, HVG_CUTOFF, SEED = 45, 30, 10.0, 0.001, 0
OVERCLUSTER, RES_LOW, RES_HIGH, BISECTIONS = 2, 0.1, 100.0, 40


def variable_genes(ad, sex):
    """Seurat v2 FindVariableGenes (dispersion method; x.low.cutoff = y.cutoff = 0.001) on one sex's cells."""
    import scanpy as sc
    sub = ad[ad.obs['sex'] == sex].copy()
    sc.pp.highly_variable_genes(sub, flavor='seurat', min_mean=HVG_CUTOFF, max_mean=np.inf, min_disp=HVG_CUTOFF)
    return set(sub.var_names[sub.var['highly_variable']])


def regress_scale(Y, obs):
    """Per gene: least-squares residuals on replicate (factor), UMI and mitochondrial fraction; z-scored, clipped."""
    reps = pd.get_dummies(obs['replicate'].astype(str), drop_first=True).to_numpy(float)
    D = np.column_stack([np.ones(len(obs)), reps, obs['n_umi'].to_numpy(float), obs['pct_mito'].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(D, Y, rcond=None)
    R = Y - D @ beta
    sd = R.std(axis=0)
    R = (R - R.mean(axis=0)) / np.where(sd > 0, sd, 1)
    return np.clip(R, -SCALE_MAX, SCALE_MAX)


def leiden_at(ad, target):
    """Leiden labels at the resolution giving `target` clusters (bisection on log resolution; else the closest count,
    ties to the larger count), with the resolution and the bisection trace."""
    import scanpy as sc
    trace, lo, hi, best = [], np.log(RES_LOW), np.log(RES_HIGH), None
    for _ in range(BISECTIONS):
        r = float(np.exp((lo + hi) / 2))
        sc.tl.leiden(ad, resolution=r, random_state=SEED, flavor='igraph', n_iterations=2, directed=False, key_added='leiden')
        n = int(ad.obs['leiden'].nunique())
        trace.append(dict(resolution=r, clusters=n))
        key = (abs(n - target), -n)
        if best is None or key < best[0]:
            best = (key, r, ad.obs['leiden'].astype(str).to_numpy().copy())
        if n == target:
            break
        lo, hi = (np.log(r), hi) if n < target else (lo, np.log(r))
    return best[2], best[1], trace


def cluster_cells(X, genes, reps, target):
    """The authors' clustering, in scanpy, on the QC'd counts (cells x genes). Returns the Leiden labels, resolution,
    bisection trace, number of variable genes and the log1p(cp10k) matrix (csr)."""
    import anndata
    import scanpy as sc
    import scipy.sparse as sp
    X = sp.csr_matrix(X, dtype=np.float32)
    names = pd.Index(genes).astype(str)
    umi = np.asarray(X.sum(axis=1)).ravel()
    mito = np.asarray(X[:, np.flatnonzero(names.str.startswith('mt:'))].sum(axis=1)).ravel()
    obs = pd.DataFrame(dict(replicate=pd.Series(reps).astype(str).to_numpy(),
                            sex=np.where(pd.Series(reps).str.contains('female'), 'female', 'male'),
                            n_umi=umi, pct_mito=mito / np.where(umi > 0, umi, 1)), index=[f'c{i}' for i in range(len(reps))])
    ad = anndata.AnnData(X=X, obs=obs)
    ad.var_names = names
    ad.var_names_make_unique()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    L = sp.csr_matrix(ad.X)
    hvg = sorted(variable_genes(ad, 'female') & variable_genes(ad, 'male'))
    cols = ad.var_names.get_indexer(hvg)
    scaled = np.zeros((ad.n_obs, len(hvg)), dtype=np.float32)
    for sex in ('female', 'male'):
        m = (ad.obs['sex'] == sex).to_numpy()
        scaled[m] = regress_scale(L[m][:, cols].toarray().astype(np.float64), ad.obs[m])
    pcs = anndata.AnnData(X=scaled, obs=ad.obs.copy())
    sc.tl.pca(pcs, n_comps=N_PCS, svd_solver='arpack', random_state=SEED)
    sc.pp.neighbors(pcs, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, random_state=SEED)
    labels, resolution, trace = leiden_at(pcs, target)
    return labels, resolution, trace, len(hvg), L


def marker_scores(L, genes, markers):
    """Run 2's cell x published-cluster score: mean per-gene z (over all cells) of each cluster's 25 strongest markers."""
    index = {g: i for i, g in enumerate(genes)}
    used = sorted({g for m in markers.values() for g in m if g in index})
    D = L[:, [index[g] for g in used]].toarray()
    Z = (D - D.mean(0)) / np.where(D.std(0) > 0, D.std(0), 1)
    zpos = {g: j for j, g in enumerate(used)}
    names = sorted(markers, key=lambda k: int(k) if k.isdigit() else k)
    S = np.column_stack([Z[:, [zpos[g] for g in markers[k] if g in zpos]].mean(1) for k in names])
    return S, names


def correspond(S, labels, names):
    """New clusters (labels) to published clusters (columns of S, named `names`): scores standardised per published
    cluster over the cells, mean per new cluster; each published cluster takes one new cluster by the Hungarian
    assignment maximising the summed score, each remaining new cluster joins its best-scoring published cluster;
    assignments below MIN_SCORE are dropped. Returns the per-cell published cluster (None if unassigned) and the
    table."""
    from scipy.optimize import linear_sum_assignment
    sd = S.std(axis=0)
    Sz = (S - S.mean(axis=0)) / np.where(sd > 0, sd, 1)
    new = sorted(set(labels), key=lambda x: int(x) if str(x).isdigit() else str(x))
    M = np.vstack([Sz[labels == c].mean(axis=0) for c in new])
    rows, cols = linear_sum_assignment(-M)
    pick = dict(zip(rows, cols))
    table, to = [], {}
    for i in range(len(new)):
        top = int(np.argmax(M[i]))
        j, step = (pick[i], 'one-to-one') if i in pick else (top, 'best match')
        keep = bool(M[i, j] >= run2.MIN_SCORE)
        table.append(dict(new_cluster=new[i], cells=int((labels == new[i]).sum()), step=step, published=names[j],
                          score=float(M[i, j]), best_published=names[top], best_score=float(M[i, top]), kept=keep))
        if keep:
            to[new[i]] = names[j]
    assigned = np.array([to.get(c) for c in labels], dtype=object)
    return assigned, pd.DataFrame(table)


def finish(assigned, L, genes, clusters, hl_markers, kept, n_cells, extra):
    """Units, label map, gates V1-V4 and outputs exactly as run 2 (sc_lineage_vnc.py main, after the assignment)."""
    from scipy import stats as st
    recovered = pd.Series(assigned[pd.notna(assigned)]).value_counts()
    clusters['recovered_cells'] = clusters['cluster'].map(recovered).fillna(0).astype(int)
    clusters.to_csv(OUT / 'clusters.csv', index=False, lineterminator='\n')
    v1 = float(st.spearmanr(clusters['recovered_cells'], clusters['published_cells']).statistic)
    unit_of = dict(zip(clusters['cluster'], clusters['unit']))
    cell_unit = np.array([unit_of.get(a) if a is not None else None for a in assigned], dtype=object)
    sizes = pd.Series(cell_unit[pd.notna(cell_unit)]).value_counts()
    units = sorted((u for u, n in sizes.items() if n >= run2.MIN_UNIT), key=lambda u: run2.hl_key(u.split('/')[0]))
    model = pd.read_parquet(MODEL_CELLS, columns=['bodyId', 'trumanHl', 'nt'])
    lmap = run2.label_map(model['trumanHl'].dropna().astype(str), set(units))
    lm = pd.DataFrame([dict(malecns_label=k, unit=u, tier=t, malecns_cells=int((model['trumanHl'] == k).sum()))
                       for k, (u, t) in lmap.items()])
    lm.to_csv(OUT / 'label_map.csv', index=False, lineterminator='\n')
    index = {g: i for i, g in enumerate(genes)}
    rows_mean, rows_frac, info = [], [], []
    for u in units:
        sub = L[cell_unit == u]
        rows_mean.append(np.asarray(sub.mean(axis=0)).ravel())
        det = sub.copy()
        det.data = np.ones_like(det.data)
        rows_frac.append(np.asarray(det.mean(axis=0)).ravel())
        labs = lm.loc[lm['unit'] == u, 'malecns_label'].tolist() if len(lm) else []
        em = model.loc[model['trumanHl'].astype(str).isin(labs), 'nt'].astype(str).value_counts()
        em = em[em.index != 'unknown']
        info.append(dict(unit=u, tier='lineage_pair' if '/' in u else 'lineage', cells=int((cell_unit == u).sum()),
                         clusters=';'.join(clusters.loc[clusters['unit'] == u, 'cluster']), malecns_labels=';'.join(labs),
                         malecns_cells=int(model['trumanHl'].astype(str).isin(labs).sum()),
                         em_transmitter=em.index[0] if len(em) else None))
    keep_g = ~pd.Index(genes).duplicated()
    mean = pd.DataFrame(np.vstack(rows_mean)[:, keep_g], index=units, columns=genes[keep_g])
    frac = pd.DataFrame(np.vstack(rows_frac)[:, keep_g], index=units, columns=genes[keep_g])
    mean.index.name = frac.index.name = 'unit'
    mean.to_parquet(OUT / 'units_allen2020vnc_mean.parquet')
    frac.to_parquet(OUT / 'units_allen2020vnc_frac.parquet')
    ui = pd.DataFrame(info)
    rel = {}
    for nt, gs in run2.NT_GENES.items():
        cols = [g for g in gs if g in mean.columns]
        v = mean[cols].mean(axis=1)
        rel[nt] = v / max(float(v.max()), 1e-12)
    ui['transcriptome_transmitter'] = pd.DataFrame(rel).idxmax(axis=1).to_numpy()
    known = ui.dropna(subset=['em_transmitter'])
    known = known[known['malecns_cells'] > 0]
    v2 = float((known['transcriptome_transmitter'] == known['em_transmitter']).mean()) if len(known) else float('nan')
    v3_hits = []
    for u, gs in hl_markers.items():
        idx = [index[g] for g in gs if g in index]
        m = cell_unit == u
        if u in units and idx and m.any():
            G = L[:, idx].toarray()
            z = (G - G.mean(0)) / np.where(G.std(0) > 0, G.std(0), 1)
            v3_hits.append(dict(unit=u, markers=len(idx), mean_z=float(z[m].mean())))
    v3 = float(np.mean([h['mean_z'] > 0 for h in v3_hits])) if v3_hits else float('nan')
    v4 = float(st.spearmanr(ui['cells'], ui['malecns_cells']).statistic) if len(ui) > 2 else float('nan')
    ui.to_csv(OUT / 'units.csv', index=False, lineterminator='\n')
    gates = dict(method='v3 cluster correspondence (amendment 0013)', **extra, qc=kept, cells_after_qc=int(n_cells),
                 cells_assigned=int(pd.notna(assigned).sum()), cells_in_units=int(np.isin(cell_unit, units).sum()), units=units,
                 label_map_labels=len(lm), V1_rho=v1, V1=bool(v1 >= 0.7), V2_concordance=v2, V2=bool(v2 >= 0.8),
                 V2_n=int(len(known)), V3_share=v3, V3=bool(v3 >= 0.8), V3_detail=v3_hits, V4_rho=v4)
    gates['passed'] = bool(gates['V1'] and gates['V2'] and gates['V3'])
    (OUT / 'gates.json').write_text(json.dumps(gates, indent=1) + '\n', encoding='utf-8', newline='\n')
    return gates, ui


def selftest():
    """Synthetic counts, two sexes x two replicates, 14 planted clusters of unequal size (60-600 cells) with 8 marker
    genes each; two clusters are published with a single marker (as 109 and 18). The pipeline must recover the sizes
    (Spearman rho >= 0.9) and label >= 95% of cells correctly; nothing is written."""
    import scipy.sparse as sp
    from scipy import stats as st
    rng = np.random.default_rng(1)
    k, g_per, n_genes = 14, 8, 800
    sizes = rng.integers(60, 600, size=k)
    truth = np.repeat(np.arange(k), sizes)
    lam = np.tile(rng.gamma(0.6, 0.6, size=n_genes), (len(truth), 1))
    for c in range(k):
        lam[truth == c, c * g_per:(c + 1) * g_per] *= 12
    reps = rng.choice(['female_rep1', 'female_rep2', 'male_rep1', 'male_rep2'], size=len(truth))
    lam[np.char.startswith(reps.astype(str), 'male'), -40:] *= 1.6          # a sex difference in 40 genes
    depth = rng.lognormal(np.log(2500), 0.3, size=len(truth))
    lam = lam / lam.sum(axis=1, keepdims=True) * depth[:, None]
    X = sp.csr_matrix(rng.poisson(lam).astype(float))
    genes = np.array([f'g{i}' for i in range(n_genes - 2)] + ['mt:a', 'mt:b'])
    labels, resolution, trace, n_hvg, L = cluster_cells(X, genes, reps, target=OVERCLUSTER * k)
    markers = {str(c): [f'g{i}' for i in range(c * g_per, (c + 1) * g_per)] for c in range(k)}
    markers['0'], markers['1'] = ['g0'], ['g8']
    S, names = marker_scores(L, genes, markers)
    assigned, table = correspond(S, labels, names)
    rec = pd.Series(assigned[pd.notna(assigned)]).value_counts().reindex(names).fillna(0)
    rho = float(st.spearmanr(rec.to_numpy(), sizes[[int(n) for n in names]]).statistic)
    accuracy = float(np.mean([a == str(t) for a, t in zip(assigned, truth)]))
    print(dict(resolution=round(resolution, 3), clusters=int(len(set(labels))), hvg=n_hvg, rho=round(rho, 4),
               accuracy=round(accuracy, 4), kept=int(table['kept'].sum())))
    assert rho >= 0.9 and accuracy >= 0.95, (len(set(labels)), rho, accuracy)
    print('selftest ok')
    return 0


def main():
    if '--selftest' in sys.argv:
        return selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    markers, clusters, hl_markers = run2.published_tables()
    X, genes, reps, kept = run2.read_dge()
    labels, resolution, trace, n_hvg, L = cluster_cells(X, genes, reps, OVERCLUSTER * len(markers))
    S, names = marker_scores(L, genes, markers)
    assigned, table = correspond(S, labels, names)
    table.to_csv(OUT / 'correspondence.csv', index=False, lineterminator='\n')
    extra = dict(variable_genes=n_hvg, leiden_resolution=resolution, leiden_clusters=int(len(set(labels))), leiden_trace=trace,
                 assignments_kept=int(table['kept'].sum()),
                 one_to_one=int((table['step'] == 'one-to-one').sum()))
    gates, ui = finish(assigned, L, genes, clusters, hl_markers, kept, X.shape[0], extra)
    write_provenance(OUT, __file__, list(run2.DGE) + sorted(run2.ELIFE.glob('*.xlsx')) + [MODEL_CELLS])
    print(json.dumps({k: v for k, v in gates.items() if k not in ('V3_detail', 'leiden_trace')}, indent=1))
    print(ui.to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
