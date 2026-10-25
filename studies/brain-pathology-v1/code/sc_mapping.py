"""T-SC1, mapping v1 (rtx5090x4, .venv-sc): per-source expression of every gene per mapped unit (M3), the mapping of the
model's 167,106 cells to those units (M1; tiers exact, alias, fuzzy, class from flybrain's type maps), QC (M4) and the
coverage report that decides Gate SC-A. No hypothesis is tested here.

    .venv-sc/bin/python studies/brain-pathology-v1/code/sc_mapping.py

Units per source (labels as in flybrain's maps):
- davis2020: driver-line samples of GSE116969 dataTable3 (<type>_<driver>_<replicate>) grouped by cell type;
  expression = mean over samples of log1p(TPM); no detection fraction (bulk);
- ozel2021: clusters of the Adult sheet of GSE142787 (mean log-normalised expression, as published);
- davie2018: `annotation` of the 57k whole-brain cells (GSE107451); mean log1p(cp10k) from counts, detection fraction;
- afca2023: `afca_annotation` of the Aging Fly Cell Atlas head (GSE218661), all ages and sexes; mean of the published
  log-normalised values (checked to be log1p(cp10k)) and detection fraction;
- adfca2025: `adfca_annotation` of the AD Fly Cell Atlas head (GSE261656), control samples only; from `umi_counts`.
Mapping (M1): model cell type -> unit through type_map_davis2020 (source_name), type_map_ozel2021 (source_cluster) and
type_map_central (davie2018 annotation; fca2022 labels applied to afca2023 and adfca2025 where the label strings match).
Each cell takes its best tier; among equal tiers the source order davis2020 > ozel2021 > davie2018 > afca2023 >
adfca2025. All candidate units are kept for the cross-source checks. Kurmangaliyev et al. 2020 (D3) is not used: its
matrix was not downloaded and Ozel's adult optic lobe covers the same types.
QC (M4): (1) the AFCA values are log1p(cp10k) (expm1 row sums 10,000); (2) Davie per-annotation values of the
transmitter and receptor genes reproduce flybrain's expression_central.csv; (3) transmitter concordance: the call per
mapped unit (marker genes relative to their highest unit) vs the majority EM-predicted transmitter of the mapped MaleCNS
type; (4) marker sanity (Kenyon cells: ey, Pka-C1; projection neurons: acj6; ring neurons: Gad1, VGAT); (5) cross-source
concordance: Spearman rho of the two profiles of a MaleCNS type mapped in two sources against random mismatched unit
pairs (genes: the 2,000 most variable genes shared by both sources).
Outputs (runs/sc-mapping/): units_<source>_mean.parquet and _frac.parquet (unit x gene), cell_map.parquet (best unit
per cell), cell_candidates.parquet (every candidate), unit_sizes.csv, coverage.csv, mapping_qc.csv, provenance.json.
"""
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_common import (DERIVED, EXT, FLYDATA, MODEL_CELLS, RUNS, SOURCE_ORDER, TIER_RANK,  # noqa: E402
                       group_mean_frac, write_provenance)

OUT = RUNS / 'sc-mapping'
DAVIS = EXT / 'GSE116969' / 'GSE116969_dataTable3.genes_x_samples_TPM.coding_genes.txt.gz'
OZEL = EXT / 'GSE142787' / 'GSE142787_Log_normalized_average_expression.xlsx'
DAVIE = EXT / 'GSE107451' / 'GSE107451_DGRP-551_w1118_WholeBrain_57k_0d_1d_3d_6d_9d_15d_30d_50d_10X_DGEM_MEX.mtx.tsv.tar.gz'
DAVIE_META = EXT / 'GSE107451' / 'GSE107451_DGRP-551_w1118_WholeBrain_57k_Metadata.tsv.gz'
AFCA_GZ = EXT / 'GSE218661' / 'GSE218661_adata_head_S_v1.0.h5ad.gz'
AFCA = DERIVED / 'GSE218661_adata_head_S_v1.0.h5ad'                  # gunzipped copy of AFCA_GZ
ADFCA = EXT / 'GSE261656' / 'GSE261656_adata_adfca_head_v1.0.h5ad'
MAPS = [FLYDATA / f for f in ('type_map_davis2020.csv', 'type_map_ozel2021.csv', 'type_map_central.csv',
                              'expression_central.csv')]
NT_MARKERS = {'acetylcholine': ['VAChT', 'ChAT'], 'gaba': ['Gad1', 'VGAT'], 'glutamate': ['VGlut'], 'histamine': ['Hdc'],
              'dopamine': ['ple', 'DAT'], 'serotonin': ['Trh', 'SerT'], 'octopamine': ['Tbh', 'Tdc2']}


def save_units(source, labels, n, mean, frac, genes):
    df = pd.DataFrame(mean, index=labels, columns=genes)
    df.index.name = 'unit'
    df.to_parquet(OUT / f'units_{source}_mean.parquet')
    if frac is not None:
        f = pd.DataFrame(frac, index=labels, columns=genes)
        f.index.name = 'unit'
        f.to_parquet(OUT / f'units_{source}_frac.parquet')
    return pd.DataFrame(dict(source=source, unit=labels, cells=n))


def davis():
    t = pd.read_csv(DAVIS, sep='\t')
    genes = t['gene_name'].astype(str).to_numpy()
    columns = [c for c in t.columns if c not in ('gene_id', 'gene_name')]
    # <type>_<driver>[_<sex>]_<replicate> (types may hold '_'; T4.T5 is profiled per sex)
    parts = {c: re.fullmatch(r'(.+)_(d\d+)(?:_(?:female|male))?_(rep\d+)', c) for c in columns}
    controls = sorted(c for c, p in parts.items() if not p)            # mock, no-driver, bead and Tm1 protocol controls
    if any(not re.search(r'mock|no_driver|beads|capture|input', c) for c in controls):
        raise RuntimeError(f'unexpected Davis sample names: {controls}')
    samples = [c for c in columns if parts[c]]
    unit = np.array([parts[c].group(1) for c in samples])
    lx = np.log1p(t[samples].to_numpy(float)).T                      # samples x genes
    labels = np.unique(unit)
    mean = np.vstack([lx[unit == u].mean(axis=0) for u in labels])
    n = np.array([(unit == u).sum() for u in labels])
    keep = ~pd.Index(genes).duplicated()
    return save_units('davis2020', labels, n, mean[:, keep], None, genes[keep]), dict(samples=len(samples), controls=controls)


def ozel():
    t = pd.read_excel(OZEL, 'Adult_average_expression')
    t = t.rename(columns={t.columns[0]: 'gene'}).drop_duplicates('gene').set_index('gene')
    labels = np.array([str(c) for c in t.columns])
    return save_units('ozel2021', labels, np.full(len(labels), -1), t.to_numpy(float).T, None, t.index.astype(str).to_numpy())


def davie():
    d = DERIVED / 'davie57k'
    if not d.exists():
        with tarfile.open(DAVIE) as t:
            t.extractall(d, filter='data')
    import scipy.io
    mtx = next(d.rglob('matrix.mtx'))
    X = scipy.io.mmread(mtx).tocsr().T.tocsr()                        # genes x cells -> cells x genes
    genes = pd.read_csv(next(d.rglob('genes.tsv')), sep='\t', header=None).iloc[:, -1].astype(str).to_numpy()
    barcodes = pd.read_csv(next(d.rglob('barcodes.tsv')), sep='\t', header=None).iloc[:, 0].astype(str).to_numpy()
    meta = pd.read_csv(DAVIE_META, sep='\t')
    key = 'new_barcode' if set(barcodes) <= set(meta['new_barcode']) else 'old_barcode'
    meta = meta.set_index(key).loc[barcodes]
    labels, n, mean, frac = group_mean_frac(X, meta['annotation'].astype(str).to_numpy(), True)
    keep = ~pd.Index(genes).duplicated()
    return save_units('davie2018', labels, n, mean[:, keep], frac[:, keep], genes[keep]), dict(barcode_key=key, cells=int(X.shape[0]))


def h5ad_units(source, path, label_col, counts_layer, subset=None):
    """Units of an h5ad atlas. counts_layer: a layer of raw counts (normalised here); None: X holds log1p(cp10k)."""
    import anndata as ad
    import scipy.sparse as sp
    a = ad.read_h5ad(path, backed='r')                                # X stays on disk; layers are read into memory
    obs = a.obs
    mask = np.ones(a.n_obs, bool) if subset is None else np.asarray(subset(obs), bool)
    labels_all = obs[label_col].astype(str).to_numpy()[mask]
    genes = a.var_names.astype(str).to_numpy()
    pieces, step = [], 50000
    for s in range(0, a.n_obs, step):                                 # contiguous reads, then the subset
        X = a.layers[counts_layer][s:s + step] if counts_layer else a.X[s:s + step]
        pieces.append(sp.csr_matrix(X)[mask[s:s + step]])
    a.file.close()
    X = sp.vstack(pieces).tocsr()
    check = None
    if not counts_layer:                                              # QC 1: log1p(cp10k)?
        rows = X[:2000].copy()
        rows.data = np.expm1(rows.data)
        sums = np.asarray(rows.sum(axis=1)).ravel()
        check = dict(expm1_row_sum_median=float(np.median(sums)), expm1_row_sum_p05=float(np.percentile(sums, 5)),
                     expm1_row_sum_p95=float(np.percentile(sums, 95)))
    labels, n, mean, frac = group_mean_frac(X, labels_all, bool(counts_layer))
    keep = ~pd.Index(genes).duplicated()
    return save_units(source, labels, n, mean[:, keep], frac[:, keep], genes[keep]), check


def type_maps():
    rows = []
    m = pd.read_csv(FLYDATA / 'type_map_davis2020.csv', comment='#', dtype=str)
    rows += [dict(source='davis2020', unit=str(r.source_name), malecns_type=r.malecns_type, tier=r.tier) for r in m.itertuples()]
    m = pd.read_csv(FLYDATA / 'type_map_ozel2021.csv', comment='#', dtype=str)
    rows += [dict(source='ozel2021', unit=str(r.source_cluster), malecns_type=r.malecns_type, tier=r.tier) for r in m.itertuples()]
    m = pd.read_csv(FLYDATA / 'type_map_central.csv', comment='#', dtype=str)
    for r in m.itertuples():
        if r.source == 'davie2018':
            rows.append(dict(source='davie2018', unit=str(r.source_name), malecns_type=r.malecns_type, tier=r.tier))
        else:                                                         # FCA labels, applied to both FCA-derived atlases
            for s in ('afca2023', 'adfca2025'):
                rows.append(dict(source=s, unit=str(r.source_name), malecns_type=r.malecns_type, tier=r.tier))
    t = pd.DataFrame(rows)
    return t[t['tier'].isin(TIER_RANK)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qc = []
    dvs, info = davis()
    qc.append(dict(check='davis_samples', value=json.dumps(info)))
    sizes = [dvs, ozel()]
    dv, info = davie()
    sizes.append(dv)
    qc.append(dict(check='davie_barcodes', value=json.dumps(info)))
    af, check = h5ad_units('afca2023', AFCA, 'afca_annotation', None)
    sizes.append(af)
    qc.append(dict(check='afca_log1p_cp10k', value=json.dumps(check)))
    ad_, _ = h5ad_units('adfca2025', ADFCA, 'adfca_annotation', 'umi_counts',
                        subset=lambda o: (o['genotype'].astype(str) == 'control').to_numpy())
    sizes.append(ad_)
    sizes = pd.concat(sizes, ignore_index=True)
    sizes.to_csv(OUT / 'unit_sizes.csv', index=False, lineterminator='\n')
    units = {s: set(sizes.loc[sizes['source'] == s, 'unit']) for s in SOURCE_ORDER}
    maps = type_maps()
    maps['unit_present'] = [u in units[s] for s, u in zip(maps['source'], maps['unit'])]
    for s in SOURCE_ORDER:
        sub = maps[maps['source'] == s]
        qc.append(dict(check=f'map_units_present_{s}', value=f"{int(sub['unit_present'].sum())}/{len(sub)} map rows; "
                                                             f"{sub.loc[sub['unit_present'], 'unit'].nunique()} units"))
    maps = maps[maps['unit_present']]
    cells = pd.read_parquet(MODEL_CELLS)
    cand = cells[['bodyId', 'type']].merge(maps, left_on='type', right_on='malecns_type', how='inner')
    cand['rank'] = cand['tier'].map(TIER_RANK) * 10 + cand['source'].map({s: i for i, s in enumerate(SOURCE_ORDER)})
    best = cand.sort_values(['bodyId', 'rank']).drop_duplicates('bodyId')
    cmap = cells.merge(best[['bodyId', 'source', 'unit', 'tier']], on='bodyId', how='left')
    cmap['tier'] = cmap['tier'].fillna('unmapped')
    cmap.to_parquet(OUT / 'cell_map.parquet', index=False)
    cand.to_parquet(OUT / 'cell_candidates.parquet', index=False)
    # coverage report (M5 input)
    sc, typ, cls = (cmap[c].fillna('').astype(str).to_numpy() for c in ('superclass', 'type', 'class'))
    ol = np.char.startswith(sc.astype('U'), 'ol_')                    # the plan's section 2 regions
    vnc = np.array([s.startswith('vnc_') or 'ascending' in s or 'descending' in s for s in sc])
    sets = {'all': np.ones(len(cmap), bool),
            'optic lobe (ol_*)': ol,
            'central brain (non-OL, non-VNC)': ~ol & ~vnc,
            'VNC with ascending and descending': vnc,
            'visual projection': sc == 'visual_projection',
            'central-brain intrinsic': sc == 'cb_intrinsic',
            'VNC intrinsic': sc == 'vnc_intrinsic',
            'descending neurons': sc == 'descending_neuron',
            'ascending neurons': sc == 'ascending_neuron',
            'readout cells': cmap['observed_by_policy'].to_numpy(bool),
            'visual input cells': cmap['direct_input'].to_numpy(bool),
            'Kenyon cells': cls == 'Kenyon_Cell',
            'ring neurons (ER)': np.array([re.match(r'ER\d', t) is not None for t in typ]),
            'central complex (class CX)': cls == 'CX',
            'antennal-lobe projection neurons (ALPN)': cls == 'ALPN',
            'dopaminergic (DAN)': cls == 'DAN',
            'mushroom-body output (MBON)': cls == 'MBON'}
    for k in ('m0', 'm0_joint', 'm0vis', 's1', 's2'):
        x = cmap[f'delta_adapt_{k}'].abs()
        sets[f'adaptation difference > 0 ({k})'] = (x > 0).to_numpy()
        sets[f'top 1% adaptation difference ({k})'] = (x >= x.quantile(0.99)).to_numpy() & (x > 0).to_numpy()
    cov = []
    for name, mask in sets.items():
        sub = cmap[mask]
        row = dict(cell_set=name, cells=int(mask.sum()))
        for t in ('exact', 'alias', 'fuzzy', 'class', 'unmapped'):
            row[t] = float((sub['tier'] == t).mean()) if len(sub) else None
        row['type_level'] = (row['exact'] or 0) + (row['alias'] or 0)
        row['type_or_fuzzy'] = row['type_level'] + (row['fuzzy'] or 0)
        row['gate_sc_a_type_level'] = row['type_level'] >= 0.5
        cov.append(row)
    pd.DataFrame(cov).to_csv(OUT / 'coverage.csv', index=False, lineterminator='\n')
    # QC 2: Davie transmitter genes vs flybrain's expression_central.csv
    ref = pd.read_csv(FLYDATA / 'expression_central.csv', comment='#')
    ref = ref[(ref['source'] == 'davie2018') & (ref['metric'] == 'mean_log1p_cp10k') & (ref['sex_subset'] == 'all')].set_index('source_name')
    ours = pd.read_parquet(OUT / 'units_davie2018_mean.parquet')
    ref.index = ref.index.astype(str)
    genes = [g for g in ref.columns if g in ours.columns]
    common = sorted(set(ref.index) & set(ours.index))
    diff = np.abs(ours.loc[common, genes].to_numpy() - ref.loc[common, genes].to_numpy(float))
    n_ours = dv.set_index('unit')['cells']
    n_diff = int((n_ours.loc[common] != ref.loc[common, 'n_cells'].astype(int)).sum())
    qc.append(dict(check='davie_vs_flybrain_expression_central',
                   value=f'{len(common)}/{len(ref)} units x {len(genes)} genes: max abs diff {float(np.nanmax(diff)):.2e} '
                         f'(values rounded to 1e-4); cell counts differ in {n_diff} units'))
    # QC 3: transmitter concordance per mapped unit
    nt_major = cells.groupby('type')['nt'].agg(lambda s: s.astype(str).value_counts().idxmax())
    nt_major = nt_major[nt_major != 'unknown']
    for s in SOURCE_ORDER:
        mean = pd.read_parquet(OUT / f'units_{s}_mean.parquet')
        rel = mean / mean.max().replace(0, np.nan)                    # per gene, relative to its highest unit
        score = pd.DataFrame({nt: rel[[g for g in gs if g in rel.columns]].mean(axis=1) for nt, gs in NT_MARKERS.items()
                              if any(g in rel.columns for g in gs)})
        calls = score.idxmax(axis=1)                                  # transcriptome call per unit
        sub = maps[(maps['source'] == s) & maps['tier'].isin(['exact', 'alias'])]
        sub = sub[sub['malecns_type'].isin(nt_major.index)]
        agree = (calls.reindex(sub['unit']).to_numpy() == nt_major.reindex(sub['malecns_type']).to_numpy())
        qc.append(dict(check=f'transmitter_concordance_{s}', value=f'{int(agree.sum())}/{len(agree)} type-level map rows '
                                                                    f'(EM transmitter known)'))
    # QC 4: marker sanity
    groups = {'Kenyon cells': (set(typ[cls == 'Kenyon_Cell']) - {''}, ['ey', 'Pka-C1']),
              'projection neurons': (set(typ[cls == 'ALPN']) - {''}, ['acj6']),
              'ring neurons (GABAergic)': (set(typ[sets['ring neurons (ER)']]), ['Gad1', 'VGAT'])}
    for s in SOURCE_ORDER:
        mean = pd.read_parquet(OUT / f'units_{s}_mean.parquet')
        z = (mean - mean.mean()) / mean.std(ddof=0).replace(0, np.nan)    # per gene across the source's units
        sub = maps[maps['source'] == s]
        for label, (types, genes_) in groups.items():
            us = sorted(set(sub.loc[sub['malecns_type'].isin(types), 'unit']))
            gs = [g for g in genes_ if g in z.columns]
            if us and gs:
                qc.append(dict(check=f'marker_{label}_{s}', value=f'mean z of {gs} over {len(us)} mapped units: '
                                                                  f'{float(np.nanmean(z.loc[us, gs].to_numpy())):.2f}'))
    # QC 5: cross-source concordance (Davis vs Ozel; Davie vs AFCA vs ADFCA) on shared variable genes
    for a_, b_ in (('davis2020', 'ozel2021'), ('davie2018', 'afca2023'), ('afca2023', 'adfca2025'), ('davie2018', 'adfca2025')):
        A, B = pd.read_parquet(OUT / f'units_{a_}_mean.parquet'), pd.read_parquet(OUT / f'units_{b_}_mean.parquet')
        shared = A.columns.intersection(B.columns)
        var = (A[shared].var() + B[shared].var()).sort_values(ascending=False).index[:2000]
        ma = maps[(maps['source'] == a_) & maps['tier'].isin(['exact', 'alias', 'fuzzy'])][['unit', 'malecns_type']]
        mb = maps[(maps['source'] == b_) & maps['tier'].isin(['exact', 'alias', 'fuzzy'])][['unit', 'malecns_type']]
        pairs = ma.merge(mb, on='malecns_type').drop_duplicates(['unit_x', 'unit_y'])
        if pairs.empty:
            qc.append(dict(check=f'cross_source_{a_}_vs_{b_}', value='no shared mapped types'))
            continue
        ra, rb = A.loc[:, var].rank(axis=1), B.loc[:, var].rank(axis=1)    # Spearman = Pearson of gene ranks
        matched = [float(np.corrcoef(ra.loc[r.unit_x], rb.loc[r.unit_y])[0, 1]) for r in pairs.itertuples()]
        rng = np.random.default_rng(20261022)
        ux, uy = pairs['unit_x'].unique(), pairs['unit_y'].unique()
        same = set(zip(pairs['unit_x'], pairs['unit_y']))
        mism = [float(np.corrcoef(ra.loc[x], rb.loc[y])[0, 1]) for x, y in
                zip(rng.choice(ux, 1000), rng.choice(uy, 1000)) if (x, y) not in same]
        qc.append(dict(check=f'cross_source_{a_}_vs_{b_}', value=f'{len(pairs)} matched unit pairs: median rho {np.median(matched):.3f}; '
                                                                 f'{len(mism)} mismatched pairs: median {np.median(mism):.3f}'))
    pd.DataFrame(qc).to_csv(OUT / 'mapping_qc.csv', index=False, lineterminator='\n')
    write_provenance(OUT, __file__, [DAVIS, OZEL, DAVIE, DAVIE_META, AFCA_GZ, AFCA, ADFCA, MODEL_CELLS, *MAPS])
    for q in qc:
        print(q['check'], '|', q['value'])
    print(pd.DataFrame(cov)[['cell_set', 'cells', 'type_level', 'type_or_fuzzy', 'class', 'unmapped']].to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
