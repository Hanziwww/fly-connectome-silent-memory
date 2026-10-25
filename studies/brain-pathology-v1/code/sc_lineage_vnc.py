"""T-SC1b on rtx5090x4 (.venv-sc): the VNC lineage tier of mapping v2 from the Allen et al. 2020 adult VNC atlas, as
declared in tasks/sc-mapping-v2-design.json and amendments 0008-0011 (QC: 0011, after the first run failed gate V1).

    .venv-sc/bin/python -I studies/brain-pathology-v1/code/sc_lineage_vnc.py

Inputs (receipt-checked, data/external/receipt.csv): data/external/GSE141807/GSM42135{96..99}_*_dge.txt.gz (genes x
cells, four replicates) and data/external/Allen2020-elife/elife-54074-fig1-data1-v1.xlsx ('All Cluster Markers'),
-fig1-data2-v1.xlsx ('Neuronal Clusters' with the predicted hemilineage; 'Non-Neuronal Clusters'), -fig4-data1-v1.xlsx
('hemilineage markers').
Procedure: the authors' QC (amendment 0011: genes in >= 3 GEMs; 1,200-10,000 UMI, >= 200 genes, <= 15% mitochondrial;
it must reproduce their 26,768 cells); log1p(cp10k); per-gene z over the cells; each cell
scored on the 25 strongest markers of each of the 120 published clusters and assigned when its best score is >= 0.25 and
beats the second by >= 0.25. Units are the Allen designations of the neuronal clusters: single hemilineages (lineage
tier) and the four pairs 13A/19A, 5B/6B, 20A/22A, 8A/24B (lineage-pair tier), each the pooled assigned cells of its
clusters (>= 30). MaleCNS trumanHl labels map to units: X -> X when designated alone, else the pair containing X; a
composite X.Y only to an identical pair. Gates V1-V3 decide whether the tier is used (V4 descriptive).
Outputs (runs/sc-lineage-vnc/): units_allen2020vnc_mean.parquet and _frac.parquet (unit x gene), label_map.csv,
clusters.csv, units.csv, gates.json, provenance.json.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_common import EXT, MODEL_CELLS, RUNS, write_provenance  # noqa: E402

OUT = RUNS / 'sc-lineage-vnc'
DGE = sorted((EXT / 'GSE141807').glob('GSM*_dge.txt.gz'))
ELIFE = EXT / 'Allen2020-elife'
# QC of the authors' Methods (amendment 0011, replacing the declared generic 300-6,000 genes and mitochondria <= 0.10):
# per replicate, genes in >= 3 GEMs; GEMs with 1,200-10,000 UMI, >= 200 genes and <= 15% mitochondrial UMI. It must
# reproduce the paper's cells per replicate within 1%.
MIN_GEMS, MIN_UMI, MAX_UMI, MIN_GENES, MAX_MITO = 3, 1200, 10000, 200, 0.15
PUBLISHED_CELLS = {'GSM4213596_female_rep1': 2590, 'GSM4213597_female_rep2': 9060, 'GSM4213598_male_rep1': 6522,
                   'GSM4213599_male_rep2': 8596}
TOP_MARKERS, MIN_SCORE, MIN_MARGIN, MIN_UNIT = 25, 0.25, 0.25, 30
NT_GENES = {'acetylcholine': ('VAChT', 'ChAT'), 'gaba': ('Gad1', 'VGAT'), 'glutamate': ('VGlut',)}


def norm_hl(label):
    """Hemilineage label -> canonical form (leading zeros and a 'lin' prefix dropped): '06B' -> '6B', 'lin0A' -> '0A'."""
    m = re.fullmatch(r'(?:lin)?0*(\d+)([A-Za-z])', str(label).strip())
    return f'{int(m.group(1))}{m.group(2).upper()}' if m else None


def hl_key(h):
    m = re.fullmatch(r'(\d+)([A-Z])', h)
    return int(m.group(1)), m.group(2)


def designation(value):
    """'4B' -> '4B'; '13A/19A' or 'lin13A & 19A' -> '13A/19A'; missing or unparsable -> None."""
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return None
    text = re.sub(r'^lin', '', text)
    parts = [norm_hl(p) for p in re.split(r'[/&]', text)]
    if None in parts:
        return None
    return '/'.join(sorted(set(parts), key=hl_key))


def published_tables():
    f1 = pd.read_excel(ELIFE / 'elife-54074-fig1-data1-v1.xlsx', 'All Cluster Markers')
    f1 = f1[(f1['avg_logFC'] > 0) & (f1['p_val_adj'] < 0.05)]
    markers = {str(k): g.sort_values('avg_logFC', ascending=False)['gene'].astype(str).head(TOP_MARKERS).tolist()
               for k, g in f1.groupby('cluster')}
    x = pd.ExcelFile(ELIFE / 'elife-54074-fig1-data2-v1.xlsx')
    neu = pd.read_excel(x, 'Neuronal Clusters')
    non = pd.read_excel(x, 'Non-Neuronal Clusters')
    clusters = pd.concat([
        pd.DataFrame(dict(cluster=neu['Neuronal Cluster'].astype(str), published_cells=neu['Cell Number'], neuronal=True,
                          published_transmitter=neu['Neurotransmitter'].astype(str),
                          unit=neu['Predicted hemilineage'].map(designation))),
        pd.DataFrame(dict(cluster=non['Non-Neuronal Cluster'].astype(str), published_cells=non['Cell Number'], neuronal=False,
                          published_transmitter=None, unit=None))], ignore_index=True)
    f4 = pd.read_excel(ELIFE / 'elife-54074-fig4-data1-v1.xlsx', 'hemilineage markers')
    f4 = f4[f4['avg_logFC'] > 0]
    hl_markers = {designation(k): g['gene'].astype(str).tolist() for k, g in f4.groupby('Lineage') if designation(k)}
    return markers, clusters, hl_markers


def read_dge():
    """The four replicates after the authors' QC: cells x genes counts (union of the kept genes), replicate labels and
    the per-replicate cell counts against the paper's."""
    import scipy.sparse as sp
    mats, genes_all, reps, kept = [], None, [], []
    for path in DGE:
        t = pd.read_csv(path, sep='\t', index_col=0)
        t.index = t.index.astype(str)
        rep = path.name.split('_dge')[0]
        t = t[(t > 0).sum(axis=1) >= MIN_GEMS]
        umi = t.sum(axis=0)
        mito = t[t.index.str.startswith('mt:')].sum(axis=0) / umi.where(umi > 0, 1)
        ok = (umi >= MIN_UMI) & (umi <= MAX_UMI) & ((t > 0).sum(axis=0) >= MIN_GENES) & (mito <= MAX_MITO)
        t = t.loc[:, ok]
        kept.append(dict(replicate=rep, gems=int(len(ok)), cells=int(ok.sum()), published=PUBLISHED_CELLS[rep]))
        mats.append(t)
        reps += [rep] * t.shape[1]
        genes_all = t.index if genes_all is None else genes_all.union(t.index)
    for k in kept:
        if abs(k['cells'] - k['published']) > 0.01 * k['published']:
            raise SystemExit(f'QC does not reproduce the paper: {kept}')
    total, want = sum(k['cells'] for k in kept), sum(PUBLISHED_CELLS.values())
    if abs(total - want) > 0.01 * want:
        raise SystemExit(f'QC does not reproduce the paper: {total} vs {want} cells')
    X = sp.vstack([sp.csr_matrix(t.reindex(genes_all, fill_value=0).to_numpy(float).T) for t in mats]).tocsr()
    return X, np.asarray(genes_all), np.asarray(reps), kept


def label_map(labels, units):
    """MaleCNS trumanHl label -> (unit, tier) under amendment 0010."""
    pairs = {u: set(u.split('/')) for u in units if '/' in u}
    out = {}
    for lab in sorted(set(labels) - {'TBD'}):
        parts = [norm_hl(p) for p in lab.split('.')]
        if None in parts:
            continue
        if len(parts) == 1:
            if parts[0] in units:
                out[lab] = (parts[0], 'lineage')
            else:
                hit = [u for u, members in pairs.items() if parts[0] in members]
                if hit:
                    out[lab] = (hit[0], 'lineage_pair')
        else:
            hit = [u for u, members in pairs.items() if members == set(parts)]
            if hit:
                out[lab] = (hit[0], 'lineage_pair')
    return out


def main():
    import scipy.sparse as sp
    from scipy import stats as st
    OUT.mkdir(parents=True, exist_ok=True)
    markers, clusters, hl_markers = published_tables()
    X, genes, reps, kept = read_dge()
    total = np.asarray(X.sum(axis=1)).ravel()
    L = sp.diags(1e4 / total) @ X
    L.data = np.log1p(L.data)
    L = L.tocsr()
    index = {g: i for i, g in enumerate(genes)}
    used = sorted({g for m in markers.values() for g in m if g in index})
    D = L[:, [index[g] for g in used]].toarray()
    Z = (D - D.mean(0)) / np.where(D.std(0) > 0, D.std(0), 1)
    zpos = {g: j for j, g in enumerate(used)}
    names = sorted(markers, key=lambda k: int(k) if k.isdigit() else k)
    S = np.column_stack([Z[:, [zpos[g] for g in markers[k] if g in zpos]].mean(1) for k in names])
    order = np.argsort(-S, axis=1)
    best, second = S[np.arange(len(S)), order[:, 0]], S[np.arange(len(S)), order[:, 1]]
    ok = (best >= MIN_SCORE) & (best - second >= MIN_MARGIN)
    assigned = np.where(ok, np.asarray(names, dtype=object)[order[:, 0]], None)
    recovered = pd.Series(assigned).value_counts()
    clusters['recovered_cells'] = clusters['cluster'].map(recovered).fillna(0).astype(int)
    clusters.to_csv(OUT / 'clusters.csv', index=False, lineterminator='\n')
    v1 = float(st.spearmanr(clusters['recovered_cells'], clusters['published_cells']).statistic)
    unit_of = dict(zip(clusters['cluster'], clusters['unit']))
    cell_unit = np.array([unit_of.get(a) if a is not None else None for a in assigned], dtype=object)
    sizes = pd.Series(cell_unit[pd.notna(cell_unit)]).value_counts()
    units = sorted((u for u, n in sizes.items() if n >= MIN_UNIT), key=lambda u: hl_key(u.split('/')[0]))
    model = pd.read_parquet(MODEL_CELLS, columns=['bodyId', 'trumanHl', 'nt'])
    lmap = label_map(model['trumanHl'].dropna().astype(str), set(units))
    lm = pd.DataFrame([dict(malecns_label=k, unit=u, tier=t, malecns_cells=int((model['trumanHl'] == k).sum()))
                       for k, (u, t) in lmap.items()])
    lm.to_csv(OUT / 'label_map.csv', index=False, lineterminator='\n')
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
    for nt, gs in NT_GENES.items():
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
    gates = dict(qc=kept, cells_after_qc=int(X.shape[0]), cells_assigned=int(ok.sum()),
                 cells_in_units=int(np.isin(cell_unit, units).sum()), units=units, label_map_labels=len(lm),
                 V1_rho=v1, V1=bool(v1 >= 0.7), V2_concordance=v2, V2=bool(v2 >= 0.8), V2_n=int(len(known)),
                 V3_share=v3, V3=bool(v3 >= 0.8), V3_detail=v3_hits, V4_rho=v4)
    gates['passed'] = bool(gates['V1'] and gates['V2'] and gates['V3'])
    (OUT / 'gates.json').write_text(json.dumps(gates, indent=1) + '\n', encoding='utf-8', newline='\n')
    write_provenance(OUT, __file__, list(DGE) + sorted(ELIFE.glob('*.xlsx')) + [MODEL_CELLS])
    print(json.dumps({k: v for k, v in gates.items() if k != 'V3_detail'}, indent=1))
    print(ui.to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
