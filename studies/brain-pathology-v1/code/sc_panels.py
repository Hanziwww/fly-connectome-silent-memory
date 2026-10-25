"""T-SC2 inputs on rtx5090x4 (.venv-sc): FlyBase IDs of the panel genes and the AD-FCA sample design, read from
identifiers and cell metadata only (no expression value is read), so the panels and the AD-FCA tests can be declared
before any of them is computed.

    .venv-sc/bin/python -I studies/brain-pathology-v1/code/sc_panels.py

Outputs (runs/sc-panels/):
- panel_genes.csv: per panel gene its roles, the FlyBase ID in Davis (gene_id), Davie (genes.tsv) and AD-FCA
  (var gene_ids), whether they agree, and the symbol's presence in each source's unit tables (runs/sc-mapping);
- adfca_samples.csv: the 14 AD-FCA libraries (genotype, age, sex, cells);
- adfca_unit_counts.csv: cells per sample and adfca_annotation unit (plus the pan-neuronal unit, every cell with
  adfca_annotation_broad 'CNS neuron'), with each unit's majority broad class and its share;
- adfca_eligible.csv: per contrast (Abeta42 vs control at 10 and 20 days; tau vs control at 20 and 30 days) the CNS
  neuron units (majority broad class 'CNS neuron') with at least ADFCA_MIN_CELLS cells in each of its 8 libraries;
- provenance.json.
"""
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_common import (AGING_GENES, DESCRIPTIVE_GENES, EXT, INDEX_GENES, K_CHANNEL_GENES, KV4_GENE, PANEL_GENES,  # noqa: E402
                       RUNS, SOURCE_ORDER, write_provenance)

OUT = RUNS / 'sc-panels'
MAPPING = RUNS / 'sc-mapping'
DAVIS = EXT / 'GSE116969' / 'GSE116969_dataTable3.genes_x_samples_TPM.coding_genes.txt.gz'
DAVIE = EXT / 'GSE107451' / 'GSE107451_DGRP-551_w1118_WholeBrain_57k_0d_1d_3d_6d_9d_15d_30d_50d_10X_DGEM_MEX.mtx.tsv.tar.gz'
ADFCA = EXT / 'GSE261656' / 'GSE261656_adata_adfca_head_v1.0.h5ad'
ADFCA_MIN_CELLS = 20
PAN_NEURONAL = 'pan-neuronal (CNS neuron)'
CONTRASTS = {'Abeta42': ('AB42', ('10', '20')), 'tau': ('hTau', ('20', '30'))}


def roles(g):
    out = []
    for name, genes in (('index', INDEX_GENES), ('kv4', (KV4_GENE,)), ('descriptive', DESCRIPTIVE_GENES),
                        ('k_channel', K_CHANNEL_GENES), ('aging', AGING_GENES)):
        if g in genes:
            out.append(name)
    return ';'.join(out)


def main():
    import h5py
    OUT.mkdir(parents=True, exist_ok=True)
    davis = pd.read_csv(DAVIS, sep='\t', usecols=['gene_id', 'gene_name'])
    with tarfile.open(DAVIE) as t:                                    # read the gene list in memory, nothing extracted
        member = next(m for m in t.getmembers() if m.name.endswith('genes.tsv'))
        davie = pd.read_csv(t.extractfile(member), sep='\t', header=None, names=['gene_id', 'gene_name'])
    with h5py.File(ADFCA, 'r') as f:
        var = f['var']

        def text(key):
            obj = var[key]
            if isinstance(obj, h5py.Group):                           # categorical: codes and categories
                cats = [x.decode() if isinstance(x, bytes) else str(x) for x in obj['categories'][:]]
                return [cats[c] for c in obj['codes'][:]]
            return [x.decode() if isinstance(x, bytes) else str(x) for x in obj[:]]
        index = text(var.attrs.get('_index', '_index'))
        adfca_ids = dict(zip(index, text('gene_ids')))
    import pyarrow.parquet as pq
    columns = {s: set(pq.read_schema(MAPPING / f'units_{s}_mean.parquet').names) for s in SOURCE_ORDER}   # names only
    rows = []
    for g in PANEL_GENES:
        ids = dict(davis=sorted(set(davis.loc[davis['gene_name'] == g, 'gene_id'])),
                   davie=sorted(set(davie.loc[davie['gene_name'] == g, 'gene_id'])),
                   adfca=[adfca_ids[g]] if g in adfca_ids else [])
        found = sorted({i for v in ids.values() for i in v})
        rows.append(dict(gene=g, roles=roles(g), fbgn=found[0] if len(found) == 1 else ';'.join(found),
                         fbgn_davis=';'.join(ids['davis']), fbgn_davie=';'.join(ids['davie']), fbgn_adfca=';'.join(ids['adfca']),
                         ids_agree=len(found) == 1 and all(len(v) == 1 for v in ids.values()),
                         **{f'in_{s}': g in columns[s] for s in SOURCE_ORDER}))
    panel = pd.DataFrame(rows)
    panel.to_csv(OUT / 'panel_genes.csv', index=False, lineterminator='\n')
    # AD-FCA design from the cell metadata
    with h5py.File(ADFCA, 'r') as f:
        obs = f['obs']

        def cat(key):
            g = obs[key]
            cats = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g['categories'][:]])
            return cats[g['codes'][:]]
        meta = pd.DataFrame({k: cat(k) for k in ('sample', 'genotype', 'age', 'sex', 'adfca_annotation',
                                                 'adfca_annotation_broad')})
    samples = meta.groupby('sample').agg(genotype=('genotype', 'first'), age=('age', 'first'), sex=('sex', 'first'),
                                         cells=('sample', 'size')).reset_index()
    if (meta.groupby('sample')[['genotype', 'age', 'sex']].nunique() != 1).any().any():
        raise RuntimeError('a library mixes genotypes, ages or sexes')
    samples.to_csv(OUT / 'adfca_samples.csv', index=False, lineterminator='\n')
    counts = meta.groupby(['adfca_annotation', 'sample']).size().unstack(fill_value=0)
    pan = meta[meta['adfca_annotation_broad'] == 'CNS neuron'].groupby('sample').size()
    counts.loc[PAN_NEURONAL] = pan.reindex(counts.columns, fill_value=0)
    counts.index.name = 'unit'
    broad = meta.groupby('adfca_annotation')['adfca_annotation_broad'].agg(lambda s: s.value_counts().idxmax())
    share = meta.groupby('adfca_annotation')['adfca_annotation_broad'].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    broad.loc[PAN_NEURONAL], share.loc[PAN_NEURONAL] = 'CNS neuron', 1.0
    counts.insert(0, 'broad_share', share.reindex(counts.index).round(4))
    counts.insert(0, 'broad', broad.reindex(counts.index))
    counts.to_csv(OUT / 'adfca_unit_counts.csv', lineterminator='\n')
    eligible = []
    for name, (genotype, ages) in CONTRASTS.items():
        libs = samples[samples['genotype'].isin(['control', genotype]) & samples['age'].isin(ages)]['sample'].tolist()
        if len(libs) != 8:
            raise RuntimeError(f'{name}: {len(libs)} libraries, expected 8')
        ok = (counts[libs].min(axis=1) >= ADFCA_MIN_CELLS) & (counts['broad'] == 'CNS neuron')
        for unit in counts.index[ok]:
            eligible.append(dict(contrast=name, unit=unit, broad=counts.loc[unit, 'broad'],
                                 broad_share=counts.loc[unit, 'broad_share'], libraries=';'.join(sorted(libs)),
                                 min_cells=int(counts.loc[unit, libs].min())))
    eligible = pd.DataFrame(eligible)
    eligible.to_csv(OUT / 'adfca_eligible.csv', index=False, lineterminator='\n')
    write_provenance(OUT, __file__, [DAVIS, DAVIE, ADFCA] + [MAPPING / f'units_{s}_mean.parquet' for s in SOURCE_ORDER])
    print(panel[['gene', 'roles', 'fbgn', 'ids_agree'] + [f'in_{s}' for s in SOURCE_ORDER]].to_string(index=False))
    print(samples.to_string(index=False))
    print(eligible.groupby('contrast').size().to_dict())
    return 0


if __name__ == '__main__':
    sys.exit(main())
