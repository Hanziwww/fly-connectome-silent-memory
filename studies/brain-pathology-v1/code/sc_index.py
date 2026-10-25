"""Unit- and cell-level single-cell values as declared in tasks/sc-panels-v1.json (amendment 0005): z-scores over each
source's mapped units, the adaptation index A and the Kv4 (Shal) weight. Reads the mapping v1 outputs
(runs/sc-mapping, locked by amendment 0004); used by sc_trace.py (T-SC3), the Kv4 operator (T-SC4) and the intrinsic
variants (T-SC6).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_common import DROPOUT, INDEX_GENES, KV4_GENE, RUNS, SOURCE_ORDER  # noqa: E402

MAPPING = RUNS / 'sc-mapping'
BULK = ('davis2020', 'ozel2021')                  # no detection fraction (bulk TPM; published cluster averages)


def cell_map():
    return pd.read_parquet(MAPPING / 'cell_map.parquet')


def units(source, kind='mean', genes=None):
    """Unit x gene table of a source (kind 'mean' or 'frac'); None for 'frac' of a bulk source."""
    if kind == 'frac' and source in BULK:
        return None
    t = pd.read_parquet(MAPPING / f'units_{source}_{kind}.parquet', columns=list(genes) if genes else None)
    t.index = t.index.astype(str)
    return t


def mapped_units(cells):
    """Per source, the units that are the best unit of at least one model cell."""
    m = cells.dropna(subset=['source'])
    return {s: sorted(set(m.loc[m['source'] == s, 'unit'].astype(str))) for s in SOURCE_ORDER}


def unit_values(cells, genes):
    """One row per (source, mapped unit): z of each gene over the source's mapped units (population SD, each unit once),
    the adaptation index A, Shal's s and r, and the detection flags of the genes (single-cell sources)."""
    rows = []
    for source, us in mapped_units(cells).items():
        mean = units(source, 'mean', genes).loc[us]
        z = (mean - mean.mean()) / mean.std(ddof=0)
        frac = units(source, 'frac', genes)
        out = pd.DataFrame({f'z_{g}': z[g] for g in genes})
        out['A'] = z[list(INDEX_GENES)].mean(axis=1)
        s = np.expm1(mean[KV4_GENE])
        out['shal_s'] = s
        out['shal_r'] = s / s.mean()
        for g in genes:
            out[f'e_{g}'] = mean[g]
            out[f'low_detection_{g}'] = (frac.loc[us, g] < DROPOUT) if frac is not None else False
        out.insert(0, 'unit', us)
        out.insert(0, 'source', source)
        rows.append(out.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def cell_values(cells, unit_table):
    """The cell table joined with its best unit's values (NaN for unmapped cells)."""
    t = cells.copy()
    t['unit'] = t['unit'].astype('object').where(t['unit'].isna(), t['unit'].astype(str))
    return t.merge(unit_table, on=['source', 'unit'], how='left')


def kv4_weights(cells, unit_table, imputation):
    """Per-cell Kv4 weight w (mean 1 over all cells): r of the best unit; unmapped cells take the mean w of the mapped
    cells of their superclass ('class'; none mapped: the global value) or of all mapped cells ('global')."""
    t = cell_values(cells, unit_table[['source', 'unit', 'shal_r']])
    w = t['shal_r'].to_numpy(float)
    mapped = ~np.isnan(w)
    glob = float(w[mapped].mean())
    if imputation == 'global':
        w = np.where(mapped, w, glob)
    elif imputation == 'class':
        sc = t['superclass'].fillna('').astype(str).to_numpy()
        means = pd.Series(w[mapped]).groupby(sc[mapped]).mean().to_dict()
        w = np.where(mapped, w, [means.get(s, glob) for s in sc])
    else:
        raise ValueError(imputation)
    return w / w.mean()
