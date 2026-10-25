"""T-SC4b on rtx5090x4 (.venv-sc): disease-state expression in the AD Fly Cell Atlas head (GSE261656), as declared in
tasks/sc-panels-v1.json (amendment 0005; families F14.SC.adfca.pan, F14.SC.adfca, F14.SC.adfca.descriptive).

    .venv-sc/bin/python -I studies/brain-pathology-v1/code/sc_adfca.py

Per library (14: genotype x age x sex, one each) and unit (the eligible CNS-neuron adfca_annotation units of
runs/sc-panels/adfca_eligible.csv and the pan-neuronal unit), umi_counts summed over the unit's cells; y = log2(CPM + 1)
with the pseudobulk total over all genes. Outcomes: Shal and the adaptation index (mean y of SK, slo, KCNQ, Shab). Per
contrast (Abeta42 vs control at 10 and 20 days; tau vs control at 20 and 30 days), on its 8 libraries: OLS
y ~ 1 + genotype + age + sex, two-sided t-test of the genotype coefficient (4 df); a constant outcome gives p = 1.
Outputs (runs/sc-adfca/): tests.csv (196 tests), descriptive.csv (every panel gene: coefficient = adjusted log2 fold
change, SE, p, mean CPM per genotype, detection per library), pseudobulk.csv, provenance.json.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_common import EXT, INDEX_GENES, KV4_GENE, PANEL_GENES, RUNS, write_provenance  # noqa: E402

ADFCA = EXT / 'GSE261656' / 'GSE261656_adata_adfca_head_v1.0.h5ad'
PANELS = RUNS / 'sc-panels'
OUT = RUNS / 'sc-adfca'
PAN = 'pan-neuronal (CNS neuron)'
CONTRASTS = {'Abeta42': 'AB42', 'tau': 'hTau'}


def ols_genotype(y, design):
    """Genotype coefficient, its SE, t and two-sided p of OLS y ~ design (columns: 1, genotype, age, sex)."""
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    df = len(y) - design.shape[1]
    rss = float(resid @ resid)
    if rss <= 1e-24:
        return float(beta[1]), 0.0, np.nan, 1.0 if abs(beta[1]) < 1e-12 else 0.0
    cov = rss / df * np.linalg.inv(design.T @ design)
    se = float(np.sqrt(cov[1, 1]))
    t = float(beta[1] / se)
    return float(beta[1]), se, t, float(2 * st.t.sf(abs(t), df))


def main():
    import anndata as ad
    import scipy.sparse as sp
    OUT.mkdir(parents=True, exist_ok=True)
    eligible = pd.read_csv(PANELS / 'adfca_eligible.csv')
    samples = pd.read_csv(PANELS / 'adfca_samples.csv').set_index('sample')
    a = ad.read_h5ad(ADFCA, backed='r')
    obs = a.obs
    genes = a.var_names.astype(str)
    gi = [genes.get_loc(g) for g in PANEL_GENES]
    X = sp.csr_matrix(a.layers['umi_counts'])
    a.file.close()
    totals = np.asarray(X.sum(axis=1)).ravel()
    panel = X[:, gi].toarray()
    detected = panel > 0
    unit_of = obs['adfca_annotation'].astype(str).to_numpy()
    neuron = (obs['adfca_annotation_broad'].astype(str) == 'CNS neuron').to_numpy()
    library = obs['sample'].astype(str).to_numpy()
    units = sorted(set(eligible['unit']))
    rows = []
    for unit in units:
        cells = neuron if unit == PAN else unit_of == unit
        for lib in samples.index:
            m = cells & (library == lib)
            n = int(m.sum())
            if n == 0:
                continue
            counts = panel[m].sum(axis=0)
            total = float(totals[m].sum())
            row = dict(unit=unit, library=lib, cells=n, total_counts=total)
            for j, g in enumerate(PANEL_GENES):
                row[f'cpm_{g}'] = 1e6 * counts[j] / total
                row[f'detected_{g}'] = float(detected[m, j].mean())
            rows.append(row)
    pb = pd.DataFrame(rows)
    pb.to_csv(OUT / 'pseudobulk.csv', index=False, lineterminator='\n', float_format='%.6g')
    tests, desc = [], []
    for _, e in eligible.iterrows():
        contrast, unit = e['contrast'], e['unit']
        libs = sorted(e['libraries'].split(';'))
        d = pb[(pb['unit'] == unit) & pb['library'].isin(libs)].set_index('library').loc[libs]
        meta = samples.loc[libs]
        older = max(meta['age'].astype(int))
        design = np.column_stack([np.ones(8), (meta['genotype'] == CONTRASTS[contrast]).astype(float),
                                  (meta['age'].astype(int) == older).astype(float), (meta['sex'] == 'male').astype(float)])
        if design[:, 1].sum() != 4:
            raise RuntimeError(f'{contrast} {unit}: unbalanced libraries')
        y = {g: np.log2(d[f'cpm_{g}'].to_numpy(float) + 1) for g in PANEL_GENES}
        y['adaptation_index'] = np.mean([y[g] for g in INDEX_GENES], axis=0)
        for outcome, values in y.items():
            beta, se, t, p = ols_genotype(values, design)
            g = None if outcome == 'adaptation_index' else outcome
            rec = dict(contrast=contrast, unit=unit, outcome=outcome, log2fc=beta, se=se, t=t, p=p,
                       cells_min=int(d['cells'].min()),
                       mean_cpm_control=float(d.loc[design[:, 1] == 0, f'cpm_{g}'].mean()) if g else None,
                       mean_cpm_disease=float(d.loc[design[:, 1] == 1, f'cpm_{g}'].mean()) if g else None,
                       detected_min=float(d[f'detected_{g}'].min()) if g else None)
            desc.append(rec)
            if outcome in (KV4_GENE, 'adaptation_index'):
                tests.append(dict(family='F14.SC.adfca.pan' if unit == PAN else 'F14.SC.adfca',
                                  test=f'{contrast}:{unit}:{outcome}', n=8, alternative='two-sided', statistic=t,
                                  estimate=beta, se=se, p_value=p))
    pd.DataFrame(tests).to_csv(OUT / 'tests.csv', index=False, lineterminator='\n')
    pd.DataFrame(desc).to_csv(OUT / 'descriptive.csv', index=False, lineterminator='\n', float_format='%.6g')
    write_provenance(OUT, __file__, [ADFCA, PANELS / 'adfca_eligible.csv', PANELS / 'adfca_samples.csv'])
    t = pd.DataFrame(tests)
    print(t.groupby('family').size().to_dict())
    print(t[t['family'] == 'F14.SC.adfca.pan'].to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
