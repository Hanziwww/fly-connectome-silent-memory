"""Post hoc sensitivity analyses declared in amendment 0008 (tasks/sc-mapping-v2-design.json), local.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\sc_sensitivity.py

- F14.SC.trace.sensitivity (4 tests): H1 v1 for M0 with the label null permuting carriers within superclass x source
  strata (class, type + fuzzy and type levels), and for the cells mapped through Davis 2020 alone (label null within
  superclass, type + fuzzy level). As sc_trace.py otherwise (one-sided, 10,000 draws, default_rng(20261022)).
- F14.SC.adfca.sensitivity (140 tests): the AD-FCA pseudobulk of runs/sc-adfca (pulled) with all 14 libraries in one OLS,
  y ~ 1 + genotype (control, AB42, hTau) + age (10, 20, 30 as factors) + sex; two-sided t-tests of the AB42 and hTau
  coefficients (8 df) on Shal and the adaptation index, for the CNS-neuron units with >= 20 cells in every library and the
  pan-neuronal unit.
Outputs: results/sc_trace_sensitivity_tests.csv, results/sc_adfca_sensitivity_tests.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_common import INDEX_GENES, KV4_GENE, LEVELS, RESULTS, RUNS  # noqa: E402
from sc_index import cell_map, cell_values  # noqa: E402
from sc_trace import SEED, p_greater, strata_draws  # noqa: E402

MIN_CELLS = 20
PAN = 'pan-neuronal (CNS neuron)'


def trace_sensitivity():
    cells = cell_map()
    table = pd.read_csv(RESULTS / 'sc_unit_values.csv', dtype={'unit': str})
    cv = cell_values(cells, table)
    carrier_all = cv['delta_adapt_m0'].fillna(0).ne(0).to_numpy()
    sc = cv['superclass'].fillna('NA').astype(str).to_numpy()
    src = cv['source'].fillna('none').astype(str).to_numpy()
    rows = []
    for level, subset, strata_kind in (('class', None, 'superclass x source'), ('type_fuzzy', None, 'superclass x source'),
                                       ('type', None, 'superclass x source'), ('type_fuzzy', 'davis2020', 'superclass')):
        mapped = cv['tier'].isin(LEVELS[level]).to_numpy().copy()
        if subset:
            mapped &= src == subset
        a = cv.loc[mapped, 'A'].to_numpy(float)
        strata = np.char.add(np.char.add(sc[mapped].astype(str), '|'), src[mapped].astype(str)) \
            if strata_kind == 'superclass x source' else sc[mapped]
        carriers = carrier_all[mapped]
        t = float(a[carriers].mean())
        null = strata_draws(a, strata, carriers, np.random.default_rng(SEED))
        name = f"m0:{level}:{'davis-only' if subset else 'source-strata'}"
        rows.append(dict(family='F14.SC.trace.sensitivity', test=name, n=int(carriers.sum()), n_reference=int(mapped.sum()),
                         alternative='greater', statistic=t, mean=t, mean_reference=float(null.mean()),
                         null_sd=float(null.std()), p_value=p_greater(t, null)))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'sc_trace_sensitivity_tests.csv', index=False, lineterminator='\n')
    return out


def adfca_sensitivity():
    pb = pd.read_csv(RUNS / 'sc-adfca' / 'pseudobulk.csv')
    samples = pd.read_csv(RUNS / 'sc-panels' / 'adfca_samples.csv').set_index('sample')
    counts = pd.read_csv(RUNS / 'sc-panels' / 'adfca_unit_counts.csv').set_index('unit')
    libs = sorted(samples.index)
    ok = (counts[libs].min(axis=1) >= MIN_CELLS) & (counts['broad'] == 'CNS neuron')
    units = sorted(counts.index[ok])
    meta = samples.loc[libs]
    age = meta['age'].astype(int)
    X = np.column_stack([np.ones(len(libs)), (meta['genotype'] == 'AB42').astype(float), (meta['genotype'] == 'hTau').astype(float),
                         (age == 20).astype(float), (age == 30).astype(float), (meta['sex'] == 'male').astype(float)])
    df = len(libs) - X.shape[1]
    xtx = np.linalg.inv(X.T @ X)
    rows = []
    for unit in units:
        d = pb[pb['unit'] == unit].set_index('library').reindex(libs)
        if d['cells'].isna().any():
            raise SystemExit(f'{unit}: a library is missing from the pseudobulk')
        y = {g: np.log2(d[f'cpm_{g}'].to_numpy(float) + 1) for g in INDEX_GENES + (KV4_GENE,)}
        y['adaptation_index'] = np.mean([y[g] for g in INDEX_GENES], axis=0)
        for outcome in (KV4_GENE, 'adaptation_index'):
            beta, *_ = np.linalg.lstsq(X, y[outcome], rcond=None)
            resid = y[outcome] - X @ beta
            s2 = float(resid @ resid) / df
            for j, contrast in ((1, 'Abeta42'), (2, 'tau')):
                se = float(np.sqrt(s2 * xtx[j, j]))
                t = float(beta[j] / se) if se > 0 else np.nan
                p = float(2 * st.t.sf(abs(t), df)) if se > 0 else 1.0
                rows.append(dict(family='F14.SC.adfca.sensitivity', test=f'{contrast}:{unit}:{outcome}:all-14-libraries', n=len(libs),
                                 df=df, alternative='two-sided', statistic=t, estimate=float(beta[j]), se=se, p_value=p,
                                 pan=unit == PAN))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'sc_adfca_sensitivity_tests.csv', index=False, lineterminator='\n')
    return out


def main():
    t = trace_sensitivity()
    print(t[['test', 'n', 'n_reference', 'statistic', 'mean_reference', 'p_value']].to_string(index=False))
    a = adfca_sensitivity()
    a['q'] = st.false_discovery_control(a['p_value'].to_numpy(), method='bh')
    print(len(a), 'AD-FCA tests;', int((a['q'] < 0.05).sum()), 'with BH q < 0.05')
    print(a.sort_values('p_value').head(12)[['test', 'estimate', 'statistic', 'p_value', 'q']].to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
