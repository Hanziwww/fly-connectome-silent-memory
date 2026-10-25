"""T-SC3, H1 and its descriptive companions (local): do the model's trace carriers have a higher adaptation index?
Declared in tasks/sc-panels-v1.json and code/testplan.py (F14.SC.trace, .secondary, .descriptive; amendment 0005).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\sc_trace.py

Inputs: runs/sc-mapping (pulled; mapping v1). Outputs (results/): sc_trace_tests.csv (22 + 2 tests), sc_unit_values.csv
(z, A and Shal per mapped unit), sc_trace_descriptive.csv, sc_trace_composition.csv. Deterministic: every test draws
from numpy default_rng(20261022).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_common import DESCRIPTIVE_GENES, INDEX_GENES, KV4_GENE, LEVELS, RESULTS  # noqa: E402
from sc_index import cell_map, cell_values, unit_values  # noqa: E402

SEED = 20261022
N = 10000
MODELS = {'m0': 'M0 readout-only', 'm0_joint': 'M0 joint', 's1': 'S1', 's2': 'S2'}
GENES = tuple(dict.fromkeys(INDEX_GENES + (KV4_GENE,) + DESCRIPTIVE_GENES))


def strata_draws(a, strata, carriers, rng):
    """Label null: N values of the mean of A over random sets with the carriers' count per stratum."""
    total = np.zeros(N)
    k_all = 0
    for s in sorted(set(strata[carriers])):
        pool = a[strata == s]
        k = int((carriers & (strata == s)).sum())
        k_all += k
        if k == len(pool):
            total += pool.sum()
            continue
        for i in range(N):
            total[i] += pool[rng.choice(len(pool), k, replace=False)].sum()
    return total / k_all


def mapping_draws(cells, carriers, a_by_type, strata_by_type, rng):
    """Mapping null: A permuted across the mapped types within superclass, carriers fixed; N values of the mean."""
    types = np.array(sorted(a_by_type))
    values = np.array([a_by_type[t] for t in types])
    groups = {s: np.flatnonzero(np.array([strata_by_type[t] for t in types]) == s) for s in sorted(set(strata_by_type.values()))}
    index = {t: i for i, t in enumerate(types)}
    carrier_types = np.array([index[t] for t in cells.loc[carriers, 'type']])
    weights = np.bincount(carrier_types, minlength=len(types)).astype(float)
    out = np.empty(N)
    for i in range(N):
        perm = values.copy()
        for s, idx in groups.items():
            perm[idx] = values[idx[rng.permutation(len(idx))]]
        out[i] = (weights * perm).sum() / weights.sum()
    return out


def p_greater(t, null):
    return (1 + int((null >= t).sum())) / (1 + len(null))


def main():
    cells = cell_map()
    table = unit_values(cells, GENES)
    table.to_csv(RESULTS / 'sc_unit_values.csv', index=False, lineterminator='\n', float_format='%.6g')
    cv = cell_values(cells, table)
    if not (cells['delta_adapt_m0vis'].fillna(0).ne(0) == cells['delta_adapt_m0'].fillna(0).ne(0)).all():
        raise SystemExit('the shared-readout carrier set differs from M0 readout-only (the declaration assumed equality)')
    strata_all = cv['superclass'].fillna('NA').astype(str).to_numpy()
    tests, desc, comp = [], [], []
    for model, label in MODELS.items():
        carrier_all = cv[f'delta_adapt_{model}'].fillna(0).ne(0).to_numpy()
        for level in ('class', 'type_fuzzy', 'type'):
            mapped = cv['tier'].isin(LEVELS[level]).to_numpy()
            sub = cv[mapped].reset_index(drop=True)
            a = sub['A'].to_numpy(float)
            if np.isnan(a).any():
                raise SystemExit(f'{model} {level}: a mapped cell has no adaptation index')
            strata = strata_all[mapped]
            carriers = carrier_all[mapped]
            t = float(a[carriers].mean())
            null_label = strata_draws(a, strata, carriers, np.random.default_rng(SEED))
            by_type = sub.groupby('type').agg(A=('A', 'first'), superclass=('superclass', lambda s: s.fillna('NA').astype(str).value_counts().idxmax()))
            if (sub.groupby('type')['A'].nunique() > 1).any():
                raise SystemExit('a type maps to more than one unit')
            null_map = mapping_draws(sub, carriers, by_type['A'].to_dict(), by_type['superclass'].to_dict(), np.random.default_rng(SEED))
            family = 'F14.SC.trace' if (model, level) == ('m0', 'class') else 'F14.SC.trace.secondary'
            base = dict(model=model, level=level, n=int(carriers.sum()), n_reference=int(mapped.sum()),
                        carriers_total=int(carrier_all.sum()), alternative='greater', statistic=t)
            tests.append(dict(family=family, test=f'{model}:{level}:matched-sets', **base, mean=t,
                              mean_reference=float(null_label.mean()), p_value=p_greater(t, null_label)))
            tests.append(dict(family=family, test=f'{model}:{level}:mapping-null', **base, mean=t,
                              mean_reference=float(null_map.mean()), p_value=p_greater(t, null_map)))
            # descriptive companions
            carrier_types = sub.loc[carriers, 'type']
            row = dict(model=model, level=level, carriers_mapped=int(carriers.sum()), carrier_types=int(carrier_types.nunique()),
                       statistic=t, matched_mean=float(null_label.mean()), matched_sd=float(null_label.std()),
                       mapping_mean=float(null_map.mean()),
                       type_weighted=float(by_type.loc[sorted(set(carrier_types)), 'A'].mean()),
                       type_weighted_noncarrier=float(by_type.loc[sorted(set(by_type.index) - set(carrier_types)), 'A'].mean()))
            rho = st.spearmanr(sub.loc[carriers, f'delta_adapt_{model}'].abs(), a[carriers])
            row['spearman_abs_delta_vs_A'], row['spearman_p'] = float(rho.statistic), float(rho.pvalue)
            for g in GENES:
                z = sub[f'z_{g}'].to_numpy(float)
                expect = float(sum(z[strata == s].mean() * (carriers & (strata == s)).sum() for s in set(strata[carriers]))
                               / carriers.sum())
                row[f'z_{g}_carriers'] = float(z[carriers].mean())
                row[f'z_{g}_matched_expectation'] = expect
                row[f'low_detection_{g}_carriers'] = float(sub.loc[carriers, f'low_detection_{g}'].astype(bool).mean())
            desc.append(row)
            for (scl, src, tier), n in sub[carriers].fillna({'superclass': 'NA'}).groupby(['superclass', 'source', 'tier']).size().items():
                comp.append(dict(model=model, level=level, superclass=scl, source=src, tier=tier, carriers=int(n)))
    tests = pd.DataFrame(tests)
    tests.to_csv(RESULTS / 'sc_trace_tests.csv', index=False, lineterminator='\n')
    pd.DataFrame(desc).to_csv(RESULTS / 'sc_trace_descriptive.csv', index=False, lineterminator='\n', float_format='%.6g')
    pd.DataFrame(comp).to_csv(RESULTS / 'sc_trace_composition.csv', index=False, lineterminator='\n')
    print(tests[['family', 'test', 'n', 'n_reference', 'statistic', 'mean_reference', 'p_value']].to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
