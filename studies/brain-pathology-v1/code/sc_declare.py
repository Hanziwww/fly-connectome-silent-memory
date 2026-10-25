"""T-SC2 lock (local): gene panels pinned to FlyBase IDs, the derived indices and the designs of T-SC3 to T-SC6, declared
with the F14.SC.* families (code/testplan.py) before any panel value, index or test is computed.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\sc_declare.py

Inputs: runs/sc-panels (sc_panels.py on rtx5090x4, pulled hash-checked: identifiers and AD-FCA cell metadata only) and the
mapping v1 lock (tasks/sc-mapping-v1.json, amendment 0004). Writes results/sc_panel_genes.csv, results/sc_adfca_design.csv
and tasks/sc-panels-v1.json (once), then tasks/lock.json and amendment 0005.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
CODE = STUDY / 'code'
RUN = STUDY / 'runs' / 'sc-panels'
RESULTS = STUDY / 'results'
TARGET = STUDY / 'tasks' / 'sc-panels-v1.json'
sys.path.insert(0, str(CODE))
import sc_common as sc  # noqa: E402

SEED = 20261022
PERMUTATIONS = 10000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if TARGET.exists():
        raise SystemExit(f'{TARGET.name} exists; task files are written once')
    prov = json.loads((RUN / 'provenance.json').read_text(encoding='utf-8'))
    for rel, digest in prov['code'].items():
        if sha256(ROOT / rel) != digest:
            raise SystemExit(f'{rel} differs from the code that made runs/sc-panels')
    panel = pd.read_csv(RUN / 'panel_genes.csv')
    if not panel['ids_agree'].all() or list(panel['gene']) != list(sc.PANEL_GENES):
        raise SystemExit('panel identifiers disagree or the panel changed')
    for g in sc.INDEX_GENES + (sc.KV4_GENE,):
        row = panel.set_index('gene').loc[g]
        if not all(row[f'in_{s}'] for s in sc.SOURCE_ORDER):
            raise SystemExit(f'{g} is missing from a source')
    eligible = pd.read_csv(RUN / 'adfca_eligible.csv')
    samples = pd.read_csv(RUN / 'adfca_samples.csv')
    shutil.copyfile(RUN / 'panel_genes.csv', RESULTS / 'sc_panel_genes.csv')
    eligible.drop(columns=['libraries']).to_csv(RESULTS / 'sc_adfca_design.csv', index=False, lineterminator='\n')
    pan = 'pan-neuronal (CNS neuron)'
    units = {c: sorted(eligible.loc[(eligible['contrast'] == c) & (eligible['unit'] != pan), 'unit']) for c in ('Abeta42', 'tau')}
    per_unit = 2 * (len(units['Abeta42']) + len(units['tau']))
    families = {'F14.SC.trace': 2, 'F14.SC.trace.secondary': 22, 'F14.SC.adfca.pan': 4, 'F14.SC.adfca': per_unit,
                'F14.SC.kv4': 8, 'F14.SC.kv4.equivalence': 8, 'F14.SC.variant': 3, 'F14.SC.variant.secondary': 11}
    sys.path.insert(0, str(ROOT))
    import importlib.util
    spec = importlib.util.spec_from_file_location('collect_tests', CODE / 'collect_tests.py')
    collect = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collect)
    for fam, n in families.items():
        if collect.EXPECTED.get(fam) != n:
            raise SystemExit(f'collect_tests.EXPECTED[{fam}] is not {n}')
    mapping_lock = STUDY / 'tasks' / 'sc-mapping-v1.json'
    design = dict(
        task='T-SC2 gene panels, indices and the designs of T-SC3 to T-SC6 (single-cell study; plan approved 2026-10-22)',
        declared='2026-10-22, before any panel value, index or test of the single-cell study was computed; the inputs '
                 'read so far are identifiers, coverage and AD-FCA cell metadata',
        code={f'code/{n}': sha256(CODE / n) for n in ('sc_common.py', 'sc_panels.py', 'sc_declare.py', 'testplan.py',
                                                      'collect_tests.py')},
        mapping=dict(file='tasks/sc-mapping-v1.json', sha256=sha256(mapping_lock)),
        inputs={f'runs/sc-panels/{p.name}': dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(RUN.iterdir())},
        panels=dict(
            index=list(sc.INDEX_GENES), kv4=sc.KV4_GENE, descriptive=list(sc.DESCRIPTIVE_GENES),
            k_channel=list(sc.K_CHANNEL_GENES), aging=list(sc.AGING_GENES),
            genes=json.loads(panel.to_json(orient='records')),
            note='FlyBase IDs agree across Davis 2020 (gene_id), Davie 2018 (genes.tsv) and AD-FCA (var gene_ids) for every '
                 'panel gene; ChAT is absent from the Ozel table (aging panel only)'),
        definitions=dict(
            unit_expression='as mapping v1: davis2020 mean log1p(TPM) over samples; ozel2021 the published log-normalised '
                            'cluster mean; davie2018, afca2023, adfca2025 mean log1p(cp10k) over cells',
            mapped_units='per source, the units that are the best unit of at least one model cell '
                         '(runs/sc-mapping/cell_map.parquet)',
            z='z(u, g) = (e(u, g) - mean over the source\'s mapped units) / population SD over them, each unit once',
            adaptation_index='A(u) = mean of z(u, g) over SK, slo, KCNQ and Shab; a cell takes A of its best unit (its tier '
                             'from mapping v1); unmapped cells have none',
            dropout=f'single-cell sources: a unit-gene value with detection fraction < {sc.DROPOUT:.0%} is flagged',
            kv4_weight='s(u) = expm1(e(u, Shal)); r(u) = s(u) / mean of s over the source\'s mapped units; w(cell) = r(best '
                       'unit); unmapped cells: class = mean w of the mapped cells of the same superclass (none: the global '
                       'value), global = mean w of all mapped cells; then w is divided by its mean over all 167,106 cells; '
                       'offset(cell) = -3.22 mV x dose x w(cell)',
            carriers='cells with a nonzero c0-visible0 delta-adapt at delay 160 ms, prefix 4 in the mechanisms-v1 '
                     'interpretation table of the model (cue 0 minus cue 1, hidden goal, at the decision)',
            levels={k: list(v) for k, v in sc.LEVELS.items()},
            primary_level='class (Gate SC-A, amendment 0004)'),
        h1=dict(
            statistic='T = mean A over the carriers mapped at the level (cell-weighted)',
            label_null='carrier labels permuted within superclass among the cells mapped at the level, counts per '
                       'superclass fixed (size- and superclass-matched random sets)',
            mapping_null='A permuted across the MaleCNS types mapped at the level within superclass (a type\'s superclass: '
                         'the majority of its cells), carriers fixed',
            p_value=f'(1 + #{{null >= T}}) / (1 + {PERMUTATIONS}), one-sided (greater); numpy default_rng({SEED}) per test',
            models=dict(primary='m0 (M0 readout-only)', secondary=['m0_joint', 's1', 's2'],
                        not_tested='m0vis (shared readout): the M0 readout-only carrier set (checked at analysis)'),
            descriptive='per-gene carrier vs matched-set means; type-weighted T; Spearman |delta-adapt| vs A among mapped '
                        'carriers; composition; dropout flags'),
        adfca=dict(
            libraries=json.loads(samples.to_json(orient='records')),
            pseudobulk='per library and unit, umi_counts summed over the unit\'s cells; y = log2(CPM + 1) with the '
                       'pseudobulk total over all genes',
            outcomes='Shal; adaptation index = mean y of SK, slo, KCNQ, Shab',
            model='per contrast on its 8 libraries: OLS y ~ 1 + genotype + age + sex; two-sided t-test of the genotype '
                  'coefficient (4 df)',
            contrasts=dict(Abeta42='AB42 vs control, ages 10 and 20 days', tau='hTau vs control, ages 20 and 30 days'),
            eligibility='CNS-neuron units (majority adfca_annotation_broad) with >= 20 cells in each of the contrast\'s 8 '
                        'libraries, plus the pan-neuronal unit (every CNS-neuron cell)',
            units=units, tests=dict(pan=4, per_unit=per_unit)),
        kv4=dict(
            operators=dict(uniform='-3.22 mV x dose on every cell; must reproduce the regime A rows of '
                                   'runs/phase3-regime-a exactly (identity check before any weighted run)',
                           weighted='definitions.kv4_weight with class and global imputation',
                           adfca='a cell-type Shal change measured in T-SC4b; declared by amendment before its runs'),
            doses=[0.5, 1.0, 1.5, 2.0],
            protocol='phase3_regime_a.py: the five memory models at all 35 timings (capped_memory_rollout, 256 decisions); '
                     'tests on the 9 trained timings (126 histories)',
            descriptive='held-out timings; dose subset 0.003-0.4x; regime B checkpoints (C01 dose 0.5, readout-only and '
                        'joint, r1-r3) under uniform, weighted (same mean) and restored; leave-one-superclass-out; weights'),
        gate_scb='if the F14.SC.trace label test (test 1) is not significant after Holm and no F14.SC.kv4 test is, stop at '
                 'annotation and report (T-SC6 not run)',
        variant=dict(
            brain='adaptation jump x 0.5 / 1 / 1.5 for mapped MaleCNS types in the low / middle / high tertile of A '
                  '(tertiles over the mapped types, each type once, class level), x 1 for unmapped cells',
            null_brain=f'the same per-type multipliers permuted across the mapped types (numpy default_rng({SEED}))',
            training='readout-only retraining with the X1 protocol (replicate_training.py: timing order seed 1000 + r, '
                     'readout jitter seed 2000 + r): M0 replicates 0-9 and S1 replicates 0-4 per brain',
            measures='net use, lock and C01 ED50 as in tasks/dy2-design.json; mechanism: prefix-0 trained net use drop '
                     'after clearing adaptation at the decision',
            references='X1 M0 readout-only r0-r9 and S1 readout-only r0-r6'),
        aging=dict(panel=list(sc.AGING_GENES), operator='declared in its own design file before any T-SC5 run (C4-C6, G2 '
                                                         'rules)'),
        families=families,
        limitations=['H1 covers the 8.9% of M0 trace carriers mapped at any tier (amendment 0004)',
                     'mRNA is not conductance: the indices and weights are coarse, monotone and declared',
                     'AD-FCA has one library per genotype, age and sex, so each contrast has 4 residual degrees of freedom'])
    text = json.dumps(design, indent=1) + '\n'
    TARGET.write_text(text, encoding='utf-8', newline='\n')
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    from studykit.registry import Study
    study = Study.named('brain-pathology-v1')
    study.lock_tasks(status='draft', note='inputs written before their runs (plan v2 batches); re-locked for amendment '
                                          '0005 (T-SC2 single-cell panels and designs)')
    amendment = study.amend('T-SC2 single-cell panels, indices and designs', dict(
        added_task_file=dict(file=f'tasks/{TARGET.name}', sha256=digest),
        why='Single-cell study, T-SC2: panels pinned to FlyBase IDs, the indices and the designs of T-SC3 (H1), T-SC4b '
            '(AD-FCA), T-SC4 (Kv4-weighted C01) and T-SC6 (intrinsic variants) fixed with the F14.SC.* families before '
            'any of their values or tests is computed.',
        families=families,
        gate_scb=design['gate_scb'],
        deviations_from_plan=dict(
            adfca='outcomes restricted to Shal and the adaptation index (the other panel genes descriptive); a separate '
                  'Holm family for the pan-neuronal unit',
            kv4='equivalence and difference as two Holm families; T19 restoration on the regime B checkpoints is '
                'descriptive (evaluation only)',
            variant='tests mirror the T-DY2 measures (net use, lock, ED50, adaptation clear)')))
    print(TARGET.name, digest, '->', amendment.name)
    print(json.dumps(families))
    return 0


if __name__ == '__main__':
    sys.exit(main())
