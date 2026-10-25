"""T-SC1 lock (M5) and Gate SC-A, local: after the hash-checked pull of runs/sc-mapping (sc_mapping.py on rtx5090x4).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\sc_lock_mapping.py

1. Checks that the pulled outputs were made by the committed code (provenance.json code hashes) from the model-cell table
   and the receipt-verified data (data/external/receipt.csv).
2. results/: sc_mapping_coverage.csv, sc_mapping_qc.csv, sc_mapping_units.csv (byte copies of the pulled tables) and
   sc_gate_sca.csv (Gate SC-A per question).
3. tasks/sc-mapping-v1.json (written once): the locked mapping (hash of every output, sources, tiers, QC, coverage, gate),
   then tasks/lock.json and amendment 0004.

Gate SC-A (SINGLE_CELL_INTEGRATION_PLAN.md section 5): a question proceeds at type or lineage level only if at least 50%
of its cells map there; otherwise at class level with the limitation stated, or it is dropped. Type level here is the
exact, alias or fuzzy tier (fuzzy: a source cluster resolved to a small group of related MaleCNS types); the stricter
exact/alias share is reported beside it. No lineage tier exists (see LINEAGE), so the fallback is the class tier.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
RUN = STUDY / 'runs' / 'sc-mapping'
RESULTS = STUDY / 'results'
TARGET = STUDY / 'tasks' / 'sc-mapping-v1.json'
CODE = STUDY / 'code'
GATE = 0.5
LINEAGE = ('No lineage tier (M2): Allen et al. 2026 (D5) Tables S2-S5 name central-brain types by transmitter '
           '(Achl_000 ...) and carry no hemilineage labels, so the planned route Allen type -> hemilineage -> itoleeHl '
           'cannot be built (manuscript/SINGLE_CELL_T-SC0_SEARCH.md). The approved fallback applies: the central brain '
           'stays at class level. The adult VNC atlas (Allen et al. 2020, GSE141807) gives hemilineages per cluster only '
           '(eLife source data; no per-cell labels deposited) and would reach at most the VNC share of the trace carriers '
           '(about 45%), below the gate; it is not used.')
# Gate SC-A questions: (question, family or use, coverage cell set, rule when the type level fails)
QUESTIONS = [
    ('Q1 H1: trace carriers, M0 readout-only (primary)', 'F14.SC.trace', 'adaptation difference > 0 (m0)',
     'H1 at class level (any tier) on the mapped carriers; type-only and type+fuzzy as sensitivity; limitation stated'),
    ('Q1 H1: trace carriers, M0 joint', 'F14.SC.trace (secondary)', 'adaptation difference > 0 (m0_joint)', 'as primary'),
    ('Q1 H1: trace carriers, shared readout', 'F14.SC.trace (secondary)', 'adaptation difference > 0 (m0vis)', 'as primary'),
    ('Q1 H1: trace carriers, S1', 'F14.SC.trace (secondary)', 'adaptation difference > 0 (s1)', 'as primary'),
    ('Q1 H1: trace carriers, S2', 'F14.SC.trace (secondary)', 'adaptation difference > 0 (s2)', 'as primary'),
    ('Q2 Kv4-weighted C01 (SC2)', 'F14.SC.kv4', 'all',
     'Shal weight per mapped cell (any tier); unmapped cells take the class mean or the global mean (two declared '
     'imputations); limitation stated'),
    ('Q3 aging operator (SC3)', 'G2 rules', 'all', 'class level by design (per-class age changes); limitation stated'),
    ('Q4 annotation: readout cells', 'descriptive', 'readout cells', 'descriptive at every tier; coverage reported'),
    ('Q4 annotation: ring neurons', 'descriptive', 'ring neurons (ER)', 'descriptive at class level'),
    ('Q4 annotation: optic lobe', 'descriptive', 'optic lobe (ol_*)', 'type level'),
    ('Q4 annotation: Kenyon cells', 'descriptive', 'Kenyon cells', 'type level (fuzzy: subtypes)'),
    ('Q5 intrinsic variants (SC4)', 'F14.SC.variant', 'all',
     'adaptation scale from the index for mapped cells (any tier); unmapped cells x1 (declared); limitation stated'),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_provenance():
    prov = json.loads((RUN / 'provenance.json').read_text(encoding='utf-8'))
    for rel, digest in prov['code'].items():
        if sha256(ROOT / rel) != digest:
            raise SystemExit(f'{rel} differs from the code that made runs/sc-mapping')
    receipt = pd.read_csv(ROOT / 'data' / 'external' / 'receipt.csv')
    by_file = dict(zip(receipt['file'], receipt['sha256']))
    checked = []
    for rel, entry in prov['inputs'].items():
        name = rel.rsplit('/', 1)[-1]
        if name in by_file:
            if by_file[name] != entry['sha256']:
                raise SystemExit(f'{rel}: not the receipt file')
            checked.append(rel)
        elif rel.startswith(('studies/', 'flybrain/')):
            if sha256(ROOT / rel) != entry['sha256']:
                raise SystemExit(f'{rel}: differs from the local file')
            checked.append(rel)
    return prov, checked


def main():
    if TARGET.exists():
        raise SystemExit(f'{TARGET.name} exists; task files are written once')
    prov, checked = check_provenance()
    for src, dst in (('coverage.csv', 'sc_mapping_coverage.csv'), ('mapping_qc.csv', 'sc_mapping_qc.csv'),
                     ('unit_sizes.csv', 'sc_mapping_units.csv')):
        shutil.copyfile(RUN / src, RESULTS / dst)
    cov = pd.read_csv(RUN / 'coverage.csv').set_index('cell_set')
    gate = []
    for question, family, cell_set, fallback in QUESTIONS:
        r = cov.loc[cell_set]
        any_tier = 1.0 - float(r['unmapped'])
        passes = float(r['type_or_fuzzy']) >= GATE
        gate.append(dict(question=question, family=family, cell_set=cell_set, cells=int(r['cells']),
                         exact_alias=round(float(r['type_level']), 4), type_or_fuzzy=round(float(r['type_or_fuzzy']), 4),
                         any_tier=round(any_tier, 4), unmapped=round(float(r['unmapped']), 4),
                         type_level_passes=passes, level='type' if passes else 'class',
                         decision='proceeds at type level' if passes else fallback))
    gate = pd.DataFrame(gate)
    gate.to_csv(RESULTS / 'sc_gate_sca.csv', index=False, lineterminator='\n')
    outputs = {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(RUN.iterdir()) if p.is_file()}
    qc = pd.read_csv(RUN / 'mapping_qc.csv')
    design = dict(
        task='T-SC1 mapping v1 (single-cell study; manuscript/SINGLE_CELL_INTEGRATION_PLAN.md M1-M5; plan approved 2026-10-22)',
        declared='2026-10-22, after the mapping and before any model analysis of the single-cell study (no adaptation '
                 'index, H1, operator or panel value computed)',
        host='rtx5090x4, .venv-sc (code/requirements-sc.txt)',
        code={f'code/{n}': sha256(CODE / n) for n in ('sc_common.py', 'sc_model_cells.py', 'sc_mapping.py',
                                                      'sc_lock_mapping.py', 'requirements-sc.txt')},
        provenance=dict(code=prov['code'], python=prov['python'], packages=prov['packages'], inputs=prov['inputs'],
                        inputs_checked=checked),
        sources=dict(
            davis2020='GSE116969 dataTable3: driver-line samples grouped by type (250 samples; 16 mock, no-driver, bead '
                      'and Tm1 protocol controls excluded); mean log1p(TPM)',
            ozel2021='GSE142787 Adult sheet: published log-normalised cluster averages',
            davie2018='GSE107451 57k whole brain: mean log1p(cp10k) and detection fraction per annotation; reproduces '
                      "flybrain's expression_central.csv to rounding",
            afca2023='GSE218661 Aging Fly Cell Atlas head: mean of the published log1p(cp10k) and detection fraction per '
                     'afca_annotation (all ages and sexes)',
            adfca2025='GSE261656 AD Fly Cell Atlas head, control samples: umi_counts -> log1p(cp10k) per adfca_annotation',
            not_used='D3 Kurmangaliyev 2020 (matrix not downloaded; Ozel covers the adult optic lobe); D5 lineage route '
                     '(see lineage)'),
        tiers='flybrain type maps (exact, alias, fuzzy, class); best tier per cell, then source order davis2020 > '
              'ozel2021 > davie2018 > afca2023 > adfca2025; fca2022 labels applied to both FCA-derived atlases',
        lineage=LINEAGE,
        outputs={f'runs/sc-mapping/{k}': v for k, v in outputs.items()},
        qc=dict(zip(qc['check'], qc['value'])),
        coverage=json.loads(cov.reset_index().to_json(orient='records')),
        gate_sca=dict(rule=f'type level (exact, alias or fuzzy) >= {GATE:.0%} of the question\'s cells, else class level '
                           'with the limitation stated',
                      questions=json.loads(gate.to_json(orient='records'))),
        limitations=['91% of the M0 trace carriers (cells with a decision-time adaptation difference at delay 160 ms, '
                     'prefix 4) are unmapped at any tier: 37% are VNC intrinsic and 32% central-brain intrinsic cells, '
                     'which no source resolves; H1 speaks for the mapped carriers (mostly visual projection and '
                     'optic-lobe cells) only',
                     'about half of all model cells are unmapped, so cell-type operators (Kv4 weighting, intrinsic '
                     'variants) act on the mapped half and use declared imputations elsewhere',
                     'single-nucleus and single-cell sources differ in depth; detection under 10% is flagged in the '
                     'panels (M3)'])
    text = json.dumps(design, indent=1) + '\n'
    TARGET.write_text(text, encoding='utf-8', newline='\n')
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    sys.path.insert(0, str(ROOT))
    from studykit.registry import Study
    study = Study.named('brain-pathology-v1')
    study.lock_tasks(status='draft', note='inputs written before their runs (plan v2 batches); re-locked for amendment '
                                          '0004 (T-SC1 single-cell mapping v1)')
    failed = gate.loc[~gate['type_level_passes'], 'question'].tolist()
    amendment = study.amend('T-SC1 single-cell mapping v1 and Gate SC-A', dict(
        added_task_file=dict(file=f'tasks/{TARGET.name}', sha256=digest),
        why='Single-cell study (manuscript/SINGLE_CELL_INTEGRATION_PLAN.md; execution plan approved by the user on '
            '2026-10-22): the mapping (M1-M3), its QC (M4) and coverage are locked (M5) before any model analysis, and '
            'Gate SC-A is applied per question.',
        gate_sca=dict(type_level=gate.loc[gate['type_level_passes'], 'question'].tolist(), class_level=failed,
                      table='results/sc_gate_sca.csv'),
        lineage=LINEAGE,
        deviations_from_plan=dict(
            d3='Kurmangaliyev et al. 2020 not used (matrix not downloaded; Ozel et al. 2021 covers the adult optic lobe)',
            d5='no lineage tier (see lineage); central brain at class level, the approved fallback',
            outputs='host outputs under runs/sc-mapping (pulled hash-checked); results/ tables derived locally')))
    print(TARGET.name, digest, '->', amendment.name)
    print(gate[['question', 'cells', 'exact_alias', 'type_or_fuzzy', 'any_tier', 'level']].to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
