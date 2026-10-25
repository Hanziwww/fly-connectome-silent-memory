"""Mapping v2 and its tests, declared before any of their data are read (local; amendment 0008).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\sc_declare_v2.py

Why: mapping v1 (amendment 0004) reaches 7.4% of the M0 trace carriers at type level, because 91% of them are VNC or
central-brain intrinsic cells that no source resolves by type. Every carrier in the VNC has a Truman hemilineage in
MaleCNS (trumanHl). The adult VNC atlas of Allen et al. 2020 (eLife 9:e54074; GEO GSE141807) assigns a predicted
hemilineage to its clusters (Figure 1-source data 2). A validated lineage tier from it would raise carrier coverage to at
most 51.7% (all 75 VNC hemilineages) and VNC coverage from 0% to at most 62%. The data are not yet downloaded (the
download awaits the user's approval); this file fixes the procedure, the validation gates and the tests first.

Declared after the H1 v1 result (null; T-SC3) was known. The v2 tests use new data (the VNC atlas) and are therefore not
post hoc, but they are not primary. The SK test follows from the v1 descriptive result (carriers higher in SK, the
small-conductance Ca2+-activated K+ channel of the slow afterhyperpolarisation) and is tested only on the new data.
Two further analyses use data already seen and are post hoc: H1 v1 with superclass x source strata, and the AD-FCA
contrasts with all 14 libraries in one model.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
TARGET = STUDY / 'tasks' / 'sc-mapping-v2-design.json'

DESIGN = dict(
    task='Mapping v2: a VNC lineage tier from the Allen et al. 2020 VNC atlas; H1 v2; post hoc sensitivity of H1 v1 and of '
         'the AD-FCA contrasts',
    declared='2026-10-22, after H1 v1 (null) and T-SC4b were known and before any VNC atlas file was downloaded',
    data=dict(
        geo='GSE141807_RAW.tar (70,584,320 bytes): GSM4213596-9 female_rep1/2, male_rep1/2 *_dge.txt.gz (genes x cells)',
        elife=['elife-54074-fig1-data1-v1.xlsx (459,067 bytes): marker genes of the 120 clusters',
               'elife-54074-fig1-data2-v1.xlsx (17,114 bytes): cluster summary with the predicted hemilineage and transmitter',
               'elife-54074-fig4-data1-v1.xlsx (46,302 bytes): hemilineage marker genes'],
        approval='each file needs the user\'s approval; receipts in data/external/receipt.csv'),
    bridge=dict(
        qc='cells with 300-6,000 detected genes and a mitochondrial UMI fraction (genes mt:*) <= 0.10',
        normalisation='log1p(counts per 10,000); per-gene z-scores over the cells kept',
        markers='per published cluster: genes with positive average log fold change and adjusted p < 0.05 in Figure 1-source '
                'data 1, the 25 largest by log fold change (fewer if fewer)',
        assignment='score(cell, cluster) = mean z of the cluster\'s markers; the cell takes the best cluster when the best '
                   'score is >= 0.25 and exceeds the second best by >= 0.25, else it stays unassigned',
        hemilineage='the cluster\'s predicted hemilineage in Figure 1-source data 2; clusters without a designation, with an '
                    'ambiguous one (two or more hemilineages, "?" or "or") or not neuronal are left out',
        unit='per hemilineage, every assigned cell of its clusters over the four replicates (>= 30 cells): mean '
             'log1p(cp10k) and detection fraction',
        malecns='trumanHl normalised by stripping leading zeros (06B -> 6B); a composite label (e.g. 20A.22A) takes the pooled '
                'cells of its components when all are present; TBD is unmapped',
        tier='lineage, between fuzzy and class (exact > alias > fuzzy > lineage > class); source allen2020vnc'),
    gates=dict(
        V1='recovered cell counts per published cluster vs the published counts: Spearman rho >= 0.7',
        V2='dominant fast transmitter of each bridged hemilineage (VAChT/ChAT vs Gad1/VGAT vs VGlut, each relative to its '
           'highest hemilineage) equals the majority EM transmitter of the MaleCNS cells with that trumanHl in >= 80% of '
           'the bridged hemilineages with a known transmitter',
        V3='each hemilineage\'s markers in Figure 4-source data 1 have a mean z > 0 in its assigned cells in >= 80% of the '
           'hemilineages listed there',
        V4='descriptive: assigned cells per hemilineage vs MaleCNS cells per trumanHl (Spearman rho)',
        rule='the lineage tier is used only if V1-V3 pass; otherwise the VNC stays unmapped and that is reported'),
    h1_v2=dict(
        values='as amendment 0005, with the lineage units as a further source (z over its bridged hemilineages)',
        level='type + lineage (exact, alias, fuzzy, lineage); if fewer than 50% of the M0 carriers map there (Gate SC-A), '
              'the tests run at any tier (adding class) and are labelled with their coverage',
        tests=['A, label null within superclass (as F14.SC.trace test 1)',
               'A, mapping null: values permuted across the mapped units (types; hemilineages for lineage-tier cells) within '
               'superclass, carriers fixed',
               'SK z-score, label null', 'SK z-score, mapping null'],
        alternative='one-sided (greater); p = (1 + #{null >= T}) / (1 + 10,000); numpy default_rng(20261022) per test',
        families={'F14.SC.trace.v2': 'M0 readout-only, 4 tests, Holm',
                  'F14.SC.trace.v2.secondary': 'M0 joint, S1, S2, 12 tests, BH'},
        central_brain='a central-brain lineage tier (Allen et al. 2026 hemilineage annotations, not public as files) would '
                      'need its own amendment before use'),
    sensitivity_v1=dict(
        family='F14.SC.trace.sensitivity (post hoc, BH, 4 tests)',
        tests=['M0, label null within superclass x source strata, class level',
               'the same at type + fuzzy level', 'the same at type level',
               'M0, cells whose best unit is from Davis 2020 only (one bulk source), label null within superclass, type + '
               'fuzzy level'],
        why='source-specific scaling of the z-scores could confound the v1 comparison when carriers and other cells of a '
            'superclass map through different sources'),
    adfca_full=dict(
        family='F14.SC.adfca.sensitivity (post hoc, BH, 140 tests)',
        model='all 14 libraries in one OLS: y ~ 1 + genotype (control, AB42, hTau) + age (10, 20, 30 as factors) + sex; '
              'two-sided t-tests of the AB42 and hTau coefficients (8 df); genotype effects assumed constant across ages',
        units='CNS-neuron units with >= 20 cells in every library (34) and the pan-neuronal unit',
        outcomes='Shal and the adaptation index, as F14.SC.adfca'),
)


def main():
    if TARGET.exists():
        raise SystemExit(f'{TARGET.name} exists; task files are written once')
    text = json.dumps(DESIGN, indent=1) + '\n'
    TARGET.write_text(text, encoding='utf-8', newline='\n')
    sys.path.insert(0, str(ROOT))
    from studykit.registry import Study
    study = Study.named('brain-pathology-v1')
    study.lock_tasks(status='draft', note='inputs written before their runs (plan v2 batches); re-locked for amendment '
                                          '0008 (mapping v2 design)')
    path = study.amend('T-SC1b mapping v2 lineage tier and H1 v2 design', dict(
        added_task_file=dict(file=f'tasks/{TARGET.name}', sha256=hashlib.sha256(text.encode('utf-8')).hexdigest()),
        why=DESIGN['task'] + '; declared before any VNC atlas file is read, after H1 v1 was known',
        families=dict(new=['F14.SC.trace.v2 (Holm, 4)', 'F14.SC.trace.v2.secondary (BH, 12)'],
                      post_hoc=['F14.SC.trace.sensitivity (BH, 4)', 'F14.SC.adfca.sensitivity (BH, 140)'])))
    print(TARGET.name, hashlib.sha256(text.encode('utf-8')).hexdigest(), path.name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
