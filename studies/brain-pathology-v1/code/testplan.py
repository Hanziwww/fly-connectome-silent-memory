"""Test families of brain-pathology-v1, declared before any test is computed (studykit/testplan.py).

Plan v2, Phase 3: the Holm primary family of each retained case is written and frozen at gate G2, before any
capacity experiment; everything else is Benjamini-Hochberg and exploratory. Each family is declared here before
its first test is computed and is never pruned after results are seen. A family declared after its data were seen
takes posthoc=True and can never be primary. No family is declared yet.
"""
from studykit.testplan import TestPlan

PLAN = TestPlan('brain-pathology-v1')
FAMILIES = PLAN.families
family = PLAN.family

# Phase 3 (declared 2026-10-19, after gate G2 and before any capacity result was computed).
# Primary (Holm), one family per question the G2 shortlist was retained for:
family('F12.C01.primary', 'phase3/C01', 'B', 'holm', True,
       'C01 (Abeta42 pan-neuronal Kv4 loss): capacity declines with severity. (1) memory: trained-timing histories solved '
       'after lesioning the five trained memory models, doses 0/0.5/1/1.5/2, exact Page trend with models as blocks; '
       '(2) learning: trained histories solved after M0 training under the lesion (regime B), doses 0/0.5/1/1.5/2, exact '
       'Page trend with replicates as blocks (readout-only and joint, one test each)')
family('F12.olfactory.primary', 'phase3/olfactory', 'B', 'holm', True,
       'C04, C13 and C16 (circuits the visual tasks do not recruit): the circuit lesion impairs olfactory cue memory more '
       'than a size-matched random lesion. Per case: held-out decoding correctness of the similar odour pair from the MBONs '
       'at the 160-ms delay, targeted vs random at dose 1, paired by trial (exact McNemar)')
# Exploratory (BH):
family('F12.regimeA.solved', 'phase3/regime-a', 'B', 'bh', False,
       'Regime A: trained-timing histories solved after each lesion vs the healthy brain, per condition and memory model '
       '(exact McNemar, paired by history)')
family('F12.olfactory', 'phase3/olfactory', 'B', 'bh', False,
       'Olfactory memory curves: held-out decoding correctness after each lesion vs the healthy brain, per condition, '
       'odour pair, delay and read-out set (exact McNemar, paired by trial)')
family('F12.regimeB', 'phase3/regime-b', 'B', 'bh', False,
       'Regime B: learning under each lesion vs the healthy replicates with the same index (updates, trained and '
       'untrained histories solved)')

# Declared 2026-10-19, before any of their runs (the second GPU batch, after the training campaign).
# G3a, circuit dependency map (code/phase3_dependency*.py): removal of 50% or 100% of each target population of the
# retained cases (Kenyon cells: C04, C16; GABAergic ALLNs: C13) vs 10 random sets matched by hop distance from the
# input cells and degree quintile (input, readout and target cells excluded); the five memory models at the 9 trained
# timings. Unit for the equivalence: the history (126 trained histories over the five models).
family('F12.dependency.equivalence', 'phase3/dependency', 'B', 'holm', False,
       'No dependency of the visual memory tasks on the retained cases\' circuits: trained-timing histories solved after '
       'the targeted removal vs healthy, paired by history over the five memory models (Tango score TOST, margin +-5 '
       'percentage points), per population and level (4 tests)')
family('F12.dependency.null', 'phase3/dependency', 'B', 'bh', False,
       'Targeted vs matched random removal: per memory model, the change in trained histories solved after the targeted '
       'removal minus the median change over the 10 matched sets; exact one-sided sign-flip over the five models '
       '(targeted less harmful), per population and level (4 tests)')
# Regime C (code/phase3_regime_c.py): the healthy M0 readout-only X1 replicates 1-3, C01 lesion at doses 0.5-2, then 100
# updates of readout-only or joint retraining (the replicate's timing order, the frozen optimiser); recovery fraction =
# (solved after retraining - solved after lesion) / (18 - solved after lesion) at the trained timings.
family('F12.regimeC', 'phase3/regime-c', 'B', 'holm', False,
       'Reserve: recovery fraction after 100 retraining updates declines with C01 dose (exact Page trend over doses '
       '0.5/1/1.5/2, replicates as blocks), per retraining mode (2 tests); joint vs readout-only recovery (exact two-sided '
       'sign-flip over the 12 dose x replicate pairs; 1 test)')
# Late rescue (code/phase3_rescue.py): Kv4 restoration modelled as removing the C01 threshold offset from the M0 models
# trained under the lesion (regime B), as Kv4 restoration rescued learning in Abeta42 flies (Ping 2015,
# doi:10.1371/journal.pgen.1005025; Feng 2017, doi:10.1016/j.neuroscience.2017.06.008), here applied after learning.
family('F12.rescue', 'phase3/rescue', 'B', 'bh', False,
       'Late rescue: trained-timing histories solved by the models trained under C01 with excitability restored vs under '
       'the lesion they learned with, paired by history over replicates 1-3 (exact McNemar), per learner and dose (8 tests)')
# Post hoc (the per-condition olfactory results were already seen): comparison between the retained cases (plan v2,
# Phase 3 item 9), on the olfactory phenotype vector (held-out decoding accuracy at 2 odour pairs x 6 delays x 4 read-out
# sets; code/phase3_comparison.py).
family('F12.comparison', 'phase3/comparison', 'B', 'holm', False,
       'Retained cases compared: Euclidean distance between the olfactory phenotype vectors of two lesions vs a permutation '
       'null that swaps the two lesions trial by trial (the same input noise in both); Abeta42 late (C04) vs tau (C16) at '
       'doses 1 and 2, and Abeta42 early (C01) vs late (C04) at dose 1 (3 tests)', posthoc=True)
# Post hoc (regime A results already seen): equivalence of the C04, C13 and C16 lesions to healthy.
family('F12.regimeA.equivalence', 'phase3/regime-a', 'B', 'holm', False,
       'Regime A: trained-timing histories solved after the C04, C13 and C16 lesions (doses 1 and 2) vs healthy, paired by '
       'history over the five memory models (Tango score TOST, margin +-5 percentage points; 6 tests)', posthoc=True)
# Dynamics solutions plan (manuscript/DYNAMICS_SOLUTIONS_PLAN.md), DY1 sensitivity diagnosis, declared 2026-10-20 before
# any of its runs (code/dy1_sensitivity.py; the memory part is phase4_dose_series.py --set micro, amendment 0002).
family('F14.DY.sensitivity', 'dy1', 'B', 'none', False,
       'Descriptive: divergence of the healthy P05 brain (the M0 readout-only network) from its unperturbed trajectory after '
       'a perturbation at cue offset, at two trained timings and both cues: uniform threshold shifts of -1e-4 to -0.1 mV, '
       'one extra spike in one cell (5 seeded cells), 5 Hz input noise (5 seeds). Spike Hamming distance per ms and L2 '
       'distances of voltage, adaptation and synaptic input every 5 ms; the growth rate (slope of the log distance, a '
       'Lyapunov-like exponent) and the decision-time distance against perturbation size (chaotic: growth then a size-'
       'independent saturation; merely sensitive: distance proportional to the size). Memory: trained-history share solved '
       'of the five memory models under static threshold shifts of -1e-4 to -3e-3 mV, beside the dose series')
# DY2 robust readout training, stage 1 (M0 readout-only), declared 2026-10-20 before any of its runs (tasks/dy2-design.json,
# amendment 0003; code/dy2_perturbation.py via replicate_training.py --perturb, code/dy2_evaluate.py). 9 training
# conditions (input noise 0.2/1/5 Hz, membrane noise 0.1/1 mV, threshold randomisation sigma 0.001/0.1/0.5 mV, input
# 1 Hz + sigma 0.1 mV) x replicates 0-4, against the X1 readout-only replicates 0-9; C01 dose 0.5 + sigma 0.5 mV x
# replicates 1-5 against regime B C01 dose 0.5 readout-only replicates 1-6. Net use and ED50 as defined in the design file.
family('F14.DY.robust', 'dy2', 'B', 'holm', True,
       'Robust readout training lowers fragility: per training condition, the C01 ED50 (shift magnitude at which trained '
       'net use first falls to <= 0.5) of its 5 replicates exceeds that of the X1 replicates (exact one-sided Mann-Whitney; '
       '9 tests); state dependence: the loss of trained net use when the C01 lesion is removed is smaller for the replicates '
       'trained under C01 dose 0.5 with threshold randomisation (sigma 0.5 mV) than for regime B C01 dose 0.5 (exact '
       'one-sided Mann-Whitney; 1 test)')
family('F14.DY.robust.secondary', 'dy2', 'B', 'bh', False,
       'Per training condition vs the X1 replicates (exact Mann-Whitney): trained net use and untrained net use at the '
       'unperturbed brain (two-sided), trained net use under the condition\'s own held-out draws (greater), random-removal '
       'ED50 (greater), updates to the end of training (two-sided); 45 tests')
family('F14.DY.robust.descriptive', 'dy2', 'B', 'none', False,
       'Descriptive: the S1 Gate 1 criterion per condition (trained net use under its own held-out draws > 0 in >= 4 of 5 '
       'replicates, and the mechanism retained: mean drops in prefix-0 trained-timing net use after clearing adaptation and '
       'after the opposite-cue swap at decision onset each at least half of the X1 mean drop); ED50 ratios with bootstrap '
       'intervals; training outcomes, updates and seconds; the full perturbation panel')

# Single-cell study (manuscript/SINGLE_CELL_INTEGRATION_PLAN.md; execution plan approved 2026-10-22), declared 2026-10-22
# before any panel value, index or test was computed (T-SC2; tasks/sc-panels-v1.json, amendment 0005). Mapping v1 and
# Gate SC-A: tasks/sc-mapping-v1.json, amendment 0004. Definitions (sc_common.py, design file): unit expression as in
# mapping v1; per source, z-scores over its mapped units (units that are the best unit of a model cell); adaptation index
# A = mean z of SK, slo, KCNQ and Shab; a cell takes the values of its best unit. Trace carriers: cells with a nonzero
# decision-time cue difference in adaptation at delay 160 ms, prefix 4 (mechanisms-v1 interpretation tables). Levels:
# class (any tier; the primary level after Gate SC-A), type + fuzzy, type (exact/alias). Permutation p-values
# (1 + #{null >= observed}) / (1 + 10,000), seeded.
family('F14.SC.trace', 'sc/trace', 'B', 'holm', True,
       'H1: trace carriers have a higher adaptation index than expected. M0 readout-only at the class level, statistic = '
       'mean A over the mapped carriers. (1) Matched random sets: carrier labels permuted within superclass among the '
       'mapped cells (size- and superclass-matched); (2) mapping null: A permuted across the mapped MaleCNS types within '
       'superclass, carriers fixed. One-sided (greater); 2 tests')
family('F14.SC.trace.secondary', 'sc/trace', 'B', 'bh', False,
       'H1 sensitivity: the two tests of F14.SC.trace for M0 readout-only at the type and type + fuzzy levels (4 tests) and '
       'for M0 joint, S1 and S2 at the class, type + fuzzy and type levels (18 tests); the shared-readout model has the M0 '
       'readout-only carrier set (same frozen brain and inputs) and is not tested again; 22 tests')
family('F14.SC.trace.descriptive', 'sc/trace', 'B', 'none', False,
       'Descriptive: per panel gene, carriers vs the matched-set mean (z units); the type-weighted statistic; Spearman rho '
       'between |adaptation difference| and A among the mapped carriers; carrier composition by superclass, source and tier; '
       'detection under 10% flagged')
family('F14.SC.annotation', 'sc/annotation', 'B', 'none', False,
       'Descriptive (Q4): mapped units, tiers, panel values, EM and transcriptome transmitters and Davis 2020 driver lines '
       'for the trace carriers, X4/X5 decision units, G3a sets, readout cells, ring neurons and C01-vulnerable cells')
# AD-FCA (T-SC4b): per library (14: genotype x age x sex, one each) and unit, pseudobulk umi_counts summed over the unit's
# cells; y = log2(CPM + 1); outcomes Shal and the adaptation index (mean y of SK, slo, KCNQ, Shab). Per contrast, its 8
# libraries (Abeta42 vs control at 10 and 20 days; tau vs control at 20 and 30 days): OLS y ~ genotype + age + sex,
# two-sided t-test of the genotype coefficient (4 df). Units: CNS-neuron adfca_annotation units with >= 20 cells in each of
# the contrast's 8 libraries (runs/sc-panels/adfca_eligible.csv: 58 for Abeta42, 38 for tau) and the pan-neuronal unit.
family('F14.SC.adfca.pan', 'sc/adfca', 'B', 'holm', False,
       'Pan-neuronal (every CNS-neuron cell): Abeta42 and tau vs control on Shal and the adaptation index; 4 tests')
family('F14.SC.adfca', 'sc/adfca', 'B', 'bh', False,
       'Per eligible CNS-neuron unit: Abeta42 vs control (58 units) and tau vs control (38 units) on Shal and the adaptation '
       'index; 192 tests')
family('F14.SC.adfca.descriptive', 'sc/adfca', 'B', 'none', False,
       'Descriptive: every panel gene per unit and contrast (coefficients, log2 fold changes, detection); units mapped to '
       'trace-carrier types flagged')
# Kv4-weighted C01 (T-SC4, evaluation only, regime A protocol of phase3_regime_a.py): per-cell offset -3.22 mV x dose x w,
# w = relative Shal expression of the cell's best unit (expm1 of the unit mean, over the source's mapped-unit mean);
# unmapped cells take the mean w of their superclass ('class') or of all mapped cells ('global'); w scaled to mean 1 over
# all cells, so the mean offset equals uniform C01. Uniform C01 rerun first must reproduce the regime A rows exactly.
family('F14.SC.kv4', 'sc/kv4', 'B', 'holm', False,
       'Kv4-weighted vs uniform C01 at the same dose (0.5, 1, 1.5, 2) and imputation (class, global): trained-timing '
       'histories solved (126 over the five memory models), paired by history, exact two-sided McNemar; 8 tests')
family('F14.SC.kv4.equivalence', 'sc/kv4', 'B', 'holm', False,
       'The same 8 comparisons as F14.SC.kv4: Tango score TOST, margin +-5 percentage points; 8 tests')
family('F14.SC.kv4.descriptive', 'sc/kv4', 'B', 'none', False,
       'Descriptive: held-out timings; the dose subset 0.003-0.4x (ED50 per model vs uniform); regime B checkpoints (C01 '
       'dose 0.5, readout-only and joint, r1-r3) under the uniform lesion they learned with, the Kv4-weighted lesion with '
       'the same mean, and restored; leave-one-superclass-out weights; the weight distribution; activity and loom checks')
# Gate SC-B: if the F14.SC.trace label test is not significant (Holm) and no F14.SC.kv4 test is, T-SC6 is not run.
# Intrinsic variants (T-SC6): adaptation jump x 0.5 / 1 / 1.5 for mapped MaleCNS types in the low / middle / high tertile
# of A (tertiles over mapped types), x 1 for unmapped cells; null brain: the same multipliers permuted across the mapped
# types (seeded). Readout-only retraining with the X1 protocol: M0 replicates 0-9 and S1 replicates 0-4 per brain; net use,
# lock and C01 ED50 as in tasks/dy2-design.json; references X1 M0 r0-9 and S1 r0-6.
family('F14.SC.variant', 'sc/variant', 'B', 'holm', False,
       'Variant brain, M0 readout-only replicates 0-9: (1) trained net use > 0; (2) lock (trained - untrained net use) > 0; '
       '(3) clearing adaptation at the decision lowers prefix-0 trained net use; exact one-sided sign tests; 3 tests')
family('F14.SC.variant.secondary', 'sc/variant', 'B', 'bh', False,
       'Variant brain vs the X1 replicates and vs the null brain, M0 (exact two-sided Mann-Whitney): trained net use, '
       'untrained net use, lock, C01 ED50 (8 tests); S1 on the variant brain: the three F14.SC.variant sign tests (3 tests); '
       '11 tests')
family('F14.SC.variant.descriptive', 'sc/variant', 'B', 'none', False,
       'Descriptive: the trace locus (adaptation removed by superclass), training updates and seconds, the S1 Gate 1-style '
       'criteria, the null brain\'s own sign tests')
# Mapping v2 (tasks/sc-mapping-v2-design.json, amendment 0008), declared 2026-10-22 after H1 v1 (null) was known and before
# any file of the VNC atlas (Allen et al. 2020, GSE141807) was read: a validated hemilineage (lineage) tier for VNC cells.
# H1 v2 uses that new data, so it is not post hoc, but it is not primary. The SK test follows from the v1 descriptive
# per-gene result and is tested only on the new data.
family('F14.SC.trace.v2', 'sc/trace', 'B', 'holm', False,
       'H1 v2, M0 readout-only at the type + lineage level of mapping v2 (any tier if under 50% of carriers map there): '
       'mean adaptation index A and mean SK z-score of the mapped carriers vs (1) label permutations within superclass x '
       'source strata and (2) the mapping null (values permuted across mapped types and hemilineages within superclass x '
       'source); one-sided; 4 tests. Strata changed from superclass alone by amendment 0009, before any v2 data, after the '
       'post hoc v1 sensitivity showed that source composition shifts the statistic')
family('F14.SC.trace.v2.secondary', 'sc/trace', 'B', 'bh', False,
       'H1 v2 for M0 joint, S1 and S2: the four F14.SC.trace.v2 tests each; 12 tests')
# Amendment 0010 (2026-10-22, after reading only the designation labels, cluster sizes and marker lists of the VNC atlas,
# before any expression or test): paired designations (13A/19A, 5B/6B, 20A/22A, 8A/24B) form a lineage-pair tier below the
# strict lineage tier; and H1 v2 in the VNC compartment, the only one where the lineage tier can reach Gate SC-A.
family('F14.SC.trace.v2.vnc', 'sc/trace', 'B', 'holm', False,
       'H1 v2 in the VNC compartment (superclass vnc_* or ascending_neuron), M0 readout-only, at the type + lineage level '
       '(lineage and lineage-pair tiers included; any tier if under 50% of the compartment\'s carriers map there): A and SK '
       'z, label permutations within superclass x source strata and the mapping null; one-sided; 4 tests')
# Post hoc (their data were seen): sensitivity of H1 v1 and of the AD-FCA contrasts.
family('F14.SC.trace.sensitivity', 'sc/trace', 'B', 'bh', False,
       'H1 v1, M0: label null within superclass x source strata at the class, type + fuzzy and type levels (3 tests); cells '
       'mapped through Davis 2020 only, label null within superclass at type + fuzzy (1 test); 4 tests', posthoc=True)
family('F14.SC.adfca.sensitivity', 'sc/adfca', 'B', 'bh', False,
       'AD-FCA with all 14 libraries in one OLS (genotype + age factors + sex; 8 df): Abeta42 and tau coefficients on Shal '
       'and the adaptation index, for the 34 CNS-neuron units with >= 20 cells in every library and the pan-neuronal unit; '
       '140 tests', posthoc=True)
family('F14.SC.aging', 'sc/aging', 'B', 'none', False,
       'Aging operator (T-SC5) under the G2 rules: C4-C6 credibility of a per-class threshold offset derived from the old '
       'vs young change of K+ channel expression (AFCA, Davie); the operator is declared in its own design file before any '
       'T-SC5 run')
# Lesion link (amendment 0015; tasks/lesion-link-design.json; code/lesion_link.py on the GPU host, code/analyze_lesion_link.py
# locally), declared 2026-10-23 before any of its runs. Scene-1 readout-trained learner: the brain is fixed, so its states
# depend on the input and the C01 lesion only. Brain conditions W-R: the cue trace written and held in state W and read in
# state R (H healthy, L<d> C01 at dose d); for W != R the full simulator state of each history is transplanted from W into
# R at the end of its delay. Readouts: healthy-trained (X1 replicates 0-9), lesion-trained (regime B replicates 1-6 at each
# dose), relearned (regime C replays, replicates 1-3 at each dose). Outcome: trained-timing histories solved (18) per
# readout and condition. Exact one-sided sign-flip over readouts (zero differences dropped) unless stated.
family('F15.link', 'link', 'B', 'holm', True,
       'Access follows the state that reads the trace rather than the state that wrote it: (1) healthy-trained readouts, dose '
       '1: solved(L1-H) - solved(H-L1) > 0 (n = 10); (2) lesion-trained readouts at the four doses: solved(H-L<d>) - '
       'solved(L<d>-H) > 0 (n = 24); (3) relearned readouts at the four doses: solved(H-L<d>) - solved(L<d>-H) > 0 (n = 12; '
       'computed only if every replay reproduces regime C exactly, otherwise reported as not computable, the correction '
       'staying over 3 tests); 3 tests')
family('F15.link.secondary', 'link', 'B', 'bh', False,
       'Test (1) of F15.link at doses 0.5, 1.5 and 2 (3); test (2) per dose, 6 readouts each (4); healthy-trained readouts at '
       'dose 1, two-sided: solved(H-L1) - solved(L1-L1) and solved(L1-H) - solved(H-H) (2); 9 tests')
family('F15.link.descriptive', 'link', 'B', 'none', False,
       'Descriptive: per readout and condition, the first-decision push (mean over the cues of the cued choice\'s logit minus '
       'the other choice\'s) and the alignment of the readout-feature cue difference with the readout\'s choice axis; per '
       'brain condition and timing, the norms and cells of the cue difference of the stored trace (adaptation at the end of '
       'the delay), of the decision-window membrane potential and adaptation and of the readout features, cosines between '
       'conditions, differential spikes in the second half of the delay and the delay firing rate; per-ms spikes for the '
       'video')
