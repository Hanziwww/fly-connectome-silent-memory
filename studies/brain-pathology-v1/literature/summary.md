# Summary - which published fly pathology states this model can represent

Scope: published, peer-reviewed findings only. 100 atlas rows from 43 primary papers
(`atlas.csv`), 24 candidate states scored on C1-C3 (`candidates.csv`), and parameter proposals
for the 16 that pass (`calibration_draft.json`). Search coverage and inclusion decisions are in
`search_log.md`.

Criteria as applied here:

- **C1, evidence in flies** - primary quantitative data, genotype controls, consistent direction,
  independent replication.
- **C2, mechanism measured and mappable** - the cellular change was measured directly, and the
  cell types map to MaleCNS v1.0 with high confidence.
- **C3, representable** - implementable with the model's existing operators only: remove neurons;
  remove synapses or connections; scale presynaptic release strength; shift excitability
  (threshold or spike-frequency-adaptation strength); scale active-zone strength;
  short-term-depression parameters.

## Shortlist (pass C1, C2 and C3)

Ranked by how much weight the evidence and the mapping can bear.

1. **C13 - Fragile X: weak GABAergic lateral inhibition (ALLN -> ALPN).**
   The best-mapped candidate in the review. Franco 2017 (10.1016/j.cub.2017.02.065) measured the
   inhibition itself: optogenetic activation of GABAergic local interneurons hyperpolarises and
   silences wild-type projection neurons but does so far more weakly in dfmr1-null flies (16 vs 16
   cells), dfmr1-null local interneurons show sustained rather than transient firing (p = 6.5e-4),
   and projection-neuron odour tuning broadens (lifetime sparseness, p = 9.8e-14). Both partners
   are named classes with non-zero weight in the model (ALLN 420 cells, ALPN 686 cells), and the
   operator is a straight presynaptic scaling.
2. **C07 - Aging: active-zone enlargement with more release at ALPN -> KC.**
   Strongest evidence base: Gupta 2016 (10.1371/journal.pbio.1002563) shows larger T-bars, larger
   BRP rings, more BRP/RBP per T-bar and a large increase in odour-evoked vesicle release at the
   identified PN-to-Kenyon-cell synapse, with spermidine suppression replicated in a second paper
   (10.1038/nn.3512), a causality test by BRP gene dosage (4xBRP young flies behave like 30-day-old
   controls), independent-circuit replication at the adult NMJ (10.1002/cne.23798) and follow-up
   work on the same brain-wide signature (10.1371/journal.pbio.3001730, 10.1038/s41467-019-09262-2).
   Implementation caveat worth flagging: release goes **up** while active zones per bouton go
   **down** and postsynaptic Kenyon-cell Ca2+ does **not** rise - a single scalar weight increase
   would not be faithful.
3. **C23 - Abeta42 photoreceptor loss (with tau's dysfunction-without-loss contrast).**
   The only well-quantified state in a cell type the model's existing visual tasks actually drive.
   Chouhan 2016 (10.1186/s40478-016-0333-4) compared the three proteins in one cell type with
   percentages: 92% -> 56% of ommatidia with an intact rhabdomere complement at 30 days for
   Abeta42, no loss but progressive ERG failure for tau, 97% -> 89% at 10 days for alpha-synuclein.
   This gives both a cell-removal target and a release-scaling target in the same paper.
4. **C04 - Abeta42 late-stage neuron loss.**
   Multi-lab, consistent: -17% mushroom body neurons at 40 days (132.10 +/- 3.17 -> 110.30 +/- 4.03,
   n = 8-9 brains; 10.1371/journal.pgen.1005025), vacuolar loss in the Kenyon cell layer from
   30 days (10.1073/pnas.0400895101), doubled vacuole number with adult-onset expression
   (10.1016/j.celrep.2018.11.098). Rescue (Kv4) and modifier (azot) arms support causality.
5. **C01 / C02 - Abeta42 Kv4 loss -> hyperexcitability (pan-neuronal; mushroom body).**
   The starting-set mechanism holds up as a measured, rescuable change: threshold current
   35.9 +/- 4.7 -> 19.5 +/- 2.2 pA and threshold charge 5132.8 +/- 1091.32 -> 2197.4 +/- 484.46
   pA*ms with unchanged passive properties, -37% Kv4 current density in MB neurons, reduction
   confirmed in the intact adult brain at 8 days, and cell-type specificity tested (IA falls in
   mushroom body neurons but not projection neurons; 10.1016/j.neuroscience.2017.06.008). Graded 1
   on C1, not 2, because of the directional conflict below.
6. **C05 / C16 - tau in Kenyon cells -> reduced excitability; C15 - tau in M4/M6 MBONs -> increased
   excitability.** Higham 2019a (10.3389/fncel.2019.00409) gives both signs with effect sizes in
   one paper: -48.1% K+-evoked Ca2+ in mushroom body lobes (p = 0.006) but 268.9 +/- 45.2% of
   control in the M4/M6 output neurons (p = 0.005), the latter reversed by nimodipine
   (98.9 +/- 21.1%) and by L-type channel knockdown, which also restores 1-h memory.
7. **C11 / C12 - Abeta42 raises then lowers release onto adult projection neurons.**
   Hahm 2018 (10.1016/j.celrep.2018.06.029) in identified adult ALPNs: inter-event interval
   5579.67 +/- 1804.34 -> 2439.60 +/- 1182.44 ms at 0-2 days (release rate x2.29) and
   1256.57 +/- 263.68 -> 2161.49 +/- 624.18 ms at 9-13 days (x0.58), localised presynaptically by
   unchanged mEPSC amplitude. These are the two most directly fittable numbers in the whole review
   and should be run as an age-staged pair, not as one static change.
8. **C14 - aging KCNQ/M-current loss in alpha/beta Kenyon cells.** Brain KCNQ falls to about 10% of
   the 5-day level, the null mutant phenocopies aged memory, and mushroom-body KCNQ restores it
   (10.1371/journal.pone.0062445). Graded 1 on C2 because the aged measurement is mRNA, not a
   current.
9. **C17 / C18 - short-term-depression states (tau and Abeta in the giant fiber pathway).**
   Kadas 2019 (10.1016/j.nbd.2018.12.004) reports isoform-specific increases in failure rate;
   Zhao 2010 (10.1523/JNEUROSCI.3699-09.2010) reports accelerated 100 Hz fatigue with a central
   transmission success rate of 39 +/- 12%, and normal aging in the same circuit drops the
   following frequency from at least 130 Hz to 60-80 Hz (10.1002/dneu.20388). The failure-rate
   measure is the closest thing in this literature to a directly fittable short-term-depression
   parameter, but the two tau papers need full-text access before a number can be used.
10. **C03 - intraneuronal Abeta lowers release probability.** Direction is solid and matches the
    late-stage ALPN result, but no magnitudes are published and the recording site is a peripheral
    synapse, so it enters the shortlist with a free magnitude only.
11. **C22 - aging does NOT remove dopaminergic neurons (control arm).** Well-powered counts at
    day 2 vs day 60 (n = 24 hemispheres; 10.3389/fnins.2010.00205), concordant with two other
    studies. Used as a null manipulation: the published locomotor decline (climbing index 0.92 ->
    0.34 by day 60) has to be produced without cell loss.

## Failures, and the criterion each fails on

| Candidate | Fails | Why |
|-----------|-------|-----|
| C08 aging DPM -> alpha/beta Kenyon-cell connectivity loss | **C3** | Evidence is strong (ITM trace lost at 30 days, alpha-tip GRASP contacts selectively reduced, DPM output requirement lost, 40% ITM rescue by raising DPM excitability; 10.1073/pnas.1118126109, 10.1523/JNEUROSCI.0978-14.2015) - but the two DPM cells are annotated dopaminergic here, so their synapses carry zero weight and both the lesion and the published rescue are functionally silent in the model. |
| C09 dopamine-neuron dysfunction (PD-like) | **C3** | Removing PAM/PPL1 cells is an available operator, but dopaminergic output carries zero weight, and the measured mechanisms (dopamine release, receptor levels) are neuromodulation. C1 is also only grade 1: whole-mount counting found no alpha-synuclein-induced cell loss (10.1002/gene.20106). |
| C10 Fragile X 5-HT2A receptor loss | **C3** | Du 2026 is a clean, quantitative, rescued result (10.1038/s41598-025-34492-4), but serotonergic synapses carry zero weight and there is no receptor-level operator. |
| C24 tau in dopaminergic neurons -> release loss | **C3** | Same structural reason as C09. |
| C19 sleep loss with AbetaArctic hyperexcitability | **C3 (in effect)** | The threshold operator exists and the measurement is clean (about 1.8-fold firing increase with sleep loss; AbetaArctic alone increases firing while Abeta42 and Abeta40 do not; 10.1016/j.cub.2015.01.016), but l-LNv is annotated serotonergic, so an imposed excitability change cannot propagate. |
| C20 alpha-synuclein plus sleep loss, D1-dependent memory loss | **C2 and C3** | Only cellular correlates are dopamine-receptor transcripts; sleep history is not a model variable (10.1093/sleep/32.8.984). |
| C06 pan-neuronal reversible tau | **C2 and C3** | The best-quantified behavioural dataset in the review (10.1523/JNEUROSCI.1374-22.2023), but the mechanism is protein-synthesis-dependent memory and tau solubility: no representable parameter was measured, and translation/plasticity are out of scope. |
| C21 aging Kenyon-cell fibre loss | **C1** | A single 1984 study of fibre profiles (10.3109/01677068409107077), contradicted by modern aging work that finds synaptic rather than cellular change. Recorded so the decision is on file: aging should be represented synaptically, not as cell loss. |

Pattern worth stating plainly: **every alpha-synuclein / PINK1 / parkin candidate and the Fragile X
5-HT2A candidate fail on the same structural ground** - not on the quality of the fly literature,
but on the model's choice to give dopamine, serotonin and octopamine synapses zero weight, plus
the annotation of DPM as dopaminergic. Four of the nine candidates in the original starting list
(aging DPM connectivity, PD dopamine dysfunction, Fragile X 5-HT2A, pan-neuronal reversible tau)
fail for reasons internal to the model rather than to the evidence.

## Known contradictions in the literature that the model has to confront

1. **Does Abeta42 make Kenyon cells more or less excitable?** Ping 2015 (Kv4 loss -> lower spike
   threshold) and Higham 2019b (-48% K+-evoked Ca2+ in the same cells) point in opposite
   directions; Tabuchi 2015 found that Abeta42 and Abeta40 changed l-LNv excitability not at all,
   only the Arctic variant did. Implement C02 and C16 as separate, oppositely signed states; do not
   average them.
2. **Does aging raise or lower effective synaptic drive onto Kenyon cells?** Presynaptic release at
   PN->KC rises (C07) while KCNQ loss predicts more excitable Kenyon cells (C14), yet measured
   postsynaptic Kenyon-cell Ca2+ is flat to slightly reduced and active zones per bouton fall.
3. **Does tau in Kenyon cells impair memory?** Yes with chronic MB drivers (25-30%;
   10.1101/lm.70804), no with adult-onset MB-restricted expression at about 14-fold higher levels
   (10.1523/JNEUROSCI.1374-22.2023). Protocol and aggregation state matter more than dose.
4. **Do PD models lose dopaminergic neurons?** Section-based studies say yes, whole-mount counting
   says no (10.1002/gene.20106), and synaptic loss consistently precedes any cell loss.

## Open questions for the later simulation checks

**C4 - the healthy model circuit behaves normally.**
- Do the baseline ALPN -> KC and KC -> MBON transformations in the healthy model match the
  published wild-type values the pathology rows are measured against (odour-evoked Kenyon-cell
  responses about 55% dF/F at the sensor level in Gupta 2016; sparse Kenyon-cell coding)? Without
  that, none of the scalings in calibration_draft.json has a reference point.
- Does deleting all 395 dopaminergic cells leave every healthy-model readout unchanged? That is the
  direct test of the zero-weight assumption on which four candidate rejections rest.
- Does the healthy model reproduce the wild-type aging baselines that are not cellular - for
  example climbing index 0.92 at day 2 (10.3389/fnins.2010.00205) - or is there no locomotor
  readout to compare against?

**C5 - the simulated change reproduces the published phenotype.**
- C23 is the only shortlisted state in a cell type the current visual tasks drive. Does deleting
  39% of R1-R6 degrade the visual memory task, and by how much relative to the 36-percentage-point
  rhabdomere loss? Is there a deletion fraction at which the task fails catastrophically rather
  than gracefully?
- C02, C05, C14, C15 and C16 all sit in the mushroom body, which the current tasks do not engage.
  Either a Kenyon-cell-dependent task is added, or these states can only be checked at the circuit
  level (KC -> MBON / APL firing), and that limit should be stated in the write-up rather than
  papered over.
- For C11/C12, does an imposed x2.29 then x0.58 change in release onto ALPNs reproduce the
  published sign reversal in net transmission with age, given that the model cannot generate the
  homeostatic compensation that the paper invokes as the cause?
- For C07, can any single parameter set satisfy all three published constraints at once
  (release up, active-zone number down, postsynaptic response flat)? If not, which one does the
  model have to violate, and does that violation change the conclusion?
- For C13, does weakening ALLN -> ALPN release broaden simulated ALPN odour tuning in the same
  direction as the published lifetime-sparseness shift?

**C6 - robustness.**
- How sensitive are the conclusions to the direction chosen for Kenyon-cell excitability? A sweep
  through both signs (C02 vs C16) is the honest way to handle contradiction 1 above.
- Several shortlisted states have no published magnitude (C03, C07, C13, C17). What range of
  fitted magnitudes still reproduces the phenotype, and is the result qualitatively stable across
  that range?
- Are the mapped cell sets robust to driver breadth? c492/c772 (Mershin 2004) also hit
  ellipsoid-body and antennal-lobe neurons, A307 (Zhao 2010) covers the whole giant fiber pathway,
  and R21D02 covers more than one MBON. Does restricting versus broadening each target set change
  the outcome?
- Do states that are reported to be reversible in flies (tau; 10.1523/JNEUROSCI.1374-22.2023) and
  rescuable (Kv4, levetiracetam, spermidine, L-type knockdown) also reverse in the model when the
  parameter is restored? A state that cannot be un-done in simulation is a sign the operator chosen
  is the wrong one.
