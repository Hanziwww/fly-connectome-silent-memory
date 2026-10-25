"""Test families, declared before any test is computed (written to qa/test_plan.csv by P0).

Every test recorded in msa.stats must name one of these families. Multiplicity is controlled
within a family (Holm or Benjamini-Hochberg); companion and descriptive families are
reported without adjustment. Families are never pruned after results are seen.

`posthoc` marks families whose tests were all added after the registered analysis (their tests carry
analysis_status 'post_hoc'). A post-hoc family cannot be primary. Correction of 2026-10-19 (freeze-v1.1,
manuscript/ERRATA.md): F4e.hidden and F5.memory_use.M0 had been declared primary although post hoc; they are
now secondary. Their tests, corrections and adjusted p-values are unchanged.
"""

FAMILIES = []


def family(name, panel, level, correction, primary, membership, posthoc=False):
    if primary and posthoc:
        raise ValueError(f'{name}: a post-hoc family cannot be primary')
    FAMILIES.append(dict(family=name, panel=panel, level=level, correction=correction,
                         primary=primary, posthoc=posthoc, membership=membership))


# Fig. 2 — acquisition (maps, n = 24)
family('F2c.primary', 'fig2c', 'A', 'holm', True,
       'Spearman rho of log(updates) and of log(simulated neuron-ms) vs certified optimal moves; '
       'Jonckheere-Terpstra trend of updates over tiers')
family('F2c.secondary', 'fig2c', 'A', 'holm', False,
       'Spearman rho of log(updates) vs tier, boxes, board size, floor cells, witness pushes and summed '
       'deadlocking alternatives; log(training seconds) vs optimum; log(updates per stage) vs optimum')
family('F2c.sensitivity22', 'fig2c', 'A', 'holm', False,
       'F2c.primary tests repeated without the two interrupted runs (n = 22)')
family('F2c.estimates', 'fig2c', 'A', 'none', False,
       'Theil-Sen log-log slopes (elasticities) of updates and neuron-ms on certified optimal moves; estimates with bootstrap CIs, no test')
family('F2d', 'fig2d', 'A', 'holm', True,
       'Wilcoxon signed-rank across maps of the per-map excess share of updates at push-entry stages '
       '(share of updates minus share of stages), and of the per-map Spearman rho between stage updates and deadlocking alternatives')
family('F2d.companion', 'fig2d', 'A', 'none', False, 'Sign tests accompanying F2d')

# Fig. 3 — brain vs readout (maps, n = 24)
family('F3c', 'fig3c', 'A', 'holm', True, 'Exact McNemar: new brain + new readout vs each of the other three pairings')
family('F3d', 'fig3d', 'A', 'holm', True,
       'Wilcoxon signed-rank across maps for the brain, readout and interaction margin terms')
family('F3d.companion', 'fig3d', 'A', 'none', False, 'Sign tests accompanying F3d')
family('F3e.consistency', 'fig3e/ed04a', 'A', 'bh', True,
       'Cross-map consistency (mean pairwise Spearman) and top-1% overlap of parameter change vs a null permuted '
       'within activity x membership x distance strata, per parameter type')
family('F3e.enrichment', 'fig3e/ed04c', 'A', 'bh', False,
       'Enrichment of superclasses and transmitters among the consistently most-changed groups vs stratified expectation')
family('F3e.magnitude', 'ed04a', 'A', 'holm', False,
       'Spearman rho across maps between the L2 norm of each core parameter change and certified optimal moves')
# Fig. 4b and memory use (histories or cue pairs within one model)
family('F4b.trained', 'fig4b', 'B', 'holm', True,
       'M0 trained timings: exact McNemar of carried-state readout-only vs reset-state and vs untrained readout, on histories and on both-cue pairs')
family('F4b.untrained', 'fig4b', 'B', 'holm', False, 'Same contrasts on untrained M0 timings')
for _fam in ('M0', 'S1', 'S2', 'M0vis'):
    family(f'F5.memory_use.{_fam}', 'ed05c/ed05d', 'B', 'holm', False,
           f'{_fam}: exact sign test of cue pairs with both cues solved vs neither, per model x split (memoryless policies solve exactly one cue per pair)',
           posthoc=True)

# Fig. 5 — limits
family('F5a.histories', 'fig5a', 'B', 'holm', True,
       'Exact Cochran-Armitage trend of per-history solution over delay rank on untrained timings, per connectome model')
family('F5a.pairs', 'fig5a', 'B', 'holm', False,
       'Exact Cochran-Armitage trend of both-cue solution over delay rank on untrained cue pairs, per connectome model', posthoc=True)
for _fam in ('M0', 'S1', 'S2'):
    for _split in ('train', 'untrained'):
        family(f'F5b.{_fam}.{_split}', 'fig5b', 'B', 'holm', _split == 'untrained',
               f'{_fam} {_split} histories: exact McNemar of the connectome readout-only reference vs every other learner on identical histories')
        family(f'F5b.pairs.{_fam}.{_split}', 'fig5b', 'B', 'holm', False,
               f'{_fam} {_split} cue pairs: exact McNemar of both-cue solution, reference vs every other learner', posthoc=True)
family('F5d.decoding', 'fig5d/ed05d', 'B', 'holm', False,
       'Exact sign test of cue pairs decoded both-correct vs both-wrong (M0-fitted decoders on M0/S1/S2; dense-delay decoder)', posthoc=True)
family('F5e', 'fig5e', 'B', 'holm', False, 'S2 readout-only: exact McNemar between orientations, paired by cue, delay and prefix, per split')

# Fig. 4c-f, 5c, 6d-e — interventions (histories within one model)
M1_UNITS = ('M0', 'M0vis.hidden', 'M0vis.visible', 'S1', 'S2')
MS_UNITS = ('M0', 'M0joint', 'M0vis.hidden', 'M0vis.visible', 'S1', 'S2')
for _u in M1_UNITS:
    family(f'F4c.primary.{_u}', 'fig4c', 'B', 'holm', True,
           f'{_u}: exact McNemar, clear or opposite-cue swap of the whole neural state at the write, hold and read windows vs baseline')
    family(f'F4c.components.{_u}', 'fig4c/ed06a', 'B', 'holm', False,
           f'{_u}: exact McNemar, clear or swap of voltage, current, adaptation, resources or delay buffer at each window vs baseline')
    family(f'F4c.margin.{_u}', 'fig4c', 'B', 'holm', False,
           f'{_u}: Wilcoxon signed-rank of first-move margin change for the F4c.primary conditions')
family('F4c.original', 'fig4c', 'B', 'holm', False,
       'Remote batch 1, M0 readout-only, decision moment (prefix 4, 7 delays): exact McNemar of each intervention vs baseline')
family('F4c.supplement', 'fig4c', 'B', 'holm', False,
       'Supplement causal probe on M0 readout-only: decision-window interventions vs no intervention, cue-offset and mid-delay interventions vs same-window sham')
family('F4d', 'fig4d', 'B', 'holm', True,
       'M0 shared visibility readout: sign test of the per-history (hidden minus visible) intervention effect, whole-state clear/swap at each window')
family('F4d.supplement', 'fig4d', 'B', 'holm', False,
       'Supplement visibility causal probe: sign test of the per-history (hidden minus visible) effect per decision-window intervention')
family('F4e.hidden', 'fig4e', 'B', 'holm', False,
       'Exact McNemar per hidden-goal model: first push follows the donor cue after a cue swap vs after norm-matched scrambling (eligible histories)',
       posthoc=True)
family('F4e.visible', 'fig4e', 'B', 'holm', False, 'Same test for the visible-goal condition of the shared readout', posthoc=True)
family('F4e.combined', 'fig4e', 'B', 'none', False,
       'Secondary: stratified exact test across the hidden-goal models with distinct brain states (combined evidence within one lineage)',
       posthoc=True)
family('F4e.sensitivity', 'fig4e', 'B', 'holm', False,
       'Exact McNemar per model on all histories whose own and donor pushes are defined and differ (not only eligible histories)', posthoc=True)
family('F4e.clustered', 'fig4e', 'B', 'holm', False,
       'Timing-clustered sensitivity: sign test across timings of (donor-following under cue swap minus under scrambling)', posthoc=True)
family('F4f', 'fig4f', 'B', 'holm', True, 'Exact Page trend test of first-decision margins over interpolation dose 0/0.25/0.5/1 (read window, combined state), per model')
family('F4f.companion', 'fig4f', 'B', 'holm', False, 'Wilcoxon signed-rank of margin (cue minus norm-matched scrambled) at each dose, per model')
for _u in MS_UNITS:
    family(f'F5c.{_u}', 'fig5c', 'B', 'holm', True,
           f'{_u}: exact McNemar of each neutral-input kind vs duration-matched blank waiting, histories pooled over durations')
    family(f'F6d.{_u}', 'fig6d', 'B', 'holm', True,
           f'{_u}: exact McNemar, full cue swap (dose 1) vs norm-matched scrambled, per state component x window')
    family(f'F6d.midpoint.{_u}', 'fig6d/ed06a', 'B', 'holm', False,
           f'{_u}: exact McNemar, cue midpoint (dose 0.5) vs norm-matched scrambled, per state component x window')
family('F6e.memory', 'fig6e', 'B', 'holm', True,
       'Memory candidate groups on the S1 confirmation family: exact McNemar, candidate vs matched group under the same suppression strength and window')
family('F6e.routes', 'fig6e', 'B', 'holm', False,
       'Direct-readout replacement at strength 1: Wilcoxon signed-rank of |margin change|, candidate vs matched group, per model')

family('F6a', 'fig6a', 'A', 'holm', False,
       'Wilcoxon signed-rank across maps: mean absolute matched-state voltage change, observed vs other cells and input vs other cells')
family('F6c', 'fig6c', 'B', 'holm', False,
       'Per model (final readout): sign test over timings of the readout row-space fraction minus the isotropic reference 3/512')
family('F6f.decoding', 'fig6f', 'B', 'holm', False,
       'Supplement held-history cue decoders: exact sign test of cue pairs decoded both-correct vs both-wrong, per model x split', posthoc=True)
family('F8.enrichment', 'ed08b', 'B', 'bh', False,
       'Exploratory, within one model: Fisher exact enrichment of superclasses and transmitters among the top-1% cue-responsive cells, per distinct state set')

# Circuit level (figure phase, added 2026-10-18 at the user's request for neuron-level analyses; post hoc, declared
# before these tests were computed; unit = cells within one distinct state set)
family('F6g.ensemble', 'fig_circuit', 'B', 'holm', False,
       'Wiring among the top-1% cue-responsive cells: within-set connections vs random sets matched by superclass, role, '
       'degree quintile and 150-um display location (1,000 draws), per distinct state set', posthoc=True)
family('F6g.sign', 'fig_circuit', 'B', 'holm', False,
       'Within that ensemble: excitatory share of connections between same-preference cells minus between opposite-preference '
       'cells, vs permutation of the cue-preference signs (1,000 draws), per distinct state set', posthoc=True)

# Follow-up analyses after freeze-v1.1 (plan v2, Phase 1B; panels/p11_followup.py). Declared before computation; post hoc
# because the questions came from the evidence audit of the frozen paper. Unit: timings or states within one model.
family('F11.timing.differential', 'followup/fu1', 'B', 'holm', False,
       'Timing lock: mean cue-differential readout margin at the 26 untrained vs the 9 trained timings, per cue pair (timing-label permutation)',
       posthoc=True)
family('F11.timing.common', 'followup/fu1', 'B', 'holm', False,
       'Timing lock: mean |cue-independent| readout margin at the 26 untrained vs the 9 trained timings, per cue pair (timing-label permutation)',
       posthoc=True)
family('F11.timing.decoder', 'followup/fu1', 'B', 'holm', False,
       'Timing lock: leave-one-timing-out shrinkage-LDA cue decoding over all 35 timings vs within-timing label swaps, per cue pair',
       posthoc=True)
family('F11.swap.history', 'followup/fu2', 'B', 'holm', False,
       'Swap specificity: histories solved after a same-cue vs an opposite-cue whole-state swap (exact McNemar; the write, hold '
       'and read windows collapsed to one record per history), per model unit', posthoc=True)
family('F11.swap.timing', 'followup/fu2', 'B', 'holm', False,
       'Swap specificity clustered by cue pair: histories solved per timing after same-cue minus opposite-cue swaps (exact sign '
       'test), per model unit', posthoc=True)
family('F11.swap.combined', 'followup/fu2', 'B', 'holm', False,
       'Swap specificity combined over the hidden-goal units M0, M0 shared readout, S1 and S2 (stratified exact McNemar)', posthoc=True)
family('F11.swap.visibility', 'followup/fu2', 'B', 'holm', False,
       'Shared readout: disruption by the opposite- vs the same-cue swap, hidden minus visible goal on matched histories (exact sign test)',
       posthoc=True)
family('F11.equivalence.memory_use', 'followup/fu3', 'B', 'holm', False,
       'Non-superiority (Tango score test, margin +10 percentage points): share of cue pairs with both cues solved minus neither at '
       'the untrained timings, per connectome memory model (hidden goal)', posthoc=True)
family('F11.equivalence.transfer', 'followup/fu3', 'B', 'holm', False,
       'Equivalence (TOST, Tango score, margin 10 pp): cue pairs decoded both-correct minus both-wrong by the M0-fitted decoder on '
       'S1 and S2, and by the fixed decoder of the dense delay scan at untrained timings', posthoc=True)
family('F11.equivalence.duration', 'followup/fu3', 'B', 'holm', False,
       'Equivalence (TOST, Tango score, margin 10 pp): histories solved after 2 neutral input windows vs duration-matched blank '
       'waiting, per unit and neutral input (the 18 comparisons of Fig. 7g)', posthoc=True)
family('F11.ensemble.strict', 'followup/fu5', 'B', 'holm', False,
       'Wiring among the top-1% cue-responsive cells vs random sets matched by cell type, synaptic distance from the inputs and '
       'degree quintile (1,000 draws), per distinct state set', posthoc=True)
family('F11.ensemble.timings', 'followup/fu5', 'B', 'holm', False,
       'Replication over timings: the top-1% ensemble at the first decision window more interconnected than the mean of 100 sets '
       'matched as in Fig. 6g, counted over the 35 timings (exact sign test), per distinct state set', posthoc=True)
family('F11.brain_change', 'followup/fu7', 'B', 'holm', False,
       '24 maps: total brain parameter change vs the brain term and the total margin change of the decomposition, controlling for '
       'log(updates) (partial Spearman, Freedman-Lane permutation), and vs log(updates) (Spearman)', posthoc=True)
family('F11.pairing', 'followup/fu6', 'B', 'holm', False,
       'M0 joint, brain x readout pairing on identical inputs: differential readout margin of matched vs mismatched brain-readout '
       'stages at the 9 trained timings (exact sign test)', posthoc=True)
# Declared 2026-10-19, before computation (remaining Phase 1B item; manuscript/PLAN.md section 14).
family('F11.sign.propagation', 'followup/fu8', 'B', 'holm', False,
       'Sign motif of the top-1% cue ensemble (Fig. 6e) vs a signed-propagation null: the same statistic in the top-1% sets of the '
       'linear response of the row-normalised signed connectome (gain 0.5 and 0.8) to the state\'s adaptation-difference pattern '
       'with random signs (200 draws, two-sided), per distinct state set and gain', posthoc=True)
