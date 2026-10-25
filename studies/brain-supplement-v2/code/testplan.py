"""Test families of brain-supplement-v2, declared before any test is computed (studykit/testplan.py).

Every Phase 1B and 1C test (plan v2) must name a family declared here before the test is first computed.
Multiplicity is controlled within a family; families are never pruned after results are seen. A family declared
after its data were seen takes posthoc=True and can never be primary. No family is declared yet.
"""
from studykit.testplan import TestPlan

PLAN = TestPlan('brain-supplement-v2')
FAMILIES = PLAN.families
family = PLAN.family

# Phase 1C X1 (declared 2026-10-19, before any replicate result existed). Unit: the replicate (training run).
# Replicates differ only by the order of the 9 trained timings and a 0.1-SD readout-initialisation jitter
# (code/replicate_training.py); replicate 0 is the frozen protocol. Learners: M0 readout-only and joint (10 each),
# S1 and S2 readout-only (7 each). Memory use of a run = (cue pairs with both cues solved - pairs with neither) / pairs.
family('F13.X1.trained', 'x1', 'B', 'holm', True,
       'Memory use at the 9 trained timings is positive across replicates, per learner (exact one-sided sign-flip test '
       'on the per-replicate net memory use)')
family('F13.X1.untrained', 'x1', 'B', 'holm', True,
       'No net memory use at the 26 untrained timings: per-replicate net memory use below +0.10, per learner (exact '
       'one-sided sign-flip test on net use - 0.10; non-superiority)')
family('F13.X1.lock', 'x1', 'B', 'holm', True,
       'Timing lock: per-replicate net memory use at trained minus untrained timings is positive, per learner (exact '
       'one-sided sign-flip test)')
family('F13.X1.descriptive', 'x1', 'B', 'bh', False,
       'Training outcome (learned_solved), updates and solved histories across replicates, per learner, against the '
       'frozen single run (descriptive intervals; replicate 0 reproduction check)')

# Phase 1C X4 and X5 (declared 2026-10-19, before any of their runs; code/x4_scrambles.py, code/x5_swaps.py). The frozen
# supplement's continuous-state interventions (mechanisms-supplement-v1, supp_replay) on its five memory models (six
# units: M0 readout-only, M0 joint, shared readout hidden and visible, S1, S2), read window (after the last pre-decision
# input), combined continuous state. Unit for the tests: the history (case x timing). Margin = first-decision logit of
# the case's reference (correct) first action minus the strongest other legal action, as in the paper's Fig. 4f;
# effect = margin after - margin at baseline.
# X4 (gap G08): the cue shift (toward the opposite cue's state by a dose a) vs 20 norm-matched scrambled shifts
# (supp_replay.matched_control with seeds offset by 1000 k, k = 1..20; k = 0 reproduces the frozen draw), at doses
# 0.125, 0.25, 0.5, 0.75 and 1, at the supplement's 6 timings.
family('F13.X4.content', 'x4', 'B', 'holm', True,
       'Content specificity: at dose 1, the margin effect of the cue shift minus the mean margin effect of the 20 '
       'scrambled shifts is negative (the cue shift moves the decision more), per unit (exact one-sided sign-flip over '
       'histories; Monte Carlo above 20 histories)')
family('F13.X4.dose', 'x4', 'B', 'holm', False,
       'Dose dependence: that difference decreases with dose over 0.125-1 (exact Page trend, histories as blocks), per unit')
family('F13.X4.rank', 'x4', 'B', 'none', False,
       'Descriptive: rank of the unit-mean cue effect among the 20 unit-mean scrambled effects, per unit and dose')
# X5 (gap G07): the full opposite-cue swap (dose 1, read window, continuous state) and the same-cue control at all 9
# trained timings; S2 also cross-orientation swaps (donor = the same or the opposite cue in the other context).
family('F13.X5.swap', 'x5', 'B', 'holm', False,
       'Opposite-cue swap vs baseline at the 9 trained timings: histories solved (exact McNemar, paired by history), per unit')
family('F13.X5.visibility', 'x5', 'B', 'holm', True,
       'Memory specificity (difference in differences): shared readout, histories disrupted by the opposite-cue swap '
       '(baseline solved, swap unsolved) with the hidden goal vs with the visible goal, matched by cue and timing (exact '
       'McNemar on the matched pairs; the same-cue control changes nothing by construction and is checked)')
family('F13.X5.cross', 'x5', 'B', 'bh', False,
       'S2 cross-orientation swaps (donor: same cue, other context; opposite cue, other context) vs baseline, histories '
       'solved (exact McNemar), and same-cue vs opposite-cue cross swaps')

# Phase 1C X8 (declared 2026-10-19, before any of its runs; code/x8_neutral_training.py). Sokoban training from a
# neutral brain (the frozen trainer's 'neutral' initialisation), readout-only (brain fixed) vs joint, on the
# representative maps D1-T0, D3-T0, D5-T0 and D7-T0. A replicate differs only by the readout jitter of X1 (0.1 SD, seed
# 2000 + r; the frozen trainer has no other randomness); r = 0 is the frozen protocol. Pilot: 3 replicates per arm and
# map; n is then fixed by simulated power and recorded as an amendment before the remaining runs. Outcome: updates to
# solve, censored at the frozen stop (stagnation or segment limit). The tests use the final n only.
family('F13.X8.primary', 'x8', 'B', 'holm', True,
       'Readout-only vs joint training from a neutral brain: updates to solve (censored at the stop), stratified log-rank '
       'over the four maps, replicates as units (permutation of the arm labels within maps; two-sided)')
family('F13.X8.map', 'x8', 'B', 'bh', False,
       'Per map: readout-only vs joint, updates to solve with unsolved runs ranked last (exact permutation Mann-Whitney), '
       'and runs solved (Fisher exact)')

# Phase 1C X6 and X7 (declared 2026-10-19, before any of their runs; code/x6_x7_prepare.py, code/x6_x7_ensemble.py).
# Ensembles: the top 1% of cells by |cue difference of v| in the Fig. 6 state of M0 readout-only and of S1 (both on the
# P05 brain; mechanisms-v1 interpretation tables). Null sets: 20 per ensemble, same size, drawn outside the ensemble and
# the input and readout cells, matched to its cell type x hop class from the inputs x degree quintile (relaxed to
# superclass x hop x degree, then hop x degree, where a stratum has no other cell; counted). Test scenes: the M0 and S1
# memory tasks at the 9 trained timings (frozen memory_rollout); each ensemble is tested in the other scene (derived on
# A, tested on B) and in its own. Margin and effects as in X4 (reference first action vs strongest legal competitor).
# X6: outgoing gate of the set at strength 1 during the first decision (the paper's group-test protocol).
# X7: every synapse between two cells of the set removed for the whole history (ensemble edges vs the within-set
# edges of the 20 null sets).
family('F13.X6.ensemble', 'x6', 'B', 'holm', True,
       'Silencing the cue ensemble moves the decision more than silencing matched sets: margin effect of the ensemble '
       'minus the mean over the 20 null sets < 0 (exact one-sided sign-flip over the 18 histories), per ensemble source and '
       'test scene (4 tests)')
family('F13.X6.rank', 'x6', 'B', 'none', False,
       'Descriptive: rank of the scene-mean ensemble effect among the 20 null sets, per ensemble source and test scene')
family('F13.X7.edges', 'x7', 'B', 'holm', False,
       'Removing the synapses within the cue ensemble moves the decision more than removing those within matched sets '
       '(as X6), per ensemble source and test scene (4 tests)')
family('F13.X7.rank', 'x7', 'B', 'none', False,
       'Descriptive: rank of the scene-mean ensemble-edge effect among the 20 null sets, per ensemble source and test scene')

# Phase 1C X3 (declared 2026-10-19, before any of its runs; code/x3_jittered_training.py). M0 readout-only, 10
# replicates; each update's timing drawn from delays 0-320 ms (step 20) x prefixes 0, 1, 2, 4, 6, 8; replicate r shares
# its readout start with X1 replicate r. Untrained timings as in X1 (26); 21 of them lie inside the jitter range, the 5
# at the 640-ms delay outside it.
family('F13.X3.lock', 'x3', 'B', 'holm', True,
       'The timing lock is a training-distribution artefact: net memory use at the 26 untrained timings after jittered '
       'training minus after fixed-timing training (X1), paired by replicate, is positive (exact one-sided sign-flip, n = 10)')
family('F13.X3.descriptive', 'x3', 'B', 'none', False,
       'Descriptive: net memory use at the trained timings, at the 21 untrained timings inside and the 5 outside the jitter '
       'range, and updates to learn, jittered vs fixed (paired by replicate)')

# Phase 1C X9 and X10 (declared 2026-10-19, before any of their runs; code/x9_replays.py, code/sokoban_replicate.py).
# X9 closes gaps G01, G06 and G10 of manuscript/results/qa/gaps.csv; X10 adds replicates to Fig. 2.
family('F13.X9.pairing', 'x9', 'A', 'bh', False,
       'G06: original-start solution with the new brain and new readout (11) vs each pairing involving the mid-training '
       'checkpoint (brain or readout from mid; 5 pairings), exact McNemar over the 24 Fig. 2 maps; withheld unless the 4 '
       'frozen pairings reproduce Fig. 3c')
family('F13.X9.descriptive', 'x9', 'A', 'none', False,
       'Descriptive: G10 original-start behaviour at every curriculum-stage checkpoint (share of maps solved by stage '
       'fraction); G01 the D7 training controls (status, updates, solved) next to the D1/D3/D5 controls')
# Post hoc (2026-10-19, after the G10 replays were seen): the declared G10 share is 0 by construction. The frozen trainer
# (studies/brain-sokoban-v1/code/training.py) stops at the first original-start solve and saves a stage checkpoint only
# when its stage passes without one, so every stage checkpoint is unsolved from the original start. Graded measures of
# the same replays instead; no test.
family('F13.X9.posthoc', 'x9', 'A', 'none', False,
       "Descriptive: at the curriculum-stage checkpoints, the share whose original-start first decision equals the final "
       "model's, and the median steps before the original-start rollout ends, by stage fraction (fifths)", posthoc=True)
family('F13.X2.trained', 'x2', 'B', 'holm', True,
       'X2 (new scenes: M1, M2, S1 mirror, S1 minus cell 45, S2 mirror, S2 minus cell 48; 7 replicates each, as X1): memory '
       'use at the 9 trained timings is positive across replicates, per scene (exact one-sided sign-flip)')
family('F13.X2.untrained', 'x2', 'B', 'holm', True,
       'X2: no net memory use at the 26 untrained timings (net use below +0.10), per scene (exact one-sided sign-flip)')
family('F13.X2.lock', 'x2', 'B', 'holm', True,
       'X2: timing lock (trained minus untrained net use > 0), per scene (exact one-sided sign-flip)')
family('F13.X2.heterogeneity', 'x2', 'B', 'none', False,
       'Descriptive: I^2 and Cochran Q of the lock effect across the original and the two new scenes of each family '
       '(replicate means and their standard errors; random-effects pooling)')
family('F13.X10.fig2', 'x10', 'A', 'none', False,
       'Fig. 2c across replicates: Spearman rho of log(updates) with the certified optimal moves over the 24 maps, per '
       'replicate set (frozen run plus 5 replicates differing by the X1 readout jitter); reported with its p per set, '
       'the median and range of rho, and rho of the per-map median updates')
