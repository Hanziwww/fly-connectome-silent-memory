# Continuous neural state, historical content, and action reading: full interpretation of the frozen-model supplement

This report covers the 20 preregistered jobs and five existing checkpoints in `mechanisms-supplement-v1`. All jobs, retrieval, and paired analyses are complete. No model training or parameter updates occurred in this round, and the original local training STOP remained in place.

**The clearest new finding is that continuous neural states produced by the opposite cue can redirect the first push toward the donor-cue direction.** This directional transfer appeared in M0, S1, and S2 and exceeded matched-magnitude scrambled controls. It extends the earlier observation that state changes affect behavior to evidence that cue-related state content affects subsequent choice. However, interventions also disrupted visible-goal behavior; the readout contribution of one candidate group did not establish necessity for complete solutions; and time/input effects did not follow simple monotonic forgetting. The main interpretation is therefore **task-specific representation–readout coordination, with historical state and current input jointly shaping action**, rather than a dedicated memory module, stable attractor, or performance superiority of the real connectome.

## 1. Completed work

| Acceptance item | Result |
|---|---|
| Scope | 5 independent reload preflights; 5 continuous-state intervention jobs; 5 propagation/direct-readout control jobs; 5 matched-duration input jobs |
| Execution | 20/20 ended normally; all have `verification.status=verified`; no technical job failure |
| Parameter freezing | 20/20 parameter audits unchanged; no optimizer or training queue invoked |
| Record count | 3,800 batched conditions; 10,640 condition–case-history behavior records, not independent map replicates |
| Retrieval | 20/20 file receipts; 8,250 manifest files, 1,359,599,110 bytes, about 1.266 GiB |
| Final checks | SHA-256 recomputed for every manifest file; no missing files, size differences, or hash differences; 69 source/provenance/input hashes matched |
| Controls | Preflight, sham, same-cue, and restoration checks passed; all matched-magnitude controls met the prespecified 1% norm tolerance |
| Actual processes | Post-completion remote `/proc` checks showed controller and workers exited; GPUs 0–3 had no experiment process |
| Resources | Cloud queue elapsed time about 50.92 minutes; final S2 continuous-state job: 474 conditions, 36.27 minutes; local free space at final verification about 106.67 GiB |

These times measure frozen replay and recording, not new training or a speedup against the local GPU. All runs derive from seed 10. The five models span three task families, M0, S1, and S2; M0 training variants and the two S2 orientations are not new independent maps.

Independent preflight reproduced original trained-history solution counts of 18/18 for M0 readout-only, 15/18 for M0 joint, 36/36 for M0 shared visible/hidden readout, 18/18 for S1, and 29/36 for S2. Partial M0 joint and S2 performance matches the original ability range; neither was retrained to full success here. The state experiments below use different fixed timings (40/160/640 ms × 0/4 prefixes), so their baseline denominators are 12 or 24, not the 18/36 above.

Acceptance records: [final integrity audit](analysis/final-integrity-audit.json), [per-job audit](analysis/final-job-audit.csv), [source audit](analysis/final-source-audit.csv), and [locked protocol](protocol.en.md).

## 2. What continuous-state transfer conveys

For histories with the same context, current input, and clock but opposite cues, denote continuous states by \(s\) and \(s'\). The intervention is

\[
s_{a}=(1-a)s+a s',\qquad a\in\{0.25,0.5,1\}.
\]

The operated variables are voltage, current, adaptation, or their combination. The midpoint \(a=0.5\) removes the paired cue difference in those variables; \(a=1\) replaces them with donor values. Delay buffers, spikes, refractory state, clock, buffer pointers, RNG state, and parameters remain with the recipient. These are partial-state interventions, not complete memory erasure or whole-brain-state replacement.

The table reports all histories for combined continuous-variable interventions immediately before the first decision window. Parentheses give paired losses/gains relative to each history's baseline, rather than net changes alone.

| Model/goal | Baseline solved | Cue midpoint | Midpoint matched control | Cue swap | Swap matched control |
|---|---:|---:|---:|---:|---:|
| M0 readout-only/hidden | 9/12 | 7/12 (4/2) | 9/12 | 4/12 (6/1) | 8/12 |
| M0 joint/hidden | 8/12 | 7/12 (3/2) | 8/12 | 5/12 (6/3) | 9/12 |
| M0 shared readout/hidden | 9/12 | 7/12 (4/2) | 10/12 | 4/12 (6/1) | 10/12 |
| M0 shared readout/visible | 12/12 | 8/12 (4/0) | 8/12 | 2/12 (10/0) | 11/12 |
| S1/hidden | 11/12 | 7/12 (4/0) | 5/12 | 2/12 (9/0) | 9/12 |
| S2/hidden | 15/24 | 11/24 (6/2) | 12/24 | 9/24 (10/4) | 13/24 |

Cue swaps disrupted complete solutions more than equal-norm scrambling in all six table conditions. Midpoint evidence is weaker: the S1 matched control was more disruptive than the cue midpoint, and both yielded 8/12 for visible-goal M0. Not every midpoint impairment can be attributed specifically to deletion of memory content.

Dose responses were not strictly monotonic. M0 readout-only solved 5/12, 7/12, and 4/12 at \(a=0.25,0.5,1\). S2 at \(a=0.25\) still totaled 15/24 while losing 4 histories and gaining 4. An unchanged net solution rate does not imply unchanged within-history behavior. Synthetic intermediate states, nonlinear dynamics, and discrete action boundaries are possible explanations that this round did not further separate.

### Directional content transfer

A post-completion descriptive audit additionally considers cases where both opposite-cue baselines solve at the same timing. Selection uses baseline behavior only, not intervention failure. It compares whether the first post-swap push follows the donor baseline direction against the equal-norm scrambled control.

| Model/goal | Case histories with both baselines solved | Follows donor after cue swap | Follows donor after equal-norm scrambling |
|---|---:|---:|---:|
| M0 readout-only/hidden | 8 | 6/8 | 0/8 |
| M0 joint/hidden | 6 | 5/6 | 0/6 |
| M0 shared readout/hidden | 8 | 6/8 | 0/8 |
| M0 shared readout/visible | 12 | 10/12 | 1/12 |
| S1/hidden | 10 | 9/10 | 2/10 |
| S2/hidden | 10 | 8/10 | 2/10 |

This is more specific than a drop in solution count: the perturbation often redirects choice toward the donor cue rather than merely inducing random error. S2 donors and recipients share context, with orientation revealed only at the formal decision; the results support an effect of historical content on context-appropriate subsequent choices. There was no cross-orientation swap or full cue–context interaction decomposition, so this does not establish general contextual reasoning.

This direction table is a supplementary conditional description, not a replacement for the all-history outcomes above or an independent-sample significance test. Raw records also retain histories outside the double-baseline-success subset.

![State content and directional transfer](analysis/final-01-state-content.png)

Figure 1: the upper panel uses all fixed timings; the lower panel uses histories where both baselines solve. Colors encode solution fraction and intervention type, and labels give actual case counts. They do not represent extra maps or training seeds.

Data: [all state pairs](analysis/final-state-factorial.csv), [direction transfer by case](analysis/directional-swap-audit.csv), and [direction summary and selection rule](analysis/directional-swap-audit.json).

## 3. State components and time windows: model- and task-dependent effects

Pre-read swaps of combined variables consistently disrupted solutions across models, but individual-variable rankings differed:

- M0 readout-only: voltage and current swaps before reading each solved 6/12, versus 9/12 for their equal-norm controls; adaptation swap and control both solved 8/12.
- M0 joint: adaptation swaps yielded 6/12 versus 10/12 before reading and 3/12 versus 8/12 after writing. Adaptation differences had a prominent behavioral effect in this existing jointly trained checkpoint, but this cannot yet be assigned to a specific parameter change during training.
- S1: pre-read adaptation swap yielded 9/12 versus 7/12, whereas the combined swap yielded 2/12 versus 9/12. The combined effect cannot simply be reduced to adaptation being the memory.
- S2: the pre-read combined swap yielded 9/24 versus 13/24, while the combined swap at the midpoint of the holding interval yielded 13/24 versus 12/24. Replacement states at different windows undergo different amounts of subsequent evolution.

Combined interventions generally change more variables. Stronger behavioral effects do not directly establish statistical or dynamical synergy. This round compares matched controls; it does not decompose every higher-order nonlinear interaction between variables.

All equal-norm controls met the prespecified tolerance. The maximum relative norm error was 0.005294, or 0.5294%, for a current intervention in M0 joint; most others were near floating-point precision. After nonnegative adaptation projection, the maximum absolute direction cosine was 0.01815, so exact orthogonality cannot be claimed throughout. Voltage/current cosines were near numerical zero. Component audits also retain the zero-boundary fraction, which includes initially zero cells and is not the fraction of cells erased by the intervention.

**Visible targets provide an important counterexample.** A cue swap reduced the shared M0 readout from 12/12 to 2/12 even with a currently visible target, often following the wrong donor cue. This is consistent with historical state overriding or interfering with current evidence, or with state–input mismatch. It does not show a lesion confined to hidden-goal memory with otherwise normal decision making.

![Components and time windows](analysis/final-02-components-windows.png)

Figure 2: each cell shows the solution-count difference between a full component cue swap and equal-norm scrambling, with its denominator. Red indicates fewer solutions after cue swap. All four components, three windows, and six model/visibility conditions are shown. The paired table retains all doses, 0.25, 0.5, and 1.

Data: [component–window–dose table](analysis/final-state-factorial.csv), [norm and direction audit](analysis/final-norm-summary.csv), and [operation-level audit](analysis/final-norm-audit.csv).

## 4. Candidate cell group: readout contribution without established necessity for full solutions

This round tested only the previously locked MNad18/MNad27 candidate group and its matched group. The candidate comprises 8 noninput cells, including 4 observed cells. Cells were not reselected using these results.

Conditions were output-propagation suppression only, direct-readout replacement only, both, and no intervention; strengths were 25%, 50%, and 100%, during the first decision or all decisions. Direct replacement interpolated selected cells' pooled contributions toward the checkpoint feature mean, retaining the denominator of 16 cells per pool, normalization, clipping, bias, and legal-action mask. It neither zeroed a whole feature pool nor biologically inactivated a cell.

Across all 3,024 nonzero channel-intervention case histories, neither candidate nor matched groups produced any new loss or gain of complete solutions. The 3024 are related condition combinations, not independent replicates. Candidate propagation-only interventions changed the first-action margin by zero, consistent with the previous effective-graph result of zero one- and two-hop mass to observed cells. Direct-readout interventions did change scores:

| Model/goal | Candidate direct readout: maximum absolute margin change | Matched direct readout: maximum absolute margin change |
|---|---:|---:|
| M0 readout-only/hidden | 0.1100 | 0.0628 |
| M0 joint/hidden | 0.1346 | 0.1280 |
| M0 shared readout/hidden | 0.1458 | 0.0582 |
| M0 shared readout/visible | 0.4574 | 0.0747 |
| S1/hidden | 0.0490 | 0.2665 |
| S2/hidden | 0.2366 | 0.1087 |

The margin contrasts the first reference action with other legal actions; that reference need not be the only successful route. Readout scales differ, so cross-model maxima are not importance rankings. The maxima demonstrate a numerical contribution of the direct channel; all doses and windows are retained separately.

The precise conclusion is: **this group contributes to action computation, but its necessity for complete solutions was not established under the tested mean-replacement and propagation interventions.** That is narrower than the cells having no effect and does not identify a memory center. In S1 the matched group changed scores more, further limiting group-specific interpretation. Unchanged complete-solution outcomes also do not guarantee identical routes at every step.

![Channel separation](analysis/final-03-readout-route.png)

Figure 3: maximum absolute change in the first reference-action margin under direct-readout intervention. Per-strength, window, and history values are retained. Every channel combination had zero complete-solution losses/gains; score changes alone do not establish behavioral necessity.

Data: [propagation/direct-readout factorial table](analysis/final-route-factorial.csv) and [within-history channel interaction](analysis/route-interactions.csv).

## 5. Matched duration: subsequent input changes the usability of historical state

Cue offset to the first formal decision was fixed at 80, 120, 160, 240, 320, 480, or 640 ms. Blank waiting was paired with an initial blank followed by 2/4/8 neutral windows. Each window lasted 40 ms. Neutral input was identical across cues and the two S2 contexts, omitted targets, and did not alter the actual solving start. S2 orientation was revealed only at the formal decision.

The main figure fixes 2 windows so all seven total durations are comparable. The examples below illustrate the full figure, not replace its complete table:

- For M0 readout-only at 240 ms, blank waiting solved 2/2 while stationary or either neutral walk solved 0/2. At 480 ms, blank waiting solved 0/2 and neutral stationary input solved 2/2. Neutral input can disrupt or restore a usable choice state.
- For S2 at 80 ms, blank waiting solved 2/4 and neutral stationary input solved 4/4; at 160 ms, the corresponding counts were 4/4 and 2/4. Input effects depend on timing and state.
- M0 joint blank-only outcomes across increasing times were 1/2, 2/2, 2/2, 0/2, 1/2, 1/2, and 0/2. Improvement, decline, and recovery rule out describing behavioral memory duration with one threshold or a simple exponential decay fit.

Matching total duration excludes the explanation that neutral-input conditions merely waited longer, but does not fully separate content, sensory drive, and the recency of input offset relative to decision. More input windows replace more blank waiting; comparisons of 2/4/8 windows are not pure causal doses varying only interference strength.

The visible-goal shared-readout condition also failed on some new timings, indicating a broader state–decision coordination problem. These data support time- and input-dependent use of history, not general stable memory independent of input.

![Matched-duration outcomes](analysis/final-04-matched-time.png)

Figure 4: each cell gives a complete-solution count, with two cue cases per M0/S1 cell and four cue×context cases per S2 cell. Columns follow time; failed timings are retained, and heatmap cells are not independent training replicates.

Data: [all durations and window counts](analysis/final-timing-summary.csv) and [within-history losses/gains relative to duration-matched blank](analysis/final-timing-pairs.csv).

## 6. Decoding and low-dimensional dynamics: useful interpretations and unestablished mechanisms

Interpretive cue decoding uses the actual normalized, clipped features from the first formal decision window. Coordinates and regression fit only discovery histories with fixed regularization and do not generate actions. The table shows held-out total durations 120/240/480/640 ms; new-input results at seen times are also retained.

| Model/goal | Correct cue decoding at held-out durations |
|---|---:|
| M0 readout-only/hidden | 26/62 |
| M0 joint/hidden | 35/62 |
| M0 shared readout/hidden | 26/62 |
| M0 shared readout/visible | 43/62 |
| S1/hidden | 35/62 |
| S2/hidden | 65/124 |

These results do not establish temporally stable linear reading of hidden cues. Visible-condition decision inputs contain the current target, so 43/62 is not direct evidence of better pure-memory decoding. Failure of this interpretive decoder also does not imply absence of information from every cell, state variable, or nonlinear readout; it tests this feature representation and fixed fitting procedure. The per-history decoder-correctness versus full-solution cross-table is retained because the two outcomes are not interchangeable.

Low-dimensional dynamics use simultaneous current/voltage summaries at 1 ms resolution, distinct from the policy's mixed 14 ms current and 40 ms voltage features. Fixed 8-dimensional coordinates and affine predictors fit discovery histories. Held-out-timing population-activity \(R^2\) for 40 ms prediction is:

| Model | Persist current state | Autonomous affine dynamics | Input-plus-phase dynamics |
|---|---:|---:|---:|
| M0 readout-only | −0.836 | −0.082 | 0.375 |
| M0 joint | −0.860 | 0.012 | 0.437 |
| M0 shared readout/hidden | −0.836 | −0.082 | 0.375 |
| S1 | −0.822 | −0.001 | 0.429 |
| S2 | −0.864 | −0.014 | 0.269 |

The readout-only M0 and shared-readout hidden condition have identical brain parameters and activity under the same histories, so their identical results are not independent replication. Adding actual input and phase improves held-out multistep prediction; autonomous evolution alone does not describe these activity changes. This is **prediction conditional on recorded future inputs**. Input and phase were added together, without separating their contributions. Much variation remains unexplained, and the models did not directly predict cue differences or intervention behavior. Positive \(R^2\) therefore does not identify a stable memory attractor or a memory-maintenance time constant.

Complete results retain all four prediction horizons, 1, 20, 40, and 80 ms, not only favorable horizons. The illustrated 40 ms horizon matches the prespecified decision window.

![Decoding and input-driven dynamics](analysis/final-05-decoding-dynamics.png)

Data: [cue decoding by history](analysis/held-history-cue-decoding.csv), [decoder/behavior cross-table](analysis/final-decoder-behavior-cross.csv), and [all prediction horizons](analysis/input-conditioned-dynamics.csv).

## 7. Connection to the acquisition study

The completed [55-job mechanism study](../mechanisms-v1/interpretation.en.md) recorded independent-reload solutions on 24 difficulty tasks. For jointly trained parameter pairs, old brain+old readout, new brain+old readout, old brain+new readout, and new brain+new readout solved 0/24, 0/24, 1/24, and 24/24, respectively. A separate readout-only training path also learned representative tasks. This supports learned coordination, not a universal requirement for brain-parameter updates in new learning. Each of four representative models scored 0/8 on the same eight untrained layouts; general Sokoban planning was not established. These are earlier results: this round added no difficulty training or generalization tests.

This supplement adds a state-level account. With parameters fixed, different histories can generate different continuous states despite the same current observation, and state content can transfer subsequent choices. However, the trained readout has limits across historical timing and current input. Local cellular contributions or decodable cues do not automatically guarantee correct complete behavior.

The supported research statement is: **in a connectome-constrained neural dynamical model, learning establishes task-specific coordination between neural representations and action readout; historical content in continuous neural state causally influences subsequent decisions, with its behavioral use constrained jointly by input timing, current goals, and context.** The subjects are the model and simulated neurons, not evidence that real flies learned Sokoban.

## 8. Mechanistic claims, alternatives, and testable predictions

| Current claim | Direct supporting data | Remaining alternatives/model conditions | Testable prediction, not an added task in this round |
|---|---|---|---|
| Continuous state carries historical content affecting choice | Same-context opposite-cue swaps often redirect the first push toward the donor, beyond equal-norm scrambling | Partial-variable replacement; subsequent evolution and current input act together; synthetic states can lie off natural-history trajectories | Cue-direction interpolation at matched history and input should change the corresponding preference, though discrete boundaries can make behavior nonmonotonic |
| Historical state is not wholly isolated from current sensory processing | Swaps also produce donor-directed choices and solution losses when goals are visible | Conflict between history and current goals, or broader state-coordination impairment | Stronger current evidence or altered presentation timing may change the weight of history; distinguish those explanations |
| The selected observed group contributes scores, without established necessity | Replacing direct pooled contributions changes margins, with zero full-solution losses/gains | Only 8 cells, 4 observed members, replacement toward a mean, artificial observation channels, and possible compensation | Larger or different functional groups may affect behavior, but the present null result does not guarantee this; do not reselect the group post hoc to rescue the hypothesis |
| Time and subsequent input change the usability of history | Duration-matched blank and neutral input produce paired losses and gains; behavior is nonmonotonic over time | Sensory drive, recent input, and trained timing distributions vary together | Equal-duration inputs arranged at different stages may produce different preferences; total delay alone need not determine choice |
| Population activity has a predictable input-driven component | Input-plus-phase models outperform both baselines at a 40 ms horizon on held-out times | Future inputs are known; population variance is explained, but not all cue information or intervention behavior | Test unseen input schedules; positive predictive ability alone is not a memory mechanism |

Unsupported stronger claims include a universal dedicated memory cell type; all memory residing solely in adaptation or voltage; a stable memory attractor generalizing across timing; full-behavior necessity of the candidate observed group; superiority of the real connectome to general networks; and extrapolation to other seeds or real animals.

## 9. Deliverables and review entry points

The five figure sets are saved as 300 dpi PNG, vector PDF, and SVG. Per-condition data, failed histories, source, and verification receipts are retained. `analysis.en.md` is the autogenerated numerical summary; this file provides the full interpretation. The machine-readable index is [final-summary.json](analysis/final-summary.json).

Raw experiment files remain read-only in `runs/`. Post-completion auditing/plotting is in `monitoring/final_evidence.py`, with directional auditing in `monitoring/paired_direction_summary.py`. Its double-baseline-success subset is explicitly a post-completion descriptive analysis, not model reselection or a registered-condition change. Both scripts read saved data without importing a simulator, GPU model, or optimizer.

The three supplementary experiment classes and their analyses are complete. This does not automatically complete the full original paper experiment matrix.
