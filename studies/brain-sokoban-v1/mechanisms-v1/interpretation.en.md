# Task acquisition, state memory, and decision reading: frozen-checkpoint mechanism results

This report covers the locked 55 jobs and 39 existing models in `mechanisms-v1`. All jobs ran, were fully retrieved, and passed acceptance, without new model training. The central account is: **learning establishes task-specific representation–readout coordination; history retained in neural state affects later choice, but its use depends strongly on timing, layout, and context.** Superiority of the real connectome is not the objective.

Three levels of conclusion are supported. First, jointly trained brain/readout pairs are strongly coordinated, while an existing brain can also acquire representative-task behavior through linear-readout-only training. Second, full neural state and components including voltage and adaptation can causally change hidden-goal choice beyond one layout. Third, single-cell cue responses, structural connections, and short-term prediction alone do not establish a specific memory-carrying cell group or a stable memory attractor.

## 1. Data and acceptance scope

| Item | Verified result |
|---|---|
| Remote jobs | 55/55 ended normally; controller and all workers exited, with four GPUs idle |
| Existing-model replay | 39/39 matched their original complete/partial solution acceptance |
| Parameter freezing | 121 stage/job freeze checks passed; no optimizer updates |
| Retrieval integrity | 5,271 manifest files, 55,451,602,323 bytes, 51.6433 GiB; complete receipts for 55/55 jobs |
| Final file verification | SHA-256 recomputed per file; no missing files, size differences, or hash differences |
| Cell index | 167,106 unique bodyIds, 11,770 shared parameter groups; 3,200 input and 4,096 observed cells, with no overlap |
| Derived analyses | 6 reviewed main figure sets in PNG/PDF; 612 candidate/matched-group traces at 1 ms resolution; cell-level Parquet and companion CSV/JSON |

See [final file verification](integrity/final-file-verification.json), [model/index checks](interpretation/integrity.json), and the [locked protocol](protocol.en.md). Training still uses only seed 10. Map families are the units of interpretation and comparison; histories, neurons, time points, and checkpoints do not add independent training replicates.

A second synchronization had converted 92 previously retrieved `progress.json`/`result.json` files to Windows line endings. After establishing that removing only the line-ending differences exactly recovered the remote SHA-256 values, the original bytes were restored and all files reverified. The earlier byte copies and [repair record](integrity/metadata-line-endings-repair.json) remain. This was a metadata transport-format issue, not a change to models, numerical results, or neural trajectories.

## 2. Task acquisition depends on the training path

All 24 existing difficulty tasks started as branches from P05 and solved after independent reload. They cover 1–3 boxes, 6×6 to 10×10 boards, and solving certificates of 9–60 moves. The 24/24 training result predates this round; the new work is its mechanism replay.

| Difficulty | Maps solved | Median updates | Median effective local training minutes | Minimum–maximum training minutes |
|---|---:|---:|---:|---:|
| D0 | 3/3 | 100 | 1.49 | 1.45–1.54 |
| D1 | 3/3 | 92 | 1.39 | 1.36–1.42 |
| D2 | 3/3 | 232 | 3.88 | 3.03–14.45 |
| D3 | 3/3 | 360 | 7.19 | 7.12–7.30 |
| D4 | 3/3 | 392 | 7.78 | 7.46–9.73 |
| D5 | 3/3 | 612 | 12.26 | 8.95–13.50 |
| D6 | 3/3 | 1,044 | 28.50 | 20.95–38.26 |
| D7 | 3/3 | 1,556 | 51.59 | 47.37–76.58 |

Higher tiers generally required more updates and time, but D0/D1 were not strictly monotonic and D2 varied substantially across maps. Board size, box count, route length, and deadlock risk covary, so these 24 maps cannot identify their separate causal effects.

Representative-task controls clarify the training bottleneck:

| Method | D1-T0 | D3-T0 | D5-T0 | D7-T0 |
|---|---|---|---|---|
| Joint training | Solved, 84 updates | Solved, 360 updates | Solved, 440 updates | Solved, 1,924 updates |
| Brain-only training | Solved, 380 updates | Segment cap after 7,020 updates; unsolved | Segment cap after 6,292 updates; unsolved | Not run in this set |
| Readout-only training | Solved, 60 updates | Solved, 240 updates | Solved, 356 updates | Solved remotely, 868 updates |
| No recovery loss | Stagnated after 4,344 updates | Stagnated after 4,604 updates | Stagnated after 4,464 updates | Not run in this set |
| Full-demonstration training | Solved, 24 updates | Solved, 72 updates | Solved, 88 updates | Solved remotely, 504 updates |

Stagnation and segment caps here are learning outcomes, not implementation errors or direct evidence of exhausted brain capacity. Brain-only training still improved average margins on fixed D3/D5 states without assembling a complete autonomous trajectory. Removing recovery loss failed on all three representative tasks. This is consistent with local action learning being insufficient for sequential behavior and with the importance of recovery signals and state coverage, but the controls do not fully separate those mechanisms.

On the same machine, D1/D3/D5 readout-only training took 4.68, 22.91, and 25.19 seconds, versus 83.56, 437.74, and 536.94 seconds for joint training. Frozen features can be reused and there is no brain backpropagation, so workloads differ. These are actual costs, not equal-simulation algorithmic efficiencies or GPU speedup factors. Remote D7 timing is not a hardware-matched comparison with local joint training.

In saved zero-update tests, the D1/D3/D5/D7 representatives each scored 0/8 on the same 8 untrained layouts. This supports task-specific acquisition, not general Sokoban planning. Four sets of 0/8 are not 32 independent maps. See the [existing remote results](../remote-20261003/progress.en.md).

![Acquisition](interpretation/figures/01-acquisition.png)

Data: [difficulty summary](interpretation/difficulty-summary.csv), [training controls](tables/training-controls.csv), and [learning events](tables/learning-events.csv).

## 3. Brain and readout: coordination does not imply universally necessary updates

At the same original task starts, combine pre/post-training brains B and complete readouts H, retaining normalization, clipping, and bias:

| Parameter combination | Complete solutions on 24 difficulty tasks |
|---|---:|
| Old brain + old readout | 0/24 |
| New brain + old readout | 0/24 |
| Old brain + new readout | 1/24, D0-T1 only |
| New brain + new readout | 24/24 |

The clearest acquisition-mechanism result is: **within parameter solutions selected by joint training, brain representations and readouts are not freely interchangeable.** This is paired substitution evidence within one lineage. It does not mean brain updates are the only route to learning, because a separate readout-only path succeeded on D1/D3/D5/D7 representatives.

The four-combination score decomposition is:

\[
\Delta q_{\rm brain}=q_{10}-q_{00},\quad
\Delta q_{\rm head}=q_{01}-q_{00},\quad
\Delta q_{\rm interaction}=q_{11}-q_{10}-q_{01}+q_{00}.
\]

The three terms sum exactly to \(q_{11}-q_{00}\), with maximum reconstruction error \(9.54\times10^{-7}\) across 34 static models. At each reference state, the competitor is fixed to the final model's highest-scoring other legal action, not reselected for each combination. States with no other legal action are separate.

Equal-map means over the 24 difficulty maps were: brain **+0.0275**, readout **+0.7575**, interaction **+0.0555** for the fixed reference-action margin. The readout term was positive in 24/24 maps; brain terms were 13 positive and 11 negative, while interactions were 16 positive and 8 negative. These describe one specified score contrast, not percentage contributions to complete behavior. D1-T1 even had total reference-margin change −0.0484 while still solving; alternative solvable actions or detours satisfy acceptance.

Complete behavior can depend on coordination while the mean margin increase is dominated by the readout term. A readout may redirect only modestly changed representations toward new actions, and boundary crossings in a few critical states may alter an entire trajectory.

![Brain/readout decomposition](interpretation/figures/02-brain-readout.png)

Data: [four-combination behavior](interpretation/four-combination-behavior.csv), [per-model decomposition](interpretation/decomposition-by-model.csv), and [shared-parameter changes](tables/parameter-changes.csv).

## 4. Whole-brain functional atlas: sparse spikes and widespread subthreshold differences

In the final stages of the 24 difficulty models, the 40 ms windows at fixed reference states contained recorded spikes in a mean **4,923/167,106 cells**, about 2.95%, with a per-map range of 4,402–5,342. Mean per-cell spike probability was 0.000779 per millisecond, equivalent to about 0.779 Hz over those windows, not a long-term firing rate during continuous natural behavior.

Separated by prespecified cell sets:

| Cell set | Cells | Mean cells with spikes per map | Mean absolute pre/post voltage change at matched states |
|---|---:|---:|---:|
| Direct input | 3,200 | 797 | 0.0106 mV |
| Readout observed | 4,096 | 941 | 0.0524 mV |
| Other | 159,810 | 3,185 | 0.00857 mV |

Observed cells changed more, but this depends on the artificial input/observation mapping and training objective; it does not identify a naturally superior Sokoban brain region. Nor does the overall mean absolute change of about 0.00968 mV imply an ineffective brain: full-behavior substitutions already show coordination dependence.

For the prespecified M0 readout-only event—160 ms delay, 4 prefix windows, hidden goal—the opposite-cue difference in mean voltage over the final reference decision window exceeded \(10^{-6}\) mV in **119,503 cells**. Adaptation differences exceeded the same numerical threshold in **6,379**, while resource differences were zero. Mean absolute cue-related voltage difference was 0.423 mV. This describes numerical coverage, not 119,503 independent significant findings or causal participation by every cell. Sparse spikes can coexist with widespread subthreshold differences.

Cell tables retain bodyId, parameter group, input/observation membership, and state differences. Shared-parameter changes are counted across 11,770 groups; copying a parameter onto cells does not create independent learning events. See the [annotation/function summary](interpretation/annotation-functional-summary.csv) for descriptive type/transmitter summaries.

Annotation limitations remain: 2,605 cells lack a type and 516 lack superclass; **145,304 lack somaNeuromere**, about 87%. Transmitter fields have no empty values but include 2,361 `unknown` entries. These data support cell/type analyses and descriptions of annotated subsets, not reliable whole-brain regional localization. Soma annotations are also not synaptic projection assignments.

![Whole-brain atlas](interpretation/figures/03-neuron-atlas.png)

Data: [cell-weighted atlas](interpretation/neuron-weighted-atlas.csv), [memory-cell summary](interpretation/memory-cell-summary.csv), and [annotation coverage](tables/annotation-coverage.csv). `atlas/` stores per-cell tables at each stage for 34 static models. `interpretation/*-memory-cells-d160-p4.parquet` contains events for the five key memory models. Type-weighted curves in the automated atlas are kept separate from this report's cell-weighted statistics.

## 5. Memory geometry: historical differences do not establish reading across time or layout

Complete hidden-goal behavior is reproduced below. Denominators count cue, delay, prefix, and context combinations within a family.

| Model/goal condition | Trained timings | Untrained timings |
|---|---:|---:|
| M0 frozen brain, readout-only | 18/18 | 26/52 |
| M0 jointly trained | 15/18 | 16/52 |
| M0 shared visible/hidden readout: hidden | 18/18 | 27/52 |
| Same model: visible goal | 18/18 | 28/52 |
| S1 frozen brain, readout-only | 18/18 | 24/52 |
| S2 context model | 29/36 | 57/104 |

Partial M0 joint and S2 solutions are reproducible learning outcomes. Both S2 orientations belong to one family, with orientation revealed after the cue for conditional choice. Untrained timing performance is often near half and is not monotonic in delay: S1, for example, solved 1/10 at 320 ms but 8/10 at 640 ms. This does not support one monotonic memory-lifetime threshold; it is more compatible with time-dependent coordination among encoding, readout direction, and current dynamic state.

For the four frozen-brain models, the maximum pre/post difference in actual 512-dimensional features under identical complete histories was **0**; for M0 joint it was 49.148 in raw mixed-feature units. Learning in the former changed readout use rather than creating new brain dynamics. Input-history differences caused by altered free trajectories are recorded separately and are not mixed into the matched-history comparison.

The actual policy uses normalized and clipped \(z\). Define

\[
W_c=(I_4-\tfrac14\mathbf1\mathbf1^\top)W.
\]

All these centered readouts have rank 3. On average, only about **0.46%–0.99%** of squared opposite-cue feature distance lies in the final hidden-goal readout row space; the remaining directions do not directly change the four current relative scores. This is not an information-utilization rate: it depends on metric, normalization, and current W, while actions also depend on legal masking. The isotropic three-dimensional/512-dimensional geometric reference is already about 0.586%, so a small ratio alone is not evidence of wasted information.

For S2 the ratio fell from 0.770% to 0.460%, and the mean centered-logit cue contrast fell from 3.923 to 2.655, while trained-timing behavior still reached 29/36. Amplifying total cue difference is not necessarily the learning objective; context-appropriate reading can matter more than total magnitude. This interpretation is consistent with the data, but is not a complete causal mediation decomposition of cue→context→action.

Interpretive decoding further limits the conclusions. A cue decoder fitted only on M0 trained histories scored 18/18 on those histories and 26/52 on M0 untrained histories; transfer to S1 yielded 8/18 on corresponding trained timings and 27/52 on untrained timings, and S2 yielded 19/36 and 53/104. Stable, general cross-layout cue reading was not obtained. Cross-time decoding of simultaneous millisecond g/v summaries had mean same-time held-history accuracy of only **50.4%–54.2%**. This is a different representation from the policy's 14/40 ms mixed features.

In regularized low-rank marginal decomposition, time was more predictable than cue. For M0 readout-only, held-out time reconstruction R² was 0.350, versus 0.0186 for cue×time. Cue×time reconstruction for M0 joint and S1 was −0.245 and −0.254. Failed interpretive models remain reported; a low-dimensional trajectory plot is not evidence of stable memory coding.

Static cross-map decoding likewise showed no reliable general neural-coding advantage. Fitting T0 and testing T1, post-training neural push-action accuracy averaged equally across maps was 49.0%, below the training-label majority baseline of 79.3%. Direct board features achieved 64.5%, also below that baseline. For solvable-direction decoding, board features reached 73.7%–86.1% and neural features 59.7%–66.7%. These results concern fixed interpretive models, do not prove information absent, and do not generate policy actions.

![Memory geometry](interpretation/figures/04-memory-geometry.png)

Data: [behavior and delay](interpretation/memory-delay-summary.csv), [geometry](interpretation/geometry-summary.csv), [cross-family memory](interpretation/cross-family-memory-summary.csv), [cross-family static decoding](interpretation/cross-family-learning-summary.csv), and [low-rank decomposition](tables/demixed-reconstruction.json).

## 6. Causal interventions: history affects choice, with conditional component and group effects

These are the six prespecified timings: delays 40/160/640 ms × prefixes 0/4. M0/S1 each have two cues; S2 has two cues×two orientations. Conditions pair the same histories. State interventions preserve clock, ring-buffer pointer, and RNG; sham, same-cue swap, and restoration checks passed.

### 6.1 Full state and components

Complete-solution counts for interventions before the first autonomous decision:

| Condition | M0 readout-only hidden | M0 shared readout hidden | Same model visible | S1 hidden | S2 hidden |
|---|---:|---:|---:|---:|---:|
| Baseline | 9/12 | 9/12 | 12/12 | 11/12 | 15/24 |
| Clear voltage | 6/12 | 6/12 | 7/12 | 6/12 | 12/24 |
| Clear current | 7/12 | 6/12 | 9/12 | 8/12 | 12/24 |
| Clear adaptation | 8/12 | 8/12 | 7/12 | 4/12 | 10/24 |
| Reset resources to 1 | 9/12 | 9/12 | 12/12 | 11/12 | 15/24 |
| Clear delay buffer | 9/12 | 9/12 | 10/12 | 8/12 | 12/24 |
| Clear all neural state | 6/12 | 6/12 | 12/12 | 6/12 | 12/24 |
| Swap in full opposite-cue state | 3/12 | 3/12 | 1/12 | 1/12 | 9/24 |

The matched visible-goal control gives relatively clear memory evidence: clearing all state before reading reduces hidden-goal solutions from 9/12 to 6/12, while the same model remains at 12/12 with a visible goal. Together with swaps and restoration, this supports a role for history in current neural state in hidden-goal decisions.

Not every impairment after a state intervention is a memory-specific impairment. Clearing adaptation reduces visible-goal behavior from 12/12 to 7/12, and full-state swaps reduce it to 1/12. State may support general sensorimotor computation, or swapping may create conflict between history and the current goal. Component clearing and full clearing are not additive and may disrupt coordination between components.

Windows also matter. Clearing adaptation after writing/at hold midpoint/before reading yields 7, 7, and 8/12 for M0 readout-only, versus 7, 4, and 4/12 for S1. Dependence on adaptation varies across tasks; the configured 200 ms constant alone does not establish a universal memory timer.

S2 net effects conceal opposite individual changes: clearing all state before reading causes **8 originally successful histories to fail and 5 failed histories to succeed**, a net loss of 3. Adaptation clearing loses 10 and gains 5, net loss 5. The [per-history pairs](interpretation/memory-intervention-pairs.csv) and [loss/gain decomposition](interpretation/lost-and-gained.csv) are therefore retained instead of reporting only aggregate means.

Synaptic resources are an explicitly unsupported mechanism in these tests. Resources never departed from 1 in static reference statistics or the prespecified 160 ms/4-prefix memory events; all resource clearing/swapping conditions above had no behavioral or first-margin effect. Although 2,952 cells have nonzero short-term-depression settings, these inputs and events did not show resource depletion contributing to memory. A configured 300 ms recovery constant is not an experimental discovery of 300 ms synaptic memory.

Across all 490 free-memory evaluations and 7,272 paired memory interventions, correct first irreversible pushes happened to coincide with complete solutions on these short probes. Both measures remain separate. The first movement and its margin are different measures and cannot replace the first push. Margin analyses fix the baseline first action and its strongest legal competitor to describe preference perturbations; they do not assume a unique correct first move.

### 6.2 Candidate groups and matched controls

Candidates were selected strictly from discovery data, without reselection based on confirmation effects. Learning groups use T0 discovery and T1 confirmation; memory groups use M0 discovery, S1 confirmation, and S2 contextual-scope testing.

| Line | Candidate type labels | Noninput cells | Observed members | Main result |
|---|---|---:|---:|---|
| Learning | IN06B056 | 9 | 0 | No solution impairment observed across the four T1 maps |
| Learning | IN19A016 | 12 | 0 | No candidate solution loss; some matched-group read-window interventions impaired solutions |
| Learning | AVLP297 | 8 | 0 | Static 14–26 ms suppression affected D1/D3, without confirmation in D5/D7 |
| Memory | MNad18,MNad27 | 8 | 4 | No behavioral effect; first-margin changes only near numerical tolerance |
| Memory | SLP324 | 10 | 0 | No behavioral effect; no actual current into the observed group measured in the representative event |
| Memory | IN17A052 | 12 | 0 | Holding-period effects depend on layout; S1 impairment did not exceed the matched group |

For AVLP297, mid-window suppression at 50%/100% lost solutions in D1-T1, and only 100% did so in D3-T1. However, its matched group also failed in D3 at read-window strengths 25%/50%/100%. The IN19A016 matched group impaired D1 and D7 during reading. A high learning-response rank alone does not establish a type-specific task mechanism.

For memory group IN17A052, 100% outgoing suppression throughout holding reduced M0 readout-only from 9/12 to 6/12 while its matched group stayed at 9/12. Both candidate and matched groups reduced S1 from 11/12 to 10/12. In S2 the candidate fell from 15/24 to 10/24 while the matched group stayed at 15/24. This supports a behavioral role under some model conditions, but the prespecified S1 confirmation did not establish a general group-specific memory effect.

Matching preserved cell count and observed-set membership count, while approximating baseline activity and log(1+degree). Mean standardized matching distances were 0.0038, 0.0142, and 0.0098 for learning groups, and 0.3077, 0, and 0.0012 for memory groups. The first memory match was relatively weak; signs and specific pathway positions were not additionally matched. Matching did not eliminate all confounds.

The interrupted channel also matters. Outgoing suppression acts on network propagation without clearing candidate cells' own g/v or removing their observed features from the policy. MNad18/MNad27 includes 4 observed cells; a null outgoing effect when outgoing paths are zero cannot exclude direct-readout participation. This tests necessity of outgoing propagation, not every possible cellular contribution.

![Causal interventions](interpretation/figures/05-causal-interventions.png)

The left panel shows net solution differences divided by paired-history counts. The right panel fixes full holding-period suppression rather than selecting the strongest dose. All doses and windows are in the [memory intervention summary](interpretation/memory-intervention-summary.csv) and [static intervention table](interpretation/static-intervention-summary.csv).

## 7. Dynamics and pathways: short-range explanation without an established long-term stable state

The eight-dimensional affine state model fits trained histories only and predicts held-out complete histories. One-step (1 ms) R² is **0.954–0.957**, above the persistence baseline **0.942–0.947**, but the gain is only about 0.010–0.012. Strong temporal continuity explains most of the apparently high prediction scores.

Recursing the same model without reinjecting observed intermediate states gives:

| Prediction interval | Held-out R² range |
|---|---:|
| 1 ms | 0.954–0.957 |
| 20 ms | 0.380–0.455 |
| 40 ms | −0.068 to −0.047 |
| 80 ms | −0.365 to −0.264 |

The model describes local propagation but does not accurately predict state one complete decision window later. It does not explicitly model time-varying external input, so longer-horizon failure may reflect input switching and nonlinearity together. Eigenvalue-derived decay of 8.3–64.4 ms and maximum singular amplification of 1.55–2.25 at 20 ms describe this coordinate system and affine approximation, not biological time constants, memory lifetime, or an attractor.

Structural analysis uses the effective directed operator and actual parameters, retains signs, and reports one- and two-hop mass separately; the two are not quantities with identical units to stack together. MNad18/MNad27 has zero one- and two-hop mass to observed cells, structurally explaining its null outgoing effect. SLP324 has indirect paths but no actual observed-group current in the representative event; a connection need not be used by this task.

The sparse-propagation recorder measured resource- and delay-adjusted outgoing signals over a 400 ms M0 history with 160 ms delay and 4 prefixes. Recorder-on/off outputs matched exactly. IN17A052 produced nonzero direct current into observed cells; its matched group did too, with greater total absolute aggregated current. These are model-current records from two cue histories and one event, not biological pathway strengths in real flies, and they cannot replace S1/S2 behavioral interventions.

![Pathways and dynamics](interpretation/figures/06-pathways-dynamics.png)

Data: [signed structural paths](tables/signed-pathways.csv), [actual direct propagation](interpretation/observed-direct-flux.csv), and [persistence and multistep prediction](interpretation/dynamics-persistence-check.csv).

## 8. Evidence chain, alternatives, and testable predictions

| Model finding | Current evidence and scope | Alternatives/model assumptions | Testable prediction |
|---|---|---|---|
| Joint acquisition establishes representation–readout coordination | Four-combination swaps on 24 maps; intact pairs solve 24/24, mixed pairs 0/24 and 1/24 | Artificial input/action mappings and P05 history; brain updates are not universally necessary | Preserving or compensating for learned coordinate changes should reduce substitution impairment |
| Historical neural state participates in hidden-goal choice | M0/S1/S2 clearing, swaps, and restoration; visible/hidden full-clear pairing before readout | Full state also supports general computation; conflicting states can severely impair visible tasks | At fixed current observation, manipulating cue-related directions while preserving general state should more selectively redirect choices |
| Voltage and adaptation contribute to some tasks | Component/window interventions produce different impairments across layouts | Parameter settings, nonlinear coordination, and general task performance can contribute | Stage-dependent effects should correspond to subsequent propagation toward readout-sensitive directions |
| Sparse spikes coexist with widespread voltage differences | Full cell index and prespecified 1 ms event/window data | Numerical differences are not causal effects; common drive can induce widespread associations | Preserving subthreshold state while changing outgoing signals should differ temporally from directly clearing voltage |
| Cue-response rankings do not locate a specific memory group by themselves | Two candidates had no behavioral effect; IN17A052 matched its control effect in S1 | Small-group redundancy, silent outputs, context-dependent function, and incomplete matching | Selection considering both cue encoding and actual propagation to readout may yield different confirmation results |
| A fixed low-dimensional model predicts only short ranges | High 1 ms R²; negative recursive 40/80 ms R² | No explicit input switching; linear approximation | Fitting within matched input phases and using input drivers may improve longer-horizon prediction |

These predictions organize biological inspiration; they are not additional completed experiments. Only testable computational hypotheses about real fly neurons are warranted. Sokoban sensory mapping, action readout, and loss are artificial constructs.

## 9. Unsupported claims and retained failures

- The data do not show that the real graph outperforms every alternative structure; that is not the research objective.
- Joint updates are not established as the only route to acquisition; successful readout-only controls directly limit that claim.
- High cue selectivity does not identify necessary memory neurons; candidate and matched interventions provide counterexamples.
- Short-term synaptic-resource memory is not established as the basis of this task; relevant resource dynamics and intervention effects were not observed.
- Stable memory coding across layouts/timings and a memory attractor are not established; decoding transfer and long-range state prediction were limited.
- Results are limited to existing seed 10 models and tested map families/timings. Cells and repeated histories are not used to manufacture independent training replicates or significance.

This round completed frozen-checkpoint mechanism recording, paired interventions, and analysis, not every training branch in the original paper plan.

## 10. Files and review entry points

- [Six main figure sets](interpretation/figures/): paired PNG/PDF files, with values from CSVs in the parent directory.
- [Machine-readable summary](analysis.json), [automated analysis index](analysis.en.md), and [execution progress](progress.en.md).
- [Paired-result and cell-summary script](interpretation/review_results.py), [additional dynamics/cell checks](interpretation/dynamics_and_cells.py), and [plotting script](interpretation/figures.py). They read saved data without loading a GPU policy or training models.
- [Final file-verification script](interpretation/verify_final_files.py) and its [result](integrity/final-file-verification.json).
- `runs/` contains retrieved raw mechanism results; `tables/`, `atlas/`, and `group-traces/` contain protocol analyses; `interpretation/` contains this review and interpretation. Original experiment outcomes and model parameters are retained.

This report develops concrete findings and failures. Automated output row counts describe scale, not evidential strength.
