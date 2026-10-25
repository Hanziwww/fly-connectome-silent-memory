# Task acquisition, state memory, and decision mechanisms in a connectome model: remote supplement analysis

Analysis data updated at 2026-10-03T05:05:12.693907+00:00. This report analyzes synchronized saved data only, without loading models or starting local training.

## Research questions

This study uses a fly-connectome-constrained neural dynamical model as a trainable, recordable, and intervenable experimental system to study acquisition, the role of persistent neural state in memory and decisions, and how these processes vary with task requirements. Ranking model performance is not the research objective.

This supplement asks whether linear readouts can learn to use fixed brain dynamics, whether hidden-goal decisions depend on prior neural state, and how these effects vary with layout, delay, and context. Parameter learning and dynamic-state memory are interpreted separately. See the full [research framing](../research-framing.en.md).

Structure, initialization, fixed-trace, and GRU controls distinguish prior experience, temporal retention, and connection organization. All results remain as references for mechanism interpretation.

## Experiment overview

The batch locked 19 queue entries, including 13 training conditions. Independent reload accepted all-history solutions for 6/13 training conditions.

| Condition | Training end status | Updates | Training-loop minutes | Trained histories solved | Untrained timings solved |
|---|---|---:|---:|---:|---:|
| supp-S1-ring-brain | learned_solved | 280 | 51.58 | 18/18 | 24/52 |
| supp-S1-ring-gru | learned_solved | 396 | 1.26 | 18/18 | 42/52 |
| supp-S2-context-brain | stagnation | 196 | 62.03 | 29/36 | 57/104 |
| supp-S2-context-gru | learned_solved | 1036 | 2.86 | 36/36 | 92/104 |
| supp-M0-visibility | learned_solved | 264 | 10.78 | 36/36 | 55/104 |
| supp-origin-S1-ring-p05-fresh | learned_solved | 276 | 17.56 | 18/18 | 32/52 |
| supp-origin-S1-ring-neutral-real | stagnation | 328 | 20.17 | 17/18 | 26/52 |
| supp-origin-S1-ring-neutral-rewired | stagnation | 376 | 20.20 | 16/18 | 27/52 |
| supp-origin-S1-ring-leaky | learned_solved | 184 | 0.39 | 18/18 | 43/52 |
| supp-origin-S2-context-p05-fresh | stagnation | 28 | 23.31 | 22/36 | 51/104 |
| supp-origin-S2-context-neutral-real | stagnation | 32 | 21.97 | 19/36 | 54/104 |
| supp-origin-S2-context-neutral-rewired | stagnation | 328 | 66.05 | 26/36 | 54/104 |
| supp-origin-S2-context-leaky | stagnation | 112276 | 60.00 | 35/36 | 78/104 |

## Directly supported results

- supp-S1-ring-brain: independently reloaded trained histories 18/18; untrained delay/prefix combinations 24/52.
- supp-S1-ring-gru: independently reloaded trained histories 18/18; untrained delay/prefix combinations 42/52.
- supp-S2-context-gru: independently reloaded trained histories 36/36; untrained delay/prefix combinations 92/104.
- supp-M0-visibility: independently reloaded trained histories 36/36; untrained delay/prefix combinations 55/104.
  One checkpoint handles visible and hidden goals at trained timings, enabling paired interventions. The frozen core changed 0 parameter elements.
- supp-origin-S1-ring-p05-fresh: independently reloaded trained histories 18/18; untrained delay/prefix combinations 32/52.
- supp-origin-S1-ring-leaky: independently reloaded trained histories 18/18; untrained delay/prefix combinations 43/52.

Freezing P05 and training the linear readout solved all registered histories on new layout S1-ring. This is acquisition after training on that layout; zero-update transfer is separate. New parameter learning is in the readout, showing that behavioral reading can use historical features provided by fixed neural dynamics.

A fixed decaying trace plus linear readout also solved every trained S1-ring history. Simple temporal traces therefore satisfy part of these tasks' memory demands and provide a reference for interpreting state memory in the connectome model. Identifying the neural-state components supporting behavior still requires within-model paired interventions.

## Partial solutions and stopping reasons

- supp-S2-context-brain: independent reload 29/36, status stagnation. Entry-margin change at the segment decision was +5.377. It lost 2 required histories and did not qualify for extension.
- supp-origin-S1-ring-neutral-real: independent reload 17/18, status stagnation. Entry-margin change was +1.982. It lost 1 required history and did not qualify for extension.
- supp-origin-S1-ring-neutral-rewired: independent reload 16/18, status stagnation. Entry-margin change was +2.792. It lost 1 required history and did not qualify for extension.
- supp-origin-S2-context-p05-fresh: independent reload 22/36, status stagnation. Entry-margin change was +1.313. It lost 3 required histories and did not qualify for extension.
- supp-origin-S2-context-neutral-real: independent reload 19/36, status stagnation. Entry-margin change was +0.829. It lost 4 required histories and did not qualify for extension.
- supp-origin-S2-context-neutral-rewired: independent reload 26/36, status stagnation. Entry-margin change was +3.988. It lost 2 required histories and did not qualify for extension.
- supp-origin-S2-context-leaky: independent reload 35/36, status stagnation. Entry-margin change was -0.190. Required histories were retained, but the margin gain did not reach 0.02, so extension was denied.

Stagnation is a registered segment-stop status, not complete cessation of improvement on every metric. Retention applies to trained histories of the same memory task, not old Sokoban levels. Independent acceptance uses the saved best checkpoint and may exceed the instantaneous stopping model. This does not establish saturated network capacity.

## Zero-update transfer of the original M0 model

- S1-ring: full hidden-goal solutions in 36/70 cue/delay/prefix combinations; execution status completed.
- S2-context: full hidden-goal solutions in 0/140 cue/delay/prefix combinations; execution status completed.

These denominators are paired histories within a layout, not independent maps. Tests do not select checkpoints.

### supp-M0-causal paired interventions

| Window | Visible goal | Intervention | Pairs | Baseline solved | Intervention solved |
|---|---|---|---:|---:|---:|
| cue_offset | False | current | 4 | 4 | 2 |
| cue_offset | False | neural_clear | 4 | 4 | 2 |
| cue_offset | False | restore | 4 | 4 | 4 |
| cue_offset | False | swap | 4 | 4 | 0 |
| cue_offset | False | voltage | 4 | 4 | 3 |
| decision | False | current | 10 | 7 | 3 |
| decision | False | neural_clear | 10 | 7 | 5 |
| decision | False | reset_each_window | 10 | 7 | 0 |
| decision | False | restore | 10 | 7 | 7 |
| decision | False | same_cue | 10 | 7 | 8 |
| decision | False | sham | 10 | 7 | 7 |
| decision | False | swap | 10 | 7 | 3 |
| decision | False | voltage | 10 | 7 | 4 |
| delay_mid | False | current | 4 | 4 | 2 |
| delay_mid | False | neural_clear | 4 | 4 | 2 |
| delay_mid | False | restore | 4 | 4 | 4 |
| delay_mid | False | swap | 4 | 4 | 0 |
| delay_mid | False | voltage | 4 | 4 | 3 |

The independent linear decoder in the dense-time scan identifies cues in 18/18 fit-time conditions and 158/312 untrained-time conditions. Regularization is fixed and fitting uses only original training times; the decoder neither generates policy actions nor selects models. Weak cross-time decoding does not prove absence of cue information throughout the neural network.

### supp-M0-visibility-causal paired interventions

| Window | Visible goal | Intervention | Pairs | Baseline solved | Intervention solved |
|---|---|---|---:|---:|---:|
| cue_offset | False | current | 4 | 4 | 2 |
| cue_offset | False | neural_clear | 4 | 4 | 2 |
| cue_offset | False | restore | 4 | 4 | 4 |
| cue_offset | False | swap | 4 | 4 | 0 |
| cue_offset | False | voltage | 4 | 4 | 2 |
| cue_offset | True | current | 4 | 4 | 2 |
| cue_offset | True | neural_clear | 4 | 4 | 2 |
| cue_offset | True | restore | 4 | 4 | 4 |
| cue_offset | True | swap | 4 | 4 | 0 |
| cue_offset | True | voltage | 4 | 4 | 2 |
| decision | False | current | 10 | 8 | 2 |
| decision | False | neural_clear | 10 | 8 | 5 |
| decision | False | reset_each_window | 10 | 8 | 0 |
| decision | False | restore | 10 | 8 | 8 |
| decision | False | same_cue | 10 | 8 | 8 |
| decision | False | sham | 10 | 8 | 8 |
| decision | False | swap | 10 | 8 | 2 |
| decision | False | voltage | 10 | 8 | 4 |
| decision | True | current | 10 | 9 | 5 |
| decision | True | neural_clear | 10 | 9 | 10 |
| decision | True | reset_each_window | 10 | 9 | 0 |
| decision | True | restore | 10 | 9 | 9 |
| decision | True | same_cue | 10 | 9 | 5 |
| decision | True | sham | 10 | 9 | 9 |
| decision | True | swap | 10 | 9 | 1 |
| decision | True | voltage | 10 | 9 | 7 |
| delay_mid | False | current | 4 | 4 | 2 |
| delay_mid | False | neural_clear | 4 | 4 | 2 |
| delay_mid | False | restore | 4 | 4 | 4 |
| delay_mid | False | swap | 4 | 4 | 0 |
| delay_mid | False | voltage | 4 | 4 | 3 |
| delay_mid | True | current | 4 | 4 | 3 |
| delay_mid | True | neural_clear | 4 | 4 | 2 |
| delay_mid | True | restore | 4 | 4 | 4 |
| delay_mid | True | swap | 4 | 4 | 0 |
| delay_mid | True | voltage | 4 | 4 | 1 |

After strictly matching timings where both cues solve in both visibility baselines, pre-decision clearing changes hidden-goal solutions from 6/6 to 3/6 while visible-goal solutions remain 6/6 to 6/6. This supports greater dependence on prior neural state for hidden-goal behavior at these timings, without extrapolation to other maps or seeds.

## Interpretation and limits

- Report GRU and brain conditions by layout; GRU success does not substitute for acceptance of a brain model, and success on one layout does not establish another.
- Trained-history solutions and untrained timing performance are distinct. Timing combinations, cues, orientations, and repeated actions are not independent maps or training seeds.
- S2 reveals orientation after a shared cue and tests conditional cue use. This manipulation is not ordinary static-board play and alone does not establish abstract goal understanding.
- Retain all visibility-paired intervention results while separately showing the subset where both baseline cues solve. Use an unintervened baseline for the decision window and same-window shams for other windows.
- Neural clearing that preserves clock, buffer pointer, and RNG is distinct from the earlier full-state reset. The verification adapter handles only duplicate metadata, not training, forward computation, or acceptance rules.
- Dense scans distinguish cue decodability from behavioral use and do not establish cross-map generalization. Equal-duration prefixes can still differ in input history.
- training_seconds measures the training loop, including behavioral evaluation and saving, not pure GPU backpropagation. Different update costs mean update counts are not equal computation.
- Initialization controls address prior experience; rewiring tests sensitivity to connection organization; traces and GRU characterize temporal processing demands. All serve learning/memory interpretation. With different actual training times, final performance differences are not equal-compute causal effects.
- This is a remote supplementary batch, not completion of the entire original paper plan; the local STOP remains.

## Figures and provenance

Grouped results are in [analysis.json](analysis.json); source files and SHA-256 are in [analysis-inputs.json](analysis-inputs.json).

![learning.png](learning.png)

![acceptance.png](acceptance.png)

![delay-decoding.png](delay-decoding.png)
