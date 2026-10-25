# Neural dynamical mechanisms of task acquisition and short-term memory

**All 55/55 jobs passed complete acceptance. See the [final mechanism interpretation](interpretation.en.md) for numerical findings, alternatives, and six reviewed figure sets.** This page retains the automated index; output row counts are not a substitute for evidence. See the [final integrity receipt](integrity/final-file-verification.json).

Research focus: how learning changes coordination between neural representations and behavioral readout in a fly-connectome-constrained model, and how historical information is stored, transformed, and used for decisions.

Reconstructed 103 existing runs and 47128 learning events; difficulty tasks independently solved: 24/24. Retrieved frozen replay jobs: 55/55.

## Scope supported by existing results

The 24 difficulty tasks support task acquisition on specific maps with one seed. Covarying box count, board size, and shortest paths are not independent causal factors; timings from different hardware are presented separately.
M0 clearing, swapping, and restoration support a behavioral role for historical state; S1/S2 and untrained timings bound its scope. Readout-parameter changes during frozen-brain training are not continued learning by the brain itself.
All existing stops, partial solutions, and implementation diagnostics remain in the run registry. Unsolved outcomes are not all model-capacity limits.

## Six analysis outputs

1. Acquisition: run-index, learning-events, and training-controls; 47128 events.
2. Brain and readout: 102 four-combination rows; fixed-competitor margins add exactly.
3. Whole-brain atlas: 1198704 type-summary rows; cell tables use bodyId, while parameter changes use shared types.
4. Memory geometry: 735 cue-difference/readout-subspace rows; cross-time decoders fit trained histories and evaluate held-out histories.
5. Causal interventions: 7488 retrieved paired rows; sham and restoration must pass, with T1/S1 separate from discovery.
6. Mechanistic paths: 12 signed-connection summary rows; paths propose mechanisms, while behavior requires independent intervention.

Cross-map interpretive validation: 160 learning rows and 280 memory rows. Coordinates and decoders fit only T0 and M0, respectively. Extracted 612 candidate/matched-group traces at 1 ms resolution.

All 55 jobs are retrieved and verified; reviewed versions of the 6 main figure sets are in `interpretation/figures/`. No new model training occurred.

## Mathematical interpretation and evidence limits

For actual normalized and clipped features z, center the four action weights to obtain Wc. Its row space has at most three dimensions. Projected cue-difference energy describes sensitivity of the four raw action scores; the legal-action mask additionally constrains behavior, so this is not a behavioral utilization rate. Bias enters score decomposition. The null space denotes insensitivity of current action differences, not irrelevance to later dynamics. Static margins use the final strongest legal competitor; states with only one legal action are separate, without fabricated competitor margins.
Simultaneous current/voltage summaries at 1 ms resolution support dynamics and decoding. The actual policy combines current at window time 14 ms with voltage at the window end; these are labeled separately. PCA, decoders, and low-dimensional state models do not generate actions.
Low-dimensional models split complete histories. Failed held-out predictions remain reported; stable eigenvalues or similar trajectories alone do not establish an attractor. Cells, time points, prefixes, and delay combinations do not add independent training replicates.

## Biological evidence table

| Model finding/question | Supporting data | Alternatives and assumptions | Testable prediction |
|---|---|---|---|
| Task acquisition and readout coordination | Original-start solutions and four-combination decomposition | Artificial sensory/action mapping and map-specific policy | Altering representation–readout coordination should selectively change relevant action margins |
| Cue state participates in decisions | M0 clearing, swapping, restoration, and new component interventions | Clearing can disrupt general computational state | Hidden-goal impairment should exceed matched visible-goal impairment, with restoration returning to baseline |
| Adaptation or synaptic resources maintaining history | Adaptation effects depend on task; resource reset had no effect, and resources stayed at 1 in prespecified events | 200/300 ms are model constants; adaptation also contributes to visible-goal computation | Separate general state coordination from cue-specific historical effects |
| History encoding converted into action | Readout geometry and cross-time/cross-layout decoding completed; stable general reading not established | Separate decodable information from policy use; legal-action masks affect decisions | Conditional readout directions may track action mechanisms better than total cue distance |
| Distributed group propagation | IN17A052 holding-period effects in M0/S2, but no advantage over matched groups in S1 | Type, degree, activity, and direct-observation confounds; outgoing gating does not remove direct readout | Confirm group specificity in independent layouts; no universal memory-cell group is claimed |

## Method references

- [Mechanistic hypotheses from connectome-constrained models](https://www.nature.com/articles/s41586-024-07939-3).
- [dPCA](https://arxiv.org/abs/1410.6031); [dynamic coding and stable readout](https://johndmurray.org/papers/murray_2017_pnas.pdf).
This study concerns simulated neurons and computational mechanisms, not direct evidence of Sokoban ability or memory mechanisms in real flies.
