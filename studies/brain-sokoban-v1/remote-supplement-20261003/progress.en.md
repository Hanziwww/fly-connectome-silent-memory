# Remote memory supplement progress

This batch is separate from the original focused remote batch; local training remains paused. Execution completion is not learning success.
Controller status: complete; 19/19 queue entries ended.

| Condition | Status | Updates/evaluated items | Effective training minutes | Independently solved training histories |
|---|---|---:|---:|---|
| supp-preflight | passed |  |  |  |
| supp-S1-ring-brain | learned_solved | 280 | 51.58 | 18/18 |
| supp-S1-ring-gru | learned_solved | 396 | 1.26 | 18/18 |
| supp-S2-context-brain | stagnation | 196 | 62.03 | 29/36 |
| supp-S2-context-gru | learned_solved | 1036 | 2.86 | 36/36 |
| supp-M0-visibility | learned_solved | 264 | 10.78 | 36/36 |
| supp-M0-causal | completed | 69 |  |  |
| supp-M0-delay | completed | 165 |  |  |
| supp-M0-visibility-causal | completed | 69 |  |  |
| supp-transfer-S1-ring | completed | 35 |  |  |
| supp-transfer-S2-context | completed | 35 |  |  |
| supp-origin-S1-ring-p05-fresh | learned_solved | 276 | 17.56 | 18/18 |
| supp-origin-S1-ring-neutral-real | stagnation | 328 | 20.17 | 17/18 |
| supp-origin-S1-ring-neutral-rewired | stagnation | 376 | 20.20 | 16/18 |
| supp-origin-S1-ring-leaky | learned_solved | 184 | 0.39 | 18/18 |
| supp-origin-S2-context-p05-fresh | stagnation | 28 | 23.31 | 22/36 |
| supp-origin-S2-context-neutral-real | stagnation | 32 | 21.97 | 19/36 |
| supp-origin-S2-context-neutral-rewired | stagnation | 328 | 66.05 | 26/36 |
| supp-origin-S2-context-leaky | stagnation | 112276 | 60.00 | 35/36 |

S1 is a new branching layout. S2 reveals orientation only after cue offset, with both orientations belonging to one map family.
This combines delayed cues with autonomous Sokoban. Revealing orientation is an experimental manipulation, not player movement.
All new training retains seed 10, one process per GPU, an 11 GiB allocation cap, and 20/40/80-minute segments.
Complete hidden-goal solving is primary; the first movement and shortest route are not acceptance gates. Untrained timings and zero-update transfer do not select models.
Neural-state clearing preserves clock, buffer pointer, and RNG; its outcomes are separate from the original batch's full-state reset.
Initialization/structure controls share a new readout initialization, with fixed branch-specific normalization calibrated only on C0/C1.
Interpretive decoders analyze data without generating actions. Map families, not cells or repeated time points, are statistical units.

## Data retrieval and analysis completed

All 29,692 supplementary files and 17,035,625,972 bytes passed file verification, with 0 missing files or SHA-256 mismatches. All 13 training conditions were independently reloaded: 6 solved every registered training history, while 7 stopped under the rules with partial solutions retained.

[Integrated interpretation](../remote-results.en.md); [detailed supplement analysis](analysis.en.md); [completion record](completion.json). Local training remains paused, and both remote batches ended.
