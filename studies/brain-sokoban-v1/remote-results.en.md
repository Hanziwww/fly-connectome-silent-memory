# Remote experiment summary: task acquisition, state memory, and decision mechanisms

2026-10-03. This report integrates the two authorized remote batches using the [research framing](research-framing.en.md).

## Completed scope

The original 16/16 queue entries and supplementary 19/19 entries ended. All 13 supplementary training conditions completed independent reload; 6 solved all registered training histories, while 7 stopped under the segment rules with partial success retained. There were also 6 preflight/fixed-model evaluation items. Queue counts are not independent maps or training replicates.

Both batches total 32,187 files and 17,607,025,354 bytes, retrieved and checked file by file with 0 missing and 0 SHA-256 mismatches. The original batch had 2,495 files and 571,399,382 bytes; the supplement had 29,692 files and 17,035,625,972 bytes. Transfer integrity and scientific acceptance are separate. Analysis read saved data without starting local model training.

## 1. Task behavior can be acquired; localize where learning occurs

The original D1/D3/D5/D7 representatives solved after independent reload in 13/25/37/60 moves, confirming preserved existing ability. Both new D7 controls from P05—readout-only and full-demonstration training—also solved independently. These findings concern acquisition on tested layouts and saving consistency.

For dynamic memory, a fixed P05 brain with a trained linear readout solved every registered training history on M0 and new layout S1. All 9 supplementary brain conditions changed 0 core-parameter elements; additional parameter learning occurred in the readout.

| Connectome-model condition | Trained histories solved after independent reload | Untrained delay/prefix combinations solved |
|---|---:|---:|
| Original M0: frozen brain, trained readout | 18/18 | 26/52 |
| S1: P05 brain/readout start, readout-only updates | 18/18 | 24/52 |
| S1: P05 brain, newly initialized readout | 18/18 | 32/52 |
| M0: shared visible/hidden-goal readout training | 36/36 | 55/104 |
| S2: P05 brain/readout start, readout-only updates | 29/36 | 57/104 |

**Direct support:** fixed neural dynamics provide historical features a linear readout can learn to use; readout learning can acquire solutions on all trained S1 histories. Parameter memory, core-parameter learning, and dynamic-state memory require separate interpretations.

## 2. Persistent state retains cue information useful for decisions

Paired hidden-goal M0 histories have matched current observations at decision. For 6 histories whose opposite-cue baselines both solved in the original M0 model, sham and restoration each gave 6/6, with restored scores matching the original behavior. Neural clearing gave 3/6. Opposite-cue state replacement gave 0/6 on the recipient task, with the first irreversible push matching the donor in 6/6.

A shared readout was additionally trained on visible and hidden goals from the same checkpoint. Strictly match identical histories whose baselines solve in both visibility conditions:

| Pre-decision intervention | Full hidden-goal solutions | Full visible-goal solutions |
|---|---:|---:|
| Baseline | 6/6 | 6/6 |
| Sham | 6/6 | 6/6 |
| Clear neural state | 3/6 | 6/6 |
| Restore original state | 6/6 | 6/6 |

This clearing preserves clock, buffer pointer, and RNG state. These 6 histories form a matched subset of one M0 family; complete pairings and other windows remain in the detailed analysis. Other interventions/windows may also affect visible-goal behavior, so not all state perturbations are memory-specific.

**Direct support:** within the tested histories and decision windows, model state carries cue-related information and causally affects hidden-goal decisions. Clearing, donor-state swapping, and restoration connect history, neural state, and actual behavior.

## 3. Task variation bounds the current readout and training method

Solving every trained history does not imply timing generalization. Original frozen-brain M0 scored 26/52 and S1 24/52 on untrained timings. Zero-update M0 transfer scored 36/70 on S1 and 0/140 on S2. Four static representatives each scored 0/8 on the same eight untrained layouts; there are still only eight untrained map families.

S2 reveals orientation after cue offset, requiring the same cue to be combined with current context. Both orientations form one family. This is a conditional-memory probe. The primary brain condition scored 29/36 after independent reload, so the training/readout method did not cover every registered history.

In the dense-delay scan, a separately fitted fixed linear decoder, distinct from the behavioral policy, identified cues in 18/18 trained-time conditions and 158/312 untrained-time conditions. This motivates separating within-training readability from cross-time stability; it does not prove that state contains no history information.

**Mechanistic interpretation:** the results support task-related state memory and learnable reading with limited temporal and layout scope. Time-varying representation and alignment between readout and training objectives are candidate explanations requiring direct separation.

## 4. Controls define mechanisms and task requirements

A fixed decaying trace reached 18/18 on S1 and 43/52 on untrained timings; for S2 it reached 35/36 and 78/104. The remaining S2 error was one cue in the second context at 160 ms delay and 0 common prefixes, with an incorrect first irreversible push. Simple temporal traces meet much of these probes' requirements and help bound the complexity of information maintenance they demand.

GRU reached 18/18 and 42/52 on S1, and 36/36 and 92/104 on S2. This shows acquisition in another trainable state system and supplies a reference for training difficulty and temporal generalization.

Initialization and connection-intervention results remain in the complete tables. They test prior experience and connection organization. Although all conditions used the same segment rules, actual training time differed; final performance differences are not equal-compute causal effects.

## 5. Partial success, stopping, and runtime anomalies remain distinct

The final two outcomes are below. Complete solutions and correct first irreversible pushes are saved separately; the counts happen to coincide for these conditions.

| Condition | Updates | Effective training minutes | Full solutions | Correct first irreversible push |
|---|---:|---:|---:|---:|
| S2 fixed decaying trace | 112,276 | 60.00 | 35/36 | 35/36 |
| S2 neutral rewired graph | 328 | 66.05 | 26/36 | 26/36 |

At the end of segment two, the decaying-trace model retained the required histories, but its fixed-entry margin changed by −0.190 and did not qualify for extension. The rewired condition improved its entry margin by about 3.988 but lost two required histories and stopped under the rule. Independent reload evaluates the saved best checkpoint, so it can outperform the behavior observed at stopping.

See the detailed analysis for other stop conditions and segment-level reasons. Stagnation is a registered stop status, not proof that every metric stopped improving. Segment checks follow complete behavioral evaluation, so measured time can exceed the nominal segment length. Training-loop time includes evaluation and saving, not just backpropagation; update counts across methods are not equal computation.

Both controllers finished normally, with no registered OOM, numerical error, or unexpected training-process loss. The verification interface was repaired for duplicate intervention metadata, and scheduling was reordered for conditions without data dependencies. The three original scientific training files, tasks, and training settings remained unchanged during those repairs. Retrieval changed from per-file SFTP to streamed tar, affecting transport only.

## Integrated interpretation and unanswered questions

The evidence supports this account: **training can produce task-specific behavior in a connectome model; persistent neural state supplies history information that a linear readout can use, and paired interventions show its involvement in the tested hidden-goal decisions.** Difficulty, delay, prefix, and context variation describe the scope of these abilities.

All runs use seed 10. Histories, orientations, cells, and checkpoints are not independent training replicates. Causal evidence concerns the implemented model's state and behavior. The original plan's full A→B→A parameter memory, deep cross-map activity mechanisms, and cell-group/time-window localization were not all complete at this snapshot.

The two authorized remote batches, data retrieval, and analysis described here are complete. Local training stays paused and no additional remote experiments are started. This does not complete the entire original paper plan.

## Data and figures

- [Original focused-batch progress](remote-20261003/progress.en.md) and [result description](remote-20261003/final.en.md)
- [Supplement progress](remote-supplement-20261003/progress.en.md) and [full analysis](remote-supplement-20261003/analysis.en.md)
- [Structured analysis](remote-supplement-20261003/analysis.json)
- [Learning curves](remote-supplement-20261003/learning.png), [independent reloads of all training conditions](remote-supplement-20261003/acceptance.png), and [delay/cue decoding](remote-supplement-20261003/delay-decoding.png)
- [Original-batch integrity manifest](remote-20261003/sync-verification.json) and [supplement integrity manifest](remote-supplement-20261003/sync-verification.json)

Each results directory retains result.json, verification.json, behavior, features, activity records, and checkpoints.
