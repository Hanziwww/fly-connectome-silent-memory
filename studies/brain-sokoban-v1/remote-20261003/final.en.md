# Completion record for the original focused remote batch

All 16/16 original queue entries ended. They include acceptance, zero-update evaluation, and training controls, not 16 learning successes. The supplement also ended and was retrieved; see the combined [remote experiment summary](../remote-results.en.md). The original paper plan is not fully complete.

D1/D3/D5/D7 representatives solved after independent reload in 13/25/37/60 moves. All four scored 0/8 on the same eight untrained layouts. The statistical units remain eight map families, not 32 independent training replicates.

The new D7 head_only and demonstration controls both solved after independent reload: 868 updates and 61.77 seconds, and 504 updates and 778.25 seconds, respectively. These times come from the same remote hardware and include each method's actual workflow; they are not a direct hardware-speedup ratio against local runs.

| M0 condition | End status | Updates | Training-loop minutes | Trained-timing solutions | Untrained-timing solutions |
|---|---|---:|---:|---:|---:|
| carry_untrained | untrained_baseline | 0 | 0 (evaluation only) | 10/18 | 24/52 |
| reset | stagnation | 6304 | 140.19 | 9/18 | 26/52 |
| head_only | learned_solved | 272 | 7.14 | 18/18 | 26/52 |
| joint | stagnation | 380 | 20.28 | 15/18 | 16/52 |
| gru | learned_solved | 468 | 0.79 | 18/18 | 42/52 |

Freezing the trained P05 brain and updating only its linear readout learned every registered M0 training history. The reset-state condition remained at 9/18 after about 140 minutes and 6,304 updates, whereas carried-state readout-only training reached 18/18. This supports a role for dynamic neural state in the tested task. Joint training stagnated at 15/18; that does not establish a capacity boundary.

M0 head_only scored 26/52 on untrained timing combinations versus 42/52 for GRU, showing limited timing generalization of this positive connectome-model result. Original swaps and restoration support a state effect on behavior; the supplement adds clock-preserving clearing, visible-goal controls, and new-layout replication.

Synchronization checked 2495 files and 571399382 bytes, with 0 missing files and 0 SHA-256 mismatches. Transfer integrity and scientific acceptance are separate records.

See memory-analysis.json. Each run retains result.json, verification.json, trajectories, features, and checkpoints. The complete transfer manifest is sync-verification.json. Local training remains paused.
