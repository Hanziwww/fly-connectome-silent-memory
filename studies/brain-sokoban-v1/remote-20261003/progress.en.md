# Focused remote experiment progress

Local training is paused. The table records remote checkpoints, experiments, and independent reloads.

| Task | Status | Updates | Effective training minutes | Independently reload-solved |
|---|---|---:|---:|---|
| import-D1 | passed |  |  |  |
| import-D3 | passed |  |  |  |
| import-D5 | passed |  |  |  |
| import-D7 | passed |  |  |  |
| generalization-D1 | completed |  |  |  |
| generalization-D3 | completed |  |  |  |
| generalization-D5 | completed |  |  |  |
| generalization-D7 | completed |  |  |  |
| remote-D7-head_only | learned_solved | 868 | 1.03 | True |
| remote-D7-demonstration | learned_solved | 504 | 12.97 | True |
| memory-history-preflight | passed |  |  |  |
| remote-dynamic-M0-gru | learned_solved | 468 | 0.79 | True |
| remote-dynamic-M0-carry_untrained | untrained_baseline | 0 | 1.01 | False |
| remote-dynamic-M0-reset | stagnation | 6304 | 140.19 | False |
| remote-dynamic-M0-head_only | learned_solved | 272 | 7.14 | True |
| remote-dynamic-M0-joint | stagnation | 380 | 20.28 | False |

## Untrained layouts

| Source checkpoint | Solved | Layouts |
|---|---:|---:|
| generalization-D1 | 0 | 8 |
| generalization-D3 | 0 | 8 |
| generalization-D5 | 0 | 8 |
| generalization-D7 | 0 | 8 |

Four representative models underwent zero-update tests on the same eight layouts. Four times eight evaluations are not thirty-two independent maps or training replicates.

Dynamic memory covers only the prespecified M0 pilot layout. Execution and verification do not imply learning success; stagnation, budget caps, and implementation errors remain separate.
Independent conditions run across GPUs, at most one experiment process per card, with seed 10, an 11 GiB allocation cap, and 20/40/80-minute segments.
Remote/local hardware and PyTorch/CUDA versions differ, so their timings are not a hardware-matched paired speedup.
