# Experiment history

This record summarizes all retained method families through 2026-10-02. The [complete index](../archive/INDEX.md) includes every run folder, preflight, diagnostic, benchmark and viewer record; the [machine-readable catalog](../archive/experiment-catalog.json) enumerates trial receipts and checkpoints. Original records remain unchanged, including unsuccessful and interrupted attempts.

## Method progression

| Stage | Question and method | Recorded result | Evidence |
|---|---|---|---|
| Four-level pilot | Internal plasticity with a frozen, previously calibrated readout; retention gates | All four tasks solved. Two were already solvable; C001/C003 were acquired through 19 accepted internal updates | [Campaign](../archive/experiments/campaign/run/ledger.json), [verification](../archive/experiments/campaign/run/verification.json) |
| Independent difficulty screen | Start each task/seed from the same pilot core | B00/B01/B02 each exhausted a 600 s budget; B05 was interrupted when training was stopped; 14 trials remained unstarted | [Summary](../archive/experiments/boundary/20261001-screen01/summary.json), [pause record](../archive/experiments/boundary/20261001-screen01/pause-status.json) |
| Learning diagnosis | Inspect gradient access, true forward changes and state interference | Gradients reached all parameter families, but measured directions could worsen true loss and correcting one decision could break others; no capacity boundary established | [Diagnosis](../archive/experiments/diagnostics/20261001-learning01/diagnosis-summary.json) |
| B00 repair A/B/C/D | Cross legacy/block proposals with legacy/behavioral selection | D solved the 11-move task for seeds 10, 11 and 12; all 12 formal trial records and four preflight arms retained | [Comparison](../archive/experiments/repair/20261001-b00-screen01/summary.json), [interventions](../archive/experiments/repair/20261001-b00-intervention01/verification.json) |
| Continuous difficulty ladder | Seed 10, inherit the D solution and grow board/boxes/moves | P00 already solved in 11 moves; P01 exhausted 600 s and deadlocked after 3 moves. P02-P07 were then unattempted | [Report](../archive/experiments/progressive/20261001-single-seed01/report.md) |
| Outcome-first + GPU population | Accept any solution; task-local counterexamples; batch candidate cores | P01 still repeated after 3 moves in 600 s. Paired candidate evaluation became 3.86-7.94 times faster; speed records preserved separately | [Report](../archive/experiments/outcome/20261001-success01/report.md), [benchmark folders](../archive/experiments/outcome) |
| Point probe + reverse lessons | First correct a critical decision, then train suffixes 1, 2, 4, 8, 14 | Critical point and suffix 4 passed, suffix 8 exhausted budget; complete P01 still failed. The critical decision later regressed | [Report](../archive/experiments/lessons/20261001-branch01/report.md) |
| Cumulative behavior protection | Preserve acquired dependencies; measured-response direction fitting | Critical behavior and suffixes 1/2/4 retained; suffix 5 and full P01 failed after 1,087.281 s | [Report](../archive/experiments/joint/20261001-retention01/report.md) |
| Loss and data-scope review | Review bottlenecks and plan paired tests | Literature review, saved-run analysis and the experiment plan are retained as research records | [Review](../archive/experiments/research/20261001-training-loss/review.md) |
| Frontier A-D, E-F, G-H | Mixed versus single-entry objectives; softplus versus hinge; step calibration; direction count | Single-entry B/D solved suffix 5; subsequent arms did not solve suffix 6 or P01. Eight models independently verified; a missing recovery signal for broken anchors was identified | [Synthesis](../archive/experiments/frontier/20261001-summary/report.md) |
| Trainable MLP H/J | Frozen inherited brain + MLP versus alternating brain/MLP updates; recovery loss | Both solved P01 in 14 moves. H changed no brain parameters; J changed 16, but replacing J's brain with its source still solved | [Report](../archive/experiments/hybrid/20261001-mlp01/report.md) |
| Readout L/M/E comparison | Freeze the same learned brain; train linear, MLP or sign-constrained E/I head | All solved P01 in 14 moves. Linear used 2,052 parameters versus 33,092 for MLP/E/I; a hidden layer was unnecessary for this task | [Report](../archive/experiments/readouts/20261001-compare01/report.md) |
| Joint brain + linear P03 | Simultaneous Adam, generic reverse suffixes and anchor recovery | 8 x 8, two boxes: solved in 22 moves after 216 updates / 410.156 s; saved-pair substitutions failed | [Report](../runs/linear-joint/20261002-p03-01/report.md) |
| Joint brain + linear P05 | Continue both components from P03 on a larger task | 10 x 10, two boxes: solved in 34 moves after 384 updates / 771.891 s; saved-pair substitutions failed | [Report](../runs/linear-joint/20261002-p05-01/report.md) |

## Retained lineage

```text
Pilot -> stopped B00 -> repair D seed 10 -> P00 -> unsuccessful P01 routes
                                              -> outcome -> lessons -> joint
                                              -> frontier B -> E -> G
                                              -> frozen-brain linear trial L on P01
                                              -> jointly trained P03 -> jointly trained P05
```

This diagram summarizes inheritance; exact source filenames and hashes remain in each run's configuration. MLP and E/I comparisons branch from the same learned G brain. Seeds 11/12 belong to the B00 repair comparison; they are not independent replications of the subsequent seed-10 lineage.

## Interpretation and preservation

The progression demonstrates that changing training and readout design overcame earlier failures. A timeout alone did not locate a brain capacity limit. A suffix success is distinct from an original-start solution. Shortest-path lengths describe task difficulty; the accepted objective is any autonomous solution. Earlier-task retention is diagnostic for the current method, and task-specific checkpoints preserve earlier successes.

Historical reports describe the state at their writing time. Statements such as "not yet added to the viewer" or "not attempted" remain historical facts in those files and do not describe the current registry. Some old reports use absolute pre-organization paths. Use this index or the [relocation manifest](../archive/maintenance/20261002/relocation-manifest.json) to locate the current files. Raw results, configurations, STOP markers, source manifests and all checkpoint bytes were retained.

Original attempt logs do not reconstruct unsaved transient candidates. The archive preserves all files present at organization time; the earlier README also records that a former 128-map pool had already been removed before this archive. Its historical aggregate scores are not held-out generalization evidence.
