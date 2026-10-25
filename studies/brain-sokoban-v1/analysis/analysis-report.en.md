# Fly-connectome Sokoban experiments: data and interpretation

At this snapshot, 68 runs were recorded, including implementation diagnostics; 31 scheduled training items remained unfinished.

The difficulty study completed 24/24 maps; 24 solved after independent reload, all 24 newly acquired. Success means autonomous completion; path length is not an acceptance gate.

D0: 3/3 reload-verified solutions, median training time 1.49 minutes; D1: 3/3, 1.39 minutes; D2: 3/3, 3.88 minutes; D3: 3/3, 7.19 minutes; D4: 3/3, 7.78 minutes; D5: 3/3, 12.26 minutes; D6: 3/3, 28.50 minutes; D7: 3/3, 51.59 minutes. These describe a fixed task set; a longer solution does not by itself imply harder neural training.

Speed calibration selected mixed precision and batch 16. Acceptance required learning and independent solutions on both calibration tasks, with route changes allowed. The same-configuration FP32 recomputation tolerance was 0.001, with actual errors saved in gradient records. Activity conclusions require separate precision-sensitivity analysis.

The fixed library contains 24 training maps, 8 untrained layouts, and 2 speed-calibration maps, all certified by bidirectional solving. Shortest-path, branching, and deadlock metrics describe task structure only.

Training controls are paired by map and parent checkpoint. Training success and independent reload success are recorded separately; already solved maps count as existing ability. Stagnation, computational-stage limits, implementation errors, numerical errors, and interruptions remain distinct.

Hidden goals in dynamic-memory tasks appear only during the cue. Both histories are tested at the same physical state, and a wrong first push is solver-certified as irreversible. Common prefixes are controlled memory probes; the policy executes the full Sokoban behavior afterward autonomously.

Activity analyses separate matched-state recordings from free trajectories. Decoders fit discovery map families and are evaluated in separate confirmation families. Decodability of an action alone does not establish task necessity; necessity is assessed using group output suppression, matched controls, and restoration.

At this snapshot, 0/4 prespecified representative models had been analyzed, and memory verification covered 0/15 layout/model combinations. Missing results remain untested; no values are imputed by speculation.

Maps and map families are the statistical units. Neurons, time points, actions, and checkpoints describe internal processes without increasing independent training replication. Training uses seed 10, and conclusions are limited to this initialization lineage and tested maps.

Files: `run-registry.csv` records all run states; `paired-training-effects.csv` contains paired controls; `generalization.csv` stores zero-update tests; `memory-delays.csv` and mechanism files are generated as their experiments finish.

Training-control observations: alternating: 3 paired maps, solution-rate difference +0.00, median time difference +0.68 minutes; board_mlp: 3, +0.00, -5.90 minutes; brain_only: 3, -0.67, +131.22 minutes; demonstration: 3, +0.00, -5.69 minutes; entry_ce: 3, +0.00, +0.51 minutes; frozen: 3, -1.00, -7.28 minutes; head_only: 3, +0.00, -6.91 minutes; neutral: 3, +0.00, -0.20 minutes; no_recovery: 3, -1.00, +52.75 minutes; rewired: 3, +0.00, -0.42 minutes; weight_shuffle: 3, +0.00, +0.34 minutes. Time depends on both success and stopping rules and must be read with solution outcomes. Structural and neutral-initialization controls change the starting point and alone cannot show that one component of an existing P05 pairing is redundant.

Completed speed results: reference total training time was 912.42 seconds, versus 494.24 seconds for the selected configuration, a ratio of 1.85. This speedup describes only the two certified calibration tasks. A configuration stopping early on numerical error cannot be called faster learning because it ran briefly.

Unanswered questions: unfinished runs support no mechanism conclusions; one seed does not estimate between-seed stability; success on fixed tasks does not establish solutions to arbitrary layouts. Model-state and intervention evidence is not directly equivalent to memory or thought in real flies.
