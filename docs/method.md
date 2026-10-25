# Formal method: joint brain and linear readout

Method ID: `joint-brain-linear-v1`. Established on 2026-10-02.
The machine-readable [registry](../method.json) selects P05 as the current model and retains P03 as its predecessor.

## Inference

`Board (4 x 10 x 10) -> full-connectome LIF brain -> 512 activity features -> fixed normalization/clipping -> affine 512-to-4 readout -> physical legal-action mask -> action`.

The MaleCNS network retains 167,106 neurons and 25,578,600 stored connection entries. Eight sensory neurons represent each input cell/channel, giving 3,200 driven neurons. Every decision resets the neural state and runs 40 ms with 1 ms steps. Currents at 14 ms and voltages relative to rest at 40 ms from 4,096 observed neurons are pooled in 256 groups of 16 to form 512 features. Activity normalization is inherited and fixed; standardized features are clipped to [-10, 10]. The readout is affine in these preprocessed features.

The head receives neural activity only. Reference solutions generate training examples and certify task difficulty; inference uses the saved neural policy. The physical mask excludes illegal moves. Learning memory resides in parameters; neural activity does not persist between decisions.

## Parameters and optimization

| Component | Trainable parameters | Constraints |
|---|---:|---|
| Brain | 35,310 | 11,770 type groups, each with outgoing gain, incoming gain and threshold offset; gains 0.8-1.25, threshold offsets +/-1 mV |
| Linear readout | 2,052 | 512 x 4 weights and 4 biases; normalization stays fixed |

Both components receive gradients from the same loss at the same parameter point and update in one Adam step. Learning rates are 0.0001 for the brain and 0.0003 for the head. Gradient norms are clipped to 1 and 5 respectively. Brain gradients use the existing surrogate spike derivative; forward evaluation retains hard LIF spikes. Connection endpoints and effective signs stay fixed.

Two-pass differentiation first obtains the loss gradient with respect to neural features, then recomputes one 16-board simulation batch at a time to accumulate brain gradients. Recomputed features must agree within 1e-6; the saved P03/P05 runs observed zero discrepancy. A feature cache is never reused across a brain update. There are four optimizer steps per behavioral assessment, and Adam is recreated at each suffix stage.

## Curriculum, loss and acceptance

Start with the last one-move suffix of a verified reference solution. After successful autonomous execution, extend the start backward by one reference move. Successful policy trajectories contribute action anchors for recovery. The witness defines lesson starts; any autonomous solving trajectory is accepted, including detours.

Let `m_entry` be the allowed-action versus disallowed-action logit margin at the new entry, and `m_i` the margin at a previously demonstrated dependency. The task loss is:

```text
entry = 0.25 * softplus((0.10 - m_entry) / 0.25)
v_i   = max(0, 0.03 - m_i)^2
loss  = entry + 0.5 * mean(v_i) + 0.5 * max(v_i)
```

The recovery term is zero when there are no anchors; the current entry is excluded from the anchor term. Healthy dependencies above the floor receive no recovery penalty.

An autonomous solution from the original start overrides auxiliary gates and ends training. Otherwise a retained candidate must keep previously acquired suffixes and either pass the current stage or improve the loss. Promotion is checked by actual neural rollouts. Prior *tasks* have no retention gate; their regression is recorded diagnostically. Evaluation detects illegal/deadlocked states and deterministic repeated-state loops, with a 512-move ceiling.

## Verified results

| Task | Board / boxes | Autonomous solution | Joint updates | Training time | Inherited model |
|---|---|---:|---:|---:|---|
| [P03](../runs/linear-joint/20261002-p03-01/report.md) | 8 x 8 / 2 | 22 moves | 216 | 410.156 s | P01 linear-readout trial L |
| [P05](../runs/linear-joint/20261002-p05-01/report.md) | 10 x 10 / 2 | 34 moves | 384 | 771.891 s | P03 joint model |

Both runs use seed 10. Training time includes optimization and curriculum assessments, excluding initialization and independent verification. Peak allocated training VRAM was about 4.395 GiB. The configured allocator ceiling is 11 GiB, with at least 13 GiB free before a serial worker starts.

| Saved P05 parameter pairing | Result |
|---|---|
| Final brain + final linear head | Solved, 34 moves |
| Previous P03 brain + final head | Repeated state, 24 moves |
| Final brain + previous P03 head | Repeated state, 10 moves |
| Cleared plasticity + final head | Repeated state, 2 moves |

Independent processes reloaded the models and reproduced actions, logits, scores, promotions and substitutions. These substitutions demonstrate dependence of the saved pairing on the learned configuration. They do not establish superiority over independently retrained alternative brains or feature models. P05 training regressed P03 to a repeated state after 2 moves; the separate P03 checkpoint remains available in the viewer. Evidence covers selected training tasks and one inherited seed lineage, not unseen-map generalization or a biological capacity limit.

## Use and continuation

```powershell
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py serve --open --port 8774
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py check
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py verify
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py train --help
```

`verify` checks the running viewer, using inference only. A newly requested training run requires an explicit task and unused output directory; the default source is the registered P05 checkpoint. For example, when P06 training is requested:

```powershell
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py train --task P06 --out runs/linear-joint/NEW-P06-RUN --seconds 1200
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py analyze --out runs/linear-joint/NEW-P06-RUN
```

Stop the viewer before a GPU training run to release its allocation. `--prepare-only` prepares and certifies a requested run without training. Completed attempts are not overwritten. The 2026-10-02 organization did not train P06 or resume historical runs.

Implementation: [brain](../sokoban/lib/brain_model.py), [readout](../sokoban/readout_models.py), [curriculum](../sokoban/linear_joint_tasks.py), [loss](../sokoban/hybrid_tasks.py), [joint optimizer](../sokoban/linear_joint_training.py), [continuation](../sokoban/run_linear_next.py).
