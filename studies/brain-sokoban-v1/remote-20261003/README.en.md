# Focused remote experiments, 2026-10-03

This batch was user-authorized on four RTX 5090 GPUs. The original local queue and STOP remain unchanged.
The remote project is `/root/autodl-tmp/brain-sokoban-v1-20261003`, the actual server data directory.

## Fixed scope

- First independently reload the D1, D3, D5, and D7 T0 formal checkpoints and solve from their original starts; then evaluate each on eight prespecified held-out layouts and registered within-map variants. Test results do not select checkpoints.
- Run readout-only and full-demonstration controls for D7-T0 from the same P05 checkpoint, retaining original learning rates and acceptance rules.
- Dynamic memory uses only the first registered layout M0, with five conditions: untrained state carry, reset, readout-only, joint, and GRU. Retain trained/untrained delays and prefixes, clearing/swapping/restoration interventions, and a 512-action cap per episode.
- Before dynamic training, compare full-history direct and checkpoint-recomputed gradients and verify one update on the longest training history. Discard those diagnostic parameters; formal runs restart from their prescribed initializations.

## Computational settings

Seed 10; serial jobs on each GPU, independent conditions across GPUs; 11 GiB allocation cap per process; unchanged 20/40/80-minute segments.
Retain mixed precision: FP64 sparse forward propagation and FP32 backward accumulation. Dynamic training batch is 1, with evaluation split into at most 8 complete histories, without shortening histories or truncating gradients.
Use an independent Python environment, matching local numpy/pandas/pyarrow/scipy. PyTorch/CUDA use versions validated on the server; exact versions are in requirements-resolved.txt.
Report remote and earlier local timings separately, without claiming same-hardware training-method speedups.

## Execution and monitoring

Remote queue entry point: `studies/brain-sokoban-v1/remote-20261003/run_remote.py`.
Successful checkpoints require independent-process reload acceptance. Queue health uses actual processes and successive updates; technical failures are not learning failures.
Only `tools/remote_study.py` reads credentials from the user-specified ssh.txt; credentials were excluded from the upload.

Local status: `flybrain\.venv\Scripts\python.exe -B -X utf8 tools/remote_study.py status`.
Local synchronization: `flybrain\.venv\Scripts\python.exe -B -X utf8 tools/remote_study.py sync`.
Synchronization retrieves results only and neither starts nor resumes local training. Ended conditions are fully synchronized; active conditions supply progress and logs.

`progress.en.md` and `latest-status.json` update during synchronization; complete results are in this directory's results/.
Other modules in the original scientific plan remain unexecuted. The 16 queue entries include migration acceptance and diagnostics, not 16 new training runs.
