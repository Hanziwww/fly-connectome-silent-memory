# Diagnosis of two training-process interruptions on 2026-10-02

The evidence attributes both interruptions to host-application and sandbox-service termination during Codex desktop updates. Windows deployment logs explicitly record the updates, forced-shutdown option, and service termination; training, supervisor, and finalizer disappeared within the same windows. The trainer left no exception-exit record. Times below are Taipei time (UTC+8).

| Time | Verified record |
|---|---|
| Around 13:01:51 | D6-T1 last saved at 380 updates; an interruption snapshot was separately retained. |
| 13:01:58–13:01:59 | Windows updated Codex from 26.928.3736.0 to 26.928.4866.0. AppX event 603 includes `ForceTargetApplicationShutdownOption`; 9648/9650 record service termination for the update. |
| 19:06:16 | Last complete D7-T0 save: 1212 updates, curriculum depth 57, effective training 2694.08 seconds (44.90 minutes). |
| 19:06:19 | The old desktop log records `update_resume_state_saved threadCount=0`, entering update-state saving. This does not prove recovery of an experiment checkpoint. |
| 19:06:27 | Windows updated Codex from 26.928.4866.0 to 26.930.2377.0, again using forced shutdown and recording successful termination of the app and sandbox service. |
| 19:06:28–19:06:33 | The new desktop app, sandbox service, and Codex host started. |
| 19:11–19:12 | The original supervisor, worker, and finalizer no longer existed; GPU utilization was about 1%. The residual supervisor.json still said running, which did not establish a live training process. |

The background launch method did not isolate training from the Codex lifecycle. A hidden-window restart had been used, but window hiding did not prevent the host update from terminating training. The exact system call that terminated Python was not recorded. Nevertheless, app updates, forced termination, and loss of the entire experiment process chain align in both occurrences.

Record this as a runtime-environment interruption. D7-T0 was not yet solved; the saved state was curriculum depth 57 during segment two, not 57 moves completed from the original start. The cumulative second-segment budget was not exhausted, and the trainer reported no OOM, numerical error, or deliberate stagnation exit. Independently accepted difficulty tasks remained at 21/24.

CPU checks confirmed resume.pt was readable and contained current/best parameters, Adam state for 5 parameter groups, Torch/CUDA/Python/NumPy random states, curriculum data, and cumulative effective training time. The configuration matched the checkpoint; all 104 registered source files were unchanged and checked parameters finite. Gradient logs reached update 1216, but the complete checkpoint ended at 1212; resumption must use the complete checkpoint. Its copy matched the original SHA-256.

Both update windows contained NVIDIA `nvlddmkm` event 153 with the generic message `Error occurred on GPUID: 100`. The original events were retained. The records cannot separately determine whether these accompanied cleanup of GPU contexts during the update or another driver problem. They do not establish memory exhaustion or model numerical divergence. System boot time remained September 27; the machine did not reboot.

This diagnostic round archived evidence without restarting training or changing the method, training source, system-update settings, or monitor. A future repair should host training in a system task or service independent of Codex and verify survival after host exit, single-instance enforcement, and complete-checkpoint continuation before formal training resumes.

Evidence directory: `../runs/difficulty-D7-T0/interruption-20261002T111847Z/`. It contains the checkpoint and training logs, Windows System/Application/AppX events with XML, desktop-update log excerpts, checkpoint-copy verification, and interruption.json. The original supervisor.json is retained as evidence; its running state was stale.
