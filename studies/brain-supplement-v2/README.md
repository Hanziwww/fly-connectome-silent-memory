# brain-supplement-v2: supplementary analyses and experiments for the frozen paper

**Status: draft.** Nothing has been registered or run.

This study tests the claims of the frozen manuscript (plan v2):

- **Phase 1B:** CPU analyses of saved data.
- **Phase 1C:** GPU experiments X1–X10 on the remote RTX PRO 6000, with the plan's pre-registered softening rules.

The frozen study `studies/brain-sokoban-v1` is a read-only input. This study never writes to it.

## Layout

| Path | Contents |
| --- | --- |
| `protocol.json` | Design, seed, replicate definition, experiments, softening rules and input hashes. It is a draft until `Study.register` freezes it. |
| `registration.json` | Hashes of the protocol, schedule and task lock at registration. Written at registration; absent until then. |
| `schedule.json` | Jobs: `id`, `dependencies`, optional `slots` (GPU slots on one lane) and `verify` (default true). |
| `tasks/lock.json` | SHA-256 of every task file in `tasks/` (`Study.lock_tasks`). |
| `amendments/` | Append-only, hash-chained changes after registration (`Study.amend`). |
| `code/testplan.py` | Test families, each declared before its tests are computed. |
| `code/study.py` | The `run`, `run_job`, `verify` and `verify_job` functions that research.py calls. |
| `runs/<job>/` | One folder per job: `stage-run.json`, `stage-verify.json`, logs and results. Written on the remote and pulled back. |
| `code-snapshots/<key>/` | Source behind each result (`Study.provenance`). |
| `supervisor.json`, `events.jsonl`, `queue.log` | Queue state, the append-only event log and the queue's output. Written on the remote. |
| `receipts/`, `sync-verification.json` | Local transfer receipts and the hash check of the pulled outputs. |
| `STOP` | Pause marker. Its presence stops new work. Remove it only for an explicit restart. |

## Commands

Run these from the project root with the bundled runtime:

```powershell
.\MIGRATION\python\python.exe -B -X utf8 MIGRATION\launch.py --script research.py --study brain-supplement-v2 status
.\MIGRATION\python\python.exe -B -X utf8 MIGRATION\launch.py --script research.py --study brain-supplement-v2 run --dry-run
```

- `status` is read-only.
- `run` and `verify` refuse until the protocol is registered and unchanged.
- On the GPU host, `python -m tools.remote launch --study brain-supplement-v2` starts `run`. It is the lane/slot queue over `schedule.json`, with one process per job stage and an independent verification process per job.
- Transfers use `python -m tools.remote push|pull|verify` (see `tools/remote/README.md`).
- Analyses use only outputs covered by a `sync-verification.json` with `transport_verified: true`.

## Rules

- Each test family is declared in `code/testplan.py` before its tests are computed. Post hoc families are flagged and never primary.
- After registration, changes are amendments only. Results are never overwritten; a repair gets a new job ID.
- Confirmatory families are tested only after all their runs are complete and verified.
- Every run records its code snapshot key, and the transport receipts record the pushed commit.
