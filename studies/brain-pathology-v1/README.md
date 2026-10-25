# brain-pathology-v1: learning, memory and movement in pathological and aged fly brains

**Status: draft.** Nothing has been registered or run.

This study asks how a pathological brain relates to its capacity for learning, memory and movement, using
interventions on the healthy connectome model (plan v2):

- **Phase 2:** a systematic review of what real fly brains show, then a credibility rubric (C1–C6) and a shortlist.
  Only conditions that pass C1–C5 go on. C5 requires an in-silico replication of the observed fly phenotype on the
  healthy model.
- **Phase 3:** capacity experiments on the retained cases only. They start after gate G2, when the atlas, scores,
  calibration and shortlist are hash-frozen in `protocol.json`.

The frozen study `studies/brain-sokoban-v1` is a read-only input. This study never writes to it.

## Layout

| Path | Contents |
| --- | --- |
| `literature/` | The Phase 2a systematic review (`atlas.csv`, search log). The review maintains it; the scaffold and research.py never write there. |
| `protocol.json` | Design, conditions, severities, targeting, regimes, rescues and input hashes. It is a draft until `Study.register` freezes it at G2. |
| `registration.json` | Hashes of the protocol, schedule and task lock at registration. Written at registration; absent until then. |
| `schedule.json` | Jobs: `id`, `dependencies`, optional `slots` (GPU slots on one lane) and `verify` (default true). |
| `tasks/lock.json` | SHA-256 of every task file in `tasks/` (`Study.lock_tasks`). |
| `amendments/` | Append-only, hash-chained changes after registration (`Study.amend`). |
| `code/testplan.py` | Test families: the Holm primary family of each retained case (frozen at G2), plus BH exploratory families. |
| `code/study.py` | The `run`, `run_job`, `verify` and `verify_job` functions that research.py calls. |
| `runs/<job>/` | One folder per job: `stage-run.json`, `stage-verify.json`, logs and results. Written on the remote and pulled back. |
| `code-snapshots/<key>/` | Source behind each result (`Study.provenance`). |
| `supervisor.json`, `events.jsonl`, `queue.log` | Queue state, the append-only event log and the queue's output. Written on the remote. |
| `receipts/`, `sync-verification.json` | Local transfer receipts and the hash check of the pulled outputs. |
| `STOP` | Pause marker. Its presence stops new work. Remove it only for an explicit restart. |

## Commands

Run these from the project root with the bundled runtime:

```powershell
.\MIGRATION\python\python.exe -B -X utf8 MIGRATION\launch.py --script research.py --study brain-pathology-v1 status
.\MIGRATION\python\python.exe -B -X utf8 MIGRATION\launch.py --script research.py --study brain-pathology-v1 run --dry-run
```

- `status` is read-only.
- `run` and `verify` refuse until the protocol is registered and unchanged.
- On the GPU host, `python -m tools.remote launch --study brain-pathology-v1` starts the queue.
- Transfers use `python -m tools.remote push|pull|verify` (see `tools/remote/README.md`).

## Rules

- No condition enters the capacity experiments without passing C1–C5, including the in-silico replication.
  Excluded cases are reported with their failing criterion.
- Conditions that need mechanisms the healthy model lacks (neuromodulation, plasticity rules, metabolism, glia) are
  excluded as not credibly replicable.
- Each test family is declared in `code/testplan.py` before its tests are computed. After registration, changes are
  amendments only.
