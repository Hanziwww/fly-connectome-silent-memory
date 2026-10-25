# tools/remote: key-based SSH transport and study queue

`tools/remote` moves code and data to the remote GPU host (RTX PRO 6000, Linux) and brings study outputs back,
with SHA-256 verification in both directions. It also runs a study's jobs there on GPU lanes and slots.

Every operation is one call to the system's OpenSSH client with these options:

- `BatchMode=yes`
- `StrictHostKeyChecking=yes`
- password and keyboard-interactive authentication off
- public keys only

So no step can ever ask for a password. Nothing here reads `~/.ssh`, keys, `local/` or `ssh.txt`. The retired
paramiko tools (`tools/remote_study.py`, `tools/remote_supplement.py`, `tools/verify_remote_sync.py`) are never
imported; `tests/test_isolation.py` enforces this.

## Setup (once)

1. **SSH key and alias.** Create a key pair (or use an existing one) and put the public key on the remote host.
   Then add a `Host` entry to `~/.ssh/config` (`C:\Users\<you>\.ssh\config` on Windows):

   ```
   Host pro6000
       HostName <address of the GPU host>
       User <remote user>
       Port <port, if not 22>
       IdentityFile ~/.ssh/id_ed25519_pro6000
       IdentitiesOnly yes
   ```

   If the key has a passphrase, load it into the ssh agent first (`ssh-add`; on Windows, start the *OpenSSH
   Authentication Agent* service). The tools never ask for it.

2. **Host key.** Connect once by hand with `ssh pro6000` and accept the host key only if its fingerprint matches the
   one shown by the provider. With `StrictHostKeyChecking=yes` the tools refuse any host that is not already in
   `~/.ssh/known_hosts`, including one whose key has changed.

3. **Settings.** Copy `tools/remote/remote.toml.example` to `remote.toml` at the project root and set:

   - `host`: the alias;
   - `remote_root`: an absolute path on the remote;
   - the GPU `lanes`, `slots_per_lane` and per-slot `memory_cap_gib`.

   `remote.toml` is git-ignored. It refuses credentials and connection details (password, identity file, user,
   port), which belong in `~/.ssh/config`.

4. **Check.**

   ```powershell
   .\MIGRATION\python\python.exe -B -X utf8 -m tools.remote check
   ```

   It reports the remote interpreter, the disk, the GPUs (`nvidia-smi`), the last push and any running study queue.

The transport needs only the standard library, so the bundled interpreter runs it directly from the project root.
`MIGRATION\launch.py --script tools\remote\__main__.py <command>` works as well. The remote side needs only
`python3`. The project environment there (`MIGRATION/setup_linux.py`) is a separate Phase 0 step.

## Commands

| Command | What it does |
| --- | --- |
| `push [--commit C] [--data LIST] [--path P] [--study S] [--dry-run]` | Sends `git archive C` (default `HEAD`; remote.toml `exclude` pathspecs left out) and listed data files (`LIST`: one project-relative path per line). Data is sent only when the remote copy is missing or different; a different remote copy is replaced only with `--replace-data`. Refused while a study queue holds its lock, unless `--allow-running`. `--study` checks that the commit holds the study's `protocol.json`, `schedule.json` and `tasks/lock.json`. |
| `pull --study S [--path P] [--dry-run]` | Fetches the study's remote outputs (`runs/`, `code-snapshots/`, `supervisor.json`, `events.jsonl`, `queue.log`) that are missing or out of date locally. Active jobs are skipped. A local file that differs from the remote is reported as a conflict and never overwritten; only mirrored state (`supervisor.json`, `events.jsonl`, `progress.json`, logs) is refreshed. |
| `verify --study S` | Once the study's remote queue has ended, checks every remote output against the local mirror. Writes `studies/S/sync-verification.json` in the schema of the old `tools/verify_remote_sync.py`. |
| `status --study S` | Remote supervisor state, STOP, worker lock, recent events, GPUs, disk and last push. |
| `launch --study S [--resume] [--clear-stop]` | Starts `research.py --study S run` on the remote, detached, with the GPU settings of remote.toml. Logs go to `studies/S/queue.log`. `--clear-stop` renames an existing STOP to `STOP-cleared-<time>.json`. |
| `stop --study S [--reason R]` | Creates `studies/S/STOP` on the remote. Workers stop at a safe point; the queue terminates the rest after a grace period. |

The frozen study `brain-sokoban-v1` is never launched, stopped or pulled. Its tracked code is pushed like any other
code, as an input.

## What is transferred

- **Pushed:** the files of a commit plus explicitly listed data, in one tar with a manifest. The remote extracts it,
  checks every file's size and SHA-256 on disk, and records `.sync/receipts/<push>.json` and `.sync/current.json`
  (commit, code and data hashes). A local copy of each receipt goes to `runtime/remote/pushes/`. Files of an earlier
  push that are absent from the new one are reported as `stale_files`, not deleted.
- **Pulled:** a remote manifest (path, size, SHA-256), then a tar of the needed files. Each file is checked before it
  replaces anything. Receipts go to `studies/S/receipts/`.
- **Never, in either direction:** `local/`, `runtime/`, `.git/`, `.sync/`, virtual environments, `remote.toml`,
  `ssh.txt`, `*known_hosts*`/`*known-hosts*`, private keys (`id_*`, `*.pem`, `*.key`, `*.ppk`), anything under
  `.ssh/`, and links or special files. Queue outputs of a study (`runs/`, `code-snapshots/`, `STOP`, …) are never
  pushed.

## The queue (`queue.py`)

`research.py --study S run` on the remote runs `Queue(study, schedule)`:

- **Lanes and slots.** Each GPU lane offers `slots_per_lane` worker slots. A job takes one slot, or `"slots": k` slots
  on one lane, and gets `CUDA_VISIBLE_DEVICES` and `STUDYKIT_MEMORY_CAP_GIB` (cap × slots) in its environment. A
  worker starts only when the device has that much free memory plus the reserve, and the disk has
  `min_free_gib` free.
- **Stages.** Each job runs `research.py --study S run --job ID`, then `verify --job ID` in a separate process (unless
  `"verify": false`), once its `dependencies` are complete.
- **Locks.** One supervisor per study (`studies/S/worker.lock`) and one lock per slot
  (`runtime/gpu-slots/lane<L>-slot<S>.lock`). Workers inherit both on Linux, so the locks stay held if the supervisor
  dies, and a new supervisor refuses to start while old workers live.
- **Failures.** A failed or lost stage is never retried; its dependents are blocked.
- **Records.**
  - `studies/S/supervisor.json`: rewritten every tick.
  - `studies/S/events.jsonl`: append-only.
  - `runs/<job>/stage-<stage>.json` and `runs/<job>/<stage>.log`: one pair per stage.
- **STOP.** No new stage starts, and workers exit at a safe point with code 75. Stopped stages run again only with
  `--resume`.
- **Exit codes.** 0 complete, 1 needs attention, 75 stopped.

## Tests

The tests are network-free: `tests/fakes.py` replaces ssh with a local run of the real agent against a temporary
"remote" folder, so tar streams, manifests and receipts are all exercised.

```powershell
.\MIGRATION\python\python.exe -B -X utf8 MIGRATION\launch.py --script tools\remote\tests\run_tests.py
```
