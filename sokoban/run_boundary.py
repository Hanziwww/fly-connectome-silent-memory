"""Prepare and supervise the finite, serial, single-task difficulty screen."""
from workspace import ARCHIVE, RUNS
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from bootstrap import HERE, ROOT
from boundary_tasks import prepare_tasks
from common import file_hash, save_json
from run_cycle import WorkerJob, acquire_worker_lock, stop_tree


def prepare(out, seeds, seconds):
    if out.exists() and any(out.iterdir()):
        raise ValueError("New experiments require an empty directory; use --resume for a prepared experiment")
    out.mkdir(parents=True, exist_ok=True)
    levels = prepare_tasks(out)
    source = ARCHIVE / "campaign/run/final.pt"
    source_hash = file_hash(source)
    shutil.copyfile(source, out / "source.pt")
    assert file_hash(out / "source.pt") == source_hash
    config = dict(experiment="independent_single_task_boundary", version=1,
        source_checkpoint=str(source.relative_to(ROOT)), source_checkpoint_sha256=source_hash,
        training_seconds=seconds, seeds=seeds, task_ids=[l["id"] for l in levels],
        frozen_readout=True, correction_enabled=False, reset_optimizer_per_trial=True,
        old_task_retention_constraint=False, old_task_retention_evaluated=True,
        training_method="Hard-forward candidate selection; surrogate Adam directions every fourth generation plus symmetric Gaussian proposals",
        teacher_scope="All states on every shortest solution from each fixed start; all optimal actions accepted",
        early_stopping="First autonomous solve, training deadline, explicit stop, or runtime error; no plateau or generation cutoff",
        primary_outcome="Autonomous fixed-start success after checkpoint reload",
        candidate_selection="Fixed from exact CPU descriptors before model evaluation",
        source_history="Four-task learned core and readout previously calibrated on 16 maps; this is task acquisition, not unseen-map generalization")
    save_json(out / "config.json", config)
    rng = np.random.default_rng(20261001)
    schedule = [dict(task_id=str(task), seed=seed) for seed in seeds
                for task in rng.permutation(config["task_ids"])]
    save_json(out / "schedule.json", schedule)
    save_json(out / "source-files.json", {str(p.relative_to(ROOT)): file_hash(p) for p in [
        HERE / "boundary_tasks.py", HERE / "boundary_worker.py", HERE / "run_boundary.py",
        HERE / "lib/brain_model.py", HERE / "lib/heads.py", HERE / "lib/evaluation.py",
        HERE / "lib/game.py", HERE / "lib/training_support.py", HERE / "continual.py"]})
    return config, schedule


def summarize(out):
    config = json.loads((out / "config.json").read_text())
    levels = json.loads((out / "levels.json").read_text())
    rows, results = [], []
    for path in sorted((out / "trials").glob("*/result.json")):
        result = json.loads(path.read_text())
        results.append(result)
        train = result.get("training", {})
        after = result.get("after", {}).get("records", [{}])[0]
        rows.append(dict(task_id=result["task_id"], seed=result["seed"], status=result["status"],
            training_seconds=train.get("training_seconds"), checkpoint_training_seconds=train.get("checkpoint_training_seconds"),
            generations=train.get("generations"), accepted_updates=train.get("accepted_updates"),
            rollout_status=after.get("status"), moves=after.get("steps"),
            teacher_correct=result.get("teacher", {}).get("correct"), teacher_states=result.get("teacher", {}).get("total"),
            old_tasks_solved=result.get("old_tasks_diagnostic", {}).get("solved"),
            simulated_boards=train.get("training_compute", {}).get("simulated_boards_including_padding"),
            peak_allocated_gib=result.get("resources", {}).get("peak_allocated_gib")))
    groups = []
    for level in levels:
        part = [r for r in rows if r["task_id"] == level["id"]]
        groups.append(dict(task_id=level["id"], name=level["name"], optimal_steps=level["optimal_steps"],
            minimum_pushes=level["minimum_pushes"], teacher_states=level["teacher_states"],
            attempted=len(part), planned=len(config["seeds"]),
            already_solved=sum(r["status"] == "already_solved" for r in part),
            learned=sum(r["status"] == "learned" for r in part),
            budget_exhausted=sum(r["status"] == "budget_exhausted" for r in part),
            errors=sum(r["status"] in ("error", "hard_stop", "stopped") for r in part)))
    summary = dict(completed=sum(r["status"] in ("already_solved", "learned", "budget_exhausted") for r in rows),
                   planned=len(levels) * len(config["seeds"]), by_task=groups)
    save_json(out / "summary.json", summary)
    if rows:
        temporary = out / "results.csv.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, out / "results.csv")
    return summary


def supervise(command, log_path, out, ceiling, identity):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process, job = None, None
    started = time.monotonic()
    status = "running"
    try:
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=flags, start_new_session=os.name != "nt")
            job = WorkerJob(process)
            identity.update(status="running", worker_pid=process.pid, hard_ceiling_seconds=ceiling)
            save_json(out / "supervisor.json", identity)
            while process.poll() is None:
                if (out / "STOP").exists():
                    status = "stopped"
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        stop_tree(process)
                    break
                if time.monotonic() - started >= ceiling:
                    status = "hard_stop"
                    stop_tree(process)
                    break
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
    finally:
        if process is not None and process.poll() is None:
            stop_tree(process)
        if job is not None:
            job.close()
    identity.update(worker_running=False, worker_exit_code=process.returncode if process else None,
                    worker_seconds=time.monotonic() - started)
    return process.returncode if process else 1, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ARCHIVE / "boundary/20261001-screen01"))
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Invalid budget or duplicate seeds")
    out = Path(args.out).resolve()
    if args.resume:
        config = json.loads((out / "config.json").read_text())
        schedule = json.loads((out / "schedule.json").read_text())
        if config["training_seconds"] != args.seconds or config["seeds"] != args.seeds:
            parser.error("Resume settings must equal the saved protocol")
    else:
        config, schedule = prepare(out, args.seeds, args.seconds)
    if args.prepare_only:
        print(json.dumps(dict(status="prepared", out=str(out), trials=len(schedule))), flush=True)
        return 0
    if (out / "STOP").exists():
        parser.error("Boundary experiment STOP file is present")
    if file_hash(out / "source.pt") != config["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint copy changed")
    lock = acquire_worker_lock()
    identity = dict(supervisor_pid=os.getpid(), status="starting", planned=len(schedule), completed=0)
    save_json(out / "supervisor.json", identity)
    try:
        for entry in schedule:
            name = f"{entry['task_id']}-s{entry['seed']}"
            trial = out / "trials" / name
            if (trial / "result.json").exists():
                old = json.loads((trial / "result.json").read_text())
                if old["status"] in ("learned", "already_solved", "budget_exhausted"):
                    continue
                raise RuntimeError(f"Preserved unsuccessful attempt {name} requires an explicit recovery decision")
            trial.mkdir(parents=True, exist_ok=True)
            identity.update(active_trial=name, completed=summarize(out)["completed"])
            command = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "boundary_worker.py"),
                       "--out", str(out), "--task", entry["task_id"], "--seed", str(entry["seed"])]
            code, state = supervise(command, trial / "worker.log", out, config["training_seconds"] + 300, identity)
            if code != 0 or state != "running":
                if not (trial / "result.json").exists():
                    save_json(trial / "result.json", dict(status=state if state != "running" else "error",
                              task_id=entry["task_id"], seed=entry["seed"], worker_exit_code=code))
                identity.update(status="stopped" if state == "stopped" else "error")
                return 1
            summary = summarize(out)
            identity.update(completed=summary["completed"])
            print(json.dumps(dict(event="trial_complete", trial=name, **summary)), flush=True)
        identity.update(active_trial="independent-verification", completed=len(schedule))
        command = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "boundary_worker.py"),
                   "--out", str(out), "--verify-suite"]
        code, state = supervise(command, out / "verification.log", out, 600, identity)
        identity["status"] = "complete" if code == 0 and state == "running" else "error"
        return 0 if identity["status"] == "complete" else 1
    except KeyboardInterrupt:
        identity["status"] = "stopped"
        return 1
    except Exception as error:
        identity.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        identity["summary"] = summarize(out)
        identity["worker_running"] = False
        save_json(out / "supervisor.json", identity)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
