"""One isolated task/seed per process, with frozen readout and bounded training."""
from workspace import ARCHIVE, RUNS
import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from bootstrap import HERE
from common import (BudgetExpired, configure, cpu_state, file_hash, save_checkpoint,
                    save_json, tensor_hash)
from boundary_tasks import TrialBudget, load_task
from brain_model import FullBrain
from heads import ActivityHead
from continual import measured, evaluate, subset, core_proposal
from training_support import gradients


class CountedBrain(FullBrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_counts()

    def reset_counts(self):
        self.counts = dict(forward_calls=0, simulated_boards_including_padding=0,
                           gradient_forward_calls=0)

    def forward(self, boards, **kwargs):
        self.counts["forward_calls"] += 1
        self.counts["simulated_boards_including_padding"] += len(boards)
        self.counts["gradient_forward_calls"] += int(torch.is_grad_enabled())
        return super().forward(boards, **kwargs)


def log(event, **values):
    print(json.dumps(dict(event=event, **values), allow_nan=False), flush=True)


def stop_check(out):
    if (out / "STOP").exists():
        raise KeyboardInterrupt("Boundary STOP file found")


def initialize(out):
    free, total = torch.cuda.mem_get_info()
    if free < 13 * 2**30:
        raise RuntimeError("Need at least 13 GiB free GPU memory")
    torch.cuda.set_per_process_memory_fraction(11 * 2**30 / total)
    source = torch.load(out / "source.pt", map_location="cpu", weights_only=True)
    encoder = CountedBrain(ARCHIVE / "campaign/run", batch=16, persist_mapping=False)
    head = ActivityHead().cuda()
    head.load_state_dict(source["head"], strict=True)
    head.requires_grad_(False)
    encoder.set_core(source["core"])
    return encoder, head, source, dict(free_before_gib=free / 2**30,
        allocator_limit_gib=11, batch=16, serial_worker=True)


def save_model(path, encoder, head, config, seed, task_id, accepted, elapsed):
    save_checkpoint(path, dict(core=cpu_state(encoder.core), head=cpu_state(head),
        source_checkpoint_sha256=config["source_checkpoint_sha256"], task_id=task_id,
        seed=seed, accepted_updates=accepted, checkpoint_training_seconds=elapsed,
        output_frozen=True, correction_enabled=False, observation_mode="snapshot",
        experiment="independent_single_task_boundary"))


def fit(encoder, head, level, data, config, seed, out, trial):
    rng = np.random.default_rng(seed)
    optimizer = dict(step=0, m={k: torch.zeros_like(v) for k, v in cpu_state(encoder.core).items()},
                     v={k: torch.zeros_like(v) for k, v in cpu_state(encoder.core).items()})
    accepted, generation, candidates = 0, 0, 0
    committed = cpu_state(encoder.core)
    encoder.reset_counts()
    budget = TrialBudget(config["training_seconds"], out / "STOP")
    phase = dict(status="training", task_id=level["id"], seed=seed,
                 training_started_unix=time.time(), training_seconds=config["training_seconds"])
    save_json(trial / "progress.json", phase)
    last_commit = 0.
    initial = latest = None
    stop = "training_budget"
    try:
        initial = latest = measured(encoder, head, data, budget.check)
        while True:
            budget.check()
            generation += 1
            base = measured(encoder, head, data, budget.check)
            saved = cpu_state(encoder.core)
            best, best_result, selected = saved, base, None
            proposals = []
            norm = None
            if generation % 4 == 1:
                chosen = rng.choice(len(data["boards"]), min(32, len(data["boards"])), replace=False)
                gradient, norm = gradients(encoder, head, subset(data, chosen), budget.check)
                optimizer["step"] += 1
                t = optimizer["step"]
                direction = {}
                for (key, value), grad in zip(saved.items(), gradient):
                    grad = grad.cpu()
                    m, v = optimizer["m"][key], optimizer["v"][key]
                    m.mul_(.9).add_(grad, alpha=.1)
                    v.mul_(.999).addcmul_(grad, grad, value=.001)
                    direction[key] = -(m / (1 - .9**t)) / ((v / (1 - .999**t)).sqrt() + 1e-8)
                for scale in (.001, .0001, .00001, -.001, -.0001, -.00001):
                    proposals.append((f"gradient:{scale}", core_proposal(saved, direction, scale)))
            for scale in (.0003, .001, .003, .01):
                direction = {k: torch.from_numpy(rng.standard_normal(v.shape).astype("float32"))
                             * (10 if k == "threshold_mv" else 1) for k, v in saved.items()}
                for sign in (1, -1):
                    proposals.append((f"gaussian:{scale}:{sign}", core_proposal(saved, direction, sign * scale)))
            for name, proposal in proposals:
                budget.check()
                encoder.set_core(proposal)
                result = measured(encoder, head, data, budget.check)
                candidates += 1
                if result["loss"] < best_result["loss"] - 1e-6:
                    best, best_result, selected = proposal, result, name
            budget.check()
            encoder.set_core(best)
            latest = best_result
            if selected is not None:
                committed = cpu_state(encoder.core)
                accepted += 1
                last_commit = time.monotonic() - budget.started
                save_model(trial / "model.pt", encoder, head, config, seed, level["id"], accepted, last_commit)
            elapsed = time.monotonic() - budget.started
            row = dict(generation=generation, accepted_updates=accepted, selected=selected,
                       loss=latest["loss"], correct=latest["correct"], teacher_states=len(data["boards"]),
                       gradient_norm=norm, candidates=candidates, seconds=elapsed, **encoder.counts)
            with (trial / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            phase.update(row)
            save_json(trial / "progress.json", phase)
            if generation == 1 or generation % 4 == 0:
                log("generation", task_id=level["id"], seed=seed, **row)
            if generation % 4 == 0 or latest["correct"] == len(data["boards"]):
                tested = evaluate(encoder, head, [level], "training-check", budget.check,
                                  [(data["boards"], latest["features"])])
                save_json(trial / "last-rollout.json", tested)
                if tested["complete"] and tested["solved"]:
                    stop = "autonomous_success"
                    break
    except BudgetExpired:
        encoder.set_core(committed)
    finally:
        # Candidate evaluation or gradient graphs never leak into the exported model.
        encoder.set_core(committed)
    elapsed = time.monotonic() - budget.started
    return dict(stop=stop, generations=generation, accepted_updates=accepted, candidates=candidates,
                training_seconds=elapsed, budget_seconds=config["training_seconds"],
                checkpoint_training_seconds=last_commit,
                initial_loss=None if initial is None else initial["loss"],
                training_compute=dict(encoder.counts))


def teacher_diagnostic(encoder, head, data):
    result = measured(encoder, head, data)
    with torch.no_grad():
        q = head(torch.as_tensor(result["features"], device="cuda"), correction=False)["brain_logits"]
        valid = torch.as_tensor(data["legal"], device="cuda").bool()
        labels = torch.as_tensor(data["labels"], device="cuda").bool()
        q = q.masked_fill(~valid, -1e9)
        margin = q.masked_fill(~labels, -1e9).max(1).values - q.masked_fill(labels | ~valid, -1e9).max(1).values
        meaningful = (valid & ~labels).any(1)
        margins = margin[meaningful].cpu().tolist()
    wrong = np.flatnonzero(~data["labels"][np.arange(len(data["boards"])), result["actions"]]).tolist()
    return dict(loss=result["loss"], correct=result["correct"], total=len(data["boards"]),
                incorrect_state_indices=wrong, minimum_choice_margin=min(margins) if margins else None,
                label_entropy=float(np.log(data["labels"].sum(1)).mean()))


def run_trial(args):
    out = Path(args.out).resolve()
    config = json.loads((out / "config.json").read_text())
    trial = out / "trials" / f"{args.task}-s{args.seed}"
    trial.mkdir(parents=True, exist_ok=True)
    if (trial / "result.json").exists() or (trial / "history.jsonl").exists():
        raise ValueError("An attempted trial cannot be overwritten; preserve it and use a new run directory")
    started = time.monotonic()
    result = dict(status="initializing", task_id=args.task, seed=args.seed, pid=os.getpid())
    save_json(trial / "progress.json", result)
    try:
        stop_check(out)
        configure(args.seed)
        level, data = load_task(out, args.task)
        encoder, head, source, resources = initialize(out)
        result["resources"] = resources
        source_core = tensor_hash(source["core"])
        source_head = tensor_hash(source["head"])
        assert tensor_hash(cpu_state(encoder.core)) == source_core
        assert tensor_hash(cpu_state(head)) == source_head
        result["source_core_sha256"] = source_core
        result["source_head_sha256"] = source_head
        check = lambda: stop_check(out)
        before = evaluate(encoder, head, [level], "source", check)
        if not before["complete"]:
            raise RuntimeError("Incomplete initial rollout")
        result["before"] = before
        log("baseline", task_id=args.task, seed=args.seed, status=before["records"][0]["status"])
        save_model(trial / "model.pt", encoder, head, config, args.seed, args.task, 0, 0.)
        if before["solved"]:
            training = dict(stop="already_solved", accepted_updates=0, generations=0, training_seconds=0.,
                            checkpoint_training_seconds=0., budget_seconds=config["training_seconds"],
                            training_compute=dict(forward_calls=0, simulated_boards_including_padding=0,
                                                  gradient_forward_calls=0))
        else:
            # Probe on the real frozen head without changing parameters; include no probe in fit accounting.
            try:
                probe, norm = gradients(encoder, head, subset(data, np.arange(min(16, len(data["boards"])))), check)
                result["gradient_probe"] = dict(norm=norm, finite=all(bool(torch.isfinite(g).all()) for g in probe))
            finally:
                encoder.set_core(source["core"])
                gc.collect()
                torch.cuda.empty_cache()
            training = fit(encoder, head, level, data, config, args.seed, out, trial)
        result["training"] = training
        save_json(trial / "progress.json", dict(status="verifying", task_id=args.task, seed=args.seed))
        reloaded = torch.load(trial / "model.pt", map_location="cpu", weights_only=True)
        head.load_state_dict(reloaded["head"], strict=True)
        encoder.set_core(reloaded["core"])
        after = evaluate(encoder, head, [level], "reloaded", check)
        result["after"] = after
        result["teacher"] = teacher_diagnostic(encoder, head, data)
        old_levels = json.loads((ARCHIVE / "campaign/pilot-levels.json").read_text())
        result["old_tasks_diagnostic"] = evaluate(encoder, head, old_levels, "old-tasks", check)
        result["checks"] = dict(output_bitwise_frozen=tensor_hash(cpu_state(head)) == source_head,
            bounds=encoder.core.bound_audit(), graph_unchanged=encoder.graph_digest() == encoder.base_digest,
            model_reload_complete=after["complete"], source_core_loaded_exactly=True)
        assert result["checks"]["output_bitwise_frozen"] and result["checks"]["graph_unchanged"]
        assert result["checks"]["bounds"]["in_bounds"] and after["complete"]
        result["status"] = "already_solved" if before["solved"] else "learned" if after["solved"] else "budget_exhausted"
        result["model_sha256"] = file_hash(trial / "model.pt")
    except KeyboardInterrupt as error:
        result.update(status="stopped", error=str(error))
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        log("error", **result)
    finally:
        if torch.cuda.is_initialized():
            result.setdefault("resources", {}).update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        result["wall_seconds"] = time.monotonic() - started
        save_json(trial / "result.json", result)
        save_json(trial / "progress.json", dict(status=result["status"], task_id=args.task, seed=args.seed))
        log("trial_finished", task_id=args.task, seed=args.seed, status=result["status"], seconds=result["wall_seconds"])
    return 0 if result["status"] in ("already_solved", "learned", "budget_exhausted") else 1


def verify_suite(args):
    out = Path(args.out).resolve()
    configure(0)
    encoder, head, source, resources = initialize(out)
    levels = json.loads((out / "levels.json").read_text())
    check = lambda: stop_check(out)
    baseline = evaluate(encoder, head, levels, "independent-source", check)
    rows = []
    for path in sorted((out / "trials").glob("*/result.json")):
        result = json.loads(path.read_text())
        if result["status"] not in ("already_solved", "learned", "budget_exhausted"):
            continue
        saved = torch.load(path.parent / "model.pt", map_location="cpu", weights_only=True)
        assert tensor_hash(saved["head"]) == tensor_hash(source["head"])
        assert file_hash(path.parent / "model.pt") == result["model_sha256"]
        head.load_state_dict(saved["head"])
        encoder.set_core(saved["core"])
        level = next(l for l in levels if l["id"] == result["task_id"])
        tested = evaluate(encoder, head, [level], "independent-reload", check)
        before = result["after"]["records"][0]
        now = tested["records"][0]
        exact = before["status"] == now["status"] and before["actions"] == now["actions"]
        logits_equal = [d["brain_logits"] for d in before["decisions"]] == [d["brain_logits"] for d in now["decisions"]]
        if not tested["complete"] or not exact or not logits_equal:
            raise AssertionError(f"Independent reload differs for {path.parent.name}")
        rows.append(dict(trial=path.parent.name, status=now["status"], steps=now["steps"],
                         actions_identical=exact, logits_identical=logits_equal))
        log("independent_verified", **rows[-1])
    expected = len(json.loads((out / "schedule.json").read_text()))
    if len(rows) != expected:
        raise AssertionError("Independent verification requires every planned trial")
    config = json.loads((out / "config.json").read_text())
    original_unchanged = file_hash(ARCHIVE / "campaign/run/final.pt") == config["source_checkpoint_sha256"]
    assert original_unchanged
    save_json(out / "verification.json", dict(status="verified", trials=rows, baseline=baseline,
        original_checkpoint_unchanged=original_unchanged, output_frozen=True,
        graph_unchanged=encoder.graph_digest() == encoder.base_digest, resources=resources))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--verify-suite", action="store_true")
    arguments = parser.parse_args()
    raise SystemExit(verify_suite(arguments) if arguments.verify_suite else run_trial(arguments))
