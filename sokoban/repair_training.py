"""Two-by-two training-method comparison, with unchanged neural architecture."""
from __future__ import annotations

import math
import time
import json
import numpy as np
import torch

from bootstrap import HERE
from common import BudgetExpired, cpu_state, save_checkpoint, save_json
from boundary_tasks import TrialBudget
from continual import core_proposal, evaluate, subset
from training_support import gradients
from game import restore, state_from, step, solved

KEYS = ("pre_log_gain", "post_log_gain", "threshold_mv")
ARMS = {"A": ("legacy", "legacy"), "B": ("legacy", "behavior"),
        "C": ("block", "legacy"), "D": ("block", "behavior")}
SETTINGS = dict(block_groups=16, pairs=4, ranked_groups=12, gradient_refresh=8,
                radius_initial=[.001, .001, .01], radius_min=[.00001, .00001, .0001],
                radius_max=[.01, .01, .1], branch_budget_fraction=.2,
                branch_candidates=32, branch_correct_drop=1, rollout_every_generations=4,
                focus_fraction=.2, loss_epsilon=1e-6, margin_epsilon=1e-5)


def loss_vectors(q, labels, legal):
    q = q.masked_fill(~legal.bool(), -1e9)
    labels = labels.bool()
    ce = -(labels.float() / labels.sum(1, keepdim=True) * q.log_softmax(1)).sum(1)
    set_loss = q.logsumexp(1) - q.masked_fill(~labels, -1e9).logsumexp(1)
    return q, ce, set_loss


def focus_loss(losses):
    k = max(1, math.ceil(len(losses) * SETTINGS["focus_fraction"]))
    return .5 * losses.mean() + .5 * losses.topk(k).values.mean()


def measure(encoder, head, data, check=lambda: None):
    features = encoder.encode(data["boards"], check)
    with torch.no_grad():
        q = head(torch.as_tensor(features, device="cuda"), correction=False)["brain_logits"]
        labels = torch.as_tensor(data["labels"], device="cuda").bool()
        legal = torch.as_tensor(data["legal"], device="cuda").bool()
        masked, ce, set_loss = loss_vectors(q, labels, legal)
        actions = masked.argmax(1).cpu().numpy()
        good = masked.masked_fill(~labels, -1e9).max(1).values
        bad = masked.masked_fill(labels | ~legal, -1e9).max(1).values
        meaningful = (legal & ~labels).any(1).cpu().numpy()
        margins = (good - bad).cpu().numpy()
        ce_values = ce.cpu().numpy()
        result = dict(loss=float(ce_values.mean()), focus_loss=float(focus_loss(set_loss)),
            set_loss=float(set_loss.mean()), correct=int(data["labels"][np.arange(len(actions)), actions].sum()),
            total=len(actions), actions=actions.tolist(), per_state_loss=ce_values.tolist(),
            per_state_set_loss=set_loss.cpu().tolist(),
            margins=[float(v) if use else None for v, use in zip(margins, meaningful)],
            min_margin=float(margins[meaningful].min()) if meaningful.any() else 0.,
            wrong=np.flatnonzero(~data["labels"][np.arange(len(actions)), actions]).tolist(),
            features=features, logits=q.cpu().numpy())
    return result


def public(result):
    return {k: v for k, v in result.items() if k not in ("features", "logits")}


def objective(result, mode):
    return result["loss" if mode == "legacy" else "focus_loss"]


def better(candidate, incumbent, mode):
    if mode == "legacy":
        return candidate["loss"] < incumbent["loss"] - SETTINGS["loss_epsilon"]
    if candidate["correct"] != incumbent["correct"]:
        return candidate["correct"] > incumbent["correct"]
    change = candidate["min_margin"] - incumbent["min_margin"]
    if abs(change) > SETTINGS["margin_epsilon"]:
        return change > 0
    return candidate["focus_loss"] < incumbent["focus_loss"] - SETTINGS["loss_epsilon"]


def branch_eligible(candidate, incumbent):
    if candidate["correct"] < incumbent["correct"] - SETTINGS["branch_correct_drop"]:
        return False
    critical = min((i for i, x in enumerate(incumbent["margins"]) if x is not None),
                   key=lambda i: incumbent["margins"][i], default=None)
    targeted = critical is not None and candidate["margins"][critical] > incumbent["margins"][critical] + SETTINGS["margin_epsilon"]
    return targeted or candidate["focus_loss"] < incumbent["focus_loss"] - SETTINGS["loss_epsilon"]


class TeacherPolicyCheck:
    """Success-only cache check using model actions, never teacher action choices.

Leaving the known states is inconclusive, not failure. The independent evaluator
still checks arbitrary trajectories at regular intervals and after model reload.
"""
    def __init__(self, level, data):
        self.map, self.start = restore(level), state_from(level["start"])
        self.index = {(int(p), (int(b),)): i for i, (p, b) in enumerate(zip(data["player"], data["box"]))}

    def solved(self, actions):
        state, visited = self.start, set()
        while state in self.index and state not in visited:
            visited.add(state)
            state = step(self.map, state, int(actions[self.index[state]]))
            if state is None:
                return False
            if solved(self.map, state):
                return True
        return False


def core_gradients(encoder, head, data, mode, check):
    if mode == "legacy":
        return gradients(encoder, head, data, check)
    # Determine the global hardest-state set before chunked backward passes.
    base = measure(encoder, head, data, check)
    n = len(data["boards"])
    k = max(1, math.ceil(n * SETTINGS["focus_fraction"]))
    selected = np.argsort(-np.asarray(base["per_state_set_loss"]), kind="stable")[:k]
    weights = np.full(n, .5 / n, dtype="float32")
    weights[selected] += .5 / k
    accum = [torch.zeros_like(p) for p in encoder.core.parameters()]
    for start in range(0, n, encoder.batch):
        check()
        count = min(n - start, encoder.batch)
        boards = np.zeros((encoder.batch, 4, 10, 10), dtype="float32")
        boards[:count] = data["boards"][start:start + count]
        features = encoder.forward(torch.from_numpy(boards).cuda())[:count]
        q = head(features, correction=False)["brain_logits"]
        labels, legal = [torch.as_tensor(data[key][start:start + count], device="cuda") for key in ("labels", "legal")]
        _, _, per_state = loss_vectors(q, labels, legal)
        (per_state * torch.as_tensor(weights[start:start + count], device="cuda")).sum().backward()
        for value, parameter in zip(accum, encoder.core.parameters()):
            value.add_(parameter.grad)
        encoder.release_graph()
    norm = float(torch.sqrt(sum(g.square().sum() for g in accum)))
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite focused surrogate gradient")
    return [g / max(1., norm) for g in accum], norm


class LegacyGenerator:
    """Original candidate construction and RNG consumption, with chosen objective."""
    def __init__(self, state, seed):
        self.rng = np.random.default_rng(seed)
        self.m = {k: torch.zeros_like(v) for k, v in state.items()}
        self.v = {k: torch.zeros_like(v) for k, v in state.items()}
        self.t = 0

    def propose(self, encoder, head, data, state, mode, generation, check, probe):
        proposals, norm = [], None
        if generation % 4 == 1:
            chosen = self.rng.choice(len(data["boards"]), min(32, len(data["boards"])), replace=False)
            grad, norm = core_gradients(encoder, head, subset(data, chosen), mode, check)
            self.t += 1
            direction = {}
            for (key, value), g in zip(state.items(), grad):
                g = g.cpu()
                self.m[key].mul_(.9).add_(g, alpha=.1)
                self.v[key].mul_(.999).addcmul_(g, g, value=.001)
                direction[key] = -(self.m[key] / (1 - .9 ** self.t)) / ((self.v[key] / (1 - .999 ** self.t)).sqrt() + 1e-8)
            for scale in (.001, .0001, .00001, -.001, -.0001, -.00001):
                proposals.append((f"gradient:{scale}", core_proposal(state, direction, scale)))
        for scale in (.0003, .001, .003, .01):
            direction = {k: torch.from_numpy(self.rng.standard_normal(v.shape).astype("float32"))
                         * (10 if k == "threshold_mv" else 1) for k, v in state.items()}
            for sign in (1, -1):
                proposals.append((f"gaussian:{scale}:{sign}", core_proposal(state, direction, sign * scale)))
        for name, proposal in proposals:
            probe(name, proposal)
        return dict(gradient_norm=norm, proposal_generator="legacy")


class BlockGenerator:
    def __init__(self, state, seed):
        self.rng = np.random.default_rng(seed)
        self.scores = np.ones((3, len(next(iter(state.values())))), dtype="float64")
        self.radius = np.array(SETTINGS["radius_initial"], dtype="float64")

    def propose(self, encoder, head, data, state, mode, generation, check, probe):
        norm = None
        if (generation - 1) % SETTINGS["gradient_refresh"] == 0:
            grad, norm = core_gradients(encoder, head, data, mode, check)
            self.scores = np.stack([g.detach().cpu().abs().numpy().astype("float64") for g in grad])
        family = (generation - 1) % 3
        key = KEYS[family]
        score = self.scores[family]
        top = np.argsort(score)[-min(256, len(score)):]
        probability = score[top] + max(float(score[top].max()), 1e-20) * 1e-5
        probability /= probability.sum()
        ranked = self.rng.choice(top, SETTINGS["ranked_groups"], replace=False, p=probability)
        remaining = np.setdiff1d(np.arange(len(score)), ranked)
        random = self.rng.choice(remaining, SETTINGS["block_groups"] - len(ranked), replace=False)
        indices = np.concatenate((ranked, random))
        bounds = (math.log(.8), math.log(1.25)) if family < 2 else (-1., 1.)
        values = state[key].numpy()[indices]
        room = np.maximum(0., np.minimum(values - bounds[0], bounds[1] - values))
        steps = np.minimum(self.radius[family], .8 * room).astype("float32")
        estimate = np.zeros(len(indices), dtype="float64")
        signals = []
        for pair in range(SETTINGS["pairs"]):
            signs = self.rng.choice([-1., 1.], len(indices)).astype("float32")
            direction = {k: torch.zeros_like(v) for k, v in state.items()}
            direction[key][indices] = torch.from_numpy(steps * signs)
            plus = probe(f"block:{key}:pair{pair}:+", core_proposal(state, direction, 1.))
            minus = probe(f"block:{key}:pair{pair}:-", core_proposal(state, direction, -1.))
            difference = objective(plus, mode) - objective(minus, mode)
            signals.append(abs(difference))
            estimate += np.divide(difference * signs, 2 * steps,
                                  out=np.zeros_like(estimate), where=steps > 1e-12)
        estimate /= SETTINGS["pairs"]
        if np.max(np.abs(estimate)) > 1e-12:
            direction = {k: torch.zeros_like(v) for k, v in state.items()}
            direction[key][indices] = torch.from_numpy((-steps * estimate / np.max(np.abs(estimate))).astype("float32"))
            probe(f"block:{key}:response-estimate", core_proposal(state, direction, 1.))
        return dict(gradient_norm=norm, proposal_generator="block", parameter_family=key,
                    group_indices=indices.tolist(), radius=float(self.radius[family]),
                    maximum_pair_signal=float(max(signals)), family_index=family)

    def feedback(self, metadata, improved, outcomes, base):
        i = metadata["family_index"]
        if metadata["maximum_pair_signal"] < 1e-5:
            self.radius[i] *= 2.
        elif improved:
            self.radius[i] *= 1.2
        elif outcomes and all(x["correct"] < base["correct"] for x in outcomes):
            self.radius[i] *= .5
        else:
            self.radius[i] *= .8
        self.radius[i] = np.clip(self.radius[i], SETTINGS["radius_min"][i], SETTINGS["radius_max"][i])


class FoundSolution(Exception):
    pass


class BranchExpired(Exception):
    pass


def fit(encoder, head, level, data, config, arm, seed, out, trial):
    generator_mode, selection_mode = ARMS[arm]
    source = cpu_state(encoder.core)
    frozen_head = cpu_state(head)
    generator = (LegacyGenerator if generator_mode == "legacy" else BlockGenerator)(source, seed)
    incumbent = source
    search = source
    policy_check = TeacherPolicyCheck(level, data)
    budget = TrialBudget(config["training_seconds"], out / "STOP")
    encoder.reset_counts()
    generation = candidates = committed = branches = 0
    branch_used = 0.
    branch_start = None
    branch_until = 0
    incumbent_result = behavior_result = None
    behavior_core = source
    last_commit = 0.
    last_status = 0.
    stop = "training_budget"
    solution = None
    branch_cap = config["training_seconds"] * SETTINGS["branch_budget_fraction"]

    def check():
        budget.check()
        if branch_start is not None and (branch_used + time.monotonic() - branch_start >= branch_cap
                                         or candidates >= branch_until):
            raise BranchExpired()

    def close_branch():
        nonlocal branch_used, branch_start, search
        if branch_start is not None:
            branch_used += time.monotonic() - branch_start
        branch_start = None
        search = incumbent

    def save(path, core, observation, elapsed):
        save_checkpoint(path, dict(core=core, head=frozen_head, arm=arm, seed=seed,
            task_id=level["id"], source_checkpoint_sha256=config["source_checkpoint_sha256"],
            accepted_updates=committed, checkpoint_training_seconds=elapsed,
            output_frozen=True, correction_enabled=False, observation_mode="snapshot",
            experiment="learning_method_repair", teacher=public(observation)))

    def verify_candidate(core, obs, tag):
        nonlocal solution, incumbent, incumbent_result, last_commit, committed, behavior_core, behavior_result
        encoder.set_core(core)
        tested = evaluate(encoder, head, [level], tag, check, [(data["boards"], obs["features"])])
        if tested["complete"] and tested["solved"]:
            incumbent, incumbent_result, solution = core, obs, tested
            behavior_core, behavior_result = core, obs
            committed += 1
            last_commit = time.monotonic() - budget.started
            save(trial / "model.pt", incumbent, incumbent_result, last_commit)
            raise FoundSolution()
        return tested

    try:
        incumbent_result = measure(encoder, head, data, check)
        initial = public(incumbent_result)
        behavior_result = incumbent_result
        save(trial / "model.pt", incumbent, incumbent_result, 0.)
        save(trial / "best-behavior.pt", incumbent, incumbent_result, 0.)
        while True:
            try:
                check()
                generation += 1
                encoder.set_core(search)
                base = measure(encoder, head, data, check)
            except BranchExpired:
                close_branch()
                continue
            origin = search
            outcomes = []
            best_core, best_obs, best_name = origin, base, None
            bridge_core, bridge_obs = None, None

            def probe(name, proposal):
                nonlocal candidates, best_core, best_obs, best_name, bridge_core, bridge_obs, behavior_core, behavior_result
                check()
                encoder.set_core(proposal)
                obs = measure(encoder, head, data, check)
                candidates += 1
                outcomes.append(obs)
                if better(obs, best_obs, selection_mode):
                    best_core, best_obs, best_name = proposal, obs, name
                if better(obs, behavior_result, "behavior"):
                    behavior_core, behavior_result = proposal, obs
                    save(trial / "best-behavior.pt", proposal, obs, time.monotonic() - budget.started)
                if selection_mode == "behavior" and branch_eligible(obs, incumbent_result):
                    if bridge_obs is None or obs["focus_loss"] < bridge_obs["focus_loss"]:
                        bridge_core, bridge_obs = proposal, obs
                with (trial / "candidates.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(candidate=candidates, generation=generation, name=name,
                        branch=branch_start is not None, loss=obs["loss"], focus_loss=obs["focus_loss"],
                        correct=obs["correct"], min_margin=obs["min_margin"], wrong=obs["wrong"],
                        seconds=time.monotonic() - budget.started), allow_nan=False) + "\n")
                if policy_check.solved(obs["actions"]):
                    verify_candidate(proposal, obs, "teacher-cache-success-check")
                return obs

            try:
                metadata = generator.propose(encoder, head, data, origin, selection_mode, generation, check, probe)
                check()
            except BranchExpired:
                close_branch()
                continue
            improved = better(best_obs, incumbent_result, selection_mode)
            if isinstance(generator, BlockGenerator):
                generator.feedback(metadata, better(best_obs, base, selection_mode), outcomes, base)
            if improved:
                incumbent, incumbent_result = best_core, best_obs
                committed += 1
                last_commit = time.monotonic() - budget.started
                save(trial / "model.pt", incumbent, incumbent_result, last_commit)
                if branch_start is not None:
                    branch_used += time.monotonic() - branch_start
                    branch_start = None
                search = incumbent
            elif branch_start is not None:
                if candidates >= branch_until or bridge_core is None:
                    branch_used += time.monotonic() - branch_start
                    branch_start = None
                    search = incumbent
                else:
                    search = bridge_core
            elif (selection_mode == "behavior" and generation % 5 == 0 and bridge_core is not None
                  and branch_used < branch_cap):
                search = bridge_core
                branch_start = time.monotonic()
                branch_until = candidates + SETTINGS["branch_candidates"]
                branches += 1
            else:
                search = incumbent
            elapsed = time.monotonic() - budget.started
            row = dict(generation=generation, candidates=candidates, accepted_updates=committed,
                selected=best_name if improved else None, loss=incumbent_result["loss"],
                focus_loss=incumbent_result["focus_loss"], correct=incumbent_result["correct"],
                min_margin=incumbent_result["min_margin"], best_behavior_correct=behavior_result["correct"],
                branch=branch_start is not None, branches=branches, seconds=elapsed, **metadata, **encoder.counts)
            with (trial / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            save_json(trial / "progress.json", dict(status="training", arm=arm, seed=seed, **row))
            if elapsed - last_status >= 20 or generation == 1:
                print(json.dumps(dict(event="generation", arm=arm, seed=seed, **row)), flush=True)
                last_status = elapsed
            if generation % SETTINGS["rollout_every_generations"] == 0:
                try:
                    tested = verify_candidate(incumbent, incumbent_result, "periodic-autonomous-check")
                except BranchExpired:
                    close_branch()
                    continue
                save_json(trial / "last-rollout.json", tested)
    except FoundSolution:
        stop = "autonomous_success"
    except BudgetExpired:
        pass
    finally:
        if branch_start is not None:
            branch_used += time.monotonic() - branch_start
        encoder.set_core(incumbent)
        if incumbent_result is not None:
            save(trial / "model.pt", incumbent, incumbent_result, last_commit)
            save(trial / "best-behavior.pt", behavior_core, behavior_result, time.monotonic() - budget.started)
    return dict(stop=stop, generations=generation, candidates=candidates, accepted_updates=committed,
        training_seconds=time.monotonic() - budget.started, budget_seconds=config["training_seconds"],
        checkpoint_training_seconds=last_commit, initial=initial, final=public(incumbent_result),
        best_behavior=public(behavior_result), branches=branches, exploration_seconds=branch_used,
        training_compute=dict(encoder.counts), solution=solution)
