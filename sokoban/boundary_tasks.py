"""Exact, CPU-only task construction for the first single-task difficulty screen."""
from workspace import ARCHIVE, RUNS
import copy
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from bootstrap import HERE
from common import BudgetExpired, save_json
from game import restore, state_from, step, solved, encode, legal


# Order is fixed before model evaluation. Coordinates use the existing 10x10 canvas.
CANDIDATES = [
    ("B00", "long_walk", "C002", (64, (65,)), 11, 1),
    ("B01", "two_pushes", "C001", (36, (45,)), 9, 2),
    ("B02", "two_pushes_long_walk", "C000", (63, (44,)), 14, 2),
    ("B03", "three_pushes", "C003", (56, (54,)), 10, 3),
    ("B04", "four_pushes", "C001", (34, (54,)), 11, 4),
    ("B05", "forced_away", "C001", (36, (35,)), 7, 3),
]


class TrialBudget:
    def __init__(self, seconds, stop_path, clock=time.monotonic):
        self.clock, self.stop_path = clock, Path(stop_path)
        self.started = clock()
        self.deadline = self.started + seconds

    def check(self):
        if self.stop_path.exists():
            raise KeyboardInterrupt("Boundary STOP file found")
        if self.clock() >= self.deadline:
            raise BudgetExpired("Single-task training budget reached")


def exact_graph(level):
    """Enumerate the entire one-box state space, including losing states."""
    m = restore(level)
    floor = sorted(set(range(100)) - m["walls"])
    if len(m["goals"]) != 1 or len(floor) > 64:
        raise ValueError("This exact teacher is restricted to small one-box tasks")
    states = [(p, (b,)) for p in floor for b in floor if p != b]
    predecessors = {s: [] for s in states}
    edges = {}
    for s in states:
        edges[s] = []
        for action in range(4):
            nxt = step(m, s, action)
            if nxt is not None:
                if nxt not in predecessors:
                    raise ValueError("Transition leaves the enumerated state space")
                weight = int(s[1] != nxt[1])
                edges[s].append((action, nxt, weight))
                predecessors[nxt].append((s, weight))
    goals = [s for s in states if solved(m, s)]
    distances = {s: 0 for s in goals}
    queue = deque(goals)
    while queue:
        t = queue.popleft()
        for s, _ in predecessors[t]:
            if s not in distances:
                distances[s] = distances[t] + 1
                queue.append(s)
    pushes = {s: 0 for s in goals}
    queue = deque(goals)
    while queue:
        t = queue.popleft()
        for s, weight in predecessors[t]:
            value = pushes[t] + weight
            if value < pushes.get(s, float("inf")):
                pushes[s] = value
                (queue.append if weight else queue.appendleft)(s)
    target = m["goals"][0]
    def manhattan(state):
        box = state[1][0]
        return abs(box // 10 - target // 10) + abs(box % 10 - target % 10)
    monotone = set(goals)
    queue = deque(goals)
    while queue:
        t = queue.popleft()
        for s, weight in predecessors[t]:
            if s not in monotone and (not weight or manhattan(t) <= manhattan(s)):
                monotone.add(s)
                queue.append(s)
    return m, edges, distances, pushes, monotone


def describe(level):
    m, edges, ds, pushes, monotone = exact_graph(level)
    start = state_from(level["start"])
    if start not in ds or solved(m, start):
        raise ValueError("Task must be solvable and initially unsolved")
    queue, discovered, training = deque([start]), {start}, []
    while queue:
        s = queue.popleft()
        if solved(m, s):
            continue
        labels = np.zeros(4, dtype=bool)
        for action, nxt, _ in edges[s]:
            if ds.get(nxt) == ds[s] - 1:
                labels[action] = True
                if nxt not in discovered:
                    discovered.add(nxt)
                    queue.append(nxt)
        if not labels.any():
            raise AssertionError("Missing optimal action in a solvable state")
        training.append((s, labels))
    reference, s = [], start
    while not solved(m, s):
        action, nxt, _ = next(e for e in edges[s] if ds.get(e[1]) == ds[s] - 1)
        reference.append(action)
        s = nxt
    data = dict(
        boards=np.stack([encode(m, s) for s, _ in training]).astype("uint8"),
        labels=np.stack([label for _, label in training]),
        legal=np.stack([legal(m, s) for s, _ in training]),
        remaining=np.array([ds[s] for s, _ in training], dtype="int64"),
        player=np.array([s[0] for s, _ in training], dtype="int64"),
        box=np.array([s[1][0] for s, _ in training], dtype="int64"),
    )
    descriptor = dict(optimal_steps=ds[start], minimum_pushes=pushes[start],
        requires_push_away=start not in monotone, reference_actions=reference,
        teacher_states=len(training), all_states=len(edges), solvable_states=len(ds),
        teacher_complete=True, teacher_scope="All nonterminal states on any shortest solution from the fixed start")
    return descriptor, data


def build_tasks():
    sources = {l["id"]: l for l in json.loads((ARCHIVE / "campaign/pilot-levels.json").read_text())}
    levels, datasets = [], {}
    for task_id, name, source, start, moves, pushes in CANDIDATES:
        old = sources[source]
        level = dict(id=task_id, name=name, source_layout=source, walls=copy.deepcopy(old["walls"]),
                     goals=list(old["goals"]), start=[start[0], list(start[1])],
                     boxes=1, size=6, split="train")
        descriptor, data = describe(level)
        assert descriptor["optimal_steps"] == moves
        assert descriptor["minimum_pushes"] == pushes
        assert descriptor["requires_push_away"] == (task_id == "B05")
        level.update(descriptor)
        levels.append(level)
        datasets[task_id] = data
    return levels, datasets


def prepare_tasks(out):
    out = Path(out)
    levels, datasets = build_tasks()
    save_json(out / "levels.json", levels)
    for task_id, data in datasets.items():
        np.savez_compressed(out / f"teacher-{task_id}.npz", **data)
    return levels


def load_task(out, task_id):
    levels = json.loads((Path(out) / "levels.json").read_text())
    level = next(l for l in levels if l["id"] == task_id)
    with np.load(Path(out) / f"teacher-{task_id}.npz", allow_pickle=False) as saved:
        data = {k: saved[k] for k in saved.files}
    return level, data
