"""CPU-certified difficulty ladder with complete optimal-action teachers."""
from workspace import ARCHIVE, RUNS
from collections import deque
import copy
import json
from math import comb
from pathlib import Path
import time

import numpy as np
from bootstrap import HERE
from common import save_json
from game import DELTAS, encode, legal, restore, solved, state_from, step


ROOM8 = ('########', '#......#', '#..#...#', '#..#.#.#', '#....#.#', '##.....#', '#......#', '########')
ROOM10 = ('##########', '#........#', '#.##..##.#', '#.##..##.#', '#...##...#',
          '#...##...#', '#.##..##.#', '#.##..##.#', '#........#', '##########')
SPECS = [
    ('P02', ROOM8, ((1, 6),), 18),
    ('P03', ROOM8, ((1, 6), (6, 1)), 22),
    ('P04', ROOM8, ((1, 6), (6, 1)), 28),
    ('P05', ROOM10, ((1, 8), (8, 1)), 34),
    ('P06', ROOM10, ((1, 8), (8, 1), (8, 8)), 40),
    ('P07', ROOM10, ((1, 8), (8, 1), (8, 8)), 48),
]


def room(rows, goals):
    size = len(rows)
    assert size in (6, 8, 10) and all(len(r) == size for r in rows)
    offset = (10 - size) // 2
    cell = lambda r, c: (r + offset) * 10 + c + offset
    floor = {cell(r, c) for r in range(size) for c in range(size) if rows[r][c] != '#'}
    assert all(rows[0][i] == rows[-1][i] == rows[i][0] == rows[i][-1] == '#' for i in range(size))
    connected, queue = set(), deque([min(floor)])
    while queue:
        p = queue.popleft()
        if p in connected:
            continue
        connected.add(p)
        queue.extend(p + d for d in DELTAS if p + d in floor and p + d not in connected)
    assert connected == floor, 'All advertised floor must be connected'
    targets = sorted(cell(r, c) for r, c in goals)
    assert set(targets) <= floor
    return dict(walls=sorted(set(range(100)) - floor), goals=targets, boxes=len(targets), size=size, split='train')


def reverse_distances(level, max_states=700000, seconds=45):
    """Exact reverse BFS; refuse truncated searches rather than label unknown states."""
    started = time.monotonic()
    m = restore(level)
    floor = set(range(100)) - m['walls']
    goals = tuple(sorted(m['goals']))
    ds = {(p, goals): 0 for p in sorted(floor - set(goals))}
    queue = deque(ds)
    while queue:
        if len(ds) > max_states or time.monotonic() - started > seconds:
            raise RuntimeError('Exact task search reached its explicit CPU bound; no incomplete teacher is exported')
        p, boxes = current = queue.popleft()
        box_set = set(boxes)
        distance = ds[current] + 1
        for d in DELTAS:
            previous_player = p - d
            if previous_player not in floor or previous_player in box_set:
                continue
            previous = (previous_player, boxes)
            if previous not in ds:
                ds[previous] = distance
                queue.append(previous)
            if p + d in box_set:
                previous = (previous_player, tuple(sorted((box_set - {p + d}) | {p})))
                if previous not in ds:
                    ds[previous] = distance
                    queue.append(previous)
    return ds, dict(exact=True, complete=True, solvable_states=len(ds),
        theoretical_states=comb(len(floor), len(goals)) * (len(floor) - len(goals)),
        maximum_distance=max(ds.values()), seconds=time.monotonic() - started)


def teacher(level, ds, max_states=128):
    m, start = restore(level), state_from(level['start'])
    if start not in ds or ds[start] == 0:
        raise ValueError('Expected an unsolved, exactly solvable start')
    queue, known, rows = deque([start]), {start}, []
    while queue:
        state = queue.popleft()
        if solved(m, state):
            continue
        labels = np.array([n is not None and ds.get(n) == ds[state] - 1
                           for n in (step(m, state, a) for a in range(4))], dtype=bool)
        assert labels.any()
        rows.append((state, labels))
        if len(rows) > max_states:
            raise ValueError('Optimal-path teacher exceeds the prespecified state cap')
        for a in np.flatnonzero(labels):
            nxt = step(m, state, int(a))
            if nxt not in known:
                known.add(nxt)
                queue.append(nxt)
    reference, state = [], start
    lookup = {s: labels for s, labels in rows}
    while not solved(m, state):
        action = int(np.flatnonzero(lookup[state])[0])
        reference.append(action)
        state = step(m, state, action)
    min_pushes = {s: 0 for s in known if solved(m, s)}
    for state, labels in sorted(rows, key=lambda r: ds[r[0]]):
        min_pushes[state] = min(int(state[1] != (nxt := step(m, state, int(a)))[1]) + min_pushes[nxt]
                               for a in np.flatnonzero(labels))
    data = dict(boards=np.stack([encode(m, s) for s, _ in rows]).astype('uint8'),
                labels=np.stack([labels for _, labels in rows]),
                legal=np.stack([legal(m, s) for s, _ in rows]),
                player=np.array([s[0] for s, _ in rows], dtype='int64'),
                boxes=np.array([s[1] for s, _ in rows], dtype='int64'),
                remaining=np.array([ds[s] for s, _ in rows], dtype='int64'))
    descriptor = dict(optimal_steps=ds[start], minimum_pushes_among_shortest_solutions=min_pushes[start],
        teacher_states=len(rows), reference_actions=reference, teacher_complete=True,
        teacher_scope='Every nonterminal state and every optimal action on all shortest solutions from this fixed start',
        floor_cells=100 - len(m['walls']))
    return descriptor, data


def build_tasks():
    old = {l['id']: l for l in json.loads((ARCHIVE / 'boundary/20261001-screen01/levels.json').read_text())}
    levels, datasets, certificates = [], {}, {}
    for task_id, source_id in (('P00', 'B00'), ('P01', 'B02')):
        level = copy.deepcopy(old[source_id])
        level.update(id=task_id, source_task=source_id)
        ds, certificate = reverse_distances(level)
        descriptor, data = teacher(level, ds)
        level.update(descriptor)
        levels.append(level)
        datasets[task_id], certificates[task_id] = data, certificate
    cache = {}
    for index, (task_id, rows, goals, target) in enumerate(SPECS):
        level = room(rows, goals)
        key = (rows, goals)
        if key not in cache:
            cache[key] = reverse_distances(level)
        ds, certificate = cache[key]
        candidates = sorted(s for s, d in ds.items() if d == target and not set(s[1]) & set(level['goals']))
        order = np.random.default_rng(100100 + index).permutation(len(candidates))
        rejected = 0
        for candidate_index in order[:500]:
            state = candidates[candidate_index]
            proposed = dict(level, id=task_id, name='difficulty_ladder', start=[state[0], list(state[1])])
            try:
                descriptor, data = teacher(proposed, ds)
            except ValueError:
                rejected += 1
                continue
            proposed.update(descriptor)
            levels.append(proposed)
            datasets[task_id] = data
            certificates[task_id] = dict(certificate, candidate_count=len(candidates), rejected_for_teacher_size=rejected,
                selection='First state in a fixed seeded permutation with <=128 exact teacher states; no model evaluation')
            break
        else:
            raise RuntimeError(f'No certified candidate for {task_id}: target={target}, available={len(candidates)}')
    for previous, current in zip(levels, levels[1:]):
        a = (previous['boxes'], previous['size'], previous['optimal_steps'])
        b = (current['boxes'], current['size'], current['optimal_steps'])
        assert all(x <= y for x, y in zip(a, b)) and a != b
        assert current['floor_cells'] >= previous['floor_cells']
    return levels, datasets, certificates


def prepare_tasks(out):
    out = Path(out)
    levels, datasets, certificates = build_tasks()
    for level in levels:
        certificates[level['id']]['independent_forward_bfs'] = forward_certificate(level)
    save_json(out / 'levels.json', levels)
    save_json(out / 'task-certificates.json', certificates)
    for task_id, data in datasets.items():
        np.savez_compressed(out / f'teacher-{task_id}.npz', **data)
    return levels


def forward_certificate(level, max_states=700000, seconds=45):
    """Independent coordinate-based forward BFS; does not reuse game.step or reverse edges."""
    started = time.monotonic()
    walls, goals = set(level['walls']), set(level['goals'])
    start = (int(level['start'][0]), tuple(sorted(level['start'][1])))
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        if len(seen) > max_states or time.monotonic() - started > seconds:
            raise RuntimeError('Independent forward certification exceeded its bound')
        (player, boxes), distance = queue.popleft()
        if set(boxes) == goals:
            assert distance == level['optimal_steps'], 'Forward and reverse shortest distances disagree'
            return dict(verified=True, optimal_steps=distance, visited_states=len(seen),
                        seconds=time.monotonic() - started)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = divmod(player, 10)
            nr, nc = r + dr, c + dc
            if not (0 <= nr < 10 and 0 <= nc < 10):
                continue
            nxt = nr * 10 + nc
            if nxt in walls:
                continue
            moved = set(boxes)
            if nxt in moved:
                br, bc = nr + dr, nc + dc
                if not (0 <= br < 10 and 0 <= bc < 10):
                    continue
                destination = br * 10 + bc
                if destination in walls or destination in moved:
                    continue
                moved.remove(nxt)
                moved.add(destination)
            candidate = (nxt, tuple(sorted(moved)))
            if candidate not in seen:
                seen.add(candidate)
                queue.append((candidate, distance + 1))
    raise AssertionError('Independent forward BFS found the selected task unsolvable')
