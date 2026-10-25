"""X2 (plan v2, Phase 1C; gap G05): two new certified scenes per memory-task family (local, CPU; before any run).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-supplement-v2\\code\\x2_scenes.py

- M family: M1 and M2, registered in the frozen study (tasks/memory.json; the M0 floor plus one or two side cells) but
  never trained or evaluated (gap G05). They are trained with the frozen legacy memory worker, like M0.
- S1 (ring) and S2 (two contexts) families, built with the frozen supplement's solver-certified constructor
  (remote-supplement-20261003/supp_tasks.task_from_geometry, which runs validate_task):
  * the left-right mirror of the scene (every cell (row, col) -> (row, 9 - col): same structure, new input pixels);
  * the first single floor-cell removal, in ascending cell order (start, box and goal cells excluded), whose task still
    certifies with the same optimal solution lengths: a new topology posing the same memory problem. (Single-cell
    additions never changed a solution; removals that did made the two cues unequal in length, a confound.)
Writes tasks/x2-tasks.json (the supplement-format tasks) and prints the summary. Deterministic.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / 'studies' / 'brain-sokoban-v1'
SUPP_SESSION = FROZEN / 'remote-supplement-20261003'
OUT = ROOT / 'studies' / 'brain-supplement-v2' / 'tasks' / 'x2-tasks.json'
S1 = dict(floor=set(range(32, 39)) | {44, 45, 46, 54, 56, 64, 65, 66, 75, 76}, start=65, box=35, goals=(32, 38), contexts=(0,))
S2 = dict(floor={14, 24, 34, 44, 54, 64, 74, 35, 36, 37, 47, 55, 56, 57, 48, 58, 68}, start=47, box=44, goals=(14, 74), contexts=(0, 1))


def mirror(c):
    return 10 * (c // 10) + 9 - c % 10


def steps(task):
    return [c['level']['optimal_steps'] for c in task['cases']]


def main():
    sys.path.insert(0, str(SUPP_SESSION))
    import supp_tasks as st
    frozen = {t['id']: t for t in st.prepare()}
    out = []
    for fam, g in (('S1', S1), ('S2', S2)):
        base_id = 'S1-ring' if fam == 'S1' else 'S2-context'
        base = st.task_from_geometry(f'{fam}-base-check', g['floor'], g['start'], g['box'], g['goals'], g['contexts'])
        if steps(base) != steps(frozen[base_id]):
            raise RuntimeError(f'{fam}: rebuilding the frozen scene changed its solutions')
        m = st.task_from_geometry(f'{fam}-mirror', {mirror(c) for c in g['floor']}, mirror(g['start']), mirror(g['box']),
                                  tuple(mirror(x) for x in g['goals']), g['contexts'])
        out.append(m)
        added = None
        for cell in sorted(g['floor'] - {g['start'], g['box'], *g['goals']}):
            try:
                t = st.task_from_geometry(f'{fam}-minus{cell}', g['floor'] - {cell}, g['start'], g['box'], g['goals'], g['contexts'])
            except (AssertionError, KeyError, StopIteration, ValueError):
                continue
            if steps(t) == steps(base):
                added = t
                break
        if added is None:
            raise RuntimeError(f'{fam}: no single-cell removal keeps the task certified with the same solutions')
        out.append(added)
        print(f"{fam}: base steps {steps(base)}; {m['id']} steps {steps(m)}; {added['id']} steps {steps(added)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding='utf-8', newline='\n')
    print(f'{len(out)} tasks -> {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
