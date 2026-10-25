"""Gate G0, local half: compare the GPU host's reproduction (runs/g0-reproduce, pulled) with the frozen records.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-supplement-v2\\code\\g0_compare.py

Pass criteria (plan v2, G0):
- P05: the mixed rollout solves the level in the recorded number of moves (result.json, after.original_steps), and the
  mixed and float64 rollouts take the same actions with every logit within 1e-3.
- Memory models: every history at the 9 trained timings has the recorded status, the same actions and every decision's
  logits within 1e-3 of mechanisms-v1/runs/record-<model>/free-after-d<d>-p<p>-behavior.json; the first-decision margin
  between the two cue-specific pushes is reported separately (|difference| <= 1e-3).
Writes studies/brain-supplement-v2/g0-report.json. Reads only; the frozen study is not modified.
For another host or GPU: --runs <pulled g0 folder> --report <report path> (default the first host's names).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-supplement-v2'
RUNS = STUDY / 'runs' / 'g0-reproduce'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
P05 = ROOT / 'runs' / 'linear-joint' / '20261002-p05-01'
TOL = 1e-3


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compare_rows(new, old):
    """Per-history comparison of two lists of memory_rollout records."""
    old_by = {(r['case'], r['delay_ms'], r['prefix_windows']): r for r in old}
    out = []
    for r in new:
        o = old_by.get((r['case'], r['delay_ms'], r['prefix_windows']))
        if o is None:
            out.append(dict(case=r['case'], delay_ms=r['delay_ms'], prefix_windows=r['prefix_windows'], found=False, ok=False))
            continue
        n = min(len(r['decisions']), len(o['decisions']))
        dl = max((float(np.max(np.abs(np.asarray(a['logits']) - np.asarray(b['logits']))))
                  for a, b in zip(r['decisions'][:n], o['decisions'][:n])), default=0.0)
        same_actions = [a['action'] for a in r['decisions']] == [b['action'] for b in o['decisions']]
        ok = r['status'] == o['status'] and same_actions and dl <= TOL
        out.append(dict(case=r['case'], delay_ms=r['delay_ms'], prefix_windows=r['prefix_windows'], found=True,
                        status_new=r['status'], status_recorded=o['status'], decisions_new=len(r['decisions']),
                        decisions_recorded=len(o['decisions']), same_actions=same_actions, max_abs_logit_difference=dl, ok=ok))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', default=str(RUNS))
    ap.add_argument('--report', default=str(STUDY / 'g0-report.json'))
    args = ap.parse_args()
    runs = Path(args.runs)
    report = dict(tolerance=TOL, runs=str(runs), environment=_read(runs / 'environment.json'))
    ok_all = True
    if (runs / 'p05.json').exists():
        p = _read(runs / 'p05.json')
        expected = _read(P05 / 'result.json')['after']['original_steps']
        mx, f64 = p['mixed'], p['float64']
        n = min(len(mx['decisions']), len(f64['decisions']))
        dl = max(float(np.max(np.abs(np.asarray(a['logits']) - np.asarray(b['logits']))))
                 for a, b in zip(mx['decisions'][:n], f64['decisions'][:n]))
        same = [d['action'] for d in mx['decisions']] == [d['action'] for d in f64['decisions']]
        ok = mx['status'] == 'solved' and mx['steps'] == expected and same and dl <= TOL
        report['p05'] = dict(status_mixed=mx['status'], steps_mixed=mx['steps'], steps_recorded=expected,
                             status_float64=f64['status'], steps_float64=f64['steps'], same_actions_mixed_float64=same,
                             max_abs_logit_difference_mixed_float64=dl, seconds_mixed=mx['seconds'],
                             seconds_float64=f64['seconds'], ok=ok)
        ok_all &= ok
    report['memory'] = {}
    for f in sorted(runs.glob('memory-*.json')):
        new = _read(f)
        mid = new['model']
        rows = []
        for d, p in sorted({(r['delay_ms'], r['prefix_windows']) for r in new['rows']}):
            old = _read(MECH / 'runs' / f'record-{mid}' / f'free-after-d{d}-p{p}-behavior.json')
            rows.extend(compare_rows([r for r in new['rows'] if (r['delay_ms'], r['prefix_windows']) == (d, p)], old))
        ok = bool(rows) and all(r['ok'] for r in rows)
        report['memory'][mid] = dict(histories=len(rows), ok_histories=sum(r['ok'] for r in rows),
                                     max_abs_logit_difference=max(r.get('max_abs_logit_difference', np.inf) for r in rows),
                                     seconds=new['seconds'], ok=ok, failures=[r for r in rows if not r['ok']][:10])
        ok_all &= ok
    report['models_passing'] = sum(v['ok'] for v in report['memory'].values())
    report['passed'] = bool(ok_all and report['models_passing'] >= 3)
    Path(args.report).write_text(json.dumps(report, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'environment'}, indent=1, default=str)[:4000])
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
