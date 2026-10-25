"""P7: method development and formal lineage (ED1), precision calibration and validation checks (ED2)."""
import json
import re

import numpy as np
import pandas as pd

from msa import csvout, io, paths, static_runs


def ed01():
    cp = paths.ARCHIVE / 'experiment-catalog.json'
    cat = io.json_load(cp)
    rows = []
    for u in cat['units']:
        for i, r in enumerate(u.get('results') or []):
            tr = r.get('training') or {}
            fe = r.get('first_evaluation_record') or {}
            rows.append(dict(record='archive_result', unit=u['id'], index=i, classification=u.get('classification'), status=r.get('status'),
                             arm=r.get('arm'), seed=r.get('seed'), task=r.get('task_id') or fe.get('level_id'),
                             training_seconds=tr.get('training_seconds') if isinstance(tr, dict) else None,
                             stop=tr.get('stop') if isinstance(tr, dict) else None, generations=tr.get('generations') if isinstance(tr, dict) else None,
                             candidates=tr.get('candidates') if isinstance(tr, dict) else None, first_status=fe.get('status'), first_steps=fe.get('steps'),
                             stage=None, method=None, recorded_result=None, path=r.get('path'), source_file=paths.rel(cp)))
    hp = paths.DOCS / 'experiment-history.md'
    lines = io.text(hp).splitlines()
    table = [l for l in lines if l.startswith('|') and not l.startswith('|---')]
    for i, l in enumerate(table[1:]):
        cells = [c.strip() for c in l.strip('|').split('|')]
        if len(cells) < 4:
            continue
        rows.append(dict(record='method_stage', unit='docs/experiment-history.md', index=i, classification='historical', status=None, arm=None, seed=None, task=None,
                         training_seconds=None, stop=None, generations=None, candidates=None, first_status=None, first_steps=None,
                         stage=cells[0], method=cells[1], recorded_result=cells[2], path=re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', cells[3]),
                         source_file=paths.rel(hp)))
    df = pd.DataFrame(rows)
    for c in ('generations', 'candidates', 'first_steps', 'seed'):
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    csvout.write_table(df, 'extended/ed01/ed01a_method_development.csv', title='Method development: archived attempts and the 14-stage progression', describe={
        'record': 'archive_result (one archived training record) or method_stage (row of the method-progression table)', 'unit': 'Archive unit',
        'index': 'Record index within the unit or table', 'classification': 'Archive classification', 'status': 'Recorded status', 'arm': 'Arm',
        'seed': 'Seed', 'task': 'Task', 'training_seconds': 'Training seconds', 'stop': 'Stop reason', 'generations': 'Generations (population methods)',
        'candidates': 'Candidates evaluated', 'first_status': 'Status of the first evaluation record', 'first_steps': 'Moves of the first evaluation record',
        'stage': 'Method stage', 'method': 'Question and method', 'recorded_result': 'Recorded result', 'path': 'Evidence path or link text'},
        key=['record', 'unit', 'index'], sort=['record', 'unit', 'index'])
    rows = []
    for name in ('20261002-p03-01', '20261002-p05-01'):
        p = paths.LINEAGE / name / 'result.json'
        r = io.json_load(p)
        tr = r.get('training') or {}
        tc = tr.get('training_compute') or {}
        src = paths.rel(p)
        rows.append(dict(model=name, item='status', value=r.get('status'), source_file=src))
        for k in ('stop', 'joint_updates', 'training_seconds', 'depth', 'assessments'):
            rows.append(dict(model=name, item=f'training.{k}', value=tr.get(k), source_file=src))
        rows.append(dict(model=name, item='training.promotions', value=len(tr.get('promotions') or []), source_file=src))
        for k, v in tc.items():
            rows.append(dict(model=name, item=f'training_compute.{k}', value=v, source_file=src))
        for suite in ('before_suite', 'after_suite'):
            s = r.get(suite) or {}
            rows.append(dict(model=name, item=f'{suite}.solved', value=s.get('solved'), source_file=src))
            rows.append(dict(model=name, item=f'{suite}.total', value=s.get('total'), source_file=src))
            for rec in s.get('records', []):
                rows.append(dict(model=name, item=f'{suite}.{rec.get("level_id")}', value=f'{rec.get("status")}:{rec.get("steps")}', source_file=src))
        aft = r.get('after') or {}
        rows.append(dict(model=name, item='after.original_status', value=aft.get('original_status'), source_file=src))
        rows.append(dict(model=name, item='after.original_steps', value=aft.get('original_steps'), source_file=src))
        for label, suite in (r.get('substitutions') or {}).items():
            recs = (suite or {}).get('records') or [{}]
            rows.append(dict(model=name, item=f'substitution.{label}', value=f'{recs[0].get("status")}:{recs[0].get("steps")}', source_file=src))
        old = r.get('old_task_after') or {}
        recs = old.get('records') or []
        if recs:
            rows.append(dict(model=name, item=f'old_task_after.{recs[0].get("level_id")}', value=f'{recs[0].get("status")}:{recs[0].get("steps")}',
                             source_file=src))
        for k, v in (r.get('checks') or {}).items():
            rows.append(dict(model=name, item=f'checks.{k}', value=v, source_file=src))
    df = pd.DataFrame(rows)
    df['value'] = df['value'].map(lambda v: v if isinstance(v, str) else json.dumps(v))
    df.insert(1, 'row', range(len(df)))
    csvout.write_table(df, 'extended/ed01/ed01b_formal_lineage.csv', title='Formal P03/P05 lineage: training, suites and saved-pair substitutions', describe={
        'row': 'Row index', 'item': 'Recorded item (substitution rows: pairing -> status:moves)', 'value': 'Value (JSON for non-strings)'}, key=['row'])


def ed02():
    bp = paths.STUDY / 'benchmark.json'
    b = io.json_load(bp)
    be_p = paths.STUDY / 'backend.json'
    backend = io.json_load(be_p)
    rows = []
    for c in b['candidates']:
        rows.append(dict(record='candidate', id=c['id'], precision=c['precision'], batch=c['batch'], eligible=c['eligible'],
                         task=None, status=None, updates=None, training_seconds=c['total_training_seconds'], wall_seconds=c['total_wall_seconds'],
                         selected=(c['id'] == backend.get('selected') or c['id'] == (backend.get('backend') or {}).get('id')), source_file=paths.rel(bp)))
    for r in b['results']:
        cfg = r.get('config') or {}
        rows.append(dict(record='run', id=r.get('id'), precision=cfg.get('precision'), batch=cfg.get('batch'), eligible=None, task=cfg.get('task'),
                         status=r.get('status'), updates=r.get('updates'), training_seconds=r.get('training_seconds'), wall_seconds=r.get('wall_seconds'),
                         selected=None, source_file=paths.rel(bp)))
    df = pd.DataFrame(rows)
    df['updates'] = pd.to_numeric(df['updates'], errors='coerce').astype('Int64')
    df['backend_record'] = json.dumps(backend, sort_keys=True)[:500]
    csvout.write_table(df, 'extended/ed02/ed02a_precision_calibration.csv', title='Precision calibration: candidates and runs', describe={
        'record': 'candidate (aggregate) or run', 'id': 'Candidate or run identifier', 'precision': 'Numerical precision', 'batch': 'Boards per call',
        'eligible': 'Both calibration tasks learned and reloaded, with passing recomputation', 'task': 'Calibration task', 'status': 'Run status',
        'updates': 'Updates', 'training_seconds': 'Training seconds (candidate: total over both tasks)', 'wall_seconds': 'Wall seconds',
        'selected': 'Selected backend', 'backend_record': 'backend.json content (truncated JSON)'}, key=['record', 'id'])
    rows = []
    for f in ('validation.json', 'validation-float64.json', 'validation-mixed.json'):
        p = paths.STUDY / f
        v = io.json_load(p)
        for k, val in v.items():
            if k in ('resume_records', 'dynamic_training_steps'):
                val = len(val) if isinstance(val, list) else val
                k = f'{k}_count'
            rows.append(dict(record='validation', check_file=f, item=k, value=json.dumps(val) if not isinstance(val, str) else val, run_id=None,
                             max_recompute_error=None, tolerance=None, updates_checked=None, duplicates_removed=None, source_file=paths.rel(p)))
    for f in ('speed-probe.json', 'speed-probe-segmented.json'):
        p = paths.STUDY / f
        v = io.json_load(p)
        items = v if isinstance(v, list) else [v]
        for i, it in enumerate(items):
            rows.append(dict(record='speed_probe', check_file=f, item=f'probe_{i}', value=json.dumps(it, sort_keys=True), run_id=None,
                             max_recompute_error=None, tolerance=None, updates_checked=None, duplicates_removed=None, source_file=paths.rel(p)))
    for rid in static_runs.run_ids():
        g = static_runs.gradients(rid)
        if not g:
            continue
        last = {}
        for row in g:
            last[row['update']] = row
        errs = [row.get('recompute_max_error') for row in last.values() if row.get('recompute_max_error') is not None]
        tol = max((row.get('recompute_tolerance') or 0) for row in last.values())
        rows.append(dict(record='recompute_audit', check_file='gradients.jsonl', item='max_recompute_error', value=None, run_id=rid,
                         max_recompute_error=float(max(errs)) if errs else None, tolerance=tol, updates_checked=len(last),
                         duplicates_removed=len(g) - len(last), source_file=paths.rel(paths.RUNS / rid / 'gradients.jsonl')))
    df = pd.DataFrame(rows)
    for c in ('updates_checked', 'duplicates_removed'):
        df[c] = df[c].astype('Int64')
    df.insert(0, 'row', range(len(df)))
    # Benchmarks are reported, not gated: the FP32 candidates were rejected precisely for failing recomputation.
    bad = df[(df.record == 'recompute_audit') & (df.max_recompute_error > df.tolerance) & ~df.run_id.fillna('').str.startswith('benchmark-')]
    df['within_tolerance'] = np.where(df.record == 'recompute_audit', df.max_recompute_error <= df.tolerance, None)
    csvout.write_table(df, 'extended/ed02/ed02b_validation_checks.csv', title='Validation checks and the per-update recomputation audit', describe={
        'row': 'Row index', 'record': 'validation, speed_probe or recompute_audit', 'check_file': 'Source file', 'item': 'Check',
        'value': 'Value (JSON)', 'max_recompute_error': 'Maximum feature recomputation error over all updates (gradient-pass consistency)',
        'tolerance': 'Recorded tolerance', 'updates_checked': 'Distinct updates audited', 'duplicates_removed': 'Duplicated update records removed (resume overlap)',
        'within_tolerance': 'Maximum error within tolerance'}, key=['row'])
    if len(bad):
        raise RuntimeError(f'recompute audit: tolerance exceeded in {bad.run_id.tolist()}')


def main():
    ed01()
    ed02()
