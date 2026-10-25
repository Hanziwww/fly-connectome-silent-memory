"""Deliverable CSV writer: fixed conventions, schema checks and data-dictionary entries.

Conventions: UTF-8 without BOM, LF line endings, shortest round-trip floats, booleans as
true/false, missing values empty, columns in the declared order, rows in a fixed order.
"""
import csv
import json
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd

from . import paths

COMMON = {
    'unit': 'Statistical unit represented by one row',
    'n': 'Number of units contributing to the row',
    'numerator': 'Count of units meeting the criterion',
    'denominator': 'Count of units evaluated',
    'estimate': 'Point estimate (proportion unless stated otherwise)',
    'ci_low': 'Lower 95% confidence limit',
    'ci_high': 'Upper 95% confidence limit',
    'ci_method': 'Method used for the confidence interval',
    'history_set_id': 'History set (see meta/history_sets.csv) defining the denominator',
    'analysis_status': 're_derived = existing result recomputed from raw records; new_registered = planned in PLAN v0.2; post_hoc = introduced after the inventory',
    'source_file': 'Project-relative path(s) of the input file(s), separated by ";"',
    'test_id': 'Identifier of the corresponding test in stats/statistical_tests.csv',
    'run_id': 'Study run identifier',
    'model': 'Model identifier as used in the study records',
    'map_id': 'Map identifier (tier-family, e.g. D3-T0)',
    'tier': 'Difficulty tier (0-7)',
    'notes': 'Free-text qualification of the row',
    'family': 'Task family (M0, S1, S2, M0vis = shared visible/hidden readout)',
    'visible': 'Goal visible at the decision (false = hidden-goal memory condition)',
    'case': 'Case identifier (c<context>-cue<cue>[-hidden|-visible])',
    'cue': 'Cue index (true cue label)',
    'context': 'Scene orientation (S2: 0 or 180 degrees)',
    'delay_ms': 'Blank delay after the cue (ms)',
    'prefix_windows': 'Common-prefix windows before the decision',
    'split': 'train (training timing) or untrained/held-out timing',
    'status': 'Harmonized end state of the rollout',
    'solved': 'Full solution within 512 actions',
    'steps': 'Moves in the rollout',
}


def _fmt(v):
    if v is None:
        return ''
    if isinstance(v, (bool, np.bool_)):
        return 'true' if v else 'false'
    if isinstance(v, Integral):
        return str(int(v))
    if isinstance(v, Real):
        f = float(v)
        if math.isnan(f):
            return ''
        if math.isinf(f):
            return 'inf' if f > 0 else '-inf'
        return repr(f)
    if v is pd.NA or v is pd.NaT:
        return ''
    if isinstance(v, (list, tuple)):
        return json.dumps(list(v), separators=(',', ':'))
    if isinstance(v, dict):
        return json.dumps(v, separators=(',', ':'), sort_keys=True)
    return str(v)


def _isnull(v):
    if v is None or v is pd.NA:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _check(df, key, relpath):
    problems = []
    if key:
        missing = [k for k in key if k not in df.columns]
        if missing:
            problems.append(f'key columns missing: {missing}')
        else:
            if df[key].isna().any().any():
                problems.append('null values in key columns')
            if df.duplicated(key).any():
                dup = df[df.duplicated(key, keep=False)][key].head(5).to_dict('records')
                problems.append(f'duplicate keys, e.g. {dup}')
    if {'numerator', 'denominator'} <= set(df.columns):
        both = df[['numerator', 'denominator']].dropna()
        if (pd.to_numeric(both['numerator']) > pd.to_numeric(both['denominator'])).any():
            problems.append('numerator > denominator')
    if {'estimate', 'ci_low', 'ci_high'} <= set(df.columns):
        t = df[['estimate', 'ci_low', 'ci_high']].apply(pd.to_numeric, errors='coerce').dropna()
        if ((t['ci_low'] - t['estimate'] > 1e-9) | (t['estimate'] - t['ci_high'] > 1e-9)).any():
            problems.append('estimate outside its confidence interval')
    for c in df.columns:
        if c == 'p_value' or c.startswith('p_') and c not in ('p_method',):
            vals = pd.to_numeric(df[c], errors='coerce').dropna()
            if ((vals < 0) | (vals > 1 + 1e-12)).any():
                problems.append(f'{c} outside [0, 1]')
    if problems:
        raise ValueError(f'{relpath}: ' + '; '.join(problems))


def write_table(df, relpath, *, title, describe, key=None, sort=None, notes=''):
    """Validate and write one deliverable table; register its data-dictionary entry."""
    df = df.reset_index(drop=True)
    descriptions = {**COMMON, **describe}
    undocumented = [c for c in df.columns if c not in descriptions]
    if undocumented:
        raise ValueError(f'{relpath}: undocumented columns {undocumented}')
    _check(df, key, relpath)
    order = sort if sort is not None else key
    if order:
        df = df.sort_values(list(order), kind='mergesort', na_position='last').reset_index(drop=True)
    target = paths.out(relpath)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = list(df.columns)
    with open(target, 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
        w.writerow(columns)
        for row in df.itertuples(index=False, name=None):
            w.writerow([_fmt(v) for v in row])
    entry = dict(file=relpath, title=title, notes=notes, rows=len(df), key=list(key or []),
                 columns=[dict(column=c, dtype=_dtype(df[c]), description=descriptions[c]) for c in columns])
    d = paths.work(f'dictionary/{relpath.replace("/", "__")}.json')
    d.parent.mkdir(parents=True, exist_ok=True)
    d.write_text(json.dumps(entry, indent=1), encoding='utf-8')
    return target


def _dtype(series):
    vals = [v for v in series.tolist() if not _isnull(v)]
    if not vals:
        return 'empty'
    if all(isinstance(v, (bool, np.bool_)) for v in vals):
        return 'boolean'
    if all(isinstance(v, Integral) and not isinstance(v, (bool, np.bool_)) for v in vals):
        return 'integer'
    if all(isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) for v in vals):
        return 'float'
    return 'string'
