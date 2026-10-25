"""Test families, declared before any test is computed (the pattern of manuscript/analysis/msa/testplan.py).

A study's code/testplan.py builds one TestPlan and declares every family with ``family()``. Every test must name a
declared family; multiplicity is controlled within a family (Holm or Benjamini-Hochberg) and companion or
descriptive families are reported without adjustment. Families are never pruned after results are seen.

New here, ``posthoc``: a family declared after its data were seen. It can never be primary and its tests are
reported as exploratory. ``record()`` gives the declarations with their hash so a registration can freeze them,
and ``compare()`` reports declarations removed or changed since then (additions are allowed).
"""
import json
import math

from .registry import SCHEMA_VERSION, object_hash

ALPHA = 0.05
CORRECTIONS = ('holm', 'bh', 'none')
FIELDS = ('family', 'panel', 'level', 'correction', 'primary', 'posthoc', 'membership')


class TestPlan:
    """Ordered family declarations of one study."""

    __test__ = False  # not a test case

    def __init__(self, study=None):
        self.study = study
        self.families = []

    def family(self, name, panel, level, correction, primary, membership, posthoc=False):
        """Declare one family. ``primary`` and ``posthoc`` are booleans; a post hoc family cannot be primary."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError('a family needs a name')
        if name in self:
            raise ValueError(f'family {name!r} is declared twice')
        if correction not in CORRECTIONS:
            raise ValueError(f'{name}: correction must be one of {CORRECTIONS}, not {correction!r}')
        if not isinstance(primary, bool) or not isinstance(posthoc, bool):
            raise ValueError(f'{name}: primary and posthoc must be True or False')
        if primary and posthoc:
            raise ValueError(f'{name}: a post hoc family cannot be primary')
        for field, value in (('panel', panel), ('level', level), ('membership', membership)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name}: {field} must be a non-empty string')
        entry = dict(family=name, panel=panel, level=level, correction=correction, primary=primary, posthoc=posthoc,
                     membership=membership)
        self.families.append(entry)
        return entry

    def __contains__(self, name):
        return any(f['family'] == name for f in self.families)

    def __getitem__(self, name):
        for f in self.families:
            if f['family'] == name:
                return f
        raise KeyError(name)

    def __len__(self):
        return len(self.families)

    def inference(self, name):
        """'confirmatory' (primary), 'exploratory' (post hoc) or 'secondary'."""
        f = self[name]
        return 'exploratory' if f['posthoc'] else 'confirmatory' if f['primary'] else 'secondary'

    def summary(self):
        return dict(families=len(self), primary=sum(f['primary'] for f in self.families),
                    posthoc=sum(f['posthoc'] for f in self.families),
                    corrections={c: sum(f['correction'] == c for f in self.families) for c in CORRECTIONS})

    def digest(self):
        return object_hash(self.families)

    def record(self):
        return dict(schema_version=SCHEMA_VERSION, study=self.study, families=[dict(f) for f in self.families],
                    sha256=self.digest())

    def compare(self, record):
        """Families of an earlier ``record()`` that are now missing or different, and families added since."""
        before = {f['family']: f for f in record['families']}
        now = {f['family']: f for f in self.families}
        return dict(removed=sorted(set(before) - set(now)),
                    changed=sorted(k for k in before if k in now and before[k] != now[k]),
                    added=sorted(set(now) - set(before)))

    def adjust(self, tests, alpha=ALPHA):
        """Family-wise adjusted p-values for a list of test rows (dicts with 'test_id', 'family' and 'p_value').

        Raises if a test names an undeclared family or a test id repeats. Rows come back as copies with
        family_correction, family_primary, family_posthoc, inference, p_adjusted and significant_adjusted.
        """
        from .stats_ext import bh, holm  # numpy/scipy only when tests are adjusted; declaring families needs neither
        rows = [dict(t) for t in tests]
        undeclared = sorted({r['family'] for r in rows if r['family'] not in self})
        if undeclared:
            raise ValueError(f'tests in undeclared families: {undeclared}')
        ids = [r['test_id'] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate test ids')
        for r in rows:
            f = self[r['family']]
            r.update(family_correction=f['correction'], family_primary=f['primary'], family_posthoc=f['posthoc'],
                     inference=self.inference(f['family']), p_adjusted=None, significant_adjusted=None)
        for f in self.families:
            members = [r for r in rows if r['family'] == f['family'] and r.get('p_value') is not None
                       and not math.isnan(float(r['p_value']))]
            if not members:
                continue
            p = [float(r['p_value']) for r in members]
            adjusted = holm(p) if f['correction'] == 'holm' else bh(p) if f['correction'] == 'bh' else p
            for r, value in zip(members, adjusted):
                r.update(p_adjusted=float(value), significant_adjusted=bool(value < alpha))
        return rows

    def to_json(self):
        return json.dumps(self.record(), indent=2, ensure_ascii=False) + '\n'
