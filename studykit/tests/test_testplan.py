"""TestPlan: declarations, the post hoc flag, family-wise adjustment and frozen-record comparison."""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from studykit.testplan import TestPlan  # noqa: E402


def make_plan():
    plan = TestPlan('demo-study')
    plan.family('X1.primary', 'x1', 'R', 'holm', True, 'Memory use per family, exact sign-flip over replicates')
    plan.family('X1.secondary', 'x1', 'R', 'bh', False, 'Every other contrast of X1')
    plan.family('X1.describe', 'x1', 'R', 'none', False, 'Descriptive estimates')
    plan.family('X6.posthoc', 'x6', 'C', 'bh', False, 'Added after the data were seen', posthoc=True)
    return plan


class Declarations(unittest.TestCase):
    def test_declare_and_summarise(self):
        plan = make_plan()
        self.assertEqual(len(plan), 4)
        self.assertEqual(plan.summary(), dict(families=4, primary=1, posthoc=1,
                                              corrections=dict(holm=1, bh=2, none=1)))
        self.assertEqual([plan.inference(f['family']) for f in plan.families],
                         ['confirmatory', 'secondary', 'secondary', 'exploratory'])
        self.assertEqual(plan['X6.posthoc']['posthoc'], True)
        self.assertEqual(plan.record()['sha256'], make_plan().digest())

    def test_invalid_declarations(self):
        plan = make_plan()
        bad = [dict(name='X1.primary'), dict(name='new', correction='bonferroni'), dict(name='new', primary=1),
               dict(name='new', primary=True, posthoc=True), dict(name='new', membership=''), dict(name=' ')]
        for change in bad:
            args = dict(name='new', panel='p', level='R', correction='holm', primary=False, membership='m')
            args.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                plan.family(**args)

    def test_compare_with_a_frozen_record(self):
        frozen = make_plan().record()
        plan = make_plan()
        plan.families[1]['membership'] = 'pruned after results'
        plan.families.pop(2)
        plan.family('X2.new', 'x2', 'R', 'holm', False, 'Declared later, before its tests')
        self.assertEqual(plan.compare(frozen), dict(removed=['X1.describe'], changed=['X1.secondary'],
                                                    added=['X2.new']))

    def test_study_test_plans_are_valid(self):
        """The plan v2 studies declare their families through their own TestPlan; none is both primary and post hoc."""
        for name in ('brain-supplement-v2', 'brain-pathology-v1'):
            path = ROOT / 'studies' / name / 'code' / 'testplan.py'
            spec = importlib.util.spec_from_file_location(f'scaffold_{name.replace("-", "_")}', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.PLAN.study, name)
            self.assertIs(module.family.__self__, module.PLAN)
            self.assertEqual([f['family'] for f in module.FAMILIES if f['primary'] and f['posthoc']], [])
            self.assertEqual(len({f['family'] for f in module.FAMILIES}), len(module.FAMILIES))


class Adjustment(unittest.TestCase):
    def test_holm_bh_and_none_within_families(self):
        plan = make_plan()
        primary = [0.01, 0.04, 0.03]
        secondary = [0.01, 0.02, 0.03, 0.2]
        tests = [dict(test_id=f'p{i}', family='X1.primary', p_value=p) for i, p in enumerate(primary)]
        tests += [dict(test_id=f's{i}', family='X1.secondary', p_value=p) for i, p in enumerate(secondary)]
        tests += [dict(test_id='d0', family='X1.describe', p_value=0.03), dict(test_id='d1', family='X1.describe',
                                                                              p_value=None),
                  dict(test_id='e0', family='X6.posthoc', p_value=0.001), dict(test_id='nan', family='X1.primary',
                                                                               p_value=float('nan'))]
        rows = {r['test_id']: r for r in plan.adjust(tests)}
        self.assertTrue(np.allclose([rows[f'p{i}']['p_adjusted'] for i in range(3)], [0.03, 0.06, 0.06]))
        self.assertTrue(np.allclose([rows[f's{i}']['p_adjusted'] for i in range(4)],
                                    st.false_discovery_control(secondary, method='bh')))
        self.assertEqual(rows['d0']['p_adjusted'], 0.03)
        self.assertIsNone(rows['d1']['p_adjusted'])
        self.assertIsNone(rows['nan']['p_adjusted'])
        self.assertEqual((rows['p0']['significant_adjusted'], rows['p1']['significant_adjusted']), (True, False))
        self.assertEqual((rows['e0']['inference'], rows['e0']['family_posthoc']), ('exploratory', True))
        self.assertEqual(rows['p0']['inference'], 'confirmatory')
        self.assertTrue(math.isnan(tests[-1]['p_value']), 'input rows must not be modified')
        self.assertNotIn('p_adjusted', tests[0])

    def test_undeclared_family_and_duplicate_ids(self):
        plan = make_plan()
        with self.assertRaisesRegex(ValueError, 'undeclared'):
            plan.adjust([dict(test_id='a', family='X9', p_value=0.1)])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            plan.adjust([dict(test_id='a', family='X1.primary', p_value=0.1)] * 2)


if __name__ == '__main__':
    unittest.main()
