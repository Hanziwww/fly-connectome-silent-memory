"""Contracts preventing false curriculum promotion and damaging replacement."""
from workspace import ARCHIVE, RUNS
import json
import unittest
import numpy as np
from bootstrap import HERE
from joint_tasks import JointLessons, can_replace, decorate, response_coefficients
from run_joint import gates, original_only


def observation(critical, statuses, loss=1.):
    records = [dict(level_id=str(i), status=s, actions=[]) for i,s in enumerate(statuses)]
    return decorate(dict(solved=statuses[0]=='solved', lesson_solved=statuses[1]=='solved',
                         complete=True, critical_margin=critical, records=records, loss=loss))


class JointTests(unittest.TestCase):
    def test_suffix_success_without_critical_is_not_promotion(self):
        obs = observation(-1., ['repeated_state','solved','solved','solved'])
        self.assertFalse(obs['joint_passed'])
        self.assertFalse(obs['retained'])

    def test_lost_previous_suffix_blocks_promotion(self):
        obs = observation(.2, ['repeated_state','solved','solved','repeated_state'])
        self.assertFalse(obs['joint_passed'])

    def test_both_behaviors_in_one_model_allow_promotion(self):
        obs = observation(.001, ['repeated_state','solved','solved','solved'])
        self.assertTrue(obs['joint_passed'])

    def test_lower_loss_with_regression_cannot_replace_champion(self):
        champion = observation(.1, ['repeated_state','repeated_state','solved','solved'])
        bad = observation(-1., ['repeated_state','solved','solved','solved'], loss=.001)
        self.assertFalse(can_replace(bad, champion))
        good = observation(.001, ['repeated_state','repeated_state','solved','solved'], loss=.1)
        self.assertTrue(can_replace(good, champion))

    def test_original_goal_overrides_every_anchor_and_length(self):
        champion = observation(.2, ['repeated_state','solved','solved','solved'], loss=0.)
        solution = observation(-10., ['solved','repeated_state','repeated_state','repeated_state'], loss=99.)
        self.assertTrue(solution['joint_passed'])
        self.assertTrue(can_replace(solution, champion))
        self.assertFalse(can_replace(champion, solution))
        self.assertFalse(can_replace(solution, solution))

    def test_auxiliary_success_is_not_original_success(self):
        obs = observation(.1, ['repeated_state','solved','solved','solved'])
        raw = dict(complete=True, total=4, solved=3, records=obs['records'])
        self.assertFalse(gates(raw, .1)['solved'])
        self.assertEqual(original_only(raw)['solved'], 0)

    def test_measured_response_combines_directions_without_predicted_damage(self):
        base, matrix = np.array([.05,-1.]), np.array([[-1.,-1.],[2.,-2.]])
        coefficients, fit = response_coefficients(base, matrix, [1.,1.], [0])
        self.assertTrue(fit['fit_success'])
        predicted = base + matrix@coefficients
        self.assertGreaterEqual(predicted[0], .03-1e-6)
        self.assertGreater(predicted[1], 0.)
        self.assertTrue(np.all(np.abs(coefficients)<=1.))

    def test_one_step_curriculum_carries_all_prior_anchors(self):
        level = json.loads((ARCHIVE/'lessons/20261001-branch01/level.json').read_text())
        lessons = JointLessons(level)
        self.assertEqual(lessons.depths, list(range(4,15)))
        self.assertEqual([l['id'] for l in lessons.levels(2)],
                         ['P01','P01-suffix-6','P01-suffix-1','P01-suffix-2','P01-suffix-4','P01-suffix-5'])
        self.assertEqual(lessons.levels(10)[1]['start'], level['start'])


if __name__ == '__main__':
    unittest.main()
