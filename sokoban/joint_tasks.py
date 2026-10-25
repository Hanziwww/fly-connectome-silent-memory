"""Behavioral retention inside P01, with one-step reverse lesson expansion."""
import numpy as np
from scipy.optimize import minimize

from bootstrap import HERE
from lesson_tasks import Lessons


class JointLessons(Lessons):
    def __init__(self, level, seed=10):
        super().__init__(level, seed)
        self.depths = list(range(4, len(self.witness) + 1))

    def at_depth(self, depth):
        state = self.witness[-depth][0]
        return dict(self.level, id=f'P01-suffix-{depth}', start=[state[0], list(state[1])])

    def levels(self, stage):
        depth = self.depths[stage]
        protected = [1, 2] + list(range(4, depth))
        return [self.level, self.at_depth(depth)] + [self.at_depth(d) for d in protected]

    def data(self, stage):
        data = super().data(stage)
        # These constraints guide a local response fit. Actual retention is
        # checked by autonomous rollouts, so alternative successful routes pass.
        protected = []
        for i, clause in enumerate(data['clauses']):
            if clause['kind'] != 'positive_witness':
                continue
            state = data['states'][clause['atoms'][0][0]]
            if state == self.critical_state or state in self.learned:
                protected.append(i)
        data['protected_clauses'] = protected
        return data


def decorate(observation):
    result = dict(observation)
    result['critical_held'] = result['critical_margin'] > 0
    result['protected_results'] = {
        r['level_id']: r['status'] == 'solved' for r in result['records'][2:]}
    result['retained'] = result['critical_held'] and all(result['protected_results'].values())
    result['joint_passed'] = result['solved'] or (result['retained'] and result['lesson_solved'])
    return result


def can_replace(candidate, champion):
    if candidate['solved'] or champion['solved']:
        return candidate['solved'] and not champion['solved']
    if not candidate['retained']:
        return False
    return candidate['loss'] < champion['loss'] - 1e-7


def response_coefficients(base_margins, response, weights, protected):
    """Fit a small trust-region update to measured hard-forward margin changes.

    The response columns are half the differences between actual +/- probes,
    already including each direction's parameter scale. No surrogate Jacobian
    is substituted for these measurements. All fitted proposals are then run
    through the real brain and behavioral gates.
    """
    base = np.asarray(base_margins, dtype='float64')
    matrix = np.asarray(response, dtype='float64')
    meaningful = np.abs(base) < 1e6
    weights = np.asarray(weights, dtype='float64') * meaningful
    weights /= max(weights.sum(), 1.)
    protected = np.array([i for i in protected if base[i] > 0 and meaningful[i]], dtype='int64')
    # Preserve a small positive predicted margin while letting excess
    # confidence decrease. This is advisory; true rollouts decide retention.
    floors = np.minimum(base[protected], .03)
    def objective(coefficients):
        prediction = base + matrix @ coefficients
        error = np.maximum(.1 - prediction, 0.)
        value = (weights * error**2).sum() + .002 * (coefficients**2).sum()
        gradient = -2 * matrix.T @ (weights * error) + .004 * coefficients
        return float(value), gradient
    constraints = []
    if len(protected):
        constraints.append(dict(type='ineq',
            fun=lambda c: base[protected] + matrix[protected] @ c - floors,
            jac=lambda c: matrix[protected]))
    result = minimize(objective, np.zeros(matrix.shape[1]), jac=True, method='SLSQP',
                      bounds=[(-1., 1.)] * matrix.shape[1], constraints=constraints,
                      options=dict(maxiter=60, ftol=1e-9))
    valid = bool(np.isfinite(result.x).all()) and bool((np.abs(result.x) <= 1.+1e-7).all()) and all(
        float(np.min(c['fun'](result.x))) >= -1e-6 for c in constraints)
    usable = valid and objective(result.x)[0] < objective(np.zeros(matrix.shape[1]))[0] - 1e-12
    # This local model is an approximation. A feasible improving fit remains
    # useful if the numerical optimizer exhausts its iteration budget; actual
    # neural/behavioral checks still govern every update.
    coefficients = result.x if usable else np.zeros(matrix.shape[1])
    return coefficients, dict(fit_success=bool(usable), optimizer_converged=bool(result.success),
        coefficients=coefficients.tolist(), protected_floors=floors.tolist(),
        predicted_margins=(base + matrix @ coefficients).tolist())
