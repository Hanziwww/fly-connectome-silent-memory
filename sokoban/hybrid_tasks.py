"""One new connection plus explicit recovery loss for its required decisions."""
import numpy as np
import torch
from scipy.optimize import minimize

from frontier_tasks import build_data, describe
from outcome_objective import losses_and_margins

TARGET, TEMPERATURE, ANCHOR_FLOOR = .1, .25, .03


def objective(margins, data):
    bridge = margins[data['bridge_clause']]
    entry = TEMPERATURE * torch.nn.functional.softplus((TARGET - bridge) / TEMPERATURE)
    ids = [i for i in data['protected_clauses'] if i != data['bridge_clause']]
    if not ids:
        return entry
    violations = (ANCHOR_FLOOR - margins[ids]).clamp(min=0).square()
    return entry + .5 * violations.mean() + .5 * violations.max()


def numpy_objective(margins, data):
    margins = np.asarray(margins, dtype='float64')
    i = data['bridge_clause']
    x = (TARGET - margins[i]) / TEMPERATURE
    value = TEMPERATURE * np.logaddexp(0, x)
    gradient = np.zeros_like(margins)
    gradient[i] = -np.exp(-np.logaddexp(0, -x))
    ids = np.asarray([j for j in data['protected_clauses'] if j != i], dtype='int64')
    if len(ids):
        positive = np.maximum(ANCHOR_FLOOR - margins[ids], 0)
        violations = positive**2
        weights = np.full(len(ids), .5 / len(ids))
        worst = violations == violations.max()
        weights[worst] += .5 / worst.sum()
        value += .5 * violations.mean() + .5 * violations.max()
        gradient[ids] += -2 * positive * weights
    return float(value), gradient


def loss(q, data):
    _, margins, correct = losses_and_margins(q, data)
    return objective(margins, data), margins, correct


def score(q, data):
    value, margins, correct = loss(q, data)
    index = data['critical_index']
    others = torch.as_tensor(data['legal'][index], device=q.device).bool().clone()
    others[3] = False
    critical = q[index, 3] - q[index].masked_fill(~others, -1e9).max()
    return dict(loss=float(value), margins=margins.cpu().tolist(),
                correct=int(correct.sum()), total=len(margins), min_margin=float(margins.min()),
                bridge_margin=float(margins[data['bridge_clause']]), critical_margin=float(critical))


def response_coefficients(base, response, data):
    base, matrix = np.asarray(base, dtype='float64'), np.asarray(response, dtype='float64')
    protected = np.asarray([i for i in data['protected_clauses'] if 0 < base[i] < 1e6], dtype='int64')
    floors = np.minimum(base[protected], ANCHOR_FLOOR)

    def f(c):
        value, derivative = numpy_objective(base + matrix @ c, data)
        return value + .002 * float(c @ c), matrix.T @ derivative + .004 * c

    constraints = []
    if len(protected):
        constraints = [dict(type='ineq', fun=lambda c: base[protected] + matrix[protected] @ c - floors,
                            jac=lambda c: matrix[protected])]
    zero = np.zeros(matrix.shape[1])
    result = minimize(f, zero, jac=True, method='SLSQP', bounds=[(-1., 1.)] * len(zero),
                      constraints=constraints, options=dict(maxiter=60, ftol=1e-9))
    valid = np.isfinite(result.x).all() and (np.abs(result.x) <= 1 + 1e-7).all()
    valid = bool(valid and all(float(c['fun'](result.x).min()) >= -1e-6 for c in constraints))
    usable = valid and f(result.x)[0] < f(zero)[0] - 1e-12
    coefficients = result.x if usable else zero
    return coefficients, dict(usable=bool(usable), converged=bool(result.success),
                              predicted_margins=(base + matrix @ coefficients).tolist())


def dataset(level, snapshot, depth):
    lessons, data, levels = build_data(level, snapshot, 'frontier', 'softplus', depth)
    data['hybrid_loss'] = 'softplus_entry_plus_squared_anchor_recovery'
    return lessons, data, levels


def descriptor(data):
    return dict(describe(data), hybrid_loss=data['hybrid_loss'], anchor_floor=ANCHOR_FLOOR,
                negative_anchors_receive_repair_loss=True, shortest_path_acceptance=False)
