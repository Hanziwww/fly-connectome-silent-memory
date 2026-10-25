"""Frozen-feedback objectives for the paired P01 frontier experiment."""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from bootstrap import HERE
from joint_tasks import JointLessons
from outcome_objective import losses_and_margins

ARMS = {
    'A': dict(scope='mixed', penalty='softplus'),
    'B': dict(scope='frontier', penalty='softplus'),
    'C': dict(scope='mixed', penalty='huber'),
    'D': dict(scope='frontier', penalty='huber'),
}
KAPPA, DELTA, TEMPERATURE = .1, .1, .25


def restore_lessons(level, snapshot):
    lessons = JointLessons(level, 10)
    state = lambda s: (int(s[0]), tuple(s[1]))
    lessons.learned = {state(s): set(a) for s, a in snapshot['learned']}
    lessons.corrections = {state(s): int(a) for s, a in snapshot['corrections']}
    lessons.failures = [
        dict(kind=c['kind'], atoms=[(state(s), tuple(a)) for s, a in c['atoms']])
        for c in snapshot['failures']]
    lessons.failure_keys = {(c['kind'], tuple(c['atoms'])) for c in lessons.failures}
    lessons.version = 12
    return lessons


def build_data(level, snapshot, scope, penalty, depth=5):
    lessons = restore_lessons(level, snapshot)
    stage = lessons.depths.index(depth)
    original = lessons.data(stage)
    bridge_state = lessons.witness[-depth][0]
    bridge = next(i for i, c in enumerate(original['clauses'])
                  if c['kind'] == 'positive_witness'
                  and original['states'][c['atoms'][0][0]] == bridge_state)
    active = list(range(len(original['clauses']))) if scope == 'mixed' else [bridge]
    selected = sorted(set(active) | set(original['protected_clauses']) | {bridge})
    clause_ids = {old: new for new, old in enumerate(selected)}
    state_ids = sorted({s for i in selected for s, _ in original['clauses'][i]['atoms']})
    state_map = {old: new for new, old in enumerate(state_ids)}
    clauses = [dict(kind=original['clauses'][i]['kind'],
                    atoms=[(state_map[s], tuple(a)) for s, a in original['clauses'][i]['atoms']])
               for i in selected]
    data = dict(boards=original['boards'][state_ids], legal=original['legal'][state_ids],
                states=[original['states'][s] for s in state_ids], clauses=clauses,
                weights=[original['weights'][i] for i in selected],
                objective_indices=[clause_ids[i] for i in active],
                protected_clauses=[clause_ids[i] for i in original['protected_clauses']],
                bridge_clause=clause_ids[bridge],
                critical_index=state_map[original['critical_index']],
                version=12, scope=scope, penalty=penalty, depth=depth)
    assert data['states'][data['critical_index']] == lessons.critical_state
    return lessons, data, lessons.levels(stage)


def torch_objective(margins, data):
    ids = torch.as_tensor(data['objective_indices'], device=margins.device)
    selected = margins[ids]
    x = KAPPA - selected
    if data['penalty'] == 'softplus':
        penalties = TEMPERATURE * F.softplus(x / TEMPERATURE)
    elif data['penalty'] == 'huber':
        positive = x.clamp(min=0)
        penalties = torch.where(positive <= DELTA, positive.square() / (2 * DELTA),
                                positive - DELTA / 2)
    else:
        raise ValueError(data['penalty'])
    weights = torch.as_tensor(data['weights'], device=margins.device, dtype=margins.dtype)[ids]
    return .75 * (weights * penalties).sum() / weights.sum() + .25 * penalties.max()


def numpy_objective(margins, data):
    """Return the same task objective and its margin derivative for local fitting."""
    margins = np.asarray(margins, dtype='float64')
    ids = np.asarray(data['objective_indices'], dtype='int64')
    x = KAPPA - margins[ids]
    if data['penalty'] == 'softplus':
        penalties = TEMPERATURE * np.logaddexp(0, x / TEMPERATURE)
        slopes = np.exp(-np.logaddexp(0, -x / TEMPERATURE))
    elif data['penalty'] == 'huber':
        positive = np.maximum(x, 0)
        penalties = np.where(positive <= DELTA, positive**2 / (2 * DELTA), positive - DELTA / 2)
        slopes = np.clip(x / DELTA, 0, 1)
    else:
        raise ValueError(data['penalty'])
    weights = np.asarray(data['weights'], dtype='float64')[ids]
    coefficients = .75 * weights / weights.sum()
    worst = penalties == penalties.max()
    coefficients[worst] += .25 / int(worst.sum())
    gradient = np.zeros_like(margins)
    gradient[ids] = -coefficients * slopes
    return float(.75 * (weights * penalties).sum() / weights.sum() + .25 * penalties.max()), gradient


def loss(q, data):
    _, margins, correct = losses_and_margins(q, data)
    return torch_objective(margins, data), margins, correct


def score(q, data):
    value, margins, correct = loss(q, data)
    ids = torch.as_tensor(data['objective_indices'], device=q.device)
    index = data['critical_index']
    competitors = torch.as_tensor(data['legal'][index], device=q.device).bool().clone()
    competitors[3] = False
    critical = q[index, 3] - q[index].masked_fill(~competitors, -1e9).max()
    return dict(loss=float(value), min_margin=float(margins[ids].min()),
                correct=int(correct[ids].sum()), total=len(ids), margins=margins.cpu().tolist(),
                critical_margin=float(critical), bridge_margin=float(margins[data['bridge_clause']]))


def response_coefficients(base, response, data):
    base, matrix = np.asarray(base, dtype='float64'), np.asarray(response, dtype='float64')
    protected = np.asarray([i for i in data['protected_clauses'] if 0 < base[i] < 1e6], dtype='int64')
    floors = np.minimum(base[protected], .03)

    def objective(c):
        value, derivative = numpy_objective(base + matrix @ c, data)
        return value + .002 * float(c @ c), matrix.T @ derivative + .004 * c

    constraints = []
    if len(protected):
        constraints.append(dict(type='ineq', fun=lambda c: base[protected] + matrix[protected] @ c - floors,
                                jac=lambda c: matrix[protected]))
    zero = np.zeros(matrix.shape[1])
    fit = minimize(objective, zero, jac=True, method='SLSQP', bounds=[(-1., 1.)] * len(zero),
                   constraints=constraints, options=dict(maxiter=60, ftol=1e-9))
    valid = np.isfinite(fit.x).all() and (np.abs(fit.x) <= 1. + 1e-7).all()
    valid = bool(valid and all(float(c['fun'](fit.x).min()) >= -1e-6 for c in constraints))
    usable = valid and objective(fit.x)[0] < objective(zero)[0] - 1e-12
    coefficients = fit.x if usable else zero
    return coefficients, dict(fit_success=bool(usable), optimizer_converged=bool(fit.success),
        coefficients=coefficients.tolist(), protected_floors=floors.tolist(),
        predicted_margins=(base + matrix @ coefficients).tolist(),
        task_function_matches_selection=True)


def describe(data):
    return dict(scope=data['scope'], penalty=data['penalty'], depth=data['depth'],
        objective_clauses=len(data['objective_indices']), measured_clauses=len(data['clauses']),
        measured_states=len(data['states']), objective_indices=data['objective_indices'],
        protected_clauses=data['protected_clauses'], states=data['states'],
        clauses=data['clauses'], weights=data['weights'], bridge_clause=data['bridge_clause'],
        kappa=KAPPA, huber_delta=DELTA, softplus_temperature=TEMPERATURE)

