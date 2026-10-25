"""Joint surrogate-gradient Adam steps for full brain and affine activity readout."""
import json
import time

import numpy as np
import torch

from boundary_tasks import TrialBudget
from common import BudgetExpired, cpu_state, save_checkpoint, save_json
from continual import evaluate
from hybrid_model import padded_forward
from hybrid_runtime import FeatureBank
from hybrid_tasks import loss
from linear_joint_tasks import SuffixCurriculum, descriptor, score
from population_gpu import logits


def joint_gradients(encoder, head, data, check=lambda: None):
    """Two-pass chain rule uses O(one simulation batch) activation storage.

    The head and core gradients refer to the same parameter point and task
    loss. The core Jacobian uses the simulator's existing surrogate spike
    derivative; forward spikes, reset and evaluation remain unchanged.
    """
    device = next(head.parameters()).device
    for parameter in list(head.parameters()) + list(encoder.core.parameters()):
        parameter.grad = None
    features = torch.as_tensor(encoder.encode(data['boards'], check), device=device).detach().requires_grad_(True)
    objective = loss(padded_forward(head, features), data)[0]
    objective.backward()
    desired = features.grad.detach()
    accum = [torch.zeros_like(p) for p in encoder.core.parameters()]
    maximum_error = 0.
    for start in range(0, len(features), encoder.batch):
        check()
        count = min(encoder.batch, len(features) - start)
        padded = np.zeros((encoder.batch, 4, 10, 10), dtype='float32')
        padded[:count] = data['boards'][start:start + count]
        encoded = encoder.forward(torch.from_numpy(padded).to(device))[:count]
        maximum_error = max(maximum_error, float((encoded.detach() - features[start:start + count].detach()).abs().max()))
        (encoded * desired[start:start + count]).sum().backward()
        for total, parameter in zip(accum, encoder.core.parameters()):
            if parameter.grad is not None:
                total.add_(parameter.grad)
        encoder.release_graph()
    if maximum_error > 1e-6:
        raise AssertionError('Recomputed brain activity differs from the gradient source')
    for parameter, gradient in zip(encoder.core.parameters(), accum):
        parameter.grad = gradient
    norms = dict(core=float(torch.sqrt(sum(g.square().sum() for g in accum))),
                 head=float(torch.sqrt(sum(p.grad.square().sum() for p in head.parameters()))))
    if not all(np.isfinite(x) for x in norms.values()):
        raise FloatingPointError('Nonfinite joint gradients')
    return dict(loss=float(objective.detach()), gradient_norms=norms, recompute_max_error=maximum_error)


def assessment(encoder, head, lessons, depth, check=lambda: None):
    data, levels = lessons.data(depth), lessons.levels(depth)
    with torch.no_grad():
        metrics = score(logits(head, encoder.encode(data['boards'], check)), data)
    suite = evaluate(encoder, head, levels, 'joint-linear-policy', check)
    records = suite['records']
    result = dict(**metrics, complete=suite['complete'],
                  solved=records[0]['status'] == 'solved', original_status=records[0]['status'],
                  original_steps=records[0]['steps'], lesson_solved=records[1]['status'] == 'solved',
                  retained=all(r['status'] == 'solved' for r in records[2:]))
    result['stage_passed'] = result['solved'] or (result['lesson_solved'] and result['retained'])
    return result, suite


def can_replace(candidate, champion):
    if candidate['solved'] or champion['solved']:
        return candidate['solved'] and not champion['solved']
    return candidate['retained'] and (candidate['stage_passed'] or candidate['loss'] < champion['loss'] - 1e-7)


def fit(encoder, head, level, config, out):
    budget = TrialBudget(config['training_seconds'], out / 'STOP')
    lessons = SuffixCurriculum(level)
    depth, updates, assessments = 1, 0, 0
    promotions, gradient_history = [], []
    champion_core, champion_head = cpu_state(encoder.core), cpu_state(head)
    champion_obs = None
    stop = 'budget'
    encoder.reset_counts()

    def save(name='model.pt'):
        save_checkpoint(out / name, dict(core=champion_core, head=champion_head, architecture=head.architecture,
            experiment='joint_linear_new_task', seed=config['seed'], task_id=level['id'],
            mode='simultaneous_joint_adam', output_frozen=False, correction_enabled=False,
            observation_mode='snapshot', recurrent_state=False, head_trainable_parameters=2052,
            source_checkpoint_sha256=config['source_checkpoint_sha256'], lesson_depth=depth,
            feedback_snapshot=lessons.snapshot(), joint_updates=updates,
            checkpoint_training_seconds=time.monotonic() - budget.started))

    def log(obs, suite, phase):
        nonlocal assessments, champion_core, champion_head, champion_obs
        assessments += 1
        record = dict(phase=phase, depth=depth, joint_updates=updates,
                      seconds=time.monotonic() - budget.started, **obs)
        with (out / 'history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        save_json(out / 'progress.json', dict(status='training', **record))
        if champion_obs is None or can_replace(obs, champion_obs):
            champion_core, champion_head, champion_obs = cpu_state(encoder.core), cpu_state(head), obs
            save()
        if assessments == 1 or assessments % 5 == 0 or obs['stage_passed']:
            print(json.dumps(dict(event='joint-linear-progress', **record)), flush=True)
        return obs['solved'] or (obs['complete'] and obs['stage_passed'])

    try:
        while depth <= len(lessons.witness):
            data = lessons.data(depth)
            save_json(out / f'data-{depth}.json', descriptor(data))
            encoder.set_core(champion_core)
            head.load_state_dict(champion_head)
            bank = FeatureBank(encoder, champion_core)
            obs, suite = assessment(bank, head, lessons, depth, budget.check)
            if not obs['complete']:
                budget.check()
                raise BudgetExpired('Incomplete stage baseline')
            assert obs['retained'] or obs['solved']
            champion_obs = obs
            save()
            passed = log(obs, suite, 'stage-baseline')
            optimizer = torch.optim.Adam([
                dict(params=list(encoder.core.parameters()), lr=config['core_learning_rate']),
                dict(params=list(head.parameters()), lr=config['head_learning_rate'])])
            while not passed:
                budget.check()
                for _ in range(config['steps_per_assessment']):
                    budget.check()
                    info = joint_gradients(encoder, head, data, budget.check)
                    torch.nn.utils.clip_grad_norm_(encoder.core.parameters(), 1.)
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
                    optimizer.step()
                    encoder.core.project()
                    encoder.release_graph()
                    updates += 1
                    info.update(update=updates, depth=depth)
                    gradient_history.append(info)
                    with (out / 'gradients.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(info) + '\n')
                bank = FeatureBank(encoder, cpu_state(encoder.core))
                obs, suite = assessment(bank, head, lessons, depth, budget.check)
                if not obs['complete'] and not obs['solved']:
                    budget.check()
                    raise BudgetExpired('Incomplete assessment')
                passed = log(obs, suite, 'joint-step')
            if obs['solved']:
                stop = 'original_solved'
                break
            name = f'suffix-{depth}.pt'
            save(name)
            record = dict(depth=depth, checkpoint=name, joint_updates=updates,
                          after=obs, after_suite=suite, seconds=time.monotonic() - budget.started)
            promotions.append(record)
            save_json(out / f'suffix-{depth}.json', record)
            for task, trajectory in zip(lessons.levels(depth), suite['records']):
                if trajectory['status'] == 'solved':
                    lessons.observe_success(task, trajectory)
            depth += 1
    except BudgetExpired:
        pass
    finally:
        encoder.set_core(champion_core)
        head.load_state_dict(champion_head)
        save()
    return dict(stop=stop, training_seconds=time.monotonic() - budget.started, joint_updates=updates,
        depth=depth, promotions=promotions, assessments=assessments,
        core_nonzero_gradient_steps=sum(x['gradient_norms']['core'] > 0 for x in gradient_history),
        head_nonzero_gradient_steps=sum(x['gradient_norms']['head'] > 0 for x in gradient_history),
        maximum_recompute_error=max((x['recompute_max_error'] for x in gradient_history), default=0.),
        training_compute=dict(encoder.counts))
