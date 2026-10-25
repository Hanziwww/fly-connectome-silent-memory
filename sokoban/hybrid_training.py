"""Bounded alternating MLP updates and measured-response internal brain updates."""
import json
import math
import time

import numpy as np
import torch

from boundary_tasks import TrialBudget
from common import BudgetExpired, cpu_state, save_checkpoint, save_json, tensor_hash
from continual import core_proposal
from hybrid_model import padded_forward
from hybrid_runtime import FeatureBank, assessment, audit_states, candidates
from hybrid_tasks import dataset, descriptor, loss, response_coefficients
from joint_tasks import can_replace

KEYS = ('pre_log_gain', 'post_log_gain', 'threshold_mv')


class Advance(Exception):
    pass


class Finished(Exception):
    pass


def compact(obs):
    return {k: v for k, v in obs.items() if k != 'records'}


def group_scores(encoder, head, bank, data, check):
    """Surrogate gradients rank groups; actual hard-forward responses choose updates."""
    features = torch.as_tensor(bank.encode(data['boards'], check), device='cuda')
    head.requires_grad_(False)
    try:
        q = padded_forward(head, features).detach().requires_grad_(True)
        desired, = torch.autograd.grad(loss(q, data)[0], q)
        accum = [torch.zeros_like(p) for p in encoder.core.parameters()]
        for start in range(0, len(features), encoder.batch):
            check()
            count = min(len(features) - start, encoder.batch)
            boards = np.zeros((encoder.batch, 4, 10, 10), dtype='float32')
            boards[:count] = data['boards'][start:start + count]
            encoded = encoder.forward(torch.from_numpy(boards).cuda())[:count]
            output = padded_forward(head, encoded)
            (output * desired[start:start + count]).sum().backward()
            for total, parameter in zip(accum, encoder.core.parameters()):
                if parameter.grad is not None:
                    total.add_(parameter.grad)
            encoder.release_graph()
        norm = float(torch.sqrt(sum(g.square().sum() for g in accum)))
        if not np.isfinite(norm):
            raise FloatingPointError('Nonfinite internal gradient')
        return np.stack([g.detach().abs().cpu().numpy() for g in accum]), norm
    finally:
        encoder.release_graph()
        head.requires_grad_(True)


def fit(encoder, head, level, snapshot, config, trial, out, initial_bank):
    budget = TrialBudget(config['training_seconds'], out / 'STOP')
    started = budget.started
    rng = np.random.default_rng(config['seed'])
    _, cache_boards, _ = audit_states(level)
    depth, generation = 6, 0
    head_steps = core_rounds = core_candidates = paired_accepts = core_accepts = 0
    stop = 'budget'
    promotions, history = [], []
    champion_core, champion_head = cpu_state(encoder.core), cpu_state(head)
    working_core, bank = champion_core, initial_bank
    champion_obs = None
    data = levels = lessons = None
    radius = np.array([.001, .001, .01])
    encoder.reset_counts()

    def check():
        budget.check()
        if (trial / 'STOP').exists():
            raise KeyboardInterrupt('Hybrid trial STOP file found')

    def save(name='model.pt'):
        save_checkpoint(trial / name, dict(core=champion_core, head=champion_head,
            architecture=head.architecture, head_trainable_parameters=sum(p.numel() for p in head.parameters()),
            seed=config['seed'], task_id=level['id'], experiment='brain_mlp_hybrid', mode=config['mode'],
            source_checkpoint_sha256=config['source_checkpoint_sha256'], output_frozen=False,
            correction_enabled=False, observation_mode='snapshot', recurrent_state=False,
            lesson_depth=depth, feedback_snapshot=snapshot, checkpoint_training_seconds=time.monotonic() - started))

    def consider(core, obs, suite, name):
        nonlocal champion_core, champion_head, champion_obs, paired_accepts
        if not obs['complete'] and not obs['solved']:
            return
        row = dict(generation=generation, phase=name, depth=depth,
                   seconds=time.monotonic() - started, **compact(obs))
        with (trial / 'candidates.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        if obs['joint_passed'] or can_replace(obs, champion_obs):
            champion_core, champion_head, champion_obs = core, cpu_state(head), obs
            paired_accepts += 1
            save()
        if obs['solved']:
            raise Finished()
        if obs['joint_passed']:
            name = f'suffix-{depth}.pt'
            save(name)
            record = dict(depth=depth, checkpoint=name, seconds=time.monotonic() - started,
                          after=compact(obs), after_suite=suite)
            promotions.append(record)
            save_json(trial / f'suffix-{depth}.json', record)
            raise Advance()

    try:
        while True:
            lessons, data, levels = dataset(level, snapshot, depth)
            save_json(trial / f'data-{depth}.json', descriptor(data))
            head.load_state_dict(champion_head)
            bank = FeatureBank(encoder, champion_core, cache_boards,
                               bank.encode(cache_boards, check)) if bank.core_hash == tensor_hash(champion_core) else FeatureBank(encoder, champion_core)
            working_core = champion_core
            champion_obs, _ = assessment(bank, head, data, levels, check)
            assert champion_obs['retained'] or champion_obs['solved']
            save()
            optimizer = torch.optim.Adam(head.parameters(), lr=config['head_learning_rate'])
            try:
                if champion_obs['solved']:
                    raise Finished()
                if champion_obs['joint_passed']:
                    consider(champion_core, champion_obs, assessment(bank, head, data, levels, check)[1], 'stage-baseline')
                while True:
                    check()
                    generation += 1
                    encoder.set_core(working_core)
                    x = torch.as_tensor(bank.encode(data['boards'], check), device='cuda')
                    head.train()
                    for _ in range(config['head_steps_per_round']):
                        check()
                        optimizer.zero_grad(set_to_none=True)
                        objective = loss(padded_forward(head, x), data)[0]
                        objective.backward()
                        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
                        if not bool(torch.isfinite(norm)):
                            raise FloatingPointError('Nonfinite MLP gradient')
                        optimizer.step()
                        head_steps += 1
                    head.eval()
                    working_obs, suite = assessment(bank, head, data, levels, check)
                    consider(working_core, working_obs, suite, 'mlp')
                    metadata = dict(core_updated=False)
                    if config['mode'] == 'alternating':
                        core_rounds += 1
                        scores, gradient_norm = group_scores(encoder, head, bank, data, check)
                        families = list(range(3)) if core_rounds % 4 == 0 else [(core_rounds - 1) % 3]
                        directions, probes = [], []
                        for pair in range(4):
                            direction = {k: torch.zeros_like(v) for k, v in working_core.items()}
                            for family in families:
                                ranks = scores[family].astype('float64')
                                top = np.argsort(ranks)[-256:]
                                probability = ranks[top] + max(float(ranks[top].max()), 1e-20) * 1e-5
                                probability /= probability.sum()
                                chosen = rng.choice(top, 12, replace=False, p=probability)
                                other = rng.choice(np.setdiff1d(np.arange(len(ranks)), chosen), 4, replace=False)
                                ids = np.concatenate((chosen, other))
                                values = working_core[KEYS[family]].numpy()[ids]
                                bounds = (math.log(.8), math.log(1.25)) if family < 2 else (-1., 1.)
                                room = np.maximum(0., np.minimum(values - bounds[0], bounds[1] - values))
                                steps = np.minimum(radius[family], .8 * room).astype('float32')
                                signs = rng.choice([-1., 1.], len(ids)).astype('float32')
                                direction[KEYS[family]][ids] = torch.from_numpy(steps * signs)
                            directions.append(direction)
                            for sign in (1., -1.):
                                probes.append((f'probe-{pair}:{sign:+g}', core_proposal(working_core, direction, sign)))
                        best_core, best_obs = working_core, working_obs

                        def probe(items):
                            def consume(index, core, obs, result_suite):
                                nonlocal core_candidates, best_core, best_obs
                                core_candidates += 1
                                consider(core, obs, result_suite, items[index][0])
                                if obs['complete'] and obs['loss'] < best_obs['loss'] - 1e-7:
                                    best_core, best_obs = core, obs
                            return candidates(encoder, head, [c for _, c in items], cache_boards,
                                              data, levels, check, consume)

                        measured = probe(probes)
                        response = np.array([(np.array(measured[2 * i]['margins']) -
                                              np.array(measured[2 * i + 1]['margins'])) / 2 for i in range(4)]).T
                        coefficients, fit_info = response_coefficients(working_obs['margins'], response, data)
                        direction = {k: sum(float(c) * d[k] for c, d in zip(coefficients, directions)) for k in KEYS}
                        if any(bool(d.count_nonzero()) for d in direction.values()):
                            probe([(f'response-fit:{scale}', core_proposal(working_core, direction, scale))
                                   for scale in (1., .5, .25)])
                        improved = best_obs['loss'] < working_obs['loss'] - 1e-7
                        if improved:
                            working_core, working_obs = best_core, best_obs
                            core_accepts += 1
                        for family in families:
                            multiplier = 2. if float(np.abs(response).max()) < 1e-6 else (1.2 if improved else .85)
                            radius[family] = np.clip(radius[family] * multiplier,
                                                    (1e-5, 1e-5, 1e-4)[family], (.01, .01, .1)[family])
                        bank = FeatureBank(encoder, working_core)
                        metadata = dict(core_updated=improved, gradient_norm=gradient_norm,
                                        radii=radius.tolist(), response_fit=fit_info)
                    row = dict(generation=generation, depth=depth, seconds=time.monotonic() - started,
                               head_steps=head_steps, core_rounds=core_rounds, core_candidates=core_candidates,
                               paired_accepts=paired_accepts, core_accepts=core_accepts,
                               working_loss=working_obs['loss'], champion=compact(champion_obs), **metadata)
                    history.append(row)
                    with (trial / 'history.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(row) + '\n')
                    save_json(trial / 'progress.json', dict(status='training', **row))
                    if generation == 1 or generation % 10 == 0:
                        print(json.dumps(dict(event='hybrid-progress', arm=config['arm'], depth=depth,
                            seconds=row['seconds'], head_steps=head_steps, core_accepts=core_accepts,
                            loss=champion_obs['loss'], bridge=champion_obs['bridge_margin'])), flush=True)
            except Advance:
                for task, record in zip(levels, champion_obs['records']):
                    if record['status'] == 'solved':
                        lessons.observe(task, record)
                snapshot = lessons.snapshot()
                depth += 1
                # New objective: re-evaluate its baseline before comparing any losses.
                head.load_state_dict(champion_head)
                bank = FeatureBank(encoder, champion_core)
                print(json.dumps(dict(event='hybrid-promotion', arm=config['arm'], depth=depth - 1,
                                      seconds=time.monotonic() - started)), flush=True)
    except Finished:
        stop = 'original_solved'
    except BudgetExpired:
        pass
    finally:
        encoder.set_core(champion_core)
        head.load_state_dict(champion_head)
        head.requires_grad_(True)
        save()
    assert champion_obs is not None and (champion_obs['retained'] or champion_obs['solved'])
    return dict(stop=stop, training_seconds=time.monotonic() - started,
                budget_seconds=config['training_seconds'], head_steps=head_steps, core_rounds=core_rounds,
                core_candidate_evaluations=core_candidates, working_core_updates=core_accepts,
                paired_champion_updates=paired_accepts, promotions=promotions, depth=depth,
                final=compact(champion_obs), records=champion_obs['records'],
                training_compute=dict(encoder.counts))
