"""Bounded point training followed by reverse lessons with persistent branches."""
import math
import time
import numpy as np
import torch

from bootstrap import HERE
from boundary_tasks import TrialBudget
from common import BudgetExpired, cpu_state, save_checkpoint, save_json
from continual import core_proposal
from lesson_population import evaluate_many
from lesson_tasks import Lessons, TARGET_MARGIN, better, loss
from population_gpu import PopulationGPU, logits

KEYS = ('pre_log_gain', 'post_log_gain', 'threshold_mv')


def public(obs):
    return {k: v for k, v in obs.items() if k not in ('records', 'seconds')}


def gradients(encoder, head, data, check):
    features = encoder.encode(data['boards'], check)
    q = logits(head, features).detach().requires_grad_(True)
    desired, = torch.autograd.grad(loss(q, data)[0], q)
    accum = [torch.zeros_like(p) for p in encoder.core.parameters()]
    for start in range(0, len(features), encoder.batch):
        check()
        count = min(len(features) - start, encoder.batch)
        boards = np.zeros((encoder.batch, 4, 10, 10), dtype='float32')
        boards[:count] = data['boards'][start:start + count]
        encoded = encoder.forward(torch.from_numpy(boards).cuda())[:count]
        output = head(encoded, correction=False)['brain_logits']
        (output * desired[start:start + count]).sum().backward()
        for value, parameter in zip(accum, encoder.core.parameters()):
            value.add_(parameter.grad)
        encoder.release_graph()
    norm = float(torch.sqrt(sum(g.square().sum() for g in accum)))
    if not np.isfinite(norm):
        raise FloatingPointError('Nonfinite lesson gradient')
    return np.stack([g.detach().abs().cpu().numpy() for g in accum]), norm


class Branch:
    def __init__(self, state, index):
        self.state, self.obs = state, None
        self.index, self.visits, self.stalled, self.version = index, 0, 0, None
        self.radius = np.array([.001, .001, .01]) * (1, 2, 4, 8)[index]
        self.scores = None

    def proposals(self, encoder, head, data, rng, check, probe):
        self.visits += 1
        norm = None
        if self.scores is None or self.version != data['version'] or (self.visits - 1) % 8 == 0:
            encoder.set_core(self.state)
            self.scores, norm = gradients(encoder, head, data, check)
            self.version = data['version']
        # Joint-family trials can cross action boundaries that single-family
        # changes fail to cross. All branches share the one seed-10 RNG stream.
        families = list(range(3)) if self.visits % 4 == 0 else [(self.visits - 1) % 3]
        coordinates = []
        for family in families:
            values = self.state[KEYS[family]].numpy()
            scores = self.scores[family].astype('float64')
            top = np.argsort(scores)[-min(256, len(scores)):]
            probability = scores[top] + max(float(scores[top].max()), 1e-20) * 1e-5
            probability /= probability.sum()
            ranked = rng.choice(top, 12, replace=False, p=probability)
            random = rng.choice(np.setdiff1d(np.arange(len(scores)), ranked), 4, replace=False)
            indices = np.concatenate((ranked, random))
            bounds = (math.log(.8), math.log(1.25)) if family < 2 else (-1., 1.)
            room = np.maximum(0., np.minimum(values[indices] - bounds[0], bounds[1] - values[indices]))
            steps = np.minimum(self.radius[family], .8 * room).astype('float32')
            coordinates.append((family, indices, steps))
        proposals, signs_by_pair = [], []
        for pair in range(4):
            direction = {k: torch.zeros_like(v) for k, v in self.state.items()}
            signs = []
            for family, indices, steps in coordinates:
                sign = rng.choice([-1., 1.], len(indices)).astype('float32')
                direction[KEYS[family]][indices] = torch.from_numpy(steps * sign)
                signs.append(sign)
            signs_by_pair.append(signs)
            for multiplier in (1., -1.):
                proposals.append((f'pair-{pair}:{multiplier:+g}', core_proposal(self.state, direction, multiplier)))
        observations = probe(proposals)
        differences = [observations[2*i]['loss'] - observations[2*i+1]['loss'] for i in range(4)]
        direction = {k: torch.zeros_like(v) for k, v in self.state.items()}
        for j, (family, indices, steps) in enumerate(coordinates):
            estimate = sum(np.divide(delta * signs_by_pair[i][j], 2 * steps,
                out=np.zeros_like(steps), where=steps > 1e-12) for i, delta in enumerate(differences)) / 4
            if np.max(np.abs(estimate)) > 1e-12:
                direction[KEYS[family]][indices] = torch.from_numpy(
                    (-steps * estimate / np.max(np.abs(estimate))).astype('float32'))
        if any(bool(d.count_nonzero()) for d in direction.values()):
            probe([('response-estimate', core_proposal(self.state, direction, 1.))])
        return dict(gradient_norm=norm, families=families,
                    maximum_pair_signal=max(abs(d) for d in differences), radii=self.radius.tolist())


class StagePassed(Exception):
    def __init__(self, state, obs):
        self.state, self.obs = state, obs


def fit(encoder, head, level, config, out):
    import json
    started = time.monotonic()
    lessons, rng = Lessons(level, config['seed']), np.random.default_rng(config['seed'])
    engine, frozen_head = PopulationGPU(encoder, head), cpu_state(head)
    current = cpu_state(encoder.core)
    encoder.reset_counts()
    count = accepted = generation = 0
    stages, stop, last_obs = [], 'training_budget', None
    overall = TrialBudget(config['critical_seconds'] + config['curriculum_seconds'], out / 'STOP')

    def save(path, state, stage):
        save_checkpoint(path, dict(core=state, head=frozen_head, seed=config['seed'], task_id=level['id'],
            source_checkpoint_sha256=config['source_checkpoint_sha256'], output_frozen=True,
            correction_enabled=False, observation_mode='snapshot', experiment='reverse_lessons',
            lesson_stage=stage, checkpoint_training_seconds=time.monotonic() - started))

    save(out / 'model.pt', current, -1)
    curriculum_budget = None
    try:
        for stage in range(-1, len(lessons.depths)):
            name = 'critical' if stage < 0 else f'suffix-{lessons.depths[stage]}'
            phase_start = time.monotonic()
            if stage < 0:
                phase_budget = TrialBudget(config['critical_seconds'], out / 'STOP')
            else:
                if curriculum_budget is None:
                    curriculum_budget = TrialBudget(config['curriculum_seconds'], out / 'STOP')
                phase_budget = curriculum_budget
            def check():
                overall.check()
                phase_budget.check()
            lesson_level = lessons.level_for(stage)
            levels = [level] + ([lesson_level] if lesson_level is not None else [])
            branches = [Branch(current, i) for i in range(4)]
            data_version = None
            data = lessons.data(stage)
            phase_initial = None
            rounds, status = 0, 'budget'
            all_rounds = []

            def observe_batch(proposals, branch_id):
                nonlocal count, current, last_obs
                observations = evaluate_many(engine, [s for _, s in proposals], levels, data, check)
                qualifying = []
                for (label, state), obs in zip(proposals, observations):
                    if not obs['complete'] and not obs['solved']:
                        continue
                    count += 1
                    with (out / 'candidates.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(dict(candidate=count, generation=generation, stage=name,
                            branch=branch_id, name=label, seconds=time.monotonic() - started, **public(obs),
                            actions=obs['records'][0]['actions'],
                            lesson_actions=obs['records'][-1]['actions'])) + '\n')
                    all_rounds.append((state, obs, label))
                    passed = obs['critical_margin'] >= TARGET_MARGIN if stage < 0 else obs['lesson_solved']
                    if obs['solved'] or passed:
                        qualifying.append((state, obs))
                if qualifying:
                    state, obs = min(qualifying, key=lambda pair: (not pair[1]['solved'], pair[1]['loss']))
                    current, last_obs = state, obs
                    raise StagePassed(state, obs)
                return observations

            try:
                while True:
                    check()
                    generation += 1
                    rounds += 1
                    all_rounds = []
                    data = lessons.data(stage)
                    if data_version != lessons.version:
                        baselines = observe_batch([(f'branch-{i}-base', b.state) for i, b in enumerate(branches)], -1)
                        for branch, obs in zip(branches, baselines):
                            branch.obs = obs
                        winner = min(branches, key=lambda b: b.obs['loss'])
                        current, last_obs = winner.state, winner.obs
                        branches[0].state, branches[0].obs = current, last_obs
                        data_version = lessons.version
                        if phase_initial is None:
                            phase_initial = public(last_obs)
                    index = (0, 1, 0, 2, 0, 3)[(rounds - 1) % 6]
                    branch = branches[index]
                    old_loss = branch.obs['loss']
                    all_rounds = []
                    metadata = branch.proposals(encoder, head, data, rng, check,
                        lambda proposals: observe_batch(proposals, index))
                    candidates = [(s, o, label) for s, o, label in all_rounds if o['complete']]
                    state, obs, selected = min(candidates, key=lambda row: row[1]['loss'])
                    improved = better(obs, branch.obs)
                    exploration = False
                    if improved:
                        branch.state, branch.obs = state, obs
                        branch.stalled = 0
                    else:
                        branch.stalled += 1
                        # Explorer branches keep bounded temporary regressions;
                        # the best model remains separately protected in branch 0.
                        tolerance = (0., .03, .10, .25)[index]
                        if index and branch.stalled >= 2 and obs['loss'] <= branch.obs['loss'] + tolerance:
                            branch.state, branch.obs = state, obs
                            branch.stalled = 0
                            exploration = True
                    if better(branch.obs, last_obs):
                        current, last_obs = branch.state, branch.obs
                        accepted += 1
                        branches[0].state, branches[0].obs = current, last_obs
                        save(out / 'model.pt', current, stage)
                    for family in metadata['families']:
                        factor = 2 if metadata['maximum_pair_signal'] < 1e-6 else (1.2 if improved else .85)
                        branch.radius[family] = np.clip(branch.radius[family] * factor,
                            (1e-5, 1e-5, 1e-4)[family], (.01, .01, .1)[family])
                    row = dict(generation=generation, stage=name, stage_round=rounds, branch=index,
                        candidates=count, accepted_updates=accepted, exploration_step=exploration,
                        branch_improved=improved, branch_loss_before=old_loss, branch_loss=branch.obs['loss'],
                        selected=selected, seconds=time.monotonic() - started,
                        training_states=len(data['states']), feedback_version=lessons.version,
                        **public(last_obs), **metadata)
                    with (out / 'history.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(row) + '\n')
                    save_json(out / 'progress.json', dict(status='training', **row))
                    save_json(out / 'last-rollouts.json', last_obs['records'])
                    if stage >= 0:
                        for lesson, record in zip(levels, branch.obs['records']):
                            lessons.observe(lesson, record)
                        save_json(out / 'lessons.json', lessons.snapshot())
                    if rounds == 1 or rounds % 5 == 0:
                        print(json.dumps(dict(event='generation', **row)), flush=True)
            except StagePassed as passed:
                current, last_obs = passed.state, passed.obs
                status = 'original_solved' if passed.obs['solved'] else 'passed'
                save(out / f'{name}.pt', current, stage)
                save(out / 'model.pt', current, stage)
                if stage >= 0:
                    lessons.observe(lesson_level, passed.obs['records'][1])
                save_json(out / 'lessons.json', lessons.snapshot())
            except BudgetExpired:
                # A deadline may interrupt a gradient or the last proposal group.
                # Retain any already completed, strictly better candidate.
                for state, obs, _ in all_rounds:
                    if obs['complete'] and (last_obs is None or better(obs, last_obs)):
                        current, last_obs = state, obs
            finally:
                stages.append(dict(stage=name, status=status, seconds=time.monotonic() - phase_start,
                    initial=phase_initial, final=public(last_obs) if last_obs else None,
                    records=last_obs['records'] if last_obs else [],
                    candidates=count, rounds=rounds))
                save_json(out / 'stages.json', stages)
                save(out / 'model.pt', current, stage)
                print(json.dumps(dict(event='stage_finished', stage=name, status=status,
                                      seconds=stages[-1]['seconds'])), flush=True)
            if status == 'original_solved':
                stop = 'autonomous_success'
                break
            if status != 'passed':
                stop = 'critical_budget' if stage < 0 else 'curriculum_budget'
                break
        else:
            # The last lesson starts at the original initial state.
            assert last_obs['solved']
            stop = 'autonomous_success'
    finally:
        encoder.set_core(current)
        save(out / 'model.pt', current, len(stages) - 2)
    return dict(stop=stop, training_seconds=time.monotonic() - started, candidates=count,
                generations=generation, accepted_updates=accepted, stages=stages,
                training_compute=dict(encoder.counts), final=public(last_obs) if last_obs else None)
