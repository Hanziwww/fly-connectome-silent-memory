"""Protected task behavior with exploratory branches and measured response fits."""
import json
import math
import time
import numpy as np
import torch

from bootstrap import HERE
from boundary_tasks import TrialBudget
from common import BudgetExpired, cpu_state, save_checkpoint, save_json
from continual import core_proposal
from joint_tasks import JointLessons, can_replace, decorate, response_coefficients
from lesson_population import evaluate_many
from lesson_training import Branch, StagePassed, gradients, public
from population_gpu import PopulationGPU

KEYS = ('pre_log_gain', 'post_log_gain', 'threshold_mv')


class ResponseBranch(Branch):
    def proposals(self, encoder, head, data, rng, check, probe):
        self.visits += 1
        norm = None
        if self.scores is None or self.version != data['version'] or (self.visits - 1) % 8 == 0:
            encoder.set_core(self.state)
            self.scores, norm = gradients(encoder, head, data, check)
            self.version = data['version']
        families = list(range(3)) if self.visits % 4 == 0 else [(self.visits - 1) % 3]
        directions, proposals = [], []
        # Four independently chosen sparse groups provide four measured basis
        # directions rather than reducing all their responses to one loss delta.
        for pair in range(4):
            direction = {k: torch.zeros_like(v) for k, v in self.state.items()}
            for family in families:
                scores = self.scores[family].astype('float64')
                top = np.argsort(scores)[-min(256, len(scores)):]
                probability = scores[top] + max(float(scores[top].max()), 1e-20) * 1e-5
                probability /= probability.sum()
                ranked = rng.choice(top, 12, replace=False, p=probability)
                random = rng.choice(np.setdiff1d(np.arange(len(scores)), ranked), 4, replace=False)
                indices = np.concatenate((ranked, random))
                values = self.state[KEYS[family]].numpy()[indices]
                bounds = (math.log(.8), math.log(1.25)) if family < 2 else (-1., 1.)
                room = np.maximum(0., np.minimum(values-bounds[0], bounds[1]-values))
                steps = np.minimum(self.radius[family], .8*room).astype('float32')
                signs = rng.choice([-1., 1.], len(indices)).astype('float32')
                direction[KEYS[family]][indices] = torch.from_numpy(steps * signs)
            directions.append(direction)
            for sign in (1., -1.):
                proposals.append((f'probe-{pair}:{sign:+g}', core_proposal(self.state, direction, sign)))
        outcomes = probe(proposals)
        response = np.array([(np.array(outcomes[2*i]['margins']) - np.array(outcomes[2*i+1]['margins'])) / 2
                             for i in range(4)]).T
        coefficients, fit = response_coefficients(self.obs['margins'], response,
                                                  data['weights'], data['protected_clauses'])
        direction = {k: sum(float(c) * d[k] for c, d in zip(coefficients, directions)) for k in self.state}
        errors = []
        if any(bool(d.count_nonzero()) for d in direction.values()):
            scales = (1., .5, .25)
            combinations = [(f'margin-fit:{scale:g}', core_proposal(self.state, direction, scale)) for scale in scales]
            verified = probe(combinations)
            mask = np.abs(self.obs['margins']) < 1e6
            for scale, obs in zip(scales, verified):
                predicted = np.asarray(self.obs['margins']) + scale * response @ coefficients
                errors.append(float(np.sqrt(np.mean((predicted[mask]-np.asarray(obs['margins'])[mask])**2))))
        return dict(gradient_norm=norm, families=families, radii=self.radius.tolist(),
            maximum_pair_signal=float(np.max(np.abs(response))), response_fit=fit,
            response_prediction_rmse=errors)


def fit(encoder, head, level, config, out):
    started = time.monotonic()
    lessons, rng = JointLessons(level, config['seed']), np.random.default_rng(config['seed'])
    engine, frozen_head = PopulationGPU(encoder, head), cpu_state(head)
    champion, champion_obs = cpu_state(encoder.core), None
    explorer_source = torch.load(out / 'explorer.pt', map_location='cpu', weights_only=True)['core']
    total_budget = TrialBudget(config['joint_seconds'] + config['extension_seconds'], out / 'STOP')
    count = generation = accepted = retention_rejections = 0
    stages, stop, extension_budget = [], 'joint_budget', None
    encoder.reset_counts()

    def save(path, state, depth):
        save_checkpoint(path, dict(core=state, head=frozen_head, seed=config['seed'], task_id=level['id'],
            source_checkpoint_sha256=config['source_checkpoint_sha256'], output_frozen=True,
            correction_enabled=False, observation_mode='snapshot', experiment='joint_retention',
            lesson_depth=depth, checkpoint_training_seconds=time.monotonic()-started))

    save(out / 'model.pt', champion, 4)
    try:
        for stage, depth in enumerate(lessons.depths):
            phase_start = time.monotonic()
            if stage == 0:
                phase_budget = TrialBudget(config['joint_seconds'], out / 'STOP')
            else:
                if extension_budget is None:
                    extension_budget = TrialBudget(config['extension_seconds'], out / 'STOP')
                phase_budget = extension_budget
            def check():
                total_budget.check()
                phase_budget.check()
            levels = lessons.levels(stage)
            branches = [ResponseBranch(champion, i) for i in range(4)]
            if stage == 0:
                branches[3].state = explorer_source
            data = lessons.data(stage)
            data_version, phase_initial, status, rounds = None, None, 'budget', 0
            round_candidates = []

            def probe(proposals, branch_index):
                nonlocal count, champion, champion_obs, accepted, retention_rejections
                observations = [decorate(o) for o in evaluate_many(engine, [s for _, s in proposals], levels, data, check)]
                passed = []
                for (name, state), obs in zip(proposals, observations):
                    if not obs['complete'] and not obs['solved']:
                        continue
                    count += 1
                    round_candidates.append((state, obs, name))
                    if obs['joint_passed']:
                        passed.append((state, obs))
                    if branch_index >= 0 and champion_obs is not None and obs['loss'] < champion_obs['loss'] and not obs['retained'] and not obs['solved']:
                        retention_rejections += 1
                    # During ordinary proposals protect completed improvements
                    # immediately, including when a later probe hits the deadline.
                    if branch_index >= 0 and champion_obs is not None and can_replace(obs, champion_obs):
                        champion, champion_obs = state, obs
                        accepted += 1
                        save(out / 'model.pt', champion, depth)
                    with (out / 'candidates.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(dict(candidate=count, generation=generation, depth=depth,
                            branch=branch_index, name=name, seconds=time.monotonic()-started, **public(obs),
                            actions=obs['records'][0]['actions'], lesson_actions=obs['records'][1]['actions']))+'\n')
                if passed:
                    state, obs = min(passed, key=lambda pair: (not pair[1]['solved'], pair[1]['loss']))
                    champion, champion_obs = state, obs
                    raise StagePassed(state, obs)
                return observations

            try:
                while True:
                    check()
                    generation += 1
                    rounds += 1
                    data = lessons.data(stage)
                    round_candidates = []
                    if data_version != lessons.version:
                        # Re-score the retained champion itself as the first row;
                        # never compare losses computed on different feedback banks.
                        branches[0].state = champion
                        observations = probe([(f'base-{i}', b.state) for i, b in enumerate(branches)], -1)
                        for branch, obs in zip(branches, observations):
                            branch.obs = obs
                        assert observations[0]['retained'], 'Protected champion lost a behavioral prerequisite'
                        champion_obs = observations[0]
                        for branch in branches[1:]:
                            if can_replace(branch.obs, champion_obs):
                                champion, champion_obs = branch.state, branch.obs
                        branches[0].state, branches[0].obs = champion, champion_obs
                        data_version = lessons.version
                        if phase_initial is None:
                            phase_initial = public(champion_obs)
                        if stage == 0 and rounds == 1:
                            # Seed replay only from behavior demonstrated by this
                            # same champion, not a different successful checkpoint.
                            for anchor, record in zip(levels[2:], champion_obs['records'][2:]):
                                lessons.observe(anchor, record)
                            if lessons.version != data_version:
                                continue
                    index = (0, 1, 0, 2, 0, 3)[(rounds-1) % 6]
                    branch = branches[index]
                    if index == 0:
                        branch.state, branch.obs = champion, champion_obs
                    before_loss = branch.obs['loss']
                    round_candidates = []
                    metadata = branch.proposals(encoder, head, data, rng, check,
                        lambda proposals: probe(proposals, index))
                    eligible = [(s, o, label) for s, o, label in round_candidates
                                if o['complete'] and (index != 0 or o['retained'])]
                    exploration, improved = False, False
                    if eligible:
                        state, obs, selected = min(eligible, key=lambda item: item[1]['loss'])
                        improved = obs['loss'] < before_loss - 1e-7
                        branch.stalled = 0 if improved else branch.stalled + 1
                        tolerance = (0., .03, .10, .25)[index]
                        exploration = bool(index and branch.stalled >= 2 and
                            obs['loss'] <= min(before_loss+tolerance, champion_obs['loss']+2*tolerance))
                        if improved or exploration:
                            branch.state, branch.obs = state, obs
                            branch.stalled = 0
                    else:
                        selected = None
                        branch.stalled += 1
                    branches[0].state, branches[0].obs = champion, champion_obs
                    for family in metadata['families']:
                        multiplier = 2. if metadata['maximum_pair_signal'] < 1e-6 else (1.2 if improved else .85)
                        branch.radius[family] = np.clip(branch.radius[family]*multiplier,
                            (1e-5, 1e-5, 1e-4)[family], (.01, .01, .1)[family])
                    row = dict(generation=generation, depth=depth, stage_round=rounds, branch=index,
                        candidates=count, accepted_updates=accepted, retention_rejections=retention_rejections,
                        exploration_step=exploration, branch_improved=improved, selected=selected,
                        branch_loss=branch.obs['loss'], seconds=time.monotonic()-started,
                        training_states=len(data['states']), feedback_version=lessons.version,
                        **public(champion_obs), **metadata)
                    with (out / 'history.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(row)+'\n')
                    save_json(out / 'progress.json', dict(status='training', **row))
                    save_json(out / 'last-rollouts.json', champion_obs['records'])
                    for task, record in zip(levels[:2], branch.obs['records'][:2]):
                        lessons.observe(task, record)
                    save_json(out / 'lessons.json', lessons.snapshot())
                    if rounds == 1 or rounds % 5 == 0:
                        print(json.dumps(dict(event='generation', **row)), flush=True)
            except StagePassed as passed:
                champion, champion_obs = passed.state, passed.obs
                status = 'original_solved' if champion_obs['solved'] else 'joint_passed'
                assert champion_obs['solved'] or champion_obs['retained']
                save(out / f'joint-{depth}.pt', champion, depth)
                save(out / 'model.pt', champion, depth)
                for task, record in zip(levels[1:], champion_obs['records'][1:]):
                    if record['status'] == 'solved':
                        lessons.observe(task, record)
                save_json(out / 'lessons.json', lessons.snapshot())
            except BudgetExpired:
                pass  # Completed eligible improvements were saved by probe().
            finally:
                stages.append(dict(depth=depth, status=status, seconds=time.monotonic()-phase_start,
                    initial=phase_initial, final=public(champion_obs) if champion_obs else None,
                    records=champion_obs['records'] if champion_obs else [], candidates=count, rounds=rounds,
                    required_anchors=[l['id'] for l in levels[2:]]))
                save_json(out / 'stages.json', stages)
                save(out / 'model.pt', champion, depth)
                print(json.dumps(dict(event='stage_finished', depth=depth, status=status,
                                      seconds=stages[-1]['seconds'])), flush=True)
            if status == 'original_solved':
                stop = 'autonomous_success'
                break
            if status != 'joint_passed':
                stop = 'joint_budget' if stage == 0 else 'extension_budget'
                break
        else:
            assert champion_obs['solved']
            stop = 'autonomous_success'
    finally:
        encoder.set_core(champion)
        save(out / 'model.pt', champion, stages[-1]['depth'] if stages else 4)
    return dict(stop=stop, training_seconds=time.monotonic()-started, candidates=count,
        generations=generation, accepted_updates=accepted, retention_rejections=retention_rejections,
        stages=stages, training_compute=dict(encoder.counts), final=public(champion_obs) if champion_obs else None)
