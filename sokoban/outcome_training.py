"""Outcome-first search with on-policy dead-end/cycle feedback and a frozen head."""
import json
import math
import time
import numpy as np
import torch

from bootstrap import HERE
from boundary_tasks import TrialBudget
from common import BudgetExpired, cpu_state, save_checkpoint, save_json
from continual import evaluate
from outcome_objective import Counterexamples, better, losses_and_margins, objective
from population_gpu import PopulationGPU, evaluate_many, logits
import repair_training as block_method


def public(obs):
    return {k: v for k, v in obs.items() if k not in ('features', 'logits', 'rollout')}


def measure(encoder, head, data, check):
    features = encoder.encode(data['boards'], check)
    with torch.no_grad():
        q = logits(head, features)
        losses, margins, correct = losses_and_margins(q, data)
        return dict(correct=int(correct.sum()), total=len(losses), min_margin=float(margins.min()),
            focus_loss=float(objective(losses)), loss=float(losses.mean()),
            per_clause_loss=losses.cpu().tolist(), margins=margins.cpu().tolist(),
            features=features, logits=q.cpu().tolist(), solved=False)


def gradients(encoder, head, data, mode, check):
    features = encoder.encode(data['boards'], check)
    with torch.no_grad():
        logits = head(torch.as_tensor(features, device='cuda'), correction=False)['brain_logits']
    q = logits.detach().requires_grad_(True)
    losses, _, _ = losses_and_margins(q, data)
    desired, = torch.autograd.grad(objective(losses), q)
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
        raise FloatingPointError('Nonfinite counterexample gradient')
    return [g / max(1., norm) for g in accum], norm


class Generator(block_method.BlockGenerator):
    def propose(self, *args):
        # This worker is a single isolated process; retain D's numeric proposals.
        original = block_method.core_gradients
        block_method.core_gradients = gradients
        try:
            return super().propose(*args)
        finally:
            block_method.core_gradients = original

    def propose_many(self, encoder, head, data, state, generation, check, probe_many):
        norm = None
        settings = block_method.SETTINGS
        if (generation - 1) % settings['gradient_refresh'] == 0:
            grad, norm = gradients(encoder, head, data, 'behavior', check)
            self.scores = np.stack([g.detach().cpu().abs().numpy().astype('float64') for g in grad])
        family = (generation - 1) % 3
        key = block_method.KEYS[family]
        score = self.scores[family]
        top = np.argsort(score)[-min(256, len(score)):]
        probability = score[top] + max(float(score[top].max()), 1e-20) * 1e-5
        probability /= probability.sum()
        ranked = self.rng.choice(top, settings['ranked_groups'], replace=False, p=probability)
        remaining = np.setdiff1d(np.arange(len(score)), ranked)
        random = self.rng.choice(remaining, settings['block_groups'] - len(ranked), replace=False)
        indices = np.concatenate((ranked, random))
        bounds = (math.log(.8), math.log(1.25)) if family < 2 else (-1., 1.)
        values = state[key].numpy()[indices]
        room = np.maximum(0., np.minimum(values - bounds[0], bounds[1] - values))
        steps = np.minimum(self.radius[family], .8 * room).astype('float32')
        proposals, all_signs = [], []
        for pair in range(settings['pairs']):
            signs = self.rng.choice([-1., 1.], len(indices)).astype('float32')
            all_signs.append(signs)
            direction = {k: torch.zeros_like(v) for k, v in state.items()}
            direction[key][indices] = torch.from_numpy(steps * signs)
            for sign in (1., -1.):
                proposals.append((f'block:{key}:pair{pair}:{sign:+g}', block_method.core_proposal(state, direction, sign)))
        outcomes = probe_many(proposals)
        estimate, signals = np.zeros(len(indices), dtype='float64'), []
        for pair, signs in enumerate(all_signs):
            difference = outcomes[2*pair]['focus_loss'] - outcomes[2*pair+1]['focus_loss']
            signals.append(abs(difference))
            estimate += np.divide(difference * signs, 2 * steps, out=np.zeros_like(estimate), where=steps > 1e-12)
        estimate /= settings['pairs']
        if np.max(np.abs(estimate)) > 1e-12:
            direction = {k: torch.zeros_like(v) for k, v in state.items()}
            direction[key][indices] = torch.from_numpy((-steps * estimate / np.max(np.abs(estimate))).astype('float32'))
            probe_many([(f'block:{key}:response-estimate', block_method.core_proposal(state, direction, 1.))])
        return dict(gradient_norm=norm, proposal_generator='population_block', parameter_family=key,
            group_indices=indices.tolist(), radius=float(self.radius[family]),
            maximum_pair_signal=float(max(signals)), family_index=family)


class FoundSolution(Exception):
    pass


def fit(encoder, head, level, config, out, before):
    bank = Counterexamples(level)
    incumbent = cpu_state(encoder.core)
    frozen_head = cpu_state(head)
    generator = Generator(incumbent, config['seed'])
    engine = PopulationGPU(encoder, head)
    rollout = before
    budget = TrialBudget(config['training_seconds'], out / 'STOP')
    encoder.reset_counts()
    generation = candidates = accepted = 0
    stop, final_obs, checkpoint_time = 'training_budget', None, 0.
    best_core, best_obs, best_rollout = incumbent, None, rollout

    def save():
        save_checkpoint(out / 'model.pt', dict(core=incumbent, head=frozen_head, seed=config['seed'],
            task_id=level['id'], source_checkpoint_sha256=config['source_checkpoint_sha256'],
            output_frozen=True, correction_enabled=False, observation_mode='snapshot',
            experiment='outcome_counterexamples', checkpoint_training_seconds=checkpoint_time,
            accepted_updates=accepted))

    save()
    try:
        if before['solved']:
            stop = 'already_solved'
            return dict(stop=stop, training_seconds=0., candidates=0, generations=0, accepted_updates=0)
        while True:
            budget.check()
            generation += 1
            discovered = bank.observe(rollout['records'][0])
            if discovered is None:
                stop = 'inconclusive_rollout_limit'
                break
            data = bank.data()
            save_json(out / 'counterexamples.json', bank.clauses)
            encoder.set_core(incumbent)
            base = measure(encoder, head, data, budget.check)
            best_core, best_obs, best_rollout = incumbent, base, rollout
            selected, outcomes = None, []

            def probe(name, proposal):
                nonlocal candidates, best_core, best_obs, best_rollout, selected
                budget.check()
                encoder.set_core(proposal)
                obs = measure(encoder, head, data, budget.check)
                actual = evaluate(encoder, head, [level], 'candidate', budget.check,
                                  [(data['boards'], obs['features'])])
                if not actual['complete']:
                    raise BudgetExpired('Candidate rollout interrupted at training deadline')
                candidates += 1
                obs.update(solved=bool(actual['solved']), rollout=actual,
                    rollout_status=actual['records'][0]['status'], rollout_steps=actual['records'][0]['steps'])
                outcomes.append(obs)
                if better(obs, best_obs):
                    best_core, best_obs, best_rollout, selected = proposal, obs, actual, name
                with (out / 'candidates.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(dict(candidate=candidates, generation=generation, name=name,
                        seconds=time.monotonic() - budget.started, **public(obs))) + '\n')
                if obs['solved']:
                    raise FoundSolution()
                return obs

            def probe_many(proposals):
                nonlocal candidates, best_core, best_obs, best_rollout, selected
                budget.check()
                observations = evaluate_many(engine, [core for _, core in proposals], level, data, budget.check)
                for (name, proposal), obs in zip(proposals, observations):
                    if not obs['rollout']['complete']:
                        continue
                    candidates += 1
                    outcomes.append(obs)
                    if better(obs, best_obs):
                        best_core, best_obs, best_rollout, selected = proposal, obs, obs['rollout'], name
                    with (out / 'candidates.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(dict(candidate=candidates, generation=generation, name=name,
                            seconds=time.monotonic() - budget.started, **public(obs))) + '\n')
                if any(obs['solved'] for obs in observations):
                    raise FoundSolution()
                return observations

            if config.get('population_gpu', True):
                metadata = generator.propose_many(encoder, head, data, incumbent, generation, budget.check, probe_many)
            else:
                metadata = generator.propose(encoder, head, data, incumbent, 'behavior', generation, budget.check, probe)
            improved = better(best_obs, base)
            generator.feedback(metadata, improved, outcomes, base)
            if improved:
                incumbent, rollout = best_core, best_rollout
                accepted += 1
                checkpoint_time = time.monotonic() - budget.started
                save()
            final_obs = best_obs
            row = dict(generation=generation, candidates=candidates, accepted_updates=accepted,
                seconds=time.monotonic() - budget.started, selected=selected, constraints=len(bank.clauses),
                states=len(data['boards']), **public(best_obs), **metadata)
            with (out / 'history.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')
            save_json(out / 'progress.json', dict(status='training', **row))
            save_json(out / 'last-rollout.json', rollout)
            if generation == 1 or generation % 5 == 0:
                print(json.dumps(dict(event='generation', **row)), flush=True)
    except FoundSolution:
        stop = 'autonomous_success'
    except BudgetExpired:
        pass
    finally:
        # Preserve a fully evaluated improvement even if time expires mid-generation.
        if best_obs is not None and (final_obs is not best_obs or stop == 'autonomous_success'):
            if best_obs['solved'] or 'base' in locals() and better(best_obs, base):
                incumbent, rollout, final_obs = best_core, best_rollout, best_obs
                accepted += 1
                checkpoint_time = time.monotonic() - budget.started
        encoder.set_core(incumbent)
        save()
    return dict(stop=stop, training_seconds=time.monotonic() - budget.started,
        budget_seconds=config['training_seconds'], candidates=candidates, generations=generation,
        accepted_updates=accepted, checkpoint_training_seconds=checkpoint_time,
        constraints=len(bank.clauses), final_constraints=public(final_obs) if final_obs else None,
        training_compute=dict(encoder.counts), last_rollout=rollout)
