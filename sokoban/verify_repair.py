"""Paired brain substitution/restoration and exact viewer audit; never trains."""
from workspace import ARCHIVE, RUNS
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from bootstrap import HERE
from common import file_hash, save_json, tensor_hash
from continual import evaluate
from curriculum import encode, legal, restore, state_from, step
from evaluation import Predictor
from viewer_catalog import INTERVENTION, REPAIR, OUT, REPAIR_MODELS, repair_bundle
from viewer_runtime import ViewerRuntime, decide


class FixedSignal:
    def __init__(self, feature):
        self.feature = np.asarray(feature, dtype='float32').copy()

    def encode(self, boards, check=lambda: None):
        check()
        return np.repeat(self.feature[None], len(boards), axis=0)


def same_rollout(first, second):
    a, b = first['records'][0], second['records'][0]
    return (first['complete'] and second['complete'] and a['status'] == b['status']
            and a['actions'] == b['actions']
            and [d['brain_logits'] for d in a['decisions']] == [d['brain_logits'] for d in b['decisions']])


def substituted_core(learned, anchor, condition, seed):
    if condition == 'repair_source':
        return {k: v.clone() for k, v in anchor.items()}
    if condition == 'original_core':
        return {k: torch.zeros_like(v) for k, v in learned.items()}
    result = {k: v.clone() for k, v in learned.items()}
    if condition.startswith('undo_'):
        key = condition.removeprefix('undo_')
        result[key] = anchor[key].clone()
    elif condition == 'shuffle_delta':
        # The same permutation preserves cross-family alignment of the deltas.
        order = torch.from_numpy(np.random.default_rng(seed).permutation(len(next(iter(anchor.values())))))
        result = {k: anchor[k] + (v - anchor[k])[order] for k, v in learned.items()}
    return result


def teacher_check(encoder, head, data, tag, check):
    predicted = Predictor(head, encoder=encoder, checkpoint_id=tag, correction=False).predict(
        data['boards'], data['legal'], check)
    actions = [r['action'] for r in predicted]
    correct = data['labels'][np.arange(len(actions)), actions]
    q = np.array([r['brain_logits'] for r in predicted])
    best = np.where(data['labels'], q, -np.inf).max(1)
    other = np.where(data['legal'] & ~data['labels'].astype(bool), q, -np.inf).max(1)
    return dict(correct=int(correct.sum()), total=len(correct), actions=actions,
                min_margin=float((best - other)[np.isfinite(other)].min()),
                original_wrong_state_action=actions[3])


def main(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError('Use a fresh output directory; audits are never overwritten.')
    out.mkdir(parents=True)
    paths, trials, level = repair_bundle()
    conditions = ['repair_source', 'original_core', 'zero_signal', 'mean_signal', 'constant_signal',
                  'undo_pre_log_gain', 'undo_post_log_gain', 'undo_threshold_mv']
    rng = np.random.default_rng(100102)
    schedule = []
    for kind in ('repair10', 'repair11', 'repair12'):
        rows = [dict(condition=c, permutation_seed=None) for c in conditions]
        rows += [dict(condition='shuffle_delta', permutation_seed=seed) for seed in range(381, 391)]
        schedule.extend(dict(model=kind, **rows[i]) for i in rng.permutation(len(rows)))
    protected = json.loads((REPAIR / 'protected-files.json').read_text())
    protected.update(json.loads((REPAIR / 'source-files.json').read_text()))
    protected.update({str(path): file_hash(path) for path in paths.values()})
    protected[str(OUT / 'viewer-verification.json')] = file_hash(OUT / 'viewer-verification.json')
    code = {str(HERE / name): file_hash(HERE / name) for name in
            ('verify_repair.py', 'viewer_runtime.py', 'viewer_catalog.py', 'lib/evaluation.py',
             'lib/heads.py', 'lib/brain_model.py', 'lib/curriculum.py', 'lib/game.py')}
    protocol = dict(question='Does autonomous B00 success depend on learned internal parameters and state-dependent neural output?',
        unit='Three fitted checkpoints from the same source brain; repeated interventions are paired technical controls.',
        training_performed=False, task='B00', fixed=['board input', 'legal-action mask', 'head weights and normalization',
        'simulation duration and per-decision reset', 'graph', 'rollout stopping rule'], schedule=schedule,
        primary_endpoint='Autonomous fixed-start completion', secondary_endpoints=['steps', 'teacher actions', 'minimum action margin'],
        restoration='Fresh disk reload of the original learned core after EVERY intervention; exact actions and logits required.',
        signal_controls=dict(zero_signal='Replace all 512 raw features by zero.',
            mean_signal='Replace features by frozen head mean, leaving readout bias and legal mask.',
            constant_signal='Use learned-brain features of the initial board at every decision.'),
        shuffle='Permute repair deltas among groups, with one shared group permutation for all three parameter families.',
        permutation_seeds=list(range(381, 391)), randomization_seed=100102, runtime_limit_seconds=300,
        no_parameter_updates=True, protected_hashes=protected, code_hashes=code,
        limits=['Single trained task, not unseen-map generalization.', 'Does not establish a unique advantage of the biological topology.',
                'The frozen readout and legal-action mask remain necessary components.'])
    save_json(out / 'protocol.json', protocol)
    started = time.monotonic()

    def check():
        if (out / 'STOP').exists():
            raise KeyboardInterrupt('Intervention STOP marker found')
        if time.monotonic() - started > 300:
            raise TimeoutError('Intervention audit exceeded its 300-second bound')

    runtime = None
    try:
        runtime = ViewerRuntime()
        head, encoder = runtime.head, runtime.encoder
        frozen_head = tensor_hash(head.state_dict())
        anchor = runtime.objects['repair_source']['core']
        with np.load(REPAIR / 'teacher-B00.npz', allow_pickle=False) as archive:
            data = {key: archive[key] for key in ('boards', 'labels', 'legal')}
        baselines, start_features, findings = {}, {}, []
        for kind in ('repair10', 'repair11', 'repair12'):
            encoder.set_core(runtime.objects[kind]['core'])
            baseline = evaluate(encoder, head, [level], kind, check)
            assert same_rollout(baseline, trials[kind]['after']) and baseline['solved'] == 1
            baselines[kind] = baseline
            start_features[kind] = encoder.encode(np.stack([encode(restore(level), state_from(level['start']))]), check)[0]
        for index, item in enumerate(schedule):
            check()
            kind, condition = item['model'], item['condition']
            core = substituted_core(runtime.objects[kind]['core'], anchor, condition, item['permutation_seed'])
            encoder.set_core(core)
            signal = (np.zeros(512, dtype='float32') if condition == 'zero_signal' else
                      head.mean.detach().cpu().numpy() if condition == 'mean_signal' else
                      start_features[kind] if condition == 'constant_signal' else None)
            source = encoder if signal is None else FixedSignal(signal)
            result = evaluate(source, head, [level], kind + ':' + condition, check)
            teacher = teacher_check(source, head, data, condition, check)
            restored = torch.load(paths[kind], map_location='cpu', weights_only=True)
            assert tensor_hash(restored['head']) == frozen_head
            encoder.set_core(restored['core'])
            recovery = evaluate(encoder, head, [level], kind + ':restored', check)
            assert same_rollout(recovery, baselines[kind]), (kind, condition)
            assert tensor_hash(head.state_dict()) == frozen_head
            record = dict(**item, result=result, teacher=teacher, restoration=recovery,
                          restoration_actions_and_logits_exact=True, core_in_bounds=encoder.core.bound_audit()['in_bounds'])
            findings.append(record)
            save_json(out / 'interventions.json', findings)
            save_json(out / 'progress.json', dict(status='substitutions', completed=index + 1, total=len(schedule),
                                                 model=kind, condition=condition, seconds=time.monotonic() - started))
            print(json.dumps(dict(model=kind, condition=condition, solved=result['solved'], restored=recovery['solved'],
                                  completed=index + 1, total=len(schedule))), flush=True)

        # Exercise the actual trace-producing viewer, not a saved route.
        viewer_results, trace_checks = {}, 0
        for kind in REPAIR_MODELS:
            check()
            encoder.set_core(runtime.objects[kind]['core'])
            actual = evaluate(encoder, head, [level], kind, check)
            runtime.cache.clear()
            runtime.kind = None
            m, state = restore(level), state_from(level['start'])
            for expected in actual['records'][0]['decisions']:
                check()
                feature, rates, trace = runtime.response(encode(m, state), kind)
                prediction = decide(head, feature, legal(m, state), runtime.ids[kind])
                assert prediction['action'] == expected['action']
                np.testing.assert_array_equal(prediction['brain_logits'], expected['brain_logits'])
                assert len(trace['events']) == 41 and rates.shape[0] == 2
                assert trace['checkpoint_id'] == runtime.ids[kind]
                trace_checks += 1
                state = step(m, state, prediction['action'])
            viewer_results[kind] = actual
        # A -> source -> B -> zero -> C -> A also verifies the checkpoint-keyed trace cache.
        board = encode(restore(level), state_from(level['start']))
        switch_features = {}
        for kind in ('repair10', 'repair_source', 'repair11', 'zero', 'repair12', 'repair10'):
            observed = runtime.response(board, kind)[0]
            encoder.set_core(runtime.objects[kind]['core'])
            runtime.kind = None
            independent = encoder.encode(board[None], check)[0]
            np.testing.assert_array_equal(observed, independent)
            switch_features[kind] = observed
        assert not np.array_equal(switch_features['repair10'], switch_features['repair_source'])
        old_levels = json.loads((ARCHIVE / 'campaign/pilot-levels.json').read_text())
        encoder.set_core(runtime.objects['brain']['core'])
        old_pilot = evaluate(encoder, head, old_levels, 'pilot-regression', check)
        historic = json.loads((OUT / 'viewer-verification.json').read_text())['results']['brain']
        assert [(r['status'], r['actions']) for r in old_pilot['records']] == [(r['status'], r['actions']) for r in historic['records']]
        assert old_pilot['solved'] == 4 and encoder.graph_digest() == encoder.base_digest
        assert tensor_hash(head.state_dict()) == frozen_head
        assert all(file_hash(path) == digest for path, digest in {**protected, **code}.items())
        groups = defaultdict(list)
        for row in findings:
            groups[row['condition']].append(row)
        summary = [dict(condition=condition, solved=sum(r['result']['solved'] for r in rows),
                        total=len(rows), restored=sum(r['restoration']['solved'] for r in rows),
                        by_model={kind: dict(solved=sum(r['result']['solved'] for r in rows if r['model'] == kind),
                                            total=sum(r['model'] == kind for r in rows)) for kind in baselines})
                   for condition, rows in groups.items()]
        report = dict(status='verified', training_performed=False, baselines=baselines, summary=summary,
            restoration_checks=len(findings), all_restorations_exact=True, frozen_head=True,
            graph_unchanged=True, protected_files_unchanged=True, results=viewer_results,
            checkpoint_ids={k: runtime.ids[k] for k in REPAIR_MODELS}, trace_decisions_verified=trace_checks,
            checkpoint_switch_cache_verified=True, original_pilot_solved=old_pilot['solved'],
            wall_seconds=time.monotonic() - started, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            interpretation_limits=protocol['limits'])
        save_json(out / 'verification.json', report)
        save_json(out / 'progress.json', dict(status='verified', seconds=report['wall_seconds']))
        print(json.dumps(dict(status='verified', summary=summary, trace_decisions=trace_checks, seconds=report['wall_seconds'])), flush=True)
    except BaseException as error:
        save_json(out / 'progress.json', dict(status='error', error=repr(error), seconds=time.monotonic() - started))
        raise
    finally:
        if runtime is not None:
            runtime.encoder.release_graph()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(INTERVENTION))
    main(parser.parse_args())
