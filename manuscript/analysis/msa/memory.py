"""Per-history memory outcomes extracted from the raw remote records (one vocabulary for both batches).

Traps handled here:
  * remote batch 1 rollouts have no first-push field: it is derived from the decisions
    (a push happens when player + DELTAS[action] is a box cell; correct push 2 for cue 0, 3 for cue 1);
  * supplement training rollouts overwrite ``cue`` with the case index: the true cue is ``cue_label``
    (observation records keep the true cue);
  * ``first_push_correct = None`` means the box never moved (not a wrong push);
  * the two batches name irreversible failures differently (harmonized to irreversible_failure).
"""
from functools import lru_cache

from . import io, paths, taxonomy

DELTAS = (-10, 10, -1, 1)
M0_PUSH = {0: 2, 1: 3}
M0_CHOICE_TARGET = {0: 3, 1: 2}   # first-step response target used by the original probes

MODELS = [
    # model, root, family, learner, kind, reference
    ('remote-dynamic-M0-head_only', 'R0', 'M0', 'connectome_readout_only', 'connectome', True),
    ('remote-dynamic-M0-joint', 'R0', 'M0', 'connectome_joint', 'connectome', False),
    ('remote-dynamic-M0-reset', 'R0', 'M0', 'connectome_readout_only_reset_state', 'connectome', False),
    ('remote-dynamic-M0-carry_untrained', 'R0', 'M0', 'connectome_untrained_readout', 'connectome', False),
    ('remote-dynamic-M0-gru', 'R0', 'M0', 'gru', 'gru', False),
    ('supp-M0-visibility', 'RS', 'M0vis', 'connectome_shared_visibility_readout', 'connectome', True),
    ('supp-S1-ring-brain', 'RS', 'S1', 'connectome_readout_only', 'connectome', True),
    ('supp-S1-ring-gru', 'RS', 'S1', 'gru', 'gru', False),
    ('supp-origin-S1-ring-p05-fresh', 'RS', 'S1', 'connectome_new_readout', 'connectome', False),
    ('supp-origin-S1-ring-neutral-real', 'RS', 'S1', 'neutral_core_real_graph', 'connectome', False),
    ('supp-origin-S1-ring-neutral-rewired', 'RS', 'S1', 'neutral_core_rewired_graph', 'connectome', False),
    ('supp-origin-S1-ring-leaky', 'RS', 'S1', 'fixed_decaying_trace', 'trace', False),
    ('supp-transfer-S1-ring', 'RS', 'S1', 'zero_update_transfer_from_M0', 'connectome', False),
    ('supp-S2-context-brain', 'RS', 'S2', 'connectome_readout_only', 'connectome', True),
    ('supp-S2-context-gru', 'RS', 'S2', 'gru', 'gru', False),
    ('supp-origin-S2-context-p05-fresh', 'RS', 'S2', 'connectome_new_readout', 'connectome', False),
    ('supp-origin-S2-context-neutral-real', 'RS', 'S2', 'neutral_core_real_graph', 'connectome', False),
    ('supp-origin-S2-context-neutral-rewired', 'RS', 'S2', 'neutral_core_rewired_graph', 'connectome', False),
    ('supp-origin-S2-context-leaky', 'RS', 'S2', 'fixed_decaying_trace', 'trace', False),
    ('supp-transfer-S2-context', 'RS', 'S2', 'zero_update_transfer_from_M0', 'connectome', False),
]
MODEL_INFO = {m[0]: dict(root=m[1], family=m[2], learner=m[3], kind=m[4], reference=m[5]) for m in MODELS}


def root(model):
    return paths.R0 if MODEL_INFO[model]['root'] == 'R0' else paths.RS


def split_label(s):
    return 'train' if s == 'train' else 'untrained'


def derive_first_push(decisions):
    for d in decisions:
        player, boxes = int(d['state'][0]), [int(b) for b in d['state'][1]]
        if player + DELTAS[int(d['action'])] in boxes:
            return int(d['action'])
    return None


@lru_cache(maxsize=None)
def verification(model):
    return io.json_load(root(model) / model / 'verification.json')


@lru_cache(maxsize=None)
def observations(model):
    return io.jsonl_load(root(model) / model / 'observations.jsonl')


def baseline_histories(model):
    """One record per evaluated history without intervention."""
    info = MODEL_INFO[model]
    src = root(model) / model
    rows = []
    if info['root'] == 'R0':
        v = verification(model)
        res = v['results']
        margins = {}
        for pr in res['probes']:
            if pr.get('intervention') is not None:
                continue
            for rec in pr['records']:
                cue = int(rec['cue'])
                lg = rec['logits']
                margins[(cue, int(pr['delay_ms']), int(pr['prefix_windows']))] = (
                    lg[M0_CHOICE_TARGET[cue]] - lg[M0_CHOICE_TARGET[1 - cue]], bool(rec['correct']))
        for ro in res['rollouts']:
            cue = int(ro['cue'])
            push = derive_first_push(ro['decisions'])
            key = (cue, int(ro['delay_ms']), int(ro['prefix_windows']))
            m = margins.get(key, (None, None))
            rows.append(dict(model=model, case=f'c0-cue{cue}', cue=cue, context=0, visible=False, delay_ms=key[1], prefix_windows=key[2],
                             split=split_label(ro['history_split']), raw_status=ro['status'], status=taxonomy.behavior(ro['status']),
                             solved=ro['status'] == 'solved', steps=int(ro['steps']), first_push_action=push,
                             first_push_correct=(None if push is None else push == M0_PUSH[cue]),
                             first_decision_margin=m[0], first_decision_correct=m[1], margin_kind='first_step_choice_logit_difference',
                             source_file=paths.rel(src / 'verification.json')))
    elif (src / 'verification.json').exists():
        v = verification(model)
        res = v['results']
        margins = {}
        for pr in res['probes']:
            if pr.get('intervention') is not None:
                continue
            for rec in pr['records']:
                margins[(rec['case'], int(pr['delay_ms']), int(pr['prefix_windows']))] = (rec.get('entry_margin'), bool(rec['correct']))
        for ro in res['rollouts']:
            if ro.get('intervention') is not None:
                continue
            key = (ro['case'], int(ro['delay_ms']), int(ro['prefix_windows']))
            m = margins.get(key, (None, None))
            rows.append(dict(model=model, case=ro['case'], cue=int(ro['cue_label']), context=int(ro['context']), visible=bool(ro['visible']),
                             delay_ms=key[1], prefix_windows=key[2], split=split_label(ro['history_split']), raw_status=ro['status'],
                             status=taxonomy.behavior(ro['status']), solved=ro['status'] == 'solved', steps=int(ro['steps']),
                             first_push_action=ro.get('first_push_action'), first_push_correct=ro.get('first_push_correct'),
                             first_decision_margin=m[0], first_decision_correct=m[1], margin_kind='entry_margin_first_move',
                             source_file=paths.rel(src / 'verification.json')))
    else:
        for line in observations(model):
            if line.get('intervention') is not None:
                continue
            d, p = int(line['delay_ms']), int(line['prefix_windows'])
            for rec in line['records']:
                rows.append(dict(model=model, case=rec['case'], cue=int(rec['cue']), context=int(rec['context']), visible=bool(rec['visible']),
                                 delay_ms=d, prefix_windows=p, split=('train' if d in (0, 40, 160) and p in (0, 1, 4) else 'untrained'),
                                 raw_status=rec['status'], status=taxonomy.behavior(rec['status']), solved=bool(rec['solved']),
                                 steps=int(rec['steps']), first_push_action=rec.get('first_push_action'),
                                 first_push_correct=rec.get('first_push_correct'), first_decision_margin=None, first_decision_correct=None,
                                 margin_kind=None, source_file=paths.rel(src / 'observations.jsonl')))
    for r in rows:
        r.update(family=info['family'], learner=info['learner'], kind=info['kind'], reference=info['reference'])
    return rows


def dense_delay_histories():
    """supp-M0-delay: delays 0-640 ms in 20-ms steps x prefixes 0/1/2/4/8 x 2 cues (head_only checkpoint)."""
    src = paths.RS / 'supp-M0-delay' / 'observations.jsonl'
    rows = []
    for line in io.jsonl_load(src):
        if line.get('intervention') is not None:
            continue
        d, p = int(line['delay_ms']), int(line['prefix_windows'])
        for rec in line['records']:
            rows.append(dict(delay_ms=d, prefix_windows=p, case=rec['case'], cue=int(rec['cue']), status=taxonomy.behavior(rec['status']),
                             solved=bool(rec['solved']), steps=int(rec['steps']), first_push_action=rec.get('first_push_action'),
                             first_push_correct=rec.get('first_push_correct'), source_file=paths.rel(src)))
    return rows
