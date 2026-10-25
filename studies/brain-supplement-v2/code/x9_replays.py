"""X9 replays (plan v2, Phase 1C; gaps G06 and G10 of manuscript/results/qa/gaps.csv). GPU host.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x9_replays.py

For each of the 24 Fig. 2 maps (mechanisms-v1 difficulty models), with the frozen mechanism policy (mechanism_model
.Frozen, batch 16) and the frozen original-start rollout (static_rollout, 512 actions):
- G06, brain x readout pairings including the mid-training checkpoint: brain (core) from before, mid or after training
  x readout (head) from before, mid or after (9 pairings; the paper's Fig. 3c replayed 4). The 4 frozen pairings must
  reproduce mechanisms-v1/interpretation/four-combination-behavior.csv (checked by analyze_x9.py).
- G10, behaviour at the stage-aligned checkpoints: every curriculum-stage champion (studies/brain-sokoban-v1/runs/
  difficulty-<map>/suffix-<depth>.pt, pushed with tools/remote/lists/x9.txt) replayed from the map's original start.
Outputs runs/x9-replays/<map>.json (status, steps, first action and first-decision logits per replay); resumable.
"""
import gc
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
FROZEN = ROOT / 'studies' / 'brain-sokoban-v1'
MECH = FROZEN / 'mechanisms-v1'
OUT = SUPP / 'runs' / 'x9-replays'
PHASES = ('before', 'mid', 'after')


def summary(r):
    d0 = r['decisions'][0] if r.get('decisions') else None
    return dict(status=r['status'], steps=r.get('steps', len(r.get('decisions', []))), first_action=d0.get('action') if d0 else None,
                first_logits=d0.get('logits') if d0 else None)


def main():
    sys.path.insert(0, str(MECH / 'code'))
    import mechanism_base as mb
    import torch
    from backend import set_pair
    from mechanism_model import Frozen, static_rollout
    from mechanism_worker import task_level
    OUT.mkdir(parents=True, exist_ok=True)
    models = [m for m in mb.read(MECH / 'models.json') if m['id'].startswith('difficulty-')]
    for model in sorted(models, key=lambda m: m['id']):
        mid_ = model['id'].replace('difficulty-', '')
        target = OUT / f'{mid_}.json'
        if target.exists():
            continue
        t0 = time.time()
        level = task_level(model['task'])
        policy = Frozen(model, OUT / 'meta' / mid_, batch=16)
        pairs = {ph: mb.load_pair(MECH / model['checkpoints'][ph]['path']) for ph in PHASES}
        rec = dict(map=mid_, model=model['id'], task=model['task'], pairings=[], stages=[])
        for b in PHASES:
            for h in PHASES:
                set_pair(policy.encoder, policy.head, dict(core=pairs[b]['core'], head=pairs[h]['head']))
                rec['pairings'].append(dict(brain=b, readout=h, **summary(static_rollout(policy, level))))
        run = FROZEN / 'runs' / model['id']
        stage_files = sorted(run.glob('suffix-*.pt'), key=lambda p: int(re.search(r'suffix-(\d+)', p.name).group(1)))
        for p in stage_files:
            saved = torch.load(p, map_location='cpu', weights_only=True)
            set_pair(policy.encoder, policy.head, dict(core=saved['core'], head=saved['head']))
            rec['stages'].append(dict(depth=int(saved['depth']), file=p.name, **summary(static_rollout(policy, level))))
        rec['stage_files_found'] = len(stage_files)
        rec['seconds'] = time.time() - t0
        target.write_text(json.dumps(rec), encoding='utf-8', newline='\n')
        print(f"{mid_}: pairings {[(x['brain'][0] + x['readout'][0], x['status']) for x in rec['pairings']]}, "
              f"{len(stage_files)} stages, solved {sum(x['status'] == 'solved' for x in rec['stages'])}, {rec['seconds']:.0f} s", flush=True)
        del policy
        gc.collect()                   # operators.apply leaves a reference cycle (enc.bind); free the GPU memory now
        torch.cuda.empty_cache()
    (OUT / 'done.json').write_text(json.dumps(dict(finished_unix=time.time())), encoding='utf-8')


if __name__ == '__main__':
    main()
