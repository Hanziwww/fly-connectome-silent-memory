"""Read-only analysis of a completed, independently verified joint-linear run."""
from workspace import ARCHIVE, RUNS
import argparse
from pathlib import Path

import torch

from bootstrap import HERE
from common import file_hash, save_json
from game import restore, solved, state_from, step
from run_linear_joint import integrity, read
from workspace import recorded_path


def board_text(level, state=None):
    game = restore(level)
    p, boxes = state_from(level['start']) if state is None else state
    size, offset = level['size'], (10-level['size'])//2
    rows = []
    for r in range(offset, offset+size):
        cells = []
        for c in range(offset, offset+size):
            cell = r*10+c
            token = '#' if cell in game['walls'] else ' '
            if cell in game['goals']:
                token = '.'
            if cell in boxes:
                token = '*' if cell in game['goals'] else '$'
            if cell == p:
                token = '+' if cell in game['goals'] else '@'
            cells.append(token)
        rows.append(''.join(cells))
    return '\n'.join(rows)


def trace_summary(level, record):
    game, state = restore(level), state_from(level['start'])
    seen, cycle = {state:0}, None
    for i, action in enumerate(record['actions'], 1):
        state = step(game, state, action)
        if state is None:
            raise AssertionError('Saved neural rollout contains an illegal move')
        if state in seen and cycle is None:
            cycle = dict(first_visit=seen[state], repeated_at=i, length=i-seen[state])
        seen[state] = i
    assert solved(game, state) == (record['status']=='solved')
    return dict(status=record['status'], steps=record['steps'], actions=record['actions'],
                final_state=[state[0],list(state[1])], boxes_on_goals=len(set(state[1]) & set(game['goals'])),
                cycle=cycle, final_board=board_text(level,state))


def analyze(out):
    config = integrity(out)
    result, verification = read(out/'result.json'), read(out/'verification.json')
    assert verification['status']=='verified'
    assert file_hash(out/'model.pt')==result['model_sha256']==verification['model_sha256']
    level, old = read(out/'level.json'), read(out/'previous-level.json')
    model = torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    assert result['checks']['source_and_code_unchanged'] and verification['actions_logits_scores_exact']
    current = trace_summary(level,result['after_suite']['records'][0])
    baseline = trace_summary(level,result['before_suite']['records'][0])
    ablations = {name:trace_summary(level,suite['records'][0]) for name,suite in result['substitutions'].items()}
    certified = [p['depth'] for p in verification['promotions']]
    cpu = read(out/'cpu-validation.json') if (out/'cpu-validation.json').exists() else dict(status='not_recorded')
    if cpu['status']=='passed':
        assert cpu['code_manifest_sha256']==file_hash(out/'source-files.json')
    summary = dict(status='complete', primary_outcome=current, before=baseline, task={
        key:level[key] for key in ('id','size','boxes','floor_cells','optimal_steps','start','goals')},
        source_checkpoint_sha256=config['source_checkpoint_sha256'], model_sha256=result['model_sha256'],
        seed=config['seed'], training=result['training'], current_depth=model['lesson_depth'],
        independently_verified_suffix_depths=certified, core_changes=result['checks']['core_changes'],
        head_changes=result['checks']['head_changes'], substitutions=ablations,
        previous_task_before=trace_summary(old,result['before_suite']['records'][1]),
        previous_task_after=trace_summary(old,result['old_task_after']['records'][0]),
        resources=result['resources'], initialization_seconds=result['initialization_seconds'],
        training_worker_seconds=result['wall_seconds'], verification_seconds=verification['wall_seconds'],
        supervisor_seconds=read(out/'supervisor.json')['wall_seconds'], cpu_validation=cpu,
        limitations=['One selected task and seed, inherited from a successful P01 model',
          'Simultaneous gradient learning uses a surrogate derivative for hard spikes',
          'A finite failed run is not a brain or affine-readout capacity bound',
          'Paired substitutions test this saved model; they do not compare independently retrained alternatives'])
    save_json(out/'analysis.json',summary)
    train=result['training']
    total_core=sum(summary['core_changes'].values())
    total_head=sum(summary['head_changes'].values())
    lines=[f"# {level['id']} joint brain and linear-readout experiment",'',
        (f"**{level['id']} was solved autonomously.**" if current['status']=='solved' else f"**{level['id']} was not solved within this bounded training attempt.**"),
        '',f"The inherited {old['id']} model initially ended with `{baseline['status']}` after {baseline['steps']} moves. "
        f"The verified final model ended with `{current['status']}` after {current['steps']} moves. "
        f"Training used {train['training_seconds']:.3f} seconds and {train['joint_updates']} joint Adam updates.",
        '', f"| Property | Previous {old['id']} | Selected {level['id']} |",'|---|---:|---:|',
        f"| Board | {old['size']} x {old['size']} | {level['size']} x {level['size']} |",
        f"| Boxes | {old['boxes']} | {level['boxes']} |",
        f"| Walkable cells | {100-len(old['walls'])} | {level['floor_cells']} |",
        f"| Certified minimum moves | {old['optimal_steps']} | {level['optimal_steps']} |",
        '', 'The minimum is a difficulty descriptor only. Any autonomous solution is accepted, including detours.',
        '', '```text', board_text(level), '```',
        '`#` wall; `@` player; `$` box; `.` goal; `*` box on goal.', '',
        '## Training and verification', '',
        f"The run starts from the verified linear {old['id']} checkpoint, with seed {config['seed']}. "
        'Both the 35,310 internal plasticity parameters and the 2,052 affine readout parameters receive '
        'gradients from one task loss and are updated in the same Adam step. '
        'Core learning rate is 0.0001; readout learning rate is 0.0003. '
        'The connectome, neural dynamics and activity normalization are preserved; gains and threshold shifts '
        'remain inside their existing bounds. Core gradients use the existing surrogate spike derivative.', '',
        'The loss retains the previous softplus entry and squared anchor-recovery terms. '
        'The P01-specific critical state has been replaced by a task-independent suffix curriculum. '
        'Successful neural rollouts provide recovery anchors. Original-start success overrides all auxiliary gates. '
        f"Old {old['id']} performance is diagnostic and does not gate training.", '',
        f"Actual saved parameter changes: **{total_core:,} internal parameters** and **{total_head:,} readout parameters**. "
        f"Nonzero gradient steps: core {train['core_nonzero_gradient_steps']}/{train['joint_updates']}, "
        f"readout {train['head_nonzero_gradient_steps']}/{train['joint_updates']}. "
        f"Maximum activity discrepancy in the two-pass gradient calculation: {train['maximum_recompute_error']:.3g}.", '',
        f"Independently verified promoted suffix depths: {certified}. Current training depth: {model['lesson_depth']}.", '',
        'A separate process reloaded final and promoted checkpoints and reproduced neural actions, logits, scores '
        'and substitutions exactly. Inference receives only the board through the full brain; the affine readout '
        'receives only 512 neural activities. There is no solver, reference-path replay, timestep input or direct '
        'board connection to the action head. The solver supplies training/certification data only.', '',
        f"CPU validation receipt: {cpu['status']}.", '',
        '## Saved-policy substitutions', '',
        f"| Parameter pairing | {level['id']} outcome | Moves | Boxes on goals at termination |",
        '|---|---|---:|---:|',
        f"| Final brain + final linear readout | {current['status']} | {current['steps']} | {current['boxes_on_goals']} |"]
    for name, r in ablations.items():
        lines.append(f"| {name} | {r['status']} | {r['steps']} | {r['boxes_on_goals']} |")
    lines += ['', 'The source-brain substitution specifically tests whether the newly changed core is required by '
        'this saved final readout. It does not test a newly retrained readout on that source brain. '
        'The zero-plasticity substitution preserves the graph while clearing all learned gains and threshold shifts.', '',
        f"Previous {old['id']} after training: `{summary['previous_task_after']['status']}`, "
        f"{summary['previous_task_after']['steps']} moves (diagnostic only).", '',
        '## Resource accounting', '',
        f"- Training: {train['training_seconds']:.3f} s of a {config['training_seconds']:.0f} s maximum.",
        f"- Initialization and baseline: {result['initialization_seconds']:.3f} s.",
        f"- Full training worker: {result['wall_seconds']:.3f} s; independent verification: {verification['wall_seconds']:.3f} s.",
        f"- Total supervisor wall time: {summary['supervisor_seconds']:.3f} s.",
        f"- Peak training allocated/reserved VRAM: {result['resources']['peak_allocated_gib']:.3f}/"
        f"{result['resources']['peak_reserved_gib']:.3f} GiB; allocator limit 11 GiB.", '',
        'Training time includes optimization and curriculum checks; baseline and final independent verification '
        'are excluded. Feature recomputation stores one differentiable simulation batch at a time. '
        'No activation cache crosses an internal parameter update.', '',
        '## Scope and artifacts', '',
        'This is one task and one seed inherited from a learned model. A failure at the time limit does not '
        'establish a capacity boundary. Joint parameter changes alone do not establish that both updates are '
        'causally necessary. The substitutions and independent reload are reported separately.', '',
        '- [Protocol](protocol.json), [task certificate](task-certificate.json), [analysis](analysis.json).',
        '- [Checkpoint](model.pt), [result](result.json), [verification](verification.json).',
        '- [Training history](history.jsonl), [gradient history](gradients.jsonl).', '',
        'The joint viewer loads this checkpoint with ActivityLinear and provides the verified '
        'substitution controls. Start it with `sokoban/serve_joint.py --open --port 8774` or '
        'the desktop launcher. Playback performs inference only.', '',
        '```powershell',
        r'.\flybrain\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s sokoban -p test_linear_joint.py -v',
        (r'.\flybrain\.venv\Scripts\python.exe -B -X utf8 sokoban/run_linear_joint.py --out runs/linear-joint/NEW-RUN --seconds 600' if level['id']=='P03' else
         f".\\flybrain\\.venv\\Scripts\\python.exe -B -X utf8 workflow.py train --source {recorded_path(config['source']).relative_to(HERE.parent).as_posix()} --task {level['id']} --out runs/linear-joint/NEW-RUN --seconds {int(config['training_seconds'])}"),
        r'.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py analyze --out runs/linear-joint/NEW-RUN',
        '```']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    import json
    result=analyze(args.out.resolve())
    print(json.dumps({k:result[k] for k in ('status','primary_outcome','core_changes','head_changes','current_depth')},indent=2))
