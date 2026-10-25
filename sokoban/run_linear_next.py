"""Bounded continuation of a verified brain-plus-linear checkpoint on a harder task."""
from workspace import ARCHIVE, RUNS
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

from bootstrap import HERE, ROOT
from common import file_hash, save_json
from game import restore, solved, state_from
from lesson_tasks import trace
from linear_joint_tasks import SuffixCurriculum
from progressive_tasks import forward_certificate, reverse_distances
from run_boundary import supervise
from run_cycle import acquire_worker_lock
import run_linear_joint as base
from workspace import CURRENT_RUN, recorded_path, source_manifest


def prepare(args, out):
    source = args.source.resolve()
    verification = base.read(source/'verification.json')
    assert verification['status']=='verified' and verification['original_solved']
    assert file_hash(source/'model.pt')==verification['model_sha256']
    # Verify the inherited evidence before starting another continuous stage.
    base.integrity(source)
    level = next(l for l in base.read(base.LEVELS) if l['id']==args.task)
    previous = base.read(source/'level.json')
    difficulty = lambda l: (l['boxes'],l['size'],l['optimal_steps'])
    assert all(a<=b for a,b in zip(difficulty(previous),difficulty(level)))
    assert difficulty(previous)!=difficulty(level)
    distances, certificate = reverse_distances(level)
    assert distances[state_from(level['start'])]==level['optimal_steps']
    certificate['independent_forward_bfs']=forward_certificate(level)
    lessons=SuffixCurriculum(level)
    game=restore(level)
    for depth in range(1,len(lessons.witness)+1):
        suffix=lessons.at_depth(depth)
        _,end=trace(game,state_from(suffix['start']),level['reference_actions'][-depth:])
        assert solved(game,end)
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source/'model.pt',out/'source.pt')
    shutil.copyfile(source/'level.json',out/'previous-level.json')
    save_json(out/'level.json',level)
    save_json(out/'task-certificate.json',certificate)
    # Freeze the code executed by this new run. The inherited run retains its
    # historical source manifest and pre-organization snapshot.
    save_json(out/'source-files.json',source_manifest())
    config=dict(base.read(source/'config.json'))
    config.update(source=str(source),source_checkpoint_sha256=file_hash(out/'source.pt'),
        task_id=level['id'],previous_task_id=previous['id'],training_seconds=args.seconds,
        curriculum=f"Generic reverse suffixes 1..{len(lessons.witness)}; original autonomous success overrides all auxiliary gates",
        source_files={str(p):file_hash(p) for p in (source/'model.pt',source/'verification.json',source/'result.json',source/'level.json',base.LEVELS)},
        copied_files={name:file_hash(out/name) for name in ('source.pt','previous-level.json','level.json','task-certificate.json')},
        code_manifest_scope='Current application, shared library, command entrypoint and formal method registry',
        suffix_witness_checks=len(lessons.witness),no_automatic_retry=True)
    save_json(out/'config.json',config)
    save_json(out/'protocol.json',dict(status='prepared',**config,
        substitutions=['source_brain_final_readout','final_brain_source_readout','zero_plasticity_final_readout'],
        interpretation='Substitutions test the saved pairing, not independently retrained alternatives'))
    print(json.dumps(dict(event='prepared',task=level['id'],previous=previous['id'],size=level['size'],
        boxes=level['boxes'],floor_cells=level['floor_cells'],reference_moves=level['optimal_steps'],
        suffix_witness_checks=len(lessons.witness))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='New result directory; completed attempts cannot be overwritten')
    parser.add_argument('--source',type=Path,default=CURRENT_RUN)
    parser.add_argument('--task',required=True,help='Explicit next task ID from the saved difficulty ladder')
    parser.add_argument('--seconds',type=float,default=1200)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    out=args.out.resolve()
    if args.worker or args.verify:
        config=base.integrity(out)
        base.SOURCE=recorded_path(config['source'])
        return base.worker(out,args.verify)
    if not 5<=args.seconds<=1200:
        parser.error('Use a 5..1200 second bounded continuation')
    if not args.resume:
        prepare(args,out)
    config=base.integrity(out)
    assert config['training_seconds']==args.seconds
    assert config['task_id']==args.task
    if args.prepare_only:
        return 0
    lock,started=acquire_worker_lock(),time.monotonic()
    identity=dict(status='starting',supervisor_pid=os.getpid(),phase='training')
    command=[sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve()),'--out',str(out),'--task',args.task]
    try:
        if (out/'result.json').exists():
            assert base.read(out/'result.json')['status'] in base.DONE,'Failed attempts are preserved'
        else:
            code,state=supervise(command+['--worker'],out/'worker.log',out,args.seconds+180,identity)
            if code or state!='running':
                identity['status']='stopped' if state=='stopped' else 'error'
                return 1
        if not (out/'verification.json').exists():
            identity['phase']='independent-verification'
            code,state=supervise(command+['--verify'],out/'verification.log',out,600,identity)
            if code or state!='running':
                identity['status']='error'
                return 1
        identity['status']='complete'
        print(json.dumps(dict(event='complete',verification=base.read(out/'verification.json'))),flush=True)
        return 0
    finally:
        identity['wall_seconds']=time.monotonic()-started
        save_json(out/'supervisor.json',identity)
        lock.close()


if __name__=='__main__':
    raise SystemExit(main())
