"""Commands for the registered, single-seed brain / Sokoban study."""
import argparse
from registry import STUDY, read


def main():
    # Frozen mechanism studies have their own queue and STOP marker. They never
    # dispatch to the training supervisor or modify its paused state.
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'mechanisms-supplement':
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mechanisms-supplement-v1' / 'code'))
        from supplement import main as supplement_main
        return supplement_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'mechanisms':
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mechanisms-v1' / 'code'))
        from mechanisms import main as mechanisms_main
        return mechanisms_main(sys.argv[2:])
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','benchmark','train','evaluate','analyze','run','status','test','validate','worker'])
    p.add_argument('--run-id')
    p.add_argument('--module')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--verify',action='store_true')
    p.add_argument('--config')
    a=p.parse_args()
    if a.command=='prepare':
        from tasks import prepare,separate_wall_families
        from memory_tasks import prepare_memory
        prepare();prepare_memory();separate_wall_families()
    elif a.command=='test':
        import unittest
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(STUDY/'code'),pattern='test_*.py'))
        return 0 if result.wasSuccessful() else 1
    elif a.command=='validate':
        from validation import validate
        return validate()
    elif a.command=='status':
        import json
        for name in ['supervisor.json','backend.json','summary.json']:
            path=STUDY/name
            if path.exists(): print(json.dumps({name:read(path)},ensure_ascii=False))
        if (STUDY/'supervisor.json').exists():
            current=read(STUDY/'supervisor.json').get('run_id')
            if current and (STUDY/'runs'/current/'progress.json').exists():
                progress=read(STUDY/'runs'/current/'progress.json')
                keys=('status','depth','updates','elapsed','solved','total','original_status','original_steps')
                print(json.dumps({'current_run':current,'progress':{k:progress[k] for k in keys if k in progress}},ensure_ascii=False))
    else:
        from execution import dispatch
        return dispatch(a)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
