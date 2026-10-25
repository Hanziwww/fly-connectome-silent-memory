"""User-facing frozen mechanism workflow; no training dispatch is reachable."""
import argparse
import json
from mechanism_base import SESSION, read


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','status','sync','analyze','report'])
    p.add_argument('--remote',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    if a.action=='prepare':
        from mechanism_prepare import prepare
        plan=prepare();print(json.dumps({'jobs':len(plan['jobs']),'mode':plan['mode']}));return 0
    if a.action in ('analyze','report'):
        from mechanism_analysis import analyze
        print(json.dumps(analyze(),ensure_ascii=False));return 0
    if a.remote or a.action=='sync':
        from mechanism_transport import dispatch
        return dispatch(a.action,a.resume)
    if a.action=='status':
        from mechanism_queue import status
        print(json.dumps(status(),ensure_ascii=False));return 0
    from mechanism_queue import run
    return run(a.resume)


if __name__=='__main__':
    raise SystemExit(main())
