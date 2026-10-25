import argparse
import json
from supp_base import prepare


def main(argv=None):
    p = argparse.ArgumentParser(description='Frozen mechanism follow-up; no training entry point')
    p.add_argument('action', choices=['prepare', 'run', 'status', 'sync', 'analyze'])
    p.add_argument('--remote', action='store_true')
    a = p.parse_args(argv)
    if a.action == 'prepare':
        plan = prepare(); print(json.dumps({'jobs': len(plan['jobs']), 'mode': plan['mode']})); return 0
    if a.action == 'analyze':
        from supp_analysis import analyze
        print(json.dumps(analyze(), ensure_ascii=False)); return 0
    if a.remote or a.action == 'sync':
        from supp_transport import dispatch
        return dispatch(a.action)
    if a.action == 'run':
        p.error('Use run --remote; local training stays paused')
    from supp_queue import status
    print(json.dumps(status(), ensure_ascii=False)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
