"""Execution-order amendment: dispatch already registered independent arms sooner."""
import argparse
import copy
import json
from pathlib import Path
import controller_v2 as prior

runner = prior.runner
original_prepare = runner.prepare
original_alive = prior.alive


def alive(pid, session=runner.SESSION):
    if original_alive(pid, session):
        return True
    if session != runner.SESSION:
        return False
    try:
        return str(session / 'controller_v3.py') in Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
    except OSError:
        return False


def prepare():
    plan = original_prepare()
    amendment = runner.read(runner.SESSION / 'execution-order-amendment.json')
    assert amendment['controller_sha256'] == runner.digest(Path(__file__))
    assert amendment['training_sha256'] == plan['code_sha256']
    assert len(amendment['jobs']) == 8
    effective = copy.deepcopy(plan)
    for job in effective['jobs']:
        if job['id'] in amendment['jobs']:
            change = amendment['jobs'][job['id']]
            assert job['id'].startswith('supp-origin-') and job['mode'] == 'train'
            assert job['dependencies'] == change['original_dependencies']
            assert change['effective_dependencies'] == ['supp-preflight']
            job['dependencies'] = change['effective_dependencies']
    return effective


prior.alive = alive
runner.alive = alive
runner.prepare = prepare


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['run', 'status'])
    if parser.parse_args().command == 'status':
        value = runner.status()
        value['supervisor']['controller_entrypoint'] = 'controller_v3.py'
        value['supervisor']['scheduling_amendment'] = 'execution-order-amendment.json'
        print(json.dumps(value))
    else:
        raise SystemExit(prior.run())
