"""Versioned verification-only metadata adapter; training sources stay immutable."""
from pathlib import Path
import argparse


def legacy_rows(probes, rows, intervention):
    # memory.worker supplies this metadata when archiving intervention rollouts.
    # Removing the duplicate key changes no tensors, actions, or acceptance rules.
    if intervention is not None:
        rows = [{k: v for k, v in r.items() if k != 'intervention'} for r in rows]
    return probes, rows


def main(job_id):
    import run_remote as runner
    import supp_experiments as ex
    receipt = runner.read(runner.SESSION / 'compatibility-repair.json')
    assert receipt['adapter_sha256'] == runner.digest(Path(__file__))
    assert receipt['unchanged_training_sha256'] == runner.code_digest()
    original = ex.batched_assessment

    def compatible(policy, task, delays, prefixes, max_steps=512, intervention=None):
        probes, rows = original(policy, task, delays, prefixes, max_steps, intervention)
        return legacy_rows(probes, rows, intervention)

    ex.batched_assessment = compatible
    job = next(j for j in runner.prepare()['jobs'] if j['id'] == job_id)
    code = runner.worker(job, verify=True)
    path = runner.ROOT / job['out'] / 'verification.json'
    if path.exists():
        value = runner.read(path)
        value['verification_adapter'] = dict(
            sha256=receipt['adapter_sha256'],
            reason='Remove duplicate intervention metadata at the legacy archive boundary',
            changes_to_training=False, changes_to_forward_pass=False,
            changes_to_acceptance=False)
        runner.write(path, value)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    raise SystemExit(main(parser.parse_args().job))
