"""Entry points that research.py dispatches to for brain-pathology-v1.

    run(study, args)              supervisor: the lane/slot queue over schedule.json
    run_job(study, job, args)     one job in this process (started by the queue as research.py ... run --job ID)
    verify(study, args)           independent re-verification of completed jobs
    verify_job(study, job, args)  one job's verification, in its own process

research.py refuses run and verify while the protocol is a draft; no capacity experiment runs before gate G2.
The job functions are not written yet.
"""


def run(study, args):
    from tools.remote.queue import Queue, settings_from_args
    return Queue(study, study.schedule(), settings_from_args(args)).run(resume=args.resume)


def run_job(study, job, args):
    raise NotImplementedError('brain-pathology-v1 has no job code yet')


def verify(study, args):
    raise NotImplementedError('brain-pathology-v1 has no verification code yet')


def verify_job(study, job, args):
    raise NotImplementedError('brain-pathology-v1 has no verification code yet')
