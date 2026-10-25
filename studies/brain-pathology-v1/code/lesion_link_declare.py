"""Amendment 0015 (local): the lesion link, declared before any of its runs (code/lesion_link.py on the GPU host,
code/analyze_lesion_link.py locally; tasks/lesion-link-design.json; families F15.link, .secondary, .descriptive in
code/testplan.py). The engineering check lesion_link.py --smoke ran before, on recorded conditions only (the healthy and
the dose-1 lesioned brain read by the healthy-trained readout 0, and the regime C replay of dose 1, replicate 1).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\lesion_link_declare.py
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
CODE = STUDY / 'code'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sys.path.insert(0, str(ROOT))
    from studykit.registry import Study
    study = Study.named('brain-pathology-v1')
    path = study.amend('Lesion link: where the cue trace is written and where it is read', dict(
        why='the user (2026-10-23) asked for the causal link between the silent trace, the drift of the read operator and the '
            'lesion and recovery effects (Fig. 7) to be tested directly; no run recorded the trace or the read under the lesion',
        design=dict(file='tasks/lesion-link-design.json', sha256=sha(STUDY / 'tasks' / 'lesion-link-design.json')),
        recordings=[
            dict(code='code/lesion_link.py', sha256=sha(CODE / 'lesion_link.py'), host='GPU',
                 what='scene-1 readout-trained learner (brain fixed): the 9 trained timings x both cues in the brain conditions '
                      'H-H, L<d>-L<d>, H-L<d> and L<d>-H (C01 doses 0.5, 1, 1.5, 2; W-R = written and held in W, read in R, '
                      'the full state transplanted at the end of the delay), read by the healthy-trained readouts 0-9, the '
                      'lesion-trained readouts (regime B replicates 1-6) and the relearned readouts (regime C replays, '
                      'replicates 1-3); per history the outcome and the first decision; per brain condition the stored trace, '
                      'the decision-window state and the readout features; per-ms spikes for the video',
                 check='the replica equals memory.assess; sham transplants and the decision-window hook change no decision; '
                       'the transplanted trace equals the donor\'s; every replay reproduces its regime C record (audit, '
                       'assessments after 0, 50 and 100 updates, every loss and gradient norm)',
                 use='Fig. 7 or S7 if favourable under the reporting rule (RESULTS_INVENTORY_V4.md), and Video S4'),
            dict(code='code/analyze_lesion_link.py', sha256=sha(CODE / 'analyze_lesion_link.py'), host='local',
                 what='checks against the earlier records (verification, regime C after 0 and 100 updates, the late '
                      'restoration), then the F15.link tests and the descriptive tables',
                 check='status and steps exactly, first-decision logits and features within 1e-4')],
        tests=dict(families=['F15.link (Holm, 3, primary)', 'F15.link.secondary (BH, 9)', 'F15.link.descriptive (none)'],
                   testplan_sha256=sha(CODE / 'testplan.py'), collect_tests_sha256=sha(CODE / 'collect_tests.py'))))
    print(path.name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
