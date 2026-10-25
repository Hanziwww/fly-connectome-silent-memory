"""T-DY2 stage 1 design (robust readout training of M0 readout-only; manuscript/DYNAMICS_SOLUTIONS_PLAN.md DY2, standby
plan S1 WP1). Writes tasks/dy2-design.json once, before any T-DY2 run (amendment 0003 records its hash).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\dy2_design.py

Designed from T-DY1 (PLAN.md section 22): transient kicks decay, but any persistent change of excitability from
-3e-4 mV on, and input noise at any rate, decorrelates the spike trajectory within 100-300 ms, and memory loss tracks it
(C01 ED50 between -1e-4 and -3e-4 mV). Hence a threshold randomisation level near that transition (sigma 1e-3 mV) beside
the planned 0.1 and 0.5 mV, and input-noise levels up to the memory curve's 5 Hz.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
TARGET = STUDY / 'tasks' / 'dy2-design.json'


def design():
    sys.path.insert(0, str(STUDY / 'code'))
    import dy2_perturbation as dy2
    import dy2_evaluate as ev
    return dict(
        task='T-DY2 stage 1: robust readout training, M0 readout-only (DYNAMICS_SOLUTIONS_PLAN.md DY2; S1 WP1, Gate 1)',
        declared='2026-10-20, before any T-DY2 run; families F14.DY.robust, F14.DY.robust.secondary, F14.DY.robust.descriptive',
        training=dict(
            trainer='studies/brain-supplement-v2/code/replicate_training.py --model remote-dynamic-M0-head_only --replicate r '
                    '--perturb <condition> (the frozen memory.worker and training_step; X1 replicate definition: timing order '
                    'seed 1000+r, readout jitter 0.1 SD seed 2000+r; the frozen wall-clock segments 1200/2400/4800 s)',
            conditions=dy2.CONDITIONS,
            replicates=[0, 1, 2, 3, 4],
            runs='studies/brain-pathology-v1/runs/dy2-<condition>-r<r>',
            state_dependence=dict(condition='th0.5', lesion='studies/brain-pathology-v1/runs/regime-b-lesions/C01-dose0.5.json '
                                  '(regime B, C01 dose 0.5: -1.61 mV on every cell)', replicates=[1, 2, 3, 4, 5],
                                  runs='studies/brain-pathology-v1/runs/dy2-C01-dose0.5-th0.5-r<r>'),
            perturbations='dy2_perturbation.py: input noise = Poisson spikes on the 3,200 visual input cells from every reset; '
                          'membrane noise = N(0, s^2) per ms on non-refractory cells, stationary SD membrane_sd_mv; '
                          'excitability = uniform threshold offset theta ~ N(0, threshold_sd_mv^2) per episode',
            acceptance='learned = all 18 trained histories solved and all probes correct unperturbed and under both '
                       'acceptance draws (noise seeds crc32(dy2-accept|<condition>|d<k>), theta -sigma and +sigma); the '
                       'verification is the frozen unperturbed one',
            clock=f'under a training condition the worker\'s segment clock runs at 1/{dy2.CLOCK_FACTOR:g} of real time '
                  '(dy2_perturbation.ScaledClock), as the merged assessment runs the frozen assessment three times; real '
                  'seconds are recorded in replicate.json',
            accept_draws={name: dy2.accept_draws(name) for name in dy2.CONDITIONS}),
        evaluation=dict(
            script='studies/brain-pathology-v1/code/dy2_evaluate.py, one job per model, frozen memory.assess',
            models=dict(dy2='the 45 healthy and 5 C01-trained T-DY2 runs',
                        x1='studies/brain-supplement-v2/runs/x1-M0-head_only-r0..r9 (the X1 replicates, the reference)',
                        regime_b='studies/brain-pathology-v1/runs/b-C01-dose0.5-M0-head_only-r1..r6 (state dependence reference)'),
            healthy_panel=dict(as_trained='full 35 timings; must reproduce verification.json',
                               shifts_mv=list(ev.SHIFTS_MV), removal='15 cell sets of the dose series (shares 0.0002-0.005, 3 '
                               'draws each)', perturbations='both held-out draws of every condition at the trained '
                               "timings; of the model's own training condition at all 35",
                               heldout_draws={name: dy2.heldout_draws(name) for name in dy2.CONDITIONS},
                               mechanism=ev.MECHANISM),
            lesion_panel='as-trained (full; must reproduce verification), mechanism, th0.5 held-out draws on the lesion, '
                         'restored (lesion removed; full)'),
        measures=dict(
            net_use='(cue pairs both solved - pairs none solved) / pairs, per timing set: trained 9 pairs, untrained 26 '
                    '(analyze_campaign.classes); lock = trained - untrained',
            ed50_c01='the shift magnitude at which trained net use first falls to <= 0.5 along shifts -3e-5 ... -3.22 mV '
                     '(as-trained at 0), log-linear interpolation between grid points; censored at 3e-5 (already <= 0.5) and '
                     '3.22 (never); 0, the lowest rank, for models whose as-trained trained net use is <= 0.5 (no memory to '
                     'lose); the geometric-mean ratios use the models with ED50 > 0',
            ed50_removal='the same over the removal shares 0.0002-0.005, net use averaged over the 3 draws per share',
            own_perturbation='trained net use averaged over the two held-out draws of the condition the model trained under '
                             '(for X1, of each condition in turn); for T-DY2 models also untrained net use and the lock '
                             'under them (descriptive)',
            restoration_loss='trained net use as-trained (lesion on) - restored (C01-trained models)',
            mechanism='drop in net use at the prefix-0 trained timings from as-trained to adaptation cleared, and to the '
                      'opposite-cue swap, at decision onset',
            cost='updates and training seconds to the end of training (result.json)'),
        tests=dict(
            primary='F14.DY.robust (Holm, 10 tests): per condition, C01 ED50 of its 5 replicates > X1 r0-r9 (exact '
                    'one-sided Mann-Whitney over all C(15, 5) splits, mid-ranks for ties and censored values); state '
                    'dependence: restoration loss of the C01 + th0.5 replicates < regime B C01 dose 0.5 readout-only r1-r6 '
                    '(exact one-sided Mann-Whitney)',
            secondary='F14.DY.robust.secondary (BH, 45 tests): per condition vs X1 r0-r9, exact Mann-Whitney: trained net '
                      'use (two-sided), untrained net use (two-sided), own-perturbation trained net use (greater), removal '
                      'ED50 (greater), updates (two-sided)',
            descriptive='F14.DY.robust.descriptive: the S1 Gate 1 criterion per condition (own-perturbation trained net '
                        'use > 0 in >= 4 of 5 replicates, and the mechanism retained: mean adaptation-clear drop and mean '
                        'swap drop each at least half of X1\'s); ED50 ratio (geometric means, condition / X1) with a '
                        'bootstrap 95% interval (10,000 resamples within groups, seed 20261020); outcomes, updates, '
                        'seconds; the full perturbation panel',
            reading='DYNAMICS_SOLUTIONS_PLAN.md section 3: robust = memory and mechanism persist (Gate 1 criterion), '
                    'ED50 ratio >= 10 and the primary ED50 test significant, state dependence smaller (test 10)'),
        stage_2='S1 readout-only (planned as M0 and S1): deferred, to be declared by amendment before any S1 run. Under the '
                'frozen wall-clock segments, S1\'s sequential assessment (about 30 s) run three times per assessment would '
                'leave fewer updates than S1 needed unperturbed (about 300 in about 3,000 s), so a failure would reflect the '
                'clock rather than robustness',
        limitations=['the frozen protocol stops on wall-clock segments, so slower perturbed updates (no feature cache, three '
                     'assessments, per-ms membrane noise) leave fewer updates per segment than X1 had; updates and seconds '
                     'are reported',
                     'the acceptance draws are fixed; the held-out panel uses other seeds and the quartile thresholds',
                     '5 replicates per condition; the exact tests use all 10 X1 replicates'])


def main():
    if TARGET.exists():
        raise SystemExit(f'{TARGET.name} exists; task files are written once, before their runs')
    text = json.dumps(design(), indent=1, sort_keys=False)
    TARGET.write_text(text + '\n', encoding='utf-8', newline='\n')
    print(TARGET.name, hashlib.sha256((text + '\n').encode('utf-8')).hexdigest())
    return 0


if __name__ == '__main__':
    sys.exit(main())
