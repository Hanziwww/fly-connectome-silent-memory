"""T-DY2 robust readout training (manuscript/DYNAMICS_SOLUTIONS_PLAN.md, DY2; family F14.DY.robust): perturbations of
the study brain during training, applied by replicate_training.py --perturb NAME without changing the frozen trainer,
and the same perturbations as evaluation draws (dy2_evaluate.py). CPU-importable; the brain parts run on the GPU host.

A perturbation acts on a constructed StudyBrain (studies/brain-sokoban-v1/code/backend.py, never edited) through three
instance patches, installed after the checkpoint, the readout jitter and any lesion operators:
- input noise: independent Poisson spikes at input_hz on every visual input cell (encoder.input_index, the memory
  curve's noise), set again after every brain reset (each training episode, each assessment row) and kept through the
  decision frames;
- membrane noise: after every 1-ms step, the voltage of every non-refractory cell + N(0, s^2), with
  s = membrane_sd_mv * sqrt(1 - a_m^2), so that a non-spiking cell's stationary voltage SD is membrane_sd_mv
  (a_m = exp(-dt / tau_m), the brain's own membrane factor);
- excitability: a uniform threshold offset theta_mv on every cell (the C01 operator's arithmetic), added at every bind
  after the brain's own parameters and any lesion operators.
With no draw active the patches leave the brain bit-identical to the frozen code (checked by dy2_check.py on the GPU).

Training (replicate_training.py): every policy reset inside a training update starts an episode, whose draws are seeded
by crc32('dy2-train|<condition>|r<replicate>|u<update>|e<episode in the update>|<part>'), so a resumed run repeats them;
theta ~ N(0, threshold_sd_mv^2) is drawn anew per episode. The readout-only trainer's feature cache is replaced by
NoCache, since a cache would freeze one realisation. Acceptance (merged_assess): the worker's assessment is the frozen
assessment unperturbed and under the condition's two fixed acceptance draws (accept_draws: noise seeds crc32(
'dy2-accept|<condition>|d<k>'), theta -sigma and +sigma). A history counts as solved, and a probe as correct, only if it is
in all three, and the choice margin is the smallest over them; so 'learned' means learned under the perturbation, and it
implies the frozen deterministic criterion, which the separate --verify step checks unperturbed as before.
Clock (ScaledClock): the frozen worker stops on wall-clock segments (1200/2400/4800 s). The merged assessment runs the
frozen assessment three times, so under a training condition the worker's clock runs at a third of real time
(CLOCK_FACTOR): each segment then holds about as many assessments and updates as in an unperturbed run. Real seconds are
recorded beside it (replicate.json).
Evaluation (heldout_draws): two draws per perturbation with noise seeds crc32('dy2-heldout|<name>|s<k>'), never used in
training or acceptance, and theta at the two quartiles of the training distribution, -0.674 sigma and +0.674 sigma.
"""
import math
import zlib

import numpy as np

CONDITIONS = {
    'in0.2': dict(input_hz=0.2),
    'in1': dict(input_hz=1.0),
    'in5': dict(input_hz=5.0),
    'mem0.1': dict(membrane_sd_mv=0.1),
    'mem1': dict(membrane_sd_mv=1.0),
    'th0.001': dict(threshold_sd_mv=0.001),
    'th0.1': dict(threshold_sd_mv=0.1),
    'th0.5': dict(threshold_sd_mv=0.5),
    'in1-th0.1': dict(input_hz=1.0, threshold_sd_mv=0.1),
}
HELDOUT = 2
CLOCK_FACTOR = 3.0                                     # the merged assessment = 3 frozen assessments (module docstring)
QUARTILE = 0.6744897501960817                         # Phi^-1(0.75): held-out thresholds at the quartiles of N(0, sigma^2)


def seed(text):
    return zlib.crc32(text.encode('utf-8'))


def draw_for(params, theta_mv, key):
    """A draw: the condition's noise levels, a threshold offset and the seed key of its random parts."""
    return dict(input_hz=float(params.get('input_hz', 0.0)), membrane_sd_mv=float(params.get('membrane_sd_mv', 0.0)),
                theta_mv=float(theta_mv), key=key)


def accept_draws(name):
    p = CONDITIONS[name]
    sd = p.get('threshold_sd_mv', 0.0)
    return [draw_for(p, (-sd, sd)[k - 1], f'dy2-accept|{name}|d{k}') for k in (1, 2)]


def heldout_draws(name):
    p = CONDITIONS[name]
    sd = p.get('threshold_sd_mv', 0.0)
    out = []
    for k in range(HELDOUT):
        out.append(draw_for(p, (-QUARTILE, QUARTILE)[k] * sd, f'dy2-heldout|{name}|s{k}'))
    return out


def describe(name):
    """The config record of a training condition (written to config.json; a verify run must match it)."""
    if name not in CONDITIONS:
        raise SystemExit(f'unknown perturbation {name}; known: {", ".join(CONDITIONS)}')
    return dict(name=name, parameters=CONDITIONS[name], accept_draws=accept_draws(name), clock_factor=CLOCK_FACTOR,
                episode_seed="crc32('dy2-train|<condition>|r<replicate>|u<update>|e<episode in the update>|<part>')",
                definition='T-DY2 (F14.DY.robust): input noise on the visual input cells, membrane voltage noise, uniform '
                           'threshold offset; per-episode draws in training; acceptance unperturbed and under accept_draws')


class ScaledClock:
    """Stands in for the time module in the frozen worker (memory.time): perf_counter runs at 1/factor of real time."""

    def __init__(self, factor):
        import time
        self._time, self.factor = time, float(factor)
        self.origin = time.perf_counter()

    def perf_counter(self):
        return self.origin + (self._time.perf_counter() - self.origin) / self.factor

    def __getattr__(self, name):
        return getattr(self._time, name)


class NoCache(dict):
    """Stands in for the readout-only trainer's feature cache: never returns or keeps features."""
    def get(self, key, default=None):
        return None

    def __setitem__(self, key, value):
        pass


class Perturbation:
    """Instance patches of one StudyBrain (module docstring). set_draw(draw) activates a draw, set_off() removes it."""

    def __init__(self, enc):
        import torch
        self.enc, self.b = enc, enc.brain
        b = self.b
        if getattr(b, '_input_target', None) is not None:
            raise RuntimeError('the brain routes its inputs elsewhere (_input_target); input noise is not defined')
        if getattr(b, '_kvec', None) is not None:
            raise RuntimeError('clocked brains are not supported (membrane factor per phase)')
        self.draw = None
        self.mgen = torch.Generator(device=b.device)
        self.a_m = float(b._a_m)
        self.step_factor = math.sqrt(1.0 - self.a_m ** 2)
        orig_reset, orig_step, prev_bind = b.reset, b.step, enc.bind

        def reset(rows=None):
            orig_reset(rows)
            d = self.draw
            if d is not None and d['input_hz'] > 0:
                b.set_poisson(enc.input_index, d['input_hz'])
            elif not bool(b.poisson_p.count_nonzero() > 0):
                b._poisson_on = False                       # the frozen state: no Poisson draws at all

        def step(n_steps=1):
            d = self.draw
            if d is None or d['membrane_sd_mv'] <= 0:
                return orig_step(n_steps)
            s = d['membrane_sd_mv'] * self.step_factor
            for _ in range(int(n_steps)):
                orig_step(1)
                noise = torch.randn(b.v.shape, generator=self.mgen, device=b.device, dtype=b.v.dtype) * s
                b.v = torch.where(b.refrac > 0, b.v, b.v + noise)

        def bind():
            prev_bind()
            d = self.draw
            if d is not None and d['theta_mv'] != 0.0:
                enc.params.v_th = enc.params.v_th + d['theta_mv']

        b.reset, b.step, enc.bind = reset, step, bind

    def set_draw(self, draw):
        """Activate a draw: seed its random parts, then refresh the threshold. The noise starts at the next reset."""
        self.draw = dict(draw)
        if draw['input_hz'] > 0:
            self.b.gen.manual_seed(seed(draw['key'] + '|poisson'))
        if draw['membrane_sd_mv'] > 0:
            self.mgen.manual_seed(seed(draw['key'] + '|membrane'))
        self.enc.bind()

    def set_off(self):
        self.draw = None
        self.b.poisson_p.zero_()
        self.b._poisson_on = False
        self.enc.bind()


class Training:
    """Per-episode draws of a training condition (replicate_training.py). begin(update) before the frozen step,
    episode() at every policy reset inside it, end() after it."""

    def __init__(self, pert, name, replicate):
        self.pert, self.name, self.replicate = pert, name, replicate
        self.params = CONDITIONS[name]
        self.update = None
        self.k = 0
        self.episodes = 0

    def begin(self, update):
        self.update, self.k = int(update), 0

    def episode(self):
        if self.update is None:
            return
        key = f'dy2-train|{self.name}|r{self.replicate}|u{self.update}|e{self.k}'
        sd = self.params.get('threshold_sd_mv', 0.0)
        theta = float(np.random.default_rng(seed(key + '|theta')).normal(0.0, sd)) if sd else 0.0
        self.pert.set_draw(draw_for(self.params, theta, key))
        self.k += 1
        self.episodes += 1

    def end(self):
        self.update = None
        self.pert.set_off()


def merge(base, draws):
    """One assessment from the unperturbed one and the perturbed draws (module docstring): solved / correct only if in
    all of them; the smallest choice margin. Rollouts keep the unperturbed decisions plus each draw's status."""
    def key(rows):
        return [(r['cue'], r['delay_ms'], r['prefix_windows']) for r in rows]
    for d in draws:
        if key(d['rollouts']) != key(base['rollouts']) or [(p['delay_ms'], p['prefix_windows']) for p in d['probes']] != \
                [(p['delay_ms'], p['prefix_windows']) for p in base['probes']]:
            raise RuntimeError('perturbed assessment histories differ from the unperturbed ones')
    rollouts = []
    for i, r in enumerate(base['rollouts']):
        statuses = [r['status']] + [d['rollouts'][i]['status'] for d in draws]
        merged = 'solved' if all(s == 'solved' for s in statuses) else next(s for s in statuses if s != 'solved')
        rollouts.append(dict(r, status=merged, draw_status=statuses))
    probes = []
    for i, p in enumerate(base['probes']):
        records = []
        for j, rec in enumerate(p['records']):
            flags = [rec['correct']] + [d['probes'][i]['records'][j]['correct'] for d in draws]
            records.append(dict(rec, correct=all(flags), draw_correct=flags))
        probes.append(dict(p, records=records, correct=sum(x['correct'] for x in records)))
    out = dict(base, probes=probes, rollouts=rollouts, probe_correct=sum(p['correct'] for p in probes),
               solved=sum(r['status'] == 'solved' for r in rollouts),
               min_choice_margin=min([base['min_choice_margin']] + [d['min_choice_margin'] for d in draws]),
               draws=[dict(solved=x['solved'], probe_correct=x['probe_correct'], min_choice_margin=x['min_choice_margin'])
                      for x in [base] + draws])
    return out


def merged_assess(frozen_assess, pert, draws):
    """The worker's assessment under a training condition (module docstring)."""
    def assess(policy, task, full=False):
        pert.set_off()
        base = frozen_assess(policy, task, full)
        results = []
        for d in draws:
            pert.set_draw(d)
            try:
                results.append(frozen_assess(policy, task, full))
            finally:
                pert.set_off()
        return merge(base, results)
    return assess
