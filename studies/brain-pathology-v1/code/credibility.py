"""Phase 2 credibility tests C4-C6 on the healthy study brain (GPU host). Plan v2; candidates and calibration targets
from literature/candidates.csv, literature/calibration_draft.json and literature/review_main.md.

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/credibility.py [--brains p05 m0joint]

C4  healthy circuit valid (shared tests):
    smell    odour panel -> PN, LN, KC, APL, MBON responses; KC sparseness per odour; PN lifetime sparseness
    apl      APL removed -> KC responses (feedback inhibition present: more KCs respond)
    ln       GABAergic ALLN removed -> PN responses (lateral inhibition present: PN tuning broadens)
    stable   memory-task input under uniform threshold offsets 0 ... -4 mV (stability margin, runaway onset)
    loom     looming-like ramp drive on LPLC2 and LC4 -> giant fibre (DNp01) burst; movement readout
    noop     every dopaminergic, serotonergic and octopaminergic cell removed -> other cells bit-identical
C5  in-silico replication of each candidate's measured change at the calibrated dose (cellular or circuit level)
C6  the C5 direction at 0.5x and 1.5x the dose and on a second brain state (M0 joint after training)

Deterministic: drives are constant currents (no Poisson input), so every number is exact and repeatable. Writes
studies/brain-pathology-v1/runs/phase2-credibility/<brain>/*.json; the scoring (score_credibility.py) runs locally.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
OUT = STUDY / 'runs' / 'phase2-credibility'
P05 = ROOT / 'runs' / 'linear-joint' / '20261002-p05-01' / 'model.pt'
BRAINS = {'p05': P05, 'm0joint': MECH / 'inputs' / 'models' / 'remote-dynamic-M0-joint' / 'model.pt'}
SETTLE_MS, ODOUR_MS = 100, 300
KC_RESPONDS = 2                       # spikes in the 300-ms odour window (>= 6.7 Hz)
OFFSETS = (0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0)
DOSES = (0.5, 1.0, 1.5)


def cell_sets(n):
    """Named cell index sets from the model's annotations (MaleCNS v1.0 subset)."""
    t, cls, nt = n['type'].fillna('').astype(str), n['class'].fillna('').astype(str), n['nt'].fillna('').astype(str)
    sel = lambda m: np.flatnonzero(np.asarray(m))          # noqa: E731
    sets = dict(orn=sel(cls == 'olfactory'), alpn=sel((cls == 'ALPN') & (nt == 'acetylcholine')),
                alpn_all=sel(cls == 'ALPN'), alln=sel(cls == 'ALLN'), alln_gaba=sel((cls == 'ALLN') & (nt == 'gaba')),
                kc=sel(t.str.startswith('KC')), kcab=sel(t.str.startswith('KCab')), apl=sel(t == 'APL'),
                mbon=sel(cls == 'MBON'), mbon_m4m6=sel(t.isin(['MBON01', 'MBON03', 'MBON04'])),
                gf=sel(t == 'DNp01'), loom=sel(t.isin(['LPLC2', 'LC4'])),
                monoamine=sel(nt.isin(['dopamine', 'serotonin', 'octopamine'])))
    return sets


class Rig:
    """One StudyBrain at batch 1 with a checkpoint's brain parameters; deterministic stimulus protocols."""

    def __init__(self, checkpoint, out):
        sys.path.insert(0, str(MECH / 'code'))
        sys.path.insert(0, str(STUDY / 'code'))
        import mechanism_base as mb  # noqa: F401  (path setup)
        import torch
        from backend import StudyBrain, set_pair
        from readout_models import ActivityLinear
        self.torch = torch
        self.enc = StudyBrain(batch=1, precision='mixed', metadata_dir=out)
        head = ActivityLinear().cuda()
        saved = mb.load_pair(checkpoint)
        set_pair(self.enc, head, saved)
        self.enc.core.requires_grad_(False)
        self.n = self.enc.brain.n
        self.sets = cell_sets(self.enc.c.neurons)
        import operators
        self.ops = operators
        from flybrain.olfaction import ODOURS
        self.odours = ODOURS
        types = self.enc.c.neurons['type'].fillna('').astype(str).to_numpy()
        self.orn_glom = np.array([x.replace('ORN_', '') for x in types[self.sets['orn']]])
        import memory_tasks
        self.memory_tasks = memory_tasks
        self.blank = torch.zeros((1, 3200), device='cuda')
        self._active0 = self.enc.brain.active.clone()
        self._jump0 = self.enc.brain.adapt_jump_vec

    def lesion(self, specs):
        """Restore the healthy brain, then apply `specs` (an empty list leaves it healthy)."""
        self.enc.brain.active.copy_(self._active0)
        self.enc.brain.adapt_jump_vec = self._jump0
        return self.ops.apply(self.enc, specs)

    def run(self, ms, drive=None, visual=None, record=None):
        """Spike counts per cell over `ms` steps (after the call's own settle period), with constant extra drive
        {cells: mV} and a constant visual drive (1, 3200) mV; record: optional list of cells for a 1-ms raster."""
        torch = self.torch
        e, b = self.enc, self.enc.brain
        with torch.no_grad():
            e.reset()
            e.bind()
            b.set_drive(e.input_index, self.blank if visual is None else visual)
            for _ in range(SETTLE_MS):
                b.step()
            for cells, mv in (drive or {}).items():
                b.set_drive(np.asarray(cells), torch.as_tensor(np.broadcast_to(mv, (1, len(cells))).copy(), dtype=torch.float32, device='cuda'))
            counts = torch.zeros(self.n, device='cuda')
            raster = []
            for _ in range(ms):
                b.step()
                counts += b.spikes[0]
                if record is not None:
                    raster.append(b.spikes[0, record].clone())
            return counts.cpu().numpy(), (torch.stack(raster).cpu().numpy() if record is not None else None)

    def odour_drive(self, name, scale=13.0):
        """Constant ORN drive for an odour: 7 mV (rheobase) + scale x saturating response, by glomerulus weight."""
        w = np.array([self.odours[name].get(g, 0.0) for g in self.orn_glom])
        resp = w / (w + 0.5)
        mv = np.where(w > 0, 7.0 + scale * resp, 0.0)
        cells = self.sets['orn'][mv > 0]
        return {tuple(cells): mv[mv > 0]}

    def odour_panel(self, ms=ODOUR_MS, record=None):
        rows, rasters = {}, {}
        for name in self.odours:
            rows[name], rasters[name] = self.run(ms, drive=self.odour_drive(name), record=record)
        return (rows, rasters) if record is not None else rows

    def memory_input(self, delay=160, prefix=4, cue=0):
        """The M0 task's visual input sequence (cue, delay, prefix, first decision board) as a list of (drive, ms)."""
        task = next(t for t in self.memory_tasks.prepare_memory() if t['id'] == 'M0')
        frames = self.memory_tasks.frames(task, cue, delay, prefix, False)
        out = []
        for f in frames:
            x = self.torch.as_tensor(f['board'], dtype=self.torch.float32, device='cuda').reshape(1, 400).repeat_interleave(8, 1) * 18
            out.append((x, f['ms']))
        return out

    def run_sequence(self, seq):
        torch = self.torch
        e, b = self.enc, self.enc.brain
        with torch.no_grad():
            e.reset()
            e.bind()
            counts, total = torch.zeros(self.n, device='cuda'), 0
            for x, ms in seq:
                b.set_drive(e.input_index, x)
                for _ in range(ms):
                    b.step()
                    counts += b.spikes[0]
                total += ms
            state = dict(v=b.v[0].clone(), g=b.g[0].clone(), adapt=b.adapt[0].clone())
        return counts.cpu().numpy(), total, state


def summarize_odours(rig, rows):
    s = rig.sets
    names = list(rows)
    kc = np.stack([rows[k][s['kc']] for k in names])
    pn = np.stack([rows[k][s['alpn']] for k in names]) / (ODOUR_MS / 1000)
    resp = kc >= KC_RESPONDS
    mean_pn = pn.mean(0)
    sq = (pn ** 2).mean(0)
    life = np.where(sq > 0, (1 - mean_pn ** 2 / np.where(sq > 0, sq, 1)) / (1 - 1 / len(names)), np.nan)
    return dict(odours=names,
                kc_fraction_responding=resp.mean(1).tolist(), kc_mean_spikes=kc.mean(1).tolist(),
                kcab_mean_spikes=[float(rows[k][s['kcab']].mean()) for k in names],
                pn_mean_hz=pn.mean(1).tolist(), pn_fraction_active=(pn > 5).mean(1).tolist(),
                pn_lifetime_sparseness_median=float(np.nanmedian(life)), pn_responsive=int(np.sum(sq > 0)),
                ln_mean_hz=[float(rows[k][s['alln']].mean() / (ODOUR_MS / 1000)) for k in names],
                apl_hz=[float(rows[k][s['apl']].mean() / (ODOUR_MS / 1000)) for k in names],
                mbon_mean_hz=[float(rows[k][s['mbon']].mean() / (ODOUR_MS / 1000)) for k in names],
                mbon_m4m6_hz=[float(rows[k][s['mbon_m4m6']].mean() / (ODOUR_MS / 1000)) for k in names],
                kc_pair_overlap=_overlap(resp))


def _overlap(resp):
    """Mean Jaccard overlap of responding KC sets between odour pairs (decorrelation)."""
    vals = []
    for i in range(len(resp)):
        for j in range(i + 1, len(resp)):
            u = np.sum(resp[i] | resp[j])
            if u:
                vals.append(np.sum(resp[i] & resp[j]) / u)
    return float(np.mean(vals)) if vals else np.nan


def c4(rig, out):
    s = rig.sets
    res = {}
    base = rig.odour_panel()
    res['smell'] = summarize_odours(rig, base)
    rig.lesion([rig.ops.spec('remove', s['apl'], label='APL removed')])
    res['apl_removed'] = summarize_odours(rig, rig.odour_panel())
    rig.lesion([rig.ops.spec('remove', s['alln_gaba'], label='GABAergic ALLN removed')])
    res['ln_removed'] = summarize_odours(rig, rig.odour_panel())
    rig.lesion([])
    seq = rig.memory_input()
    stable = []
    for off in OFFSETS:
        rig.lesion([rig.ops.spec('threshold_offset', np.arange(rig.n), off, label=f'uniform {off} mV')])
        c, total, _ = rig.run_sequence(seq)
        hz = c / (total / 1000)
        stable.append(dict(offset_mv=off, mean_hz=float(hz.mean()), fraction_over_50hz=float((hz > 50).mean()),
                           fraction_over_100hz=float((hz > 100).mean()), fraction_active=float((hz > 1).mean())))
    res['stable'] = stable
    rig.lesion([])
    ramp = np.linspace(0, 20, 400)
    loom = []
    for label, mv in (('blank', None), ('loom', ramp)):
        torch = rig.torch
        e, b = rig.enc, rig.enc.brain
        with torch.no_grad():
            e.reset()
            e.bind()
            b.set_drive(e.input_index, rig.blank)
            for _ in range(SETTLE_MS):
                b.step()
            gf = []
            for t in range(400):
                if mv is not None:
                    b.set_drive(s['loom'], torch.full((1, len(s['loom'])), float(mv[t]), device='cuda'))
                b.step()
                gf.append(float(b.spikes[0, s['gf']].sum()))
        loom.append(dict(condition=label, gf_spikes=float(np.sum(gf)), gf_first_spike_ms=(int(np.argmax(np.array(gf) > 0)) + 1) if np.sum(gf) else None))
    res['loom'] = loom
    c0, _, st0 = rig.run_sequence(seq)
    rig.lesion([rig.ops.spec('remove', s['monoamine'], label='monoamine cells removed')])
    c1, _, st1 = rig.run_sequence(seq)
    keep = np.setdiff1d(np.arange(rig.n), s['monoamine'])
    same = all(bool(rig.torch.equal(st0[k][keep], st1[k][keep])) for k in st0) and bool(np.array_equal(c0[keep], c1[keep]))
    res['noop'] = dict(monoamine_cells=int(len(s['monoamine'])), other_cells_bit_identical=same)
    rig.lesion([])
    (out / 'c4.json').write_text(json.dumps(res), encoding='utf-8', newline='\n')
    return res


CANDIDATES = {
    # id: (operator kind, target set, calibrated value, what is measured)
    'C01': ('threshold_offset', 'all', None, 'rheobase ratio 0.54 (ramp current) / 0.43 (charge); network rates'),
    'C02': ('threshold_offset', 'kc', None, 'KC odour responses up (Kv4 loss, same rheobase shift as C01)'),
    'C16': ('threshold_offset', 'kc', None, 'KC evoked response ratio 0.519'),
    'C15': ('threshold_offset', 'mbon_m4m6', None, 'M4/M6 MBON evoked response ratio 2.689'),
    'C14': ('adaptation_scale', 'kcab', 0.10, 'KCab adaptation 0.10 of young: less adaptation to a sustained odour'),
    'C04': ('remove', 'kc', 0.17, '17% of KCs removed: MB output to MBONs'),
    'C13': ('release_scale', 'alln_gaba', 0.5, 'weaker GABAergic LN release: PN lifetime sparseness down'),
    'C11': ('input_scale', 'alpn_all', 2.29, 'all input onto ALPNs x2.29: PN responses up'),
    'C12': ('input_scale', 'alpn_all', 0.581, 'all input onto ALPNs x0.581: PN responses down'),
    'C07': ('release_scale', 'alpn_all', 1.5, 'ALPN release up: KC response flat or down (published triple constraint)'),
}


def _rheobase_offset(target_ratio):
    """Threshold offset (mV) whose isolated-cell rheobase is target_ratio of the healthy one (constant-drive rheobase)."""
    from operators import rheobase
    base = rheobase(0.0)
    lo, hi = -6.9, 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if rheobase(mid) / base > target_ratio:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _fit_offset(rig, cells, target_ratio, measure, lo=-6.5, hi=8.0, steps=16):
    """Threshold offset on `cells` that scales the positive response `measure(rig)` by target_ratio. The response
    falls monotonically as the threshold rises, so bisection on the offset converges."""
    rig.lesion([])
    base = measure(rig)
    for _ in range(steps):
        mid = (lo + hi) / 2
        rig.lesion([rig.ops.spec('threshold_offset', cells, mid)])
        if measure(rig) / base > target_ratio:
            lo = mid
        else:
            hi = mid
    rig.lesion([])
    return (lo + hi) / 2, base


def _adaptation(rig, record):
    """Adaptation of `record` cells to a sustained 1-s odour (apple): rate in the last 500 ms over the first 100 ms."""
    c, raster = rig.run(1000, drive=rig.odour_drive('apple'), record=record)
    first, last = raster[:100].sum(0) / 0.1, raster[500:].sum(0) / 0.5
    ok = first > 0
    return dict(cells_firing=int(ok.sum()), mean_first_hz=float(first[ok].mean()) if ok.any() else None,
                mean_last_hz=float(last[ok].mean()) if ok.any() else None,
                median_index=float(np.median(last[ok] / first[ok])) if ok.any() else None)


def _kc_drive_response(rig, cells_out):
    """Mean spike count of `cells_out` under a strong uniform KC drive (K+-depolarisation analogue: 12 mV for 300 ms)."""
    c, _ = rig.run(ODOUR_MS, drive={tuple(rig.sets['kc']): 12.0})
    return float(c[cells_out].mean())


def c5_c6(rig, out, c4res):
    from operators import ramp_threshold, adaptation_index
    s = rig.sets
    rng = np.random.default_rng(10)
    res = {}
    # C01/C02: rheobase calibration in the isolated model cell (ramp protocol, Ping 2015)
    off = _rheobase_offset(0.54)
    i0, q0 = ramp_threshold(0.02)
    i1, q1 = ramp_threshold(0.02, v_th_offset=off)
    res['calibration_rheobase'] = dict(offset_mv=off, ramp_current_ratio=i1 / i0, ramp_charge_ratio=q1 / q0,
                                       reported_current_ratio=0.54, reported_charge_ratio=0.43)
    seq = rig.memory_input()
    for cid, (kind, target, value, what) in CANDIDATES.items():
        rows = []
        for dose in DOSES:
            if cid in ('C01', 'C02'):
                v = off * dose
            elif cid == 'C16':
                v = None
            elif cid == 'C15':
                v = None
            elif kind in ('adaptation_scale',):
                v = 1 - (1 - value) * dose
            elif kind == 'remove':
                v = value * dose
            else:
                v = 1 + (value - 1) * dose
            cells = np.arange(rig.n) if target == 'all' else s[target]
            row = dict(dose=dose)
            if cid == 'C16':
                v, b0 = _fit_offset(rig, s['kc'], 1 - 0.481 * dose, lambda r: _kc_drive_response(r, r.sets['kc']))
                row['fitted_offset_mv'] = v
            if cid == 'C15':
                v, b0 = _fit_offset(rig, s['mbon_m4m6'], 1 + 1.689 * dose, lambda r: _kc_drive_response(r, r.sets['mbon_m4m6']) + 1e-6)
                row['fitted_offset_mv'] = v
            if kind == 'remove':
                chosen = rng.choice(cells, int(round(v * len(cells))), replace=False)
                specs = [rig.ops.spec('remove', chosen)]
            else:
                specs = [rig.ops.spec(kind, cells, v)]
            row['value'] = float(v) if v is not None else None
            rig.lesion(specs)
            panel = summarize_odours(rig, rig.odour_panel())
            c, total, _ = rig.run_sequence(seq)
            hz = c / (total / 1000)
            row.update(odour=panel, task_mean_hz=float(hz.mean()), task_fraction_over_100hz=float((hz > 100).mean()),
                       kc_drive_mbon=_kc_drive_response(rig, s['mbon']), kc_drive_m4m6=_kc_drive_response(rig, s['mbon_m4m6']),
                       kc_drive_kc=_kc_drive_response(rig, s['kc']))
            if cid in ('C14', 'C01', 'C02'):
                row['kcab_sustained_odour'] = _adaptation(rig, s['kcab'])
            rig.lesion([])
            rows.append(row)
        res[cid] = dict(kind=kind, target=target, what=what, rows=rows)
    rig.lesion([])
    res['kcab_sustained_odour_healthy'] = _adaptation(rig, s['kcab'])
    ai0 = adaptation_index(10.0)
    ai1 = adaptation_index(10.0, jump_scale=0.10)
    res['calibration_adaptation'] = dict(index_young=ai0, index_scaled_0p10=ai1)
    (out / 'c5_c6.json').write_text(json.dumps(res), encoding='utf-8', newline='\n')
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--brains', nargs='*', default=list(BRAINS))
    args = ap.parse_args()
    for name in args.brains:
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        t = time.time()
        rig = Rig(BRAINS[name], out / 'meta')
        r4 = c4(rig, out)
        print(f'{name}: C4 done in {time.time() - t:.0f} s; KC responding {np.round(r4["smell"]["kc_fraction_responding"], 3)}', flush=True)
        c5_c6(rig, out, r4)
        print(f'{name}: C5/C6 done in {time.time() - t:.0f} s', flush=True)
        del rig
    (OUT / 'done.json').write_text(json.dumps(dict(finished_unix=time.time(), brains=args.brains)), encoding='utf-8')


if __name__ == '__main__':
    main()
