"""Video S2 (Cell; related to Figure 2) - write, hold, read: one memory trial, millisecond by millisecond.

Scene 1, readout trained; 160-ms delay, 4 input windows; hidden goal. Three dark CNS panels, brain on top as in Figure 2C:
the cells spiking after cue 0 and after cue 1 (cell-group colours, Figure 1A, with a 4-ms afterglow), and the trace, each
cell's adaptation difference between the two cues (cue 1 minus cue 0) on the dark cue map of Figure 2C with the cells spiking
differently in that millisecond as white sparks. The trace is rebuilt from the spikes with the model's adaptation rule (P12
checks the rebuild against the record at every ms). Small boards show what the agent sees in each phase. Below, two live
charts over the trial's phases: per ms, the cells spiking differently and the cells holding the trace (trial_timecourse, P12);
and the trace's strength, the norm of the adaptation difference, against the adaptation's own 200-ms decay (dashed), drawn from
the last cell spiking differently as the cue's activity fades: from there to the end of the delay the strength follows it
exactly (checked here).

Data: manuscript/results/plan_v2/video_trial_spikes.csv, video_trial_cells.csv, video_trial_phases.csv, trial_timecourse.csv;
the frozen memory-task boards (studies/brain-sokoban-v1/tasks/memory.json, read only); soma positions from fd_cells.csv.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_video_s2.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import palette as pal  # noqa: E402
from render import project as rp, scenes as sc, video as vd  # noqa: E402

OUT = FIG / "v4" / "videos"
PLAN = rp.RESULTS / "plan_v2"
MEMORY = rp.ROOT / "studies" / "brain-sokoban-v1" / "tasks" / "memory.json"
DELAY, PREFIX = 160, 4
STAGE = (560, 620)
XS = (60, 680, 1300)
Y0 = 186
GLOW_TAU, GLOW_MIN = 4.0, 0.05              # afterglow of a spike (ms) and the smallest shown
DA_LO, DA_HI = 1e-3, 3.0                     # trace colour scale, as the Figure 2C tiles
PHASE_COLOUR = {"cue": (214, 160, 60), "delay": (90, 90, 90), "input": (92, 200, 255), "decision": (236, 236, 236)}
GOLD, REF = (214, 160, 60), (150, 150, 150)


def frames_per_ms(phase):
    return 2 if phase == "delay" else 3


def phase_of(t, ph):
    if t <= ph["cue_end_ms"]:
        return "cue"
    if t <= ph["delay_end_ms"]:
        return "delay"
    if t <= ph["input_end_ms"]:
        return "input"
    return "decision"


def board_for(task, cue, phase, t, ph):
    """What the agent sees: the board with its goal in the cue, nothing in the delay, the board without the goal after."""
    lv = task["levels"][cue]
    walls = set(lv["walls"])
    if phase == "delay":
        return None
    if phase == "cue":
        return walls, set(lv["goals"]), [34], 44, False
    if phase == "input":
        k = (t - ph["delay_end_ms"] - 1) // 40                  # prefix windows: 44, 54, 44, 54 (memory_tasks.prefix_states)
        return walls, set(lv["goals"]), [34], (44 if k % 2 == 0 else 54), True
    return walls, set(lv["goals"]), [34], 44, True


def main():
    t0 = time.time()
    task = next(t for t in json.loads(MEMORY.read_text(encoding="utf-8")) if t["id"] == "M0")
    ph = pd.read_csv(PLAN / "video_trial_phases.csv").set_index("delay_ms").loc[DELAY]
    sp = pd.read_csv(PLAN / "video_trial_spikes.csv")
    sp = sp[sp["delay_ms"] == DELAY]
    jump = pd.read_csv(PLAN / "video_trial_cells.csv")["adapt_jump"].to_numpy(np.float64)
    tc = pd.read_csv(PLAN / "trial_timecourse.csv").groupby("ms")[["differential_spike_cells", "trace_cells"]].sum()
    fd = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False)
    s, xyz = sc.somata(fd)
    cell_of = s["cell"].to_numpy()
    group = s["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other").to_numpy()
    colour = rp.srgb_to_linear(np.array([to_rgb(pal.ATLAS_ON_DARK[g]) for g in group]))
    stages = [vd.Stage(xyz, *STAGE) for _ in range(3)]
    px, py = stages[0].project(xyz)
    dens = sc.density_weights(stages[0].view, xyz, power=0.45)
    n = len(jump)
    decay_a = np.exp(-1.0 / 200.0)
    decay_g = np.exp(-1.0 / GLOW_TAU)
    a = [np.zeros(n), np.zeros(n)]
    glow = [np.zeros(n), np.zeros(n)]
    by_ms = {(c, t): g["cell"].to_numpy() for (c, t), g in sp.groupby(["cue", "ms"])}
    last = int(ph["decision_end_ms"])

    # live metrics: the cells spiking differently and holding the trace (trial_timecourse), and the trace's strength, the norm of
    # the adaptation difference; with no cell spiking differently, the difference only decays, by exp(-1/200) per ms
    norm, a_, differs = np.zeros(last + 1), [np.zeros(n), np.zeros(n)], np.zeros(last + 1, bool)
    for t in range(1, last + 1):
        ons = [np.sort(by_ms.get((c, t), np.zeros(0, np.int64))) for c in (0, 1)]
        for c in (0, 1):
            a_[c] *= decay_a
            a_[c][ons[c]] += jump[ons[c]]
        norm[t] = np.linalg.norm(a_[1] - a_[0])
        differs[t] = not np.array_equal(ons[0], ons[1])
    cue_end, delay_end, input_end = int(ph["cue_end_ms"]), int(ph["delay_end_ms"]), int(ph["input_end_ms"])
    # the cue's own activity fades over the delay's first ms; from the last cell spiking differently on, the trace only decays
    quiet = int(max([t for t in range(cue_end + 1, delay_end + 1) if differs[t]], default=cue_end))
    ref_t = np.arange(quiet, delay_end + 1)
    ref = norm[quiet] * decay_a ** (ref_t - quiet)
    decay_err = float(np.max(np.abs(norm[ref_t] / ref - 1)))
    if decay_err > 1e-9:
        raise RuntimeError(f"the trace does not follow the adaptation's decay in the silent delay ({decay_err:.2e})")
    phases = [("cue", 1, cue_end), ("delay", cue_end + 1, delay_end), ("input", delay_end + 1, input_end), ("decision", input_end + 1, last)]
    bands = [(lo - 0.5, hi + 0.5, tuple(v // 7 for v in PHASE_COLOUR[nm])) for nm, lo, hi in phases]
    xt = [((lo + hi) / 2, nm) for nm, lo, hi in phases]
    ms = np.arange(1, last + 1)
    differ = tc.loc[ms, "differential_spike_cells"].to_numpy(float)
    held = tc.loc[ms, "trace_cells"].to_numpy(float)
    top = 10 ** int(np.ceil(np.log10(max(differ.max(), held.max()))))
    c1 = vd.Chart((150, 880, 880, 1020), (0.5, last + 0.5), (0, top), "cells per ms", [0] + [10 ** k for k in range(1, int(np.log10(top)) + 1)],
                  log="symlog", bands=bands, xticks=xt, ytick_fmt="{:,.0f}")
    pos_ = norm[1:][norm[1:] > 0]
    k_lo, k_hi = int(np.floor(np.log10(pos_.min()))), int(np.ceil(np.log10(pos_.max())))
    c2 = vd.Chart((1090, 880, 1800, 1020), (0.5, last + 0.5), (10.0 ** k_lo, 10.0 ** k_hi), "trace strength",
                  [10.0 ** k for k in range(k_lo, k_hi + 1)], log=True, bands=bands, xticks=xt, ytick_fmt=lambda v: f"{v:,.0f}" if v >= 1 else f"{v:g}")

    title = "Write, hold, read: one memory trial, millisecond by millisecond"
    sub = f"Scene 1, readout trained; {DELAY}-ms delay, {PREFIX} input windows. The cue is held in adaptation while no cell spikes differently."
    enc = vd.Encoder(OUT / "Video_S2.mp4")
    blank = Image.new("RGB", (vd.W, vd.H))
    poster, meta_frames = None, 0
    for t in range(1, last + 1):
        spikes = []
        for c in (0, 1):
            on = by_ms.get((c, t), np.zeros(0, np.int64))
            s01 = np.zeros(n)
            s01[on] = 1.0
            a[c] = a[c] * decay_a + jump * s01
            glow[c] = glow[c] * decay_g + s01
            spikes.append(s01)
        da = (a[1] - a[0])[cell_of]
        diff = (spikes[0] != spikes[1])[cell_of]
        imgs = []
        for c in (0, 1):
            g = glow[c][cell_of]
            i = np.flatnonzero(g > GLOW_MIN)
            imgs.append(stages[c].frame(px[i], py[i], colour[i], np.minimum(g[i], 1.5) * dens[i]))
        on = np.flatnonzero(np.abs(da) > DA_LO)
        u = np.clip(np.log10(np.abs(da[on]) / DA_LO) / np.log10(DA_HI / DA_LO), 0, 1)
        col = rp.srgb_to_linear(pal.SIGNED_ON_DARK(0.5 + np.sign(da[on]) * (0.26 + 0.24 * u))[:, :3])
        w = (0.25 + 1.3 * u ** 1.5) * dens[on]
        sparks = np.flatnonzero(diff)
        imgs.append(stages[2].frame(np.r_[px[on], px[sparks]], np.r_[py[on], py[sparks]],
                                    np.r_[col, np.ones((len(sparks), 3))], np.r_[w, np.full(len(sparks), 1.4)]))
        phase = phase_of(t, ph)
        img = blank.copy()
        for x, im in zip(XS, imgs):
            img.paste(im, (x, Y0))
        d = ImageDraw.Draw(img)
        vd.text(d, (60, 44), title, 38, bold=True)
        vd.text(d, (60, 98), sub, 23, fill=vd.MUTED)
        for x, lab in zip(XS, ("after cue 0: cells spiking now", "after cue 1: cells spiking now", "the trace: adaptation, cue 1 minus cue 0")):
            vd.text(d, (x, Y0 - 34), lab, 24, bold=True)
        for c in (0, 1):
            b = board_for(task, c, phase, t, ph)
            x, y = XS[c] + STAGE[0] - 8 * 17 - 6, Y0 + 6
            if b is None:
                d.rectangle([x, y, x + 8 * 17, y + 4 * 17], outline=vd.DIM)
                vd.text(d, (x + 4, y + 22), "blank", 18, fill=vd.DIM)
            else:
                walls, goals, boxes, player, hidden = b
                crop = Image.new("RGB", (170, 170))
                vd.board(ImageDraw.Draw(crop), 0, 0, 17, walls, goals, boxes, player, hidden_goals=hidden)
                img.paste(crop.crop((17 * 2, 17 * 3, 17 * 8, 17 * 6)), (x, y))
        vd.text(d, (vd.W - 60, 50), f"{t} ms, {phase}", 30, bold=True, anchor="ra")
        c1.frame(d, t)
        c1.series(d, ms, held, GOLD, t)
        c1.series(d, ms, differ, vd.INK, t)
        c1.keys(d, [("spiking differently", vd.INK, None), ("holding the trace", GOLD, None)])
        c2.frame(d, t)
        c2.dashed(d, ref_t, ref, REF)
        c2.series(d, ms, norm[1:], GOLD, t, label_fmt="{:,.1f}")
        c2.keys(d, [("200-ms decay", REF, "dash")])
        reps = frames_per_ms(phase) + (60 if t == last else 0) + (45 if t == 1 else 0)
        enc.add(img, reps)
        meta_frames += reps
        if t == int(ph["delay_end_ms"]):
            poster = img
    frames = enc.close()
    poster.save(OUT / "Video_S2_poster.png", optimize=False)
    meta = dict(video="Video_S2.mp4", frames=frames, fps=vd.FPS, seconds=frames / vd.FPS, delay_ms=DELAY, prefix_windows=PREFIX,
                ms=last, frames_per_ms={"delay": 2, "other": 3}, stage_px=list(STAGE), da_range=[DA_LO, DA_HI],
                trace_strength={str(k): float(norm[k]) for k in (cue_end, quiet, delay_end, input_end, last)}, silent_from_ms=quiet + 1,
                delay_decay_max_rel_error=decay_err)
    (OUT / "Video_S2.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    print(f"Video S2: {frames} frames ({frames / vd.FPS:.1f} s), {time.time() - t0:.0f} s; delay decay error {decay_err:.2e}", flush=True)


if __name__ == "__main__":
    main()
