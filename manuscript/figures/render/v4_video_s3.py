"""Video S3 (Cell; related to Figure 5) - one trace, two readings: the trained 160-ms delay and the untrained 640-ms delay.

Scene 1, readout trained, 4 input windows, hidden goal. Two dark CNS panels, brain on top: the trace (each cell's adaptation
difference between the cues, on the dark cue map of Figure 2C) with the cells spiking differently in that millisecond as
white sparks. The two trials run phase by phase in step: the cue and the input at the same speed, the delay compressed to
the same screen time, so the 640-ms delay runs four times faster. At each decision a mark per cue shows whether the history
was solved. Three live charts on the shared phase axis: the cosine between the two trials' traces at matching moments (the
same pattern, only weaker, through the silent delay), each trace's strength (the norm of the adaptation difference), and from
the decision on each cue's push toward its target at the first decision (Figure 5F: the push shared by both cues added for cue
1 and subtracted for cue 0). The end card shows the decision-time cue difference after 160, 320 and 640 ms (Figure 5A) with
each pattern's cosine to the trained one.

Data: manuscript/results/plan_v2/video_trial_spikes.csv, video_trial_cells.csv, video_trial_phases.csv (P12, the trace rebuilt
from the spikes and checked against the record at every ms), fig5_record_outcomes.csv, fig5_code.csv, fig5_decision_cells.csv,
fig5_decision_cosines.csv; soma positions from fd_cells.csv.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_video_s3.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import palette as pal  # noqa: E402
from render import project as rp, scenes as sc, video as vd  # noqa: E402

OUT = FIG / "v4" / "videos"
PLAN = rp.RESULTS / "plan_v2"
DELAYS, PREFIX = (160, 640), 4
STAGE = (500, 720)
XS = (60, 600)
Y0 = 186
CX0, CX1 = 1290, 1800                                                   # the chart column
FRAMES = {"cue": 80, "delay": 160, "input": 320, "decision": 80}       # the same screen time per phase for both trials
DA_LO, DA_HI = 1e-3, 3.0                                                # trace colour scale, as Video S2 and Figure 2C
DV_LO, DV_HI = 0.05, 5.0                                                # decision-time pattern colour scale (mV)
OK, NO = (120, 200, 120), (230, 90, 90)
TRIAL_COL = {160: (214, 160, 60), 640: (110, 170, 255)}
PHASE_COLOUR = {"cue": (214, 160, 60), "delay": (90, 90, 90), "input": (92, 200, 255), "decision": (236, 236, 236)}


def signed(values, lo, hi):
    on = np.abs(values) > lo
    u = np.clip(np.log10(np.abs(values[on]) / lo) / np.log10(hi / lo), 0, 1)
    col = rp.srgb_to_linear(pal.SIGNED_ON_DARK(0.5 + np.sign(values[on]) * (0.26 + 0.24 * u))[:, :3])
    return np.flatnonzero(on), col, 0.25 + 1.3 * u ** 1.5


def ms_at(frame, ph):
    """The trial's ms shown at a frame, phase by phase."""
    bounds = [("cue", 0, ph["cue_end_ms"]), ("delay", ph["cue_end_ms"], ph["delay_end_ms"]), ("input", ph["delay_end_ms"], ph["input_end_ms"]),
              ("decision", ph["input_end_ms"], ph["decision_end_ms"])]
    f = frame
    for name, a, b in bounds:
        n = FRAMES[name]
        if f < n:
            return name, int(a + 1 + min(b - a - 1, int(f * (b - a) / n)))
        f -= n
    return "decision", int(ph["decision_end_ms"])


def main():
    t0 = time.time()
    phs = pd.read_csv(PLAN / "video_trial_phases.csv").set_index("delay_ms")
    sp = pd.read_csv(PLAN / "video_trial_spikes.csv")
    jump = pd.read_csv(PLAN / "video_trial_cells.csv")["adapt_jump"].to_numpy(np.float64)
    rec = pd.read_csv(PLAN / "fig5_record_outcomes.csv")
    rec = rec[(rec["scene"] == "scene 1") & (rec["trained_parameters"] == "readout") & (rec["prefix_windows"] == PREFIX)].set_index("delay_ms")
    dec = pd.read_csv(PLAN / "fig5_decision_cells.csv")
    cosv = pd.read_csv(PLAN / "fig5_decision_cosines.csv").set_index(["timing_a", "timing_b"])["cosine"]
    fd = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False)
    s, xyz = sc.somata(fd)
    cell_of = s["cell"].to_numpy()
    stage = vd.Stage(xyz, *STAGE)
    px, py = stage.project(xyz)
    dens = sc.density_weights(stage.view, xyz, power=0.4)
    n = len(jump)
    decay = np.exp(-1.0 / 200.0)
    state = {}
    for d in DELAYS:                                   # each trial advances ms by ms as the frames need it
        q = sp[sp["delay_ms"] == d]
        by = {(c, t): g["cell"].to_numpy() for (c, t), g in q.groupby(["cue", "ms"])}
        state[d] = dict(by=by, a=[np.zeros(n), np.zeros(n)], t=0, diff=np.zeros(len(cell_of), bool), img=None, shown=None)

    # live metrics, frame by frame: the cosine between the two trials' traces at matching moments and each trace's strength (the
    # norm of the adaptation difference over all cells); from the decision on, each cue's push toward its target (Figure 5F)
    total = sum(FRAMES.values())
    a_, t_ = {d: [np.zeros(n), np.zeros(n)] for d in DELAYS}, dict.fromkeys(DELAYS, 0)
    cosf, normf = np.full(total, np.nan), {d: np.zeros(total) for d in DELAYS}
    for f in range(total):
        da_ = {}
        for d in DELAYS:
            _, t = ms_at(f, phs.loc[d])
            while t_[d] < t:
                t_[d] += 1
                for c in (0, 1):
                    on = state[d]["by"].get((c, t_[d]), np.zeros(0, np.int64))
                    a_[d][c] *= decay
                    a_[d][c][on] += jump[on]
            da_[d] = a_[d][1] - a_[d][0]
            normf[d][f] = np.linalg.norm(da_[d])
        if min(normf[d][f] for d in DELAYS) > 0:
            cosf[f] = float(da_[DELAYS[0]] @ da_[DELAYS[1]] / (normf[DELAYS[0]][f] * normf[DELAYS[1]][f]))
    code = pd.read_csv(PLAN / "fig5_code.csv")
    code = code[(code["scene"] == "scene 1") & (code["trained_parameters"] == "readout") & (code["prefix_windows"] == PREFIX)].set_index("delay_ms")
    push = {d: (float(code.loc[d, "differential"] - code.loc[d, "common"]), float(code.loc[d, "differential"] + code.loc[d, "common"])) for d in DELAYS}
    for d in DELAYS:                                   # the per-cue pushes agree with the record's correct first pushes
        if sum(p > 0 for p in push[d]) != int(rec.loc[d, "first_pushes_correct"]):
            raise RuntimeError(f"the pushes at {d} ms disagree with the record's correct first pushes")
    edges = np.cumsum([0] + list(FRAMES.values()))
    bands = [(edges[k] - 0.5, edges[k + 1] - 0.5, tuple(v // 7 for v in PHASE_COLOUR[nm])) for k, nm in enumerate(FRAMES)]
    xt = [((edges[k] + edges[k + 1] - 1) / 2, nm) for k, nm in enumerate(FRAMES)]
    frames_x = np.arange(total)
    c_lo = min(0.0, float(np.floor(np.nanmin(cosf) * 2) / 2))
    ch1 = vd.Chart((CX0, 240, CX1, 380), (-0.5, total - 0.5), (c_lo, 1.0), "same trace? (cosine)", sorted({c_lo, 0.0, 0.5, 1.0}),
                   bands=bands, xticks=xt, ytick_fmt=lambda v: f"{v:g}".replace("-", "\u2212"))
    nz = np.r_[normf[160], normf[640]]
    nz = nz[nz > 0]
    k_lo, k_hi = int(np.floor(np.log10(nz.min()))), int(np.ceil(np.log10(nz.max())))
    ch2 = vd.Chart((CX0, 480, CX1, 620), (-0.5, total - 0.5), (10.0 ** k_lo, 10.0 ** k_hi), "trace strength",
                   [10.0 ** k for k in range(k_lo, k_hi + 1)], log=True, bands=bands, xticks=xt,
                   ytick_fmt=lambda v: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    big = max(abs(v) for p in push.values() for v in p)
    pm, tick = 1.5 * big, float(np.ceil(big / 2) * 2)          # head room for the values at the bar ends
    ch3 = vd.Chart((CX0, 720, CX1, 900), (-0.7, 4.7), (-pm, pm), "push toward the cued target", [-tick, 0, tick],
                   xticks=[(0, "cue 0"), (1, "cue 1"), (3, "cue 0"), (4, "cue 1")],
                   ytick_fmt=lambda v: f"{v:+g}".replace("-", "\u2212") if v else "0")

    def charts(d_, f):
        ch1.frame(d_, f)
        ch1.series(d_, frames_x, cosf, vd.INK, f, label_fmt="{:.2f}")
        ch2.frame(d_, f)
        for d in DELAYS[::-1]:
            ch2.series(d_, frames_x, normf[d], TRIAL_COL[d], f, label_fmt=None)
        ch2.tips(d_, [(f, normf[d][f], TRIAL_COL[d], f"{normf[d][f]:,.1f}") for d in DELAYS if normf[d][f] > 0])
        ch2.keys(d_, [(f"{d} ms", TRIAL_COL[d], None) for d in DELAYS])
        ch3.frame(d_)
        _, yz = ch3.px(0, 0)
        d_.line([ch3.x0, yz, ch3.x1, yz], fill=vd.DIM, width=1)
        grow = min(1.0, max(0.0, (f - edges[3] + 1) / 20))
        for g, d in enumerate(DELAYS):
            for c in (0, 1):
                xc, yv = ch3.px(3 * g + c, push[d][c] * grow)
                if grow > 0:
                    d_.rectangle([xc - 26, min(yz, yv), xc + 26, max(yz, yv)], fill=TRIAL_COL[d])
                if grow == 1:
                    up = push[d][c] > 0
                    vd.text(d_, (xc, yv - 6 if up else yv + 6), f"{push[d][c]:+.1f}".replace("-", "\u2212"), 18, fill=TRIAL_COL[d],
                            anchor="md" if up else "ma")
            gx, _ = ch3.px(3 * g + 0.5, 0)
            vd.text(d_, (gx, ch3.y1 + 32), f"{d} ms", 18, fill=TRIAL_COL[d], bold=True, anchor="ma")

    def render(d, t):
        st = state[d]
        while st["t"] < t:
            st["t"] += 1
            sp01 = []
            for c in (0, 1):
                v = np.zeros(n)
                v[st["by"].get((c, st["t"]), np.zeros(0, np.int64))] = 1.0
                st["a"][c] = st["a"][c] * decay + jump * v
                sp01.append(v)
            st["diff"] = (sp01[0] != sp01[1])[cell_of]
        if st["shown"] != t:
            da = (st["a"][1] - st["a"][0])[cell_of]
            i, col, w = signed(da, DA_LO, DA_HI)
            sparks = np.flatnonzero(st["diff"])
            st["img"] = stage.frame(np.r_[px[i], px[sparks]], np.r_[py[i], py[sparks]], np.r_[col, np.ones((len(sparks), 3))],
                                    np.r_[w * dens[i], np.full(len(sparks), 1.4)])
            st["shown"] = t
        return st["img"]

    enc = vd.Encoder(OUT / "Video_S3.mp4")
    blank = Image.new("RGB", (vd.W, vd.H))
    title, sub = "One trace, two readings", f"Scene 1, readout trained, {PREFIX} input windows: the trained 160-ms and the untrained 640-ms delay"
    for f in range(total):
        img = blank.copy()
        d_ = ImageDraw.Draw(img)
        vd.text(d_, (60, 44), title, 38, bold=True)
        vd.text(d_, (60, 98), sub, 23, fill=vd.MUTED)
        for x, d in zip(XS, DELAYS):
            ph = phs.loc[d]
            phase, t = ms_at(f, ph)
            img.paste(render(d, t), (x, Y0))
            d_.ellipse([x, Y0 - 29, x + 16, Y0 - 13], fill=TRIAL_COL[d])
            vd.text(d_, (x + 26, Y0 - 34), f"{'trained' if d == 160 else 'untrained'}: {d} ms", 26, bold=True)
            ty, w = Y0 + STAGE[1] + 10, STAGE[0]
            vd.text(d_, (x, ty + 22), f"{t} ms, {phase}" + (" (4× faster)" if phase == "delay" and d == 640 else ""), 22, fill=vd.MUTED)
            if phase == "decision":
                r = rec.loc[d]
                for c in (0, 1):
                    solved = str(r[f"cue{c}_solved"]).lower() == "true"
                    cx0 = x + w - 190 + 95 * c
                    d_.ellipse([cx0, ty + 22, cx0 + 22, ty + 44], fill=OK if solved else NO)
                    vd.text(d_, (cx0 + 28, ty + 24), f"cue {c}", 20, fill=vd.MUTED)
        charts(d_, f)
        enc.add(img, 1 + (45 if f == 0 else 0) + (45 if f == total - 1 else 0))
    # end card: the decision-time pattern after 160, 320 and 640 ms, with its cosine to the trained one
    card = blank.copy()
    d_ = ImageDraw.Draw(card)
    vd.text(d_, (60, 44), "The same trace, read after three delays", 38, bold=True)
    vd.text(d_, (60, 98), "Cue difference of the membrane potential at the first decision; cosine with the trained 160-ms pattern", 23, fill=vd.MUTED)
    small = vd.Stage(xyz, 500, 700)
    spx, spy = small.project(xyz)
    sdens = sc.density_weights(small.view, xyz, power=0.4)
    for k, d in enumerate((160, 320, 640)):
        v = dec.set_index("cell").loc[cell_of, f"dv_d{d}_p4_mv"].to_numpy(float)
        i, col, w = signed(v, DV_LO, DV_HI)
        x = 140 + k * 580
        card.paste(small.frame(spx[i], spy[i], col, w * sdens[i]), (x, 200))
        c = float(cosv.loc[("dv_d160_p4_mv", f"dv_d{d}_p4_mv")])
        vd.text(d_, (x, 166), f"{d} ms" + ("  (trained)" if d == 160 else ""), 26, bold=True)
        vd.text(d_, (x + 250, 920), "reference" if d == 160 else f"cosine {c:+.2f}".replace("-", "−"), 28, bold=d != 160, anchor="ma")
    enc.add(card, 150)
    frames = enc.close()
    card.save(OUT / "Video_S3_poster.png", optimize=False)
    meta = dict(video="Video_S3.mp4", frames=frames, fps=vd.FPS, seconds=frames / vd.FPS, delays_ms=list(DELAYS), prefix_windows=PREFIX,
                frames_per_phase=FRAMES, da_range=[DA_LO, DA_HI], dv_range=[DV_LO, DV_HI],
                cosines={str(d): float(cosv.loc[("dv_d160_p4_mv", f"dv_d{d}_p4_mv")]) for d in (160, 320, 640)},
                trace_cosine={nm: float(cosf[edges[k + 1] - 1]) for k, nm in enumerate(FRAMES)},
                trace_strength={str(d): {nm: float(normf[d][edges[k + 1] - 1]) for k, nm in enumerate(FRAMES)} for d in DELAYS},
                push={str(d): {"cue 0": push[d][0], "cue 1": push[d][1]} for d in DELAYS})
    (OUT / "Video_S3.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    print(f"Video S3: {frames} frames ({frames / vd.FPS:.1f} s), {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
