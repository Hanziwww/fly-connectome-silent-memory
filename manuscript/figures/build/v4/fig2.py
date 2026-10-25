"""v4 Fig. 2 (Cell) - Working memory is held in an activity-silent intrinsic trace (174 mm wide).

A  the trace through one trial (scene 1, readout trained): cells with a voltage difference, and the cells holding the
   adaptation trace by superclass group, ms by ms
B  differential spikes of scene 1 under the trial strip: they stop early in the delay; input restarts them
C  dark filmstrip, the trace through the same trial at five moments (end of cue, 80 ms, end of delay, 240 ms, decision):
   adaptation difference on the cue map, white for cells that spiked differently in the 40 ms before. Written by spikes,
   held silently, read out by input
D  decay of the cue difference by state variable: synaptic current within ms, adaptation with its 200-ms time constant
E  the voltage difference is carried by adaptation at the end of the delay and by synaptic input at the decision
F  transplanting the end-of-delay state recreates the decision; adaptation alone does so at long delays
G  swapping in the other cue's state flips decisions (four learners, family-wise Holm)
H  input reads the trace: the decision-time drive is synaptic at all 35 timings; and the cue drive into readout types comes
   mostly from the few types that held the trace (by the readout types' region, against the share of types that hold it)
I  holding the trace costs no spikes (the one-trial spike and information budget is in the source data; it is the healthy
   point of Fig. 7K)
J  without input the trace fades passively; input renews it
K  input reads the trace through the cells that hold it, across the CNS: a typed network in the grammar of MaleCNS Fig. 5F
   (brain and VNC split at the dashed rule; modules by anatomy; gold: the type held the trace through the silent delay;
   outlined: readout types; edges: cue drive at the first decision, synapse weight x presynaptic cue spiking difference summed
   per type pair, four equal bins, coloured and headed by the sender's transmitter)

Every number in a label is read from manuscript/results (plan_v2 via P12, followup, figure_data) or from the render metadata.
Reader-facing names come from plan_v2/names.csv (NARRATIVE_V4_OUTLINE.md section 12). Class (c) results are not shown
(RESULTS_INVENTORY_V4.md): adaptation alone at the 160-ms delay, and the shared hidden/visible-goal readout units of the swap
family. The four-learner coverage ladder and the delay-end trace of scenes 2 and 3 (the first build) are in Figure S2.

Run after P12 and render/v4_trial_tiles.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\fig2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
from lib import circuit as ci, figstyle as fs, palette as pal, qc  # noqa: E402

cm.use_v4()
TRIAL_DIR = cm.RENDERS / "v4" / "trial"
TRIAL = json.loads((TRIAL_DIR / "tiles.json").read_text(encoding="utf-8"))
TW, TH = TRIAL["trace_mm"]
U = fs.unit(1)
XS = [fs.unit_x(i) for i in range(4)]
R1_H = 45.0
BAND_Y = R1_H + 4.2
KEY_Y = BAND_Y + TH + 2.8
ROW3 = BAND_Y + TH + 6.0
R3_H = 48.0
ROW4 = ROW3 + R3_H + 3.4
R4_H = 60.0
H = ROW4 + R4_H
WHITE = fs.ON_DARK
NL_ = chr(10)
B_ = "brain and readout"
# learner key (P12), model code (fu4, fu9), state key (fd_coverage_graded, palette)
LEARNERS = [("s1_readout", "remote-dynamic-M0-head_only", "m0"), ("s1_brain", "remote-dynamic-M0-joint", "m0_joint"),
            ("s2_readout", "supp-S1-ring-brain", "s1"), ("s3_readout", "supp-S2-context-brain", "s2_o0")]
COL = {k: pal.STATE_SET[s] for k, _, s in LEARNERS}
CODE_COL = {c: pal.STATE_SET[s] for _, c, s in LEARNERS}
SWAP_UNITS = {"M0": "remote-dynamic-M0-head_only", "M0joint": "remote-dynamic-M0-joint", "S1": "supp-S1-ring-brain",
              "S2": "supp-S2-context-brain"}
SPIKE_ON_WHITE = "#B07A00"
SEGMENTS = [("cue", "#E4E4E4"), ("delay", "#3A3A3A"), ("prefix", "#BDBDBD"), ("decision", "#E4E4E4")]
GROUP_ORDER = ["optic lobe intrinsic", "central brain intrinsic", "VNC intrinsic", "visual projection & centrifugal",
               "descending & ascending", "motor & efferent", "sensory", "other"]
SNAP_LABEL = {40: "Cue end", 80: "Delay", 200: "Delay end", 240: "Prefix", 400: "Decision"}
HELD = "#EBC05A"                                       # H, K: the types that held the trace through the silent delay


def names():
    return cm.table("plan_v2/names.csv")


def label(code, sep=", "):
    r = names()
    r = r[(r["code"] == code) & (r["kind"] == "model")].iloc[0]
    return r["scene"].capitalize() + sep + ("brain + readout" if r["trained"] == B_ else "readout") + " trained"


def models(codes=None):
    return [c for _, c, _ in LEARNERS] if codes is None else codes


def tile_text(cv, x, y, s, size=None, **kw):
    t = cv.text(x, y, s, size=size or fs.PT_SMALL, color=WHITE, **kw)
    t.set_path_effects(fs.dark_halo(0.9))
    return t


def white_letter(cv, x, y, s):
    cv.letter(x, y, s)
    cv.fig.texts[-1].set_color(WHITE)


def title(cv, x0, y0, letter, text):
    cv.letter(x0, y0, letter)
    cv.text(x0 + 4.0, y0 + 0.5, text, size=fs.PT, va="top", fontweight="bold")


def strip(ax, events, y, hgt):
    """Trial strip in data coordinates: events [(kind, start_ms, end_ms)]."""
    col = dict(SEGMENTS)
    for kind, a, b in events:
        ax.add_patch(Rectangle((a, y), b - a, hgt, facecolor=col[kind], edgecolor=fs.AXIS, linewidth=0.3, clip_on=False))


def test_rows(family):
    t = cm.table("plan_v2/tests.csv")
    return t[t["family"] == family]


def gradient(cv, x, y, w, h, colours):
    ax = cv.axes(x, y, w, h)
    ax.imshow(np.asarray(colours)[:, None, :3] * np.ones((1, 2, 1)), aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_axis_off()


# ---------------------------------------------------------------- A: the trace through one trial, by class

def panel_course(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "A", "The trace through one trial")
    tc = cm.table("plan_v2/trial_timecourse.csv")
    tc = tc.assign(group=tc["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other"))
    s = cm.table("plan_v2/fig2_delay_end_summary.csv").set_index("learner").loc["s1_readout"]
    cue, end_delay = int(s["cue_offset_ms"]), int(s["end_of_delay_ms"])
    t_end = int(tc["ms"].max())
    first_dec = t_end - 40
    events = [("cue", 0, cue), ("delay", cue, end_delay), ("prefix", end_delay, first_dec), ("decision", first_dec, t_end)]
    left, aw = x0 + 12.6, w - 14.2
    ax_s = cv.axes(left, y0 + 10.4, aw, 1.2)
    strip(ax_s, events, 0, 1)
    ax_s.set_xlim(0, t_end)
    ax_s.set_ylim(0, 1)
    ax_s.set_axis_off()
    for kind, a, b in events:
        cv.text(left + aw * (a + b) / 2 / t_end, y0 + 9.8, kind, size=fs.PT_SMALL, ha="center", va="bottom")
    # voltage difference: cells above 0.01 mV (log)
    v = tc.groupby("ms")["voltage_cells"].sum()
    ax_v = cv.axes(left, y0 + 12.6, aw, 9.6)
    ax_v.plot(v.index, v.to_numpy(), color=fs.TEXT2, lw=0.8, zorder=3)
    v_end = int(v.loc[end_delay])
    ax_v.scatter([end_delay], [v_end], s=5, facecolor=fs.TEXT, edgecolor="none", zorder=4)
    ax_v.set_yscale("log")
    ax_v.set_ylim(500, 3e5)
    ax_v.set_yticks([1e3, 1e5], ["10$^3$", "10$^5$"])
    ax_v.minorticks_off()
    ax_v.set_xlim(0, t_end)
    ax_v.set_xticks([])
    ax_v.set_ylabel(f"Voltage{NL_}cells", labelpad=0.8, linespacing=1.0)
    fs.clean(ax_v, bottom=False)
    ax_v.spines["bottom"].set_visible(False)
    ax_v.text(end_delay + 22, 620, f"{v_end:,}", fontsize=fs.PT_SMALL, ha="left", va="bottom")
    # the trace: cells with an adaptation difference, by group (stacked)
    g = tc.groupby(["ms", "group"])["trace_cells"].sum().unstack(fill_value=0)
    groups = [grp for grp in GROUP_ORDER if grp in g and g[grp].max() > 0]
    ax = cv.axes(left, y0 + 23.6, aw, h - 31.6)
    ms = g.index.to_numpy()
    bottom = np.zeros(len(ms))
    for grp in groups:
        val = g[grp].to_numpy(float)
        ax.fill_between(ms, bottom, bottom + val, color=pal.on_white(pal.ATLAS_ON_DARK[grp]), lw=0, step="mid", zorder=2)
        bottom += val
        sd.add("A", "trace cells", group=grp, ms=ms.tolist(), cells=g[grp].astype(int).tolist())
    ax.set_xlim(0, t_end)
    ax.set_ylim(0, bottom.max() * 1.08)
    ax.set_yticks([0, 3000, 6000], ["0", "3,000", "6,000"])
    ax.set_xlabel("Time from cue onset (ms)", labelpad=0.8)
    ax.set_ylabel(f"Trace{NL_}cells", labelpad=0.8, linespacing=1.0)
    fs.clean(ax)
    held = int(bottom[np.searchsorted(ms, end_delay)])
    ax.text((cue + end_delay) / 2, held * 1.04, f"{held:,}", fontsize=fs.PT_SMALL, ha="center", va="bottom")
    end = cv.key(x0 + 4.0, y0 + 5.6, [([dict(kind="box", fc=pal.on_white(pal.ATLAS_ON_DARK[grp]))], pal.SHORT_GROUP[grp]) for grp in groups],
                 size=fs.PT_SMALL, gap=1.3)
    if end > XS[2] - 1.0:
        raise RuntimeError(f"the group key of A runs into B ({end:.1f} mm)")
    sd.add("A", "voltage cells", ms=v.index.tolist(), cells=v.astype(int).tolist(), threshold_mv=0.01)
    sd.add("A", "summary", trace_cells_at_delay_end=held, voltage_cells_at_delay_end=v_end, learner=label(LEARNERS[0][1]))


# ---------------------------------------------------------------- B: raster

def panel_raster(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "B", "Spikes that tell the cues apart stop early in the delay")
    r = cm.table("plan_v2/fig2_raster.csv")
    s = cm.table("plan_v2/fig2_delay_end_summary.csv").set_index("learner").loc["s1_readout"]
    d = cm.table("followup/fu4_state_dynamics.csv")
    d = d[(d["model"] == LEARNERS[0][1]) & (d["context"] == 0) & (d["visibility"] == "hidden") & (d["delay_ms"] == s["delay_ms"])
          & (d["prefix_windows"] == s["prefix_windows"])].sort_values("ms")
    t_end = int(d["ms"].max())
    cue, end_delay = int(s["cue_offset_ms"]), int(s["end_of_delay_ms"])
    first_dec = t_end - 40
    events = [("cue", 0, cue), ("delay", cue, end_delay), ("prefix", end_delay, first_dec), ("decision", first_dec, t_end)]
    ymax = float(d["energy_spikes"].max())
    left, aw = x0 + 9.0, w - 11.0
    ax_strip = cv.axes(left, y0 + 10.4, aw, 1.2)
    strip(ax_strip, events, 0, 1)
    ax_strip.set_xlim(0, t_end)
    ax_strip.set_ylim(0, 1)
    ax_strip.set_axis_off()
    for kind, a, b in events:
        cv.text(left + aw * (a + b) / 2 / t_end, y0 + 9.8, kind, size=fs.PT_SMALL, ha="center", va="bottom")
    ax_top = cv.axes(left, y0 + 12.4, aw, 6.4)
    ax_top.fill_between(d["ms"] - 0.5, d["energy_spikes"], step="post", color="#6E6E6E", lw=0, zorder=2)
    ax_top.set_xlim(0, t_end)
    ax_top.set_ylim(0, ymax * 1.05)
    ax_top.set_xticks([])
    ax_top.set_yticks([0, 200], ["0", "200"])
    ax_top.set_ylabel("Cells/ms", labelpad=0.8)
    fs.clean(ax_top)
    ax_top.spines["bottom"].set_visible(False)
    ax = cv.axes(left, y0 + 19.4, aw, h - 25.6)
    first = r.groupby("cell")["ms"].min().sort_values(kind="mergesort")
    rank = pd.Series(np.arange(len(first)), index=first.index)
    for sign, colour in ((-1, pal.CUE[0]), (1, pal.CUE[1])):
        q = r[r["sign"] == sign]
        ax.scatter(q["ms"] - 0.5, rank.loc[q["cell"]].to_numpy(), s=0.25, c=colour, linewidths=0, rasterized=True, zorder=3)
    silent_a = cue + int(s["last_differential_spike_ms_after_cue_offset"])
    ax.axvspan(silent_a, end_delay, color="#F2F2F2", lw=0, zorder=1)
    ax.text((silent_a + end_delay) / 2, len(first) * 0.5, f"{end_delay - silent_a} ms",
            fontsize=fs.PT_SMALL, ha="center", va="center", linespacing=1.0)
    ax.set_xlim(0, t_end)
    ax.set_ylim(len(first), 0)
    ax.set_yticks([len(first)], [f"{len(first):,}"])
    ax.set_xlabel("Time from cue onset (ms)", labelpad=0.8)
    ax.set_ylabel("Cells", labelpad=0.8)
    fs.clean(ax)
    cv.key(x0 + 4.0, y0 + 5.6, [([dict(kind="mark", s=5, fc=pal.CUE[0], ec=pal.CUE[0])], "spike only after cue 0"),
                                ([dict(kind="mark", s=5, fc=pal.CUE[1], ec=pal.CUE[1])], "only after cue 1")], size=fs.PT_SMALL, gap=1.6)
    cv.text(x0 + w - 0.8, y0 + 5.6, label(LEARNERS[0][1]) + f", {int(s['delay_ms'])}-ms delay", size=fs.PT_SMALL, ha="right", va="center",
            color=fs.TEXT2)
    sd.add("B", "raster", events=len(r), cells=len(first), learner=label(LEARNERS[0][1]), delay_ms=int(s["delay_ms"]),
           prefix_windows=int(s["prefix_windows"]), silent_from_ms=silent_a, silent_to_ms=end_delay)


# ---------------------------------------------------------------- C: dark filmstrip, the trace through the trial

def panel_band(cv, sd):
    cells = cm.table("plan_v2/trial_cells.csv")
    snaps = TRIAL["snapshots_ms"]
    xs = [k * (TW + fs.GUTTER) for k in range(len(snaps))]
    for k, (t, x) in enumerate(zip(snaps, xs)):
        name = f"trace_t{t}"
        meta = json.loads((TRIAL_DIR / f"{name}.json").read_text(encoding="utf-8"))
        cv.tile(x, BAND_Y, np.asarray(Image.open(TRIAL_DIR / f"{name}.png").convert("RGB")), width=TW, height=TH, group="band", label=name)
        held = int((cells[f"dadapt_t{t}"] != 0).sum())
        sparks = int((cells[f"diffspike_w{t}"] > 0).sum())
        tx = x + (4.6 if k == 0 else 1.2)
        tile_text(cv, tx, BAND_Y + 1.2, f"{SNAP_LABEL[t]} · {t} ms", va="top", size=fs.PT)
        tile_text(cv, tx, BAND_Y + 4.0, f"{held:,} hold the trace", va="top")
        tile_text(cv, tx, BAND_Y + 6.4, "no cell spikes differently" if sparks == 0 else f"{sparks:,} spike differently", va="top")
        sd.add("C", "moment", ms=int(t), trace_cells=held, spiking_differently_last_40_ms=sparks, trace_cells_drawn=int(meta["trace_cells_located"]))
    white_letter(cv, 1.2, BAND_Y + 0.9, "C")
    # key, black space right of the VNC in the first tile: adaptation difference (log, both signs) and the white sparks
    lo, hi = TRIAL["da_range"]
    bx, by, half, gap = xs[0] + TW - 2.4, BAND_Y + 27.1, 4.6, 2.2
    u = np.linspace(1, 0, 96)
    gradient(cv, bx, by, 1.3, half, pal.SIGNED_ON_DARK(0.5 + 0.26 + 0.24 * u))
    gradient(cv, bx, by + half + gap, 1.3, half, pal.SIGNED_ON_DARK(0.5 - 0.26 - 0.24 * u[::-1]))
    tile_text(cv, bx - 0.4, by, f"+{hi:g}", va="center", ha="right")
    tile_text(cv, bx - 0.4, by + 2 * half + gap, f"{fs.MINUS}{hi:g}", va="center", ha="right")
    tile_text(cv, bx + 1.3, by - 1.6, f"adaptation,{NL_}cue 1 {fs.MINUS} cue 0", va="bottom", ha="right", linespacing=1.0)
    cv.mark(bx + 0.65, by + 2 * half + gap + 2.6, s=5, fc=WHITE, ec=WHITE)
    tile_text(cv, bx - 0.4, by + 2 * half + gap + 2.6, "spikes", va="center", ha="right")
    # scale bar (last tile)
    last = f"trace_t{snaps[-1]}"
    meta = json.loads((TRIAL_DIR / f"{last}.json").read_text(encoding="utf-8"))
    img = np.asarray(Image.open(TRIAL_DIR / f"{last}.png").convert("RGB"))
    length = 100 / meta["scale_um_per_px"] / (img.shape[1] / TW)
    cv.line([xs[-1] + TW - 1.4 - length, xs[-1] + TW - 1.4], [BAND_Y + TH - 2.0, BAND_Y + TH - 2.0], WHITE, lw=1.0, solid_capstyle="butt")
    sd.add("C", "view", scale_bar_um=100, um_per_px=meta["scale_um_per_px"], adaptation_range=[lo, hi], learner=label(LEARNERS[0][1]),
           delay_ms=160, prefix_windows=4)


# ---------------------------------------------------------------- D: decay by variable

def panel_decay(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "D", "What persists is adaptation")
    s = cm.table("followup/fu4_state_summary.csv")
    s = s[s["model"].isin(models())]
    d = cm.table("followup/fu4_state_dynamics.csv")
    delay = int(s["delay_ms"].max())
    d = d[(d["model"] == LEARNERS[0][1]) & (d["context"] == 0) & (d["visibility"] == "hidden") & (d["delay_ms"] == delay)
          & (d["prefix_windows"] == 4)].sort_values("ms")
    cue = int(d.loc[d["segment"] == "cue", "ms"].max())
    dd = d[(d["segment"] == "delay")]
    t = dd["ms"].to_numpy() - cue
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 15.6)
    curves = [("energy_adapt", "adaptation", fs.TEXT, 1.0), ("energy_v", "voltage", "#7A7A7A", 0.7),
              ("energy_g", "synaptic current", "#B5B5B5", 0.7)]
    for col, lab, colour, lw in curves:
        e = dd[col].to_numpy(float)
        ref = e[t == 20][0]
        y = np.clip(e / ref, 1e-12, None)
        ax.plot(t, y, color=colour, lw=lw, zorder=3)
        sd.add("D", "energy", variable=lab, learner=label(LEARNERS[0][1]), delay_ms=delay, reference_ms_after_offset=20,
               ms_after_offset=t.tolist(), relative_energy=y.tolist())
    ax.set_yscale("log")
    ax.set_xlim(0, delay)
    ax.set_ylim(1e-12, 30)
    ax.set_yticks([1e-12, 1e-6, 1], ["10$^{-12}$", "10$^{-6}$", "1"])
    ax.set_xticks([0, delay // 2, delay])
    ax.set_xlabel("ms after cue offset", labelpad=0.8)
    ax.set_ylabel("Cue difference (energy)", labelpad=0.8)
    fs.minor(ax, x=False, y=False)
    fs.clean(ax)
    tg = (s["tau_eff_g_ms"].min(), s["tau_eff_g_ms"].max())
    ta = (s["tau_eff_adapt_ms"].min(), s["tau_eff_adapt_ms"].max())
    r2a = s["r2_adapt"].min()
    ax.text(delay * 0.98, 0.08, f"adaptation{NL_}τ {ta[0]:.0f}–{ta[1]:.0f} ms", fontsize=fs.PT_SMALL, ha="right", va="bottom",
            linespacing=1.0)
    ax.text(delay * 0.98, 1e-6, "voltage", fontsize=fs.PT_SMALL, ha="right", va="top", color="#5E5E5E")
    ax.text(30, 3e-11, f"synaptic current{NL_}τ {tg[0]:.1f}–{tg[1]:.1f} ms", fontsize=fs.PT_SMALL, ha="left", va="bottom",
            color="#5E5E5E", linespacing=1.0)
    cv.text(x0 + w - 0.8, y0 + 6.0, "four learners", size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)
    sd.add("D", "time constants", tau_synaptic_ms=list(map(float, tg)), tau_adaptation_ms=list(map(float, ta)), r2_adaptation_min=float(r2a),
           rows=int(len(s)))


def learner_key(cv, sd):
    items = [([dict(kind="mark", s=7, fc=COL[k], ec=COL[k])], label(c)) for k, c, _ in LEARNERS]
    cv.key(XS[0] + 1.0, KEY_Y, items, size=fs.PT_SMALL, gap=3.0)
    cv.text(fs.FULL - 0.6, KEY_Y, "learners in D–I", size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)
    for k, c, _ in LEARNERS:
        sd.add("key", "learner", key=k, label=label(c), colour=COL[k])


# ---------------------------------------------------------------- E: voltage split

def panel_split(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "E", "Adaptation holds, input reads")
    s = cm.table("followup/fu4_state_summary.csv")
    s = s[s["model"].isin(models()) & (s["visibility"] == "hidden")]
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 16.6)
    groups = [("share_v_adaptation_end_of_delay", 0), ("share_v_synaptic_first_decision", 1)]
    rng = np.random.default_rng(2)
    for col, gx in groups:
        for i, (_, r) in enumerate(s.iterrows()):
            fc = CODE_COL[r["model"]] if r["delay_ms"] == s["delay_ms"].min() else "white"
            ax.scatter([gx + (list(CODE_COL).index(r["model"]) - 1.5) * 0.12 + rng.uniform(-0.02, 0.02)], [100 * r[col]], s=8,
                       facecolor=fc, edgecolor=CODE_COL[r["model"]], linewidth=0.5, zorder=3)
            sd.add("E", col, learner=label(r["model"]), context=int(r["context"]), delay_ms=int(r["delay_ms"]), share=float(r[col]))
    lo_a, hi_a = 100 * s["share_v_adaptation_end_of_delay"].min(), 100 * s["share_v_adaptation_end_of_delay"].max()
    lo_s = 100 * s["share_v_synaptic_first_decision"].min()
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 50, 100])
    ax.set_xticks([0, 1], ["end of delay", "decision"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Share of the voltage difference (%)", labelpad=0.8)
    fs.clean(ax)
    ax.text(0, 70, f"adaptation{NL_}{lo_a:.1f}–{min(hi_a, 100):.0f}%", fontsize=fs.PT_SMALL, ha="center", va="top", linespacing=1.0)
    ax.text(1, 70, f"synapses{NL_}≥ {lo_s:.0f}%", fontsize=fs.PT_SMALL, ha="center", va="top", linespacing=1.0)
    d0, d1 = int(s["delay_ms"].min()), int(s["delay_ms"].max())
    cv.key(x0 + 9.0, y0 + 6.0, [([dict(kind="mark", s=7, fc=fs.TEXT2, ec=fs.TEXT2)], f"{d0}-ms delay"),
                                ([dict(kind="mark", s=7, fc="white", ec=fs.TEXT2)], f"{d1} ms")], size=fs.PT_SMALL, gap=1.6)


# ---------------------------------------------------------------- F: transplant

def panel_transplant(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "F", "Transplanted state suffices")
    t = cm.table("plan_v2/fig2_transplant.csv")
    delays = sorted(t["delay_ms"].unique())
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 15.6)
    xi = {dl: i for i, dl in enumerate(delays)}
    for p, fc in ((4, fs.TEXT), (0, "white")):
        q = t[t["prefix_windows"] == p]
        ax.scatter([xi[d] + (0.12 if p == 0 else -0.12) for d in q["delay_ms"]], q["full_state_cosine"], s=8, facecolor=fc, edgecolor=fs.TEXT,
                   linewidth=0.5, zorder=3)
    a = t.dropna(subset=["adaptation_only_cosine"])
    ax.scatter([xi[d] for d in a["delay_ms"]], a["adaptation_only_cosine"], s=10, marker="D", facecolor=SPIKE_ON_WHITE, edgecolor=SPIKE_ON_WHITE,
               linewidth=0.4, zorder=4)
    for _, r in a.iterrows():
        ax.text(xi[r["delay_ms"]], r["adaptation_only_cosine"] - 0.06, f"{r['adaptation_only_cosine']:.2f}", fontsize=fs.PT_SMALL, ha="center",
                va="top", color=fs.TEXT)
    for _, r in t.iterrows():
        sd.add("F", "transplant", prefix_windows=int(r["prefix_windows"]), delay_ms=int(r["delay_ms"]), full_state_cosine=float(r["full_state_cosine"]),
               full_state_margin=float(r["full_state_margin"]), native_margin=float(r["native_margin"]),
               adaptation_only_cosine=None if pd.isna(r["adaptation_only_cosine"]) else float(r["adaptation_only_cosine"]))
    dm = float((t["full_state_margin"] - t["native_margin"]).abs().max())
    ax.set_xlim(-0.6, len(delays) - 0.4)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks(range(len(delays)), [str(int(d)) for d in delays])
    ax.set_xlabel("Delay (ms)", labelpad=0.8)
    ax.set_ylabel("Cosine with the native decision state", labelpad=0.8)
    fs.clean(ax)
    lo = float(t["full_state_cosine"].min())
    cos = f"cosine {lo:.3f}" if lo >= 0.9995 else f"cosine ≥ {lo:.3f}"
    cv.key(x0 + 9.0, y0 + 6.0, [([dict(kind="mark", s=8, fc=fs.TEXT, ec=fs.TEXT)], "full state"),
                                ([dict(kind="mark", s=9, marker="D", fc=SPIKE_ON_WHITE, ec=SPIKE_ON_WHITE)], "adaptation alone")],
           size=fs.PT_SMALL, gap=1.6)
    sd.add("F", "summary", full_state_cosine_min=lo, margin_difference_max=dm)


# ---------------------------------------------------------------- G: swaps

def panel_swaps(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "G", "Swapped state flips choices")
    s = cm.table("plan_v2/fig2_swaps.csv")
    tests = test_rows("F13.X5.swap")
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 15.6)
    codes = list(SWAP_UNITS)
    for i, unit in enumerate(codes):
        q = s[s["code"] == unit]
        col = CODE_COL[SWAP_UNITS[unit]]
        b, a = 100 * q["baseline_solved"].astype(str).str.lower().eq("true").mean(), 100 * q["swapped_solved"].astype(str).str.lower().eq("true").mean()
        ax.plot([i - 0.18, i + 0.18], [b, a], color=col, lw=0.8, zorder=2)
        ax.scatter([i - 0.18], [b], s=10, facecolor=col, edgecolor=col, linewidth=0.4, zorder=3)
        ax.scatter([i + 0.18], [a], s=10, facecolor="white", edgecolor=col, linewidth=0.6, zorder=3)
        p = float(tests[tests["test"].astype(str).str.startswith(f"{unit}: opposite-cue swap")]["p_adjusted"].iloc[0])
        ax.text(i, 104 + 8 * (i % 2), fs.fmt_p(p), fontsize=fs.PT_SMALL, ha="center", va="bottom")
        sd.add("G", "swap", learner=label(SWAP_UNITS[unit]), histories=int(len(q)), baseline_solved_pct=b, swapped_solved_pct=a, holm_p=p)
    ax.set_xlim(-0.6, len(codes) - 0.4)
    ax.set_ylim(0, 124)
    ax.set_yticks([0, 50, 100])
    ax.set_xticks([])
    ax.set_ylabel("Trained histories solved (%)", labelpad=0.8)
    fs.clean(ax)
    ax.spines["bottom"].set_visible(False)
    cv.key(x0 + 9.0, y0 + 6.0, [([dict(kind="mark", s=8, fc=fs.TEXT2, ec=fs.TEXT2)], "own state"),
                                ([dict(kind="mark", s=8, fc="white", ec=fs.TEXT2)], "swapped")], size=fs.PT_SMALL, gap=1.6)
    cv.text(x0 + w - 0.8, y0 + 6.0, f"Holm {fs.P_SYM}", size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- H: drive split

def panel_drive(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "H", "Input reads the trace")
    d = cm.table("followup/fu9_drive_split.csv")
    d = d[d["model"].isin(models()) & (d["visibility"] == "hidden")]
    ax = cv.axes(x0 + 9.0, y0 + 6.4, w - 10.6, 19.6)
    rng = np.random.default_rng(3)
    for i, code in enumerate(models()):
        q = d[d["model"] == code]
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(q)), 100 * q["share_drive_synaptic"], s=3.5, facecolor=CODE_COL[code], edgecolor="none",
                   alpha=0.85, zorder=3)
        for _, r in q.iterrows():
            sd.add("H", "timing", learner=label(code), context=int(r["context"]), delay_ms=int(r["delay_ms"]),
                   prefix_windows=int(r["prefix_windows"]), share_drive_synaptic=float(r["share_drive_synaptic"]))
    lo = 100 * float(d["share_drive_synaptic"].min())
    n_t = int(d.groupby(["model", "context"]).size().max())
    ax.axhline(lo, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
    ax.set_xlim(-0.6, len(models()) - 0.4)
    ax.set_ylim(50, 108)
    ax.set_yticks([50, 75, 100])
    ax.set_xticks([])
    ax.set_ylabel("Drive from synapses (%)", labelpad=0.8)
    fs.clean(ax)
    ax.spines["bottom"].set_visible(False)
    ax.axhline(lo, color=fs.TEXT2, lw=0.5, ls=(0, (2.0, 1.4)), zorder=1)
    ax.text(len(models()) - 0.5, lo - 2, f"{lo:.0f}%", fontsize=fs.PT_SMALL, ha="right", va="top")
    sd.add("H", "summary", minimum_share_pct=lo, timings=n_t, measure="share of the decision-time drive that is synaptic")
    # below: the cue drive into readout types, the share from the types that held the trace, by the readout types' region
    ty = cm.table("plan_v2/fig2_cue_flow_types.csv").set_index("type")
    e = cm.table("plan_v2/fig2_cue_flow_edges.csv")
    ro = set(ty[ty["role"].astype(str).str.contains("readout")].index)
    into = e[e["post"].isin(ro) & (e["pre"] != e["post"]) & (e["pre"] != "untyped") & (e["post"] != "untyped")].copy()
    sc = ty["superclass"].astype(str)
    into["region"] = into["post"].map(lambda p: "VNC" if sc.get(p, "").startswith("vnc") else ("descending" if "descending" in sc.get(p, "") else "brain"))
    into["held"] = into["pre"].map(ty["hold"]) > 0
    rows = [("brain", into[into["region"] == "brain"]), ("descending", into[into["region"] == "descending"]),
            ("VNC", into[into["region"] == "VNC"]), ("all", into)]
    base = 100 * float((ty["hold"] > 0).mean())
    cv.text(x0 + 1.0, y0 + 31.4, "Cue drive into readout types", size=fs.PT_SMALL, va="center")
    bx = cv.axes(x0 + 13.6, y0 + 35.6, w - 15.2, 15.0)
    yy = np.arange(len(rows))[::-1]
    for yv, (name, d) in zip(yy, rows):
        share = 100 * float(d.loc[d["held"], "abs_drive"].sum() / d["abs_drive"].sum())
        bx.barh(yv, share, height=0.7, color=HELD, linewidth=0)
        bx.barh(yv, 100 - share, left=share, height=0.7, color=ci.GREY_CHIP, linewidth=0)
        bx.text(share / 2, yv, f"{share:.0f}%", ha="center", va="center", fontsize=fs.PT_SMALL, fontweight="bold" if name == "all" else "normal")
        sd.add("H", "cue drive into readout types", region=name, share_from_held_types_pct=share, type_pairs=int(len(d)),
               drive_total=float(d["abs_drive"].sum()))
    bx.axvline(base, color=fs.TEXT, linewidth=0.6, linestyle=(0, (2.0, 1.5)))
    bx.text(base + 2.0, len(rows) - 0.42, f"{base:.0f}% of types", fontsize=fs.PT_SMALL, va="bottom", ha="left")
    bx.set_yticks(yy, [r[0] for r in rows])
    bx.set_xlim(0, 100)
    bx.set_ylim(-0.6, len(rows) + 0.2)
    bx.set_xticks([0, 50, 100], ["0", "50", "100%"])
    bx.set_xlabel("From types holding the trace", labelpad=0.8)
    fs.clean(bx)
    bx.tick_params(axis="y", length=0)
    sd.add("H", "types holding the trace", share_of_types_pct=base, types=int(len(ty)))


# ---------------------------------------------------------------- I: cost

def panel_cost(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "I", "Holding the trace costs no spikes")
    d = cm.table("followup/fu4_state_dynamics.csv")
    delay = int(d["delay_ms"].max())
    d = d[d["model"].isin(models()) & (d["visibility"] == "hidden") & (d["delay_ms"] == delay) & (d["prefix_windows"] == 4)]
    e = cm.table("plan_v2/fig2_energy.csv").iloc[0]
    ax = cv.axes(x0 + 9.0, y0 + 9.2, w - 10.6, h - 16.0)
    silent = []
    for (code, ctx), q in d.groupby(["model", "context"]):
        q = q.sort_values("ms")
        cum = np.cumsum(q["energy_spikes"].to_numpy(float))
        ax.plot(q["ms"], cum / 1000, color=CODE_COL[code], lw=0.7 if ctx == 0 else 0.5, ls="-" if ctx == 0 else (0, (2, 1)), zorder=3)
        dl = q[q["segment"] == "delay"]
        nz = dl.loc[dl["energy_spikes"] > 0, "ms"]
        silent.append(int(dl["ms"].max() - (nz.max() if len(nz) else dl["ms"].min())))
        sd.add("I", "cumulative differential spikes", learner=label(code), context=int(ctx), delay_ms=delay, ms=q["ms"].tolist(), cumulative=cum.tolist())
    cue = int(d.loc[d["segment"] == "cue", "ms"].max())
    t_end = int(d["ms"].max())
    first_dec = t_end - 40
    ax.set_xlim(0, t_end)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax)
    strip(ax, [("cue", 0, cue), ("delay", cue, cue + delay), ("prefix", cue + delay, first_dec), ("decision", first_dec, t_end)], ymax * 1.04,
          ymax * 0.06)
    ax.axvspan(cue + delay - min(silent), cue + delay, color="#F2F2F2", lw=0, zorder=1)
    ax.text(cue + delay - min(silent) / 2, ymax * 0.5, f"none for{NL_}≥ {min(silent)} ms", fontsize=fs.PT_SMALL, ha="center",
            va="center", linespacing=1.0)
    ax.set_xlabel("Time from cue onset (ms)", labelpad=0.8)
    ax.set_ylabel(f"Differential spikes{NL_}(thousands)", labelpad=0.8, linespacing=1.0)
    fs.clean(ax)
    sd.add("I", "budget (one trial, scene 1; shown as the healthy point of Fig. 7K)", spikes_per_trial=float(e["spikes_per_trial"]), cue_bits_per_decision=float(e["cue_bits_per_decision"]),
           silent_ms_min=min(silent), delay_ms=delay)


# ---------------------------------------------------------------- J: decay vs refresh

def panel_refresh(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "J", "Input renews the trace")
    e = cm.table("followup/fu4_decision_energy.csv")
    e = e[(e["model"] == LEARNERS[0][1]) & (e["visibility"] == "hidden")]
    f = cm.table("followup/fu9_trace_decay.csv")
    f = f[f["model"].isin(models()) & (f["visibility"] == "hidden")]
    ax = cv.axes(x0 + 9.0, y0 + 8.6, w - 10.6, h - 15.0)
    prefixes = sorted(e["prefix_windows"].unique())
    shades = {p: c for p, c in zip(prefixes, [SPIKE_ON_WHITE, "#C9C9C9", "#A6A6A6", "#6E6E6E", "#1A1A1A"])}   # no input: gold, as F
    for p in prefixes:
        q = e[e["prefix_windows"] == p].sort_values("delay_ms")
        ax.plot(q["delay_ms"], q["energy_adapt"], color=shades[p], lw=1.0 if p == 0 else 0.7, marker="o", ms=1.6, zorder=3 if p == 0 else 2)
        for _, r in q.iterrows():
            sd.add("J", "adaptation energy at the decision", learner=label(LEARNERS[0][1]), prefix_windows=int(p), delay_ms=int(r["delay_ms"]),
                   energy_adapt=float(r["energy_adapt"]))
    ax.set_yscale("log")
    ax.set_xlim(-20, 660)
    ax.set_xticks([0, 320, 640])
    ax.set_xlabel("Delay (ms)", labelpad=0.8)
    ax.set_ylabel("Trace at the decision", labelpad=0.8)
    fs.minor(ax, x=False, y=False)
    fs.clean(ax)
    p0 = f[f["prefix_windows"] == 0]
    tau = (p0["tau_eff_adapt_ms"].min(), p0["tau_eff_adapt_ms"].max())
    ax.text(30, float(e[e["prefix_windows"] == 0]["energy_adapt"].min()) * 1.3, f"no input: τ {tau[0]:.0f}–{tau[1]:.0f} ms", fontsize=fs.PT_SMALL,
            ha="left", va="bottom")
    items = [([dict(kind="hline", colour=shades[p], lw=1.0 if p == 0 else 0.7, w=1.8)], f"{p}" if p != prefixes[-1] else f"{p} prefix windows")
             for p in prefixes]
    cv.key(x0 + 4.0, y0 + 5.6, items, size=fs.PT_SMALL, gap=0.9)
    sd.add("J", "passive decay", tau_ms=list(map(float, tau)), r2_min=float(p0["r2_adapt"].min()), intrinsic_tau_ms=float(p0["intrinsic_tau_adapt_ms"].iloc[0]),
           rows=int(len(p0)), key="prefix windows before the decision")


# ---------------------------------------------------------------- K: the typed network (MaleCNS Fig. 5F)

K_THRESH, K_MAX_EDGES = 120.0, 34                      # cue drive (mV) at or above which a type pair is a candidate; edges drawn
K_CHIP_H, K_PITCH, K_COL_GAP, K_PAD_TOP, K_PAD = 2.8, 4.1, 2.6, 3.6, 1.0
K_COLS = {"central brain": 2, "visual projection": 1, "descending": 2, "ascending": 1, "VNC interneurons": 3, "motor neurons": 1}
K_ROWS = [["central brain", "visual projection", "descending"], ["ascending", "VNC interneurons", "motor neurons"]]
K_TARGETS = (("visual projection", 3), ("descending", 3), ("central brain", 2), ("motor neurons", 3))     # readout types per module
K_SENDERS = (("central brain", 6), ("visual projection", 1), ("VNC interneurons", 6), ("ascending", 1), ("descending", 2))


def k_module(sc):
    s = str(sc)
    if s in ("visual_projection", "visual_centrifugal"):
        return "visual projection"
    if s.startswith("cb_"):
        return "central brain"
    if "descending" in s:
        return "descending"
    if "ascending" in s:
        return "ascending"
    if s == "vnc_intrinsic":
        return "VNC interneurons"
    if s in ("vnc_motor", "vnc_efferent"):
        return "motor neurons"
    return None


def k_select(ty, e, ro):
    """The readout types with the most incoming cue drive per module, their strongest senders, then the strongest edges."""
    g = ty["superclass"].map(k_module)
    e = e.assign(g_pre=e["pre"].map(g), g_post=e["post"].map(g)).dropna(subset=["g_pre", "g_post"])
    e = e[e["abs_drive"] >= K_THRESH]
    into = e[e["post"].isin(ro)]
    targets = []
    for mod, n in K_TARGETS:
        s = into[into["g_post"] == mod].groupby("post")["abs_drive"].sum().sort_values(ascending=False)
        targets += list(s.index[:n])
    senders = []
    for mod, n in K_SENDERS:
        s = e[e["post"].isin(targets) & (e["g_pre"] == mod) & ~e["pre"].isin(targets)].groupby("pre")["abs_drive"].sum().sort_values(ascending=False)
        senders += list(s.index[:n])
    nodes = list(dict.fromkeys(targets + senders))
    shown = e[e["pre"].isin(nodes) & e["post"].isin(nodes)].sort_values("abs_drive", ascending=False).head(K_MAX_EDGES)
    used = set(shown["pre"]) | set(shown["post"])
    return [n for n in nodes if n in used], shown, g


def k_boxes(members, widths, x0, y0, net_w):
    """Module sizes from their slot grids, packed in two rows: brain above the dashed rule, VNC below."""
    size = {}
    for mod, names in members.items():
        cols = min(K_COLS[mod], max(1, len(names)))
        rows = int(np.ceil(len(names) / cols))
        colw = max(widths[n] for n in names) + K_COL_GAP
        size[mod] = (cols, rows, colw, cols * colw + 2 * K_PAD - K_COL_GAP + 1.0, K_PAD_TOP + rows * K_PITCH + K_PAD)
    boxes, y, rule = {}, y0, None
    for k_, row in enumerate(K_ROWS):
        mods = [m for m in row if m in size]
        hh = max(size[m][4] for m in mods)
        ws = [size[m][3] for m in mods]
        chan = (net_w - 1.0 - sum(ws)) / max(len(mods) - 1, 1)
        if chan < 1.6:
            raise RuntimeError(f"K: the modules of row {k_ + 1} do not fit in {net_w:.1f} mm")
        x = x0 + 0.5
        for m, w_ in zip(mods, ws):
            boxes[m] = (x, y, x + w_, y + hh)
            x += w_ + chan
        y += hh
        if k_ == 0:
            rule = y + 2.2
            y = rule + 2.2
    return boxes, size, rule, y


def k_slots(box, cols, colw):
    bx0, by0, bx1, by1 = box
    xs = [bx0 + K_PAD + 0.5 + colw / 2 - K_COL_GAP / 2 + c * colw for c in range(cols)]
    rows_fit = int(round((by1 - by0 - K_PAD_TOP - K_PAD) / K_PITCH))
    ys = [by0 + K_PAD_TOP + K_PITCH * (r + 0.5) for r in range(rows_fit)]
    return [(x, y) for x in xs for y in ys]


def k_assign(members, boxes, size, shown):
    """Chips to slots: sources left, readout right, then greedy moves and swaps that cut the drive-weighted Manhattan length
    (an edge running right to left counts three times its horizontal span)."""
    pos, slot_of, free, nb = {}, {}, {}, {}
    for (a, b), v in shown.set_index(["pre", "post"])["abs_drive"].items():
        nb.setdefault(a, []).append((b, v, True))
        nb.setdefault(b, []).append((a, v, False))
    has_in, has_out = set(shown["post"]), set(shown["pre"])
    for mod, names in members.items():
        cols, _, colw, _, _ = size[mod]
        sl = k_slots(boxes[mod], cols, colw)
        rank = sorted(names, key=lambda n: (n in has_in and n not in has_out, n in has_in))
        by_x = sorted(range(len(sl)), key=lambda s_: (sl[s_][0], sl[s_][1]))
        for n, s_ in zip(rank, by_x):
            pos[n], slot_of[n] = sl[s_], (mod, s_)
        free[mod] = sl

    def cost(n, p=None):
        x, y = p or pos[n]
        c = 0.0
        for o, v, out in nb.get(n, []):
            ox, oy = pos[o]
            dx = (ox - x) if out else (x - ox)
            c += v * (abs(dx) + abs(oy - y) + (2.0 * -dx if dx < 0 else 0.0))
        return c

    for _ in range(30):
        moved = False
        for n in list(pos):
            mod, s_ = slot_of[n]
            here = cost(n)
            occupant = {slot_of[m][1]: m for m in members[mod]}
            best, best_gain = None, 1e-6
            for s2, p2 in enumerate(free[mod]):
                if s2 == s_:
                    continue
                m = occupant.get(s2)
                if m is None:
                    gain = here - cost(n, p2)
                else:
                    before = here + cost(m)
                    pos[n], pos[m] = p2, free[mod][s_]
                    after = cost(n) + cost(m)
                    pos[n], pos[m] = free[mod][s_], p2
                    gain = before - after
                if gain > best_gain:
                    best, best_gain = (s2, m), gain
            if best:
                s2, m = best
                if m is not None:
                    pos[m], slot_of[m] = free[mod][s_], (mod, s_)
                pos[n], slot_of[n] = free[mod][s2], (mod, s2)
                moved = True
        if not moved:
            break
    return pos


def panel_network(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "K", "Input reads the trace through the cells that hold it")
    ty = cm.table("plan_v2/fig2_cue_flow_types.csv").set_index("type")
    e = cm.table("plan_v2/fig2_cue_flow_edges.csv")
    e = e[(e["pre"] != "untyped") & (e["post"] != "untyped") & (e["pre"] != e["post"])]
    ro = set(ty[ty["role"].astype(str).str.contains("readout")].index)
    nodes, shown, g = k_select(ty, e, ro)
    members = {}
    for n in nodes:
        members.setdefault(g[n], []).append(n)
    widths = {n: fs.text_width(n, fs.PT_SMALL) + 2 * ci.CHIP_PAD for n in nodes}
    ky = y0 + h - 4.4                                         # keys: two rows at the bottom
    boxes, size, rule, bottom = k_boxes(members, widths, x0, y0 + 5.6, w)
    if bottom > ky - 4.4:
        raise RuntimeError(f"K: the network ({bottom - y0:.1f} mm) runs into its keys")
    pos = k_assign(members, boxes, size, shown)
    for mod, (bx0, by0, bx1, by1) in boxes.items():
        ci.module(cv, bx0, by0, bx1 - bx0, by1 - by0, mod, size=fs.PT_SMALL)
    ci.dashed_rule(cv, x0 + 0.5, x0 + w - 0.5, rule, above=None, below=None)
    cv.text(x0 + w - 0.5, rule - 0.7, "brain", size=fs.PT_SMALL, va="bottom", ha="right")
    cv.text(x0 + w - 0.5, rule + 0.7, "VNC", size=fs.PT_SMALL, va="top", ha="right")
    labels = [(bx0 + 1.2, by0 + 1.0, bx0 + 1.6 + fs.text_width(m, fs.PT_SMALL), by0 + 3.4) for m, (bx0, by0, bx1, by1) in boxes.items()]
    net = ci.Board(cv, (x0, y0 + 5.0, x0 + w, bottom + 1.0), cell=0.5, obstacles=labels)
    for n in nodes:
        fill = HELD if ty.loc[n, "hold"] > 0 else ("#FFFFFF" if n in ro else ci.GREY_CHIP)
        net.add(n, n, *pos[n], h=K_CHIP_H, size=fs.PT_SMALL, fill=fill, edge=fs.TEXT if n in ro else "none")
        sd.add("K", "type", type=n, module=g[n], held_trace=bool(ty.loc[n, "hold"] > 0), readout=n in ro, cells=int(ty.loc[n, "cells"]))
    bins = list(np.linspace(K_THRESH, float(shown["abs_drive"].max()), 5))
    bins[-1] += 1e-6
    widths_ = tuple(v * 0.85 for v in ci.WIDTHS)
    for r in shown.itertuples():
        b = ci.bin_of(r.abs_drive, bins)
        tr = str(ty.loc[r.pre, "transmitter"])
        net.link(r.pre, r.post, widths_[b], ci.colour_for(tr, b), ci.head_for(tr))
        sd.add("K", "edge", pre=r.pre, post=r.post, cue_drive=float(r.abs_drive), synapses=float(r.synapses), width_bin=b + 1, transmitter=tr)
    failed = net.draw()
    if failed:
        raise RuntimeError(f"K: {len(failed)} edges could not be routed")
    cv.text(x0 + 0.6, ky - 3.0, "cue drive at the decision (mV, summed over connections)", size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    xe = ci.width_key(cv, x0 + 0.6, ky, [f"{bins[i]:.0f}–{bins[i + 1]:.0f}" for i in range(4)], widths=widths_, length=3.6)
    kx = max(x0 + 31.0, xe + 2.4)                             # the other keys start after the width key
    ci.transmitter_key(cv, kx, ky)
    cv.key(kx, ky + 3.0, [([dict(kind="box", size=2.0, fc=HELD)], "held the trace"),
                          ([dict(kind="box", size=2.0, fc="#FFFFFF", ec=fs.TEXT, lw=0.6)], "readout type")], size=fs.PT_SMALL, gap=1.6)
    sd.add("K", "selection", threshold=K_THRESH, edges=int(len(shown)), types=int(len(nodes)), bins=[float(b) for b in bins])
    return tuple(sorted(n for n in nodes if "_" in n))


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("fig2")
    for rel in ("plan_v2/names.csv", "plan_v2/trial_cells.csv", "plan_v2/trial_timecourse.csv", "plan_v2/fig2_delay_end_summary.csv",
                "plan_v2/fig2_raster.csv", "plan_v2/fig2_transplant.csv", "plan_v2/fig2_swaps.csv", "plan_v2/fig2_energy.csv", "plan_v2/tests.csv",
                "followup/fu4_state_dynamics.csv", "followup/fu4_state_summary.csv", "followup/fu4_decision_energy.csv", "followup/fu9_drive_split.csv",
                "followup/fu9_trace_decay.csv", "plan_v2/fig2_cue_flow_types.csv", "plan_v2/fig2_cue_flow_edges.csv"):
        sd.source(cm.RESULTS / rel)
    for t in TRIAL["snapshots_ms"]:
        sd.source(TRIAL_DIR / f"trace_t{t}.png")
        sd.source(TRIAL_DIR / f"trace_t{t}.json")
    panel_course(cv, sd, XS[0], 0.0, fs.unit(2), R1_H)
    panel_raster(cv, sd, XS[2], 0.0, fs.unit(2), R1_H)
    panel_band(cv, sd)
    learner_key(cv, sd)
    panel_decay(cv, sd, XS[0], ROW3, U, R3_H)
    panel_split(cv, sd, XS[1], ROW3, U, R3_H)
    panel_transplant(cv, sd, XS[2], ROW3, U, R3_H)
    panel_swaps(cv, sd, XS[3], ROW3, U, R3_H)
    panel_drive(cv, sd, XS[0], ROW4, U, R4_H)
    panel_cost(cv, sd, XS[1], ROW4, U, 29.0)
    panel_refresh(cv, sd, XS[1], ROW4 + 31.0, U, R4_H - 31.0)
    published = panel_network(cv, sd, XS[2], ROW4, fs.unit(2), R4_H)
    cm.finish(cv, "fig2", sd, allow_names=published)


if __name__ == "__main__":
    main()
