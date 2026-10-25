"""v4 Figure S2 (Cell), related to Figure 2 - The silent trace in every learner: written, held, read (174 mm wide).

A  dark band (the first build of Fig. 2): where the cue writes the trace, the cells that spike differently during the cue
   (scene 1), and the voltage trace held at the end of the delay in scenes 1-3, with the cells that carry an adaptation
   difference
B  dark gallery: the cue difference at the decision in five state sets (scene 1 readout and brain + readout trained, scene 2,
   scene 3 in both contexts)
C  a sparse trace, read out widely: share of cells above each |cue difference| at the end of the delay and at the decision
D  time constants of the cue difference by state variable in every learner, timing and context
E  input reads the trace at every timing: the synaptic share of the decision-time drive on the delay x prefix grid
F  the transplanted state reproduces the decision margin at every delay
G  scene 3: the state carries the context with the cue (opposite-cue and other-context swaps)
H  adaptation persists in every learner: cue-difference energy by variable through the 640-ms delay

Every number in a label is read from manuscript/results (followup, figure_data; plan_v2 via P12) or from the render
metadata. Reader-facing names only. Not shown (class c, RESULTS_INVENTORY_V4.md): the swaps of the readout shared by hidden-
and visible-goal training (they bear on T12), adaptation alone at the 160-ms delay (T10), the cue decoder on held-out
histories (near chance; A7) and the affine-dynamics fits (mixed, T14 context).

Run after P12, render/v4_fig2_tiles.py (first build) and render/v4_s2_tiles.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\s2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
from lib import figstyle as fs, palette as pal, qc  # noqa: E402

cm.use_v4()
BAND_DIR = cm.RENDERS / "v4" / "fig2"
BAND = json.loads((BAND_DIR / "tiles.json").read_text(encoding="utf-8"))
TW, TH = BAND["tile_mm"]
GAL_DIR = cm.RENDERS / "v4" / "s2"
GAL = json.loads((GAL_DIR / "tiles.json").read_text(encoding="utf-8"))
GW, GH = GAL["tile_mm"]
U = fs.unit(1)
XS = [fs.unit_x(i) for i in range(4)]
GAL_Y = TH + 4.2
ROW3 = GAL_Y + GH + 4.2
R_H = 46.0
ROW4 = ROW3 + R_H + 3.4
H = ROW4 + R_H
WHITE = fs.ON_DARK
NL_ = chr(10)
B_ = "brain and readout"
SPIKE = "#FFE7A3"
LEARNERS = [("s1_readout", "remote-dynamic-M0-head_only", "m0"), ("s1_brain", "remote-dynamic-M0-joint", "m0_joint"),
            ("s2_readout", "supp-S1-ring-brain", "s1"), ("s3_readout", "supp-S2-context-brain", "s2_o0")]
CODE_COL = {c: pal.STATE_SET[s] for _, c, s in LEARNERS}
GALLERY = {"m0": ("Scene 1", "readout trained"), "m0_joint": ("Scene 1", "brain + readout trained"), "s1": ("Scene 2", "readout trained"),
           "s2_o0": ("Scene 3, context 1", "readout trained"), "s2_o180": ("Scene 3, context 2", "readout trained")}


def names():
    return cm.table("plan_v2/names.csv")


def label(code, sep=", "):
    r = names()
    r = r[(r["code"] == code) & (r["kind"] == "model")].iloc[0]
    return r["scene"].capitalize() + sep + ("brain + readout" if r["trained"] == B_ else "readout") + " trained"


def short(code):
    r = names()
    r = r[(r["code"] == code) & (r["kind"] == "model")].iloc[0]
    return r["scene"].capitalize() + (", brain + readout" if r["trained"] == B_ else "")


def models():
    return [c for _, c, _ in LEARNERS]


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


def gradient(cv, x, y, w, h, colours):
    ax = cv.axes(x, y, w, h)
    ax.imshow(np.asarray(colours)[:, None, :3] * np.ones((1, 2, 1)), aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_axis_off()


def signed_key(cv, bx, by, half, gap, lo, hi, unit_label=None):
    u = np.linspace(1, 0, 96)
    gradient(cv, bx, by, 1.3, half, pal.SIGNED_ON_DARK(0.5 + 0.26 + 0.24 * u))
    gradient(cv, bx, by + half + gap, 1.3, half, pal.SIGNED_ON_DARK(0.5 - 0.26 - 0.24 * u[::-1]))
    if unit_label:
        tile_text(cv, bx + 1.3, by - 0.9, unit_label, va="bottom", ha="right")
    tile_text(cv, bx - 0.4, by, f"+{hi:g}", va="center", ha="right")
    tile_text(cv, bx - 0.4, by + half, f"+{lo:g}", va="center", ha="right")
    tile_text(cv, bx - 0.4, by + half + gap, f"{fs.MINUS}{lo:g}", va="center", ha="right")
    tile_text(cv, bx - 0.4, by + 2 * half + gap, f"{fs.MINUS}{hi:g}", va="center", ha="right")
    tile_text(cv, bx + 1.3, by + 2 * half + gap + 1.2, f"cue 1{NL_}{fs.MINUS} cue 0" + ("" if unit_label else f"{NL_}(mV)"), va="top", ha="right",
              linespacing=1.0)


# ---------------------------------------------------------------- A: written and held (the first build's band)

def panel_band(cv, sd):
    xs = [k * (TW + fs.GUTTER) for k in range(4)]
    keys = ["write"] + [f"trace_{k}" for k in BAND["traces"]]
    imgs = {k: np.asarray(Image.open(BAND_DIR / f"{k}.png").convert("RGB")) for k in keys}
    meta = {k: json.loads((BAND_DIR / f"{k}.json").read_text(encoding="utf-8")) for k in keys}
    summ = cm.table("plan_v2/fig2_delay_end_summary.csv").set_index("learner")
    cells = cm.table("plan_v2/fig2_delay_end_cells.csv")
    for x, k in zip(xs, keys):
        cv.tile(x, 0, imgs[k], width=TW, height=TH, group="band", label=k)
    white_letter(cv, 1.2, 0.9, "A")
    s1 = summ.loc["s1_readout"]
    lag = int(s1["last_differential_spike_ms_after_cue_offset"])
    rest = int(s1["end_of_delay_ms"] - s1["cue_offset_ms"]) - lag
    tile_text(cv, xs[0] + 4.6, 1.3, "Written: spikes that differ", va="top", size=fs.PT)
    tile_text(cv, xs[0] + 4.6, 3.9, "between the cues (scene 1)", va="top")
    trace_titles = {"s1_readout": "Held: scene 1", "s2_readout": "Held: scene 2", "s3_readout": "Held: scene 3"}
    for x, k in zip(xs[1:], BAND["traces"]):
        tile_text(cv, x + 1.4, 1.3, trace_titles[k], va="top", size=fs.PT)
        tile_text(cv, x + 1.4, 3.9, "voltage trace, end of delay", va="top")
    w = meta["write"]
    kx, ky = xs[0] + 0.9, TH - 19.0
    cv.mark(kx + 0.7, ky, s=7.0, fc=SPIKE, ec=SPIKE, lw=0.3)
    tile_text(cv, kx + 1.8, ky, f"{w['cells']:,} cells", va="center")
    for i, line in enumerate(["in the cue", f"and {lag} ms", "after it;", "none in the", f"last {rest} ms"]):
        tile_text(cv, kx, ky + 2.5 + 2.3 * i, line, va="center")
    sd.add("A", "written", cells=int(w["cells"]), cells_located=int(w["cells_located"]), last_ms_after_offset=lag, silent_rest_ms=rest)
    lo, hi = BAND["dv_mv"]
    raster = cm.table("plan_v2/fig2_raster.csv")
    written = set(raster.loc[raster["ms"] <= s1["end_of_delay_ms"], "cell"])
    for x, k in zip(xs[1:], BAND["traces"]):
        dv = cells[f"dv_{k}_mv"].abs()
        n_above = int((dv > lo).sum())
        held = set(cells.loc[cells[f"dadapt_{k}"] != 0, "cell"])
        same = k == "s1_readout" and held == written
        lines = [f"{n_above:,} cells", f"> {lo:g} mV;", f"{len(held):,} carry", "an adaptation", "difference" + (":" if same else "")]
        lines += ["the written", "cells"] if same else []
        for i, line in enumerate(lines):
            tile_text(cv, x + 1.0, TH - 18.0 + 2.3 * i, line, va="center")
        sd.add("A", "held", tile=k, learner=label(summ.loc[k, "code"]), cells_above=n_above, threshold_mv=lo, cells_adaptation_trace=len(held),
               written_cells_identical=bool(same))
    signed_key(cv, xs[1] + TW - 2.6, 34.0, 6.0, 3.0, lo, hi, "mV")
    length = 100 / meta[keys[-1]]["scale_um_per_px"] / (imgs[keys[-1]].shape[1] / TW)
    cv.line([xs[3] + TW - 1.6 - length, xs[3] + TW - 1.6], [TH - 2.0, TH - 2.0], WHITE, lw=1.0, solid_capstyle="butt")
    sd.add("A", "view", scale_bar_um=100, colour_range_mv=[lo, hi])


# ---------------------------------------------------------------- B: the cue at the decision in five state sets

def panel_gallery(cv, sd):
    lo, hi = GAL["dv_mv"]
    for k, key in enumerate(GAL["sets"]):
        x = k * (GW + fs.GUTTER)
        t = GAL["tiles"][key]
        cv.tile(x, GAL_Y, np.asarray(Image.open(GAL_DIR / f"decision_{key}.png").convert("RGB")), width=GW, height=GH, group="gallery",
                label=f"decision_{key}")
        head, sub = GALLERY[key]
        tx = x + (4.4 if k == 0 else 1.2)
        tile_text(cv, tx, GAL_Y + 1.1, head, va="top", size=fs.PT)
        tile_text(cv, tx, GAL_Y + 3.9, sub, va="top")
        tile_text(cv, x + 1.0, GAL_Y + GH - 6.2, f"{t['cells_located']:,} cells", va="center")
        tile_text(cv, x + 1.0, GAL_Y + GH - 3.9, f"> {lo:g} mV", va="center")
        sd.add("B", "state set", state_set=head + ", " + sub, cells_above=int(t["cells_located"]), positive=int(t["positive"]), negative=int(t["negative"]),
               located=int(t["located"]), threshold_mv=lo)
    white_letter(cv, 1.2, GAL_Y + 0.9, "B")
    signed_key(cv, GW - 2.4, GAL_Y + 23.0, 3.6, 2.8, lo, hi)
    cv.text(fs.FULL - 0.6, GAL_Y - 1.4, "cue difference at the decision (160-ms delay, prefix 4)", size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- C: coverage

def panel_coverage(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "C", "A sparse trace, read out widely")
    end = cm.table("plan_v2/fig2_coverage.csv")
    dec = cm.table("figure_data/fd_coverage_graded.csv")
    dec = dec[(dec["variable"] == "v") & (dec["population"] == "all")]
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 15.6)
    for key, code, skey in LEARNERS:
        e = end[(end["learner"] == key) & (end["cells_above"] > 0)].sort_values("threshold_mv")
        g = dec[(dec["state_key"] == skey) & (dec["cells_above"] > 0)].sort_values("threshold")
        ax.plot(g["threshold"], 100 * g["fraction_above"], color="#9A9A9A", lw=0.6, zorder=2)
        ax.plot(e["threshold_mv"], 100 * e["fraction_above"], color=fs.TEXT, lw=0.6, zorder=3)
        for _, r in e.iterrows():
            sd.add("C", "end of delay", learner=label(code), threshold_mv=float(r["threshold_mv"]), fraction_above=float(r["fraction_above"]))
        for _, r in g.iterrows():
            sd.add("C", "decision", learner=label(code), threshold_mv=float(r["threshold"]), fraction_above=float(r["fraction_above"]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 3)
    ax.set_ylim(5e-3, 100)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1], ["0.001", "0.01", "0.1", "1"])
    ax.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    fs.minor(ax, x=False, y=False)
    ax.set_xlabel("|Cue difference| (mV)", labelpad=0.8)
    ax.set_ylabel("Cells above (%)", labelpad=0.8)
    fs.clean(ax)
    e1 = end[end["learner"] == "s1_readout"].set_index("threshold_mv")["fraction_above"]
    g1 = dec[dec["state_key"] == "m0"].set_index("threshold")["fraction_above"]
    near = lambda s, t: float(s.iloc[np.argmin(np.abs(np.log10(s.index.to_numpy()) - np.log10(t)))])  # noqa: E731
    fe, fd = near(e1, 0.01), near(g1, 0.01)
    ax.text(0.0012, 32, f"decision{NL_}{100 * fd:.0f}%", fontsize=fs.PT_SMALL, color=fs.TEXT2, va="top", ha="left", linespacing=1.0)
    ax.text(0.0012, 100 * fe * 0.5, f"delay end{NL_}{100 * fe:.1f}%", fontsize=fs.PT_SMALL, color=fs.TEXT, va="top", ha="left",
            linespacing=1.0)
    cv.text(x0 + w - 0.8, y0 + 6.0, "four learners", size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)
    sd.add("C", "scene 1 at 0.01 mV", end_of_delay=fe, decision=fd)


# ---------------------------------------------------------------- D: time constants by variable

def panel_tau(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "D", "Adaptation's own clock")
    s = cm.table("followup/fu4_state_summary.csv")
    s = s[s["model"].isin(models())]
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 16.6)
    vars_ = [("g", f"synaptic{NL_}current"), ("v", "voltage"), ("adapt", "adaptation")]
    rng = np.random.default_rng(91)
    for j, (v, lab) in enumerate(vars_):
        ax.plot([j - 0.38, j + 0.38], [float(s[f"intrinsic_tau_{v}_ms"].iloc[0])] * 2, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
        for _, r in s.iterrows():
            fc = CODE_COL[r["model"]] if r["delay_ms"] == s["delay_ms"].min() else "white"
            ax.scatter([j + rng.uniform(-0.22, 0.22)], [r[f"tau_eff_{v}_ms"]], s=8, facecolor=fc, edgecolor=CODE_COL[r["model"]], linewidth=0.5, zorder=3)
            sd.add("D", v, learner=label(r["model"]), context=int(r["context"]), delay_ms=int(r["delay_ms"]), tau_ms=float(r[f"tau_eff_{v}_ms"]),
                   r2=float(r[f"r2_{v}"]), intrinsic_tau_ms=float(r[f"intrinsic_tau_{v}_ms"]))
    ax.set_yscale("log")
    ax.set_ylim(2, 600)
    ax.set_yticks([5, 20, 200], ["5", "20", "200"])
    ax.minorticks_off()
    ax.set_xlim(-0.6, len(vars_) - 0.4)
    ax.set_xticks(range(len(vars_)), [lab for _, lab in vars_], linespacing=1.0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Decay time constant (ms)", labelpad=0.8)
    fs.clean(ax)
    d0, d1 = int(s["delay_ms"].min()), int(s["delay_ms"].max())
    cv.key(x0 + 9.0, y0 + 6.0, [([dict(kind="mark", s=7, fc=fs.TEXT2, ec=fs.TEXT2)], f"{d0}-ms delay"),
                                ([dict(kind="mark", s=7, fc="white", ec=fs.TEXT2)], f"{d1} ms"),
                                ([dict(kind="hline", colour=fs.GRID, lw=fs.LW_THIN, w=2.0)], "intrinsic")], size=fs.PT_SMALL, gap=1.2)


# ---------------------------------------------------------------- E: synaptic drive at every timing

def panel_drive(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "E", "Input reads the trace at every timing")
    d = cm.table("followup/fu9_drive_split.csv")
    d = d[d["model"].isin(models()) & (d["visibility"] == "hidden") & (d["context"] == 0)]
    delays = sorted(d["delay_ms"].unique())
    prefixes = sorted(d["prefix_windows"].unique())
    lo, hi = 0.75, 1.0
    pw, gap = (w - 9.0 - 3 * 1.6 - 9.0) / 4, 1.6
    top, ph = y0 + 9.4, h - 17.4
    for k, code in enumerate(models()):
        q = d[d["model"] == code]
        m = q.pivot(index="prefix_windows", columns="delay_ms", values="share_drive_synaptic").reindex(index=prefixes, columns=delays).to_numpy(float)
        ax = cv.axes(x0 + 9.0 + k * (pw + gap), top, pw, ph)
        u = np.clip((m - lo) / (hi - lo), 0, 1)
        ax.imshow(pal.MAGNITUDE_ON_DARK(0.1 + 0.9 * u), aspect="auto", interpolation="nearest", extent=(-0.5, len(delays) - 0.5, len(prefixes) - 0.5, -0.5))
        trained = q[q["trained"].astype(str).str.lower().eq("true")]
        for _, r in trained.iterrows():
            ax.add_patch(Rectangle((delays.index(r["delay_ms"]) - 0.5, prefixes.index(r["prefix_windows"]) - 0.5), 1, 1, facecolor="none", edgecolor="white",
                                   linewidth=0.4, zorder=3))
        ax.set_xticks([0, len(delays) - 1], [str(delays[0]), str(delays[-1])])
        ax.set_yticks([0, len(prefixes) - 1], [str(prefixes[0]), str(prefixes[-1])] if k == 0 else ["", ""])
        ax.tick_params(length=0, labelsize=fs.PT_SMALL)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cv.text(x0 + 9.0 + k * (pw + gap) + pw / 2, top - 0.8, short(code), size=fs.PT_SMALL, ha="center", va="bottom")
        for _, r in q.iterrows():
            sd.add("E", "timing", learner=label(code), delay_ms=int(r["delay_ms"]), prefix_windows=int(r["prefix_windows"]),
                   share_drive_synaptic=float(r["share_drive_synaptic"]), trained=bool(str(r["trained"]).lower() == "true"))
    cv.text(x0 + 9.0, y0 + h - 2.6, "Delay (ms)", size=fs.PT_SMALL + 0.5, ha="left", va="center")
    cv.text(x0 + 6.6, top + ph / 2, "Prefix windows", size=fs.PT_SMALL + 0.5, ha="center", va="center", rotation=90)
    bx = x0 + w - 8.4
    gradient(cv, bx, top, 1.4, ph, pal.MAGNITUDE_ON_DARK(0.1 + 0.9 * np.linspace(1, 0, 96)))
    cv.text(bx + 2.0, top + 0.4, f"{100 * hi:.0f}%", size=fs.PT_SMALL, va="center")
    cv.text(bx + 2.0, top + ph - 0.4, f"{100 * lo:.0f}%", size=fs.PT_SMALL, va="center")
    cv.text(x0 + w - 0.8, y0 + h - 2.6, "synaptic share of the drive; framed: trained timings", size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)
    sd.add("E", "range", min_share=float(d["share_drive_synaptic"].min()), colour_range=[lo, hi])


# ---------------------------------------------------------------- F: transplant margins

def panel_transplant(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "F", "Transplanted state, same decision")
    t = cm.table("plan_v2/fig2_transplant.csv")
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 15.6)
    lim = float(np.ceil(max(t["native_margin"].abs().max(), t["full_state_margin"].abs().max())))
    ax.plot([-lim, lim], [-lim, lim], color=fs.GRID, lw=fs.LW_THIN, zorder=1)
    for p, fc in ((4, fs.TEXT), (0, "white")):
        q = t[t["prefix_windows"] == p]
        ax.scatter(q["native_margin"], q["full_state_margin"], s=9, facecolor=fc, edgecolor=fs.TEXT, linewidth=0.5, zorder=3)
        for _, r in q.iterrows():
            sd.add("F", "timing", prefix_windows=int(p), delay_ms=int(r["delay_ms"]), native_margin=float(r["native_margin"]),
                   transplant_margin=float(r["full_state_margin"]))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Margin, own state", labelpad=0.8)
    ax.set_ylabel("Margin, transplanted state", labelpad=0.8)
    fs.clean(ax)
    dm = float((t["full_state_margin"] - t["native_margin"]).abs().max())
    cv.key(x0 + 9.0, y0 + 6.0, [([dict(kind="mark", s=8, fc=fs.TEXT, ec=fs.TEXT)], "prefix 4"),
                                ([dict(kind="mark", s=8, fc="white", ec=fs.TEXT)], "prefix 0")], size=fs.PT_SMALL, gap=1.4)


# ---------------------------------------------------------------- G: scene 3 carries its context

def panel_context(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "G", "Scene 3 carries its context")
    c = cm.table("plan_v2/s2_context.csv")
    t = cm.table("plan_v2/tests.csv")
    sw = t[(t["family"] == "F13.X5.swap") & t["test"].astype(str).str.startswith("S2:")].iloc[0]
    cr = t[(t["family"] == "F13.X5.cross") & t["test"].astype(str).str.contains("cross_same_cue swap vs baseline")].iloc[0]
    cols = [("own_state_solved", f"own{NL_}state", None), ("opposite_cue_state_solved", f"opposite{NL_}cue", f"Holm {fs.P_SYM} = {fs.fmt_p(float(sw['p_adjusted']))}"),
            ("other_context_same_cue_solved", f"other{NL_}context", f"BH {fs.Q_SYM} = {fs.fmt_p(float(cr['p_adjusted']))}")]
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 16.6)
    for i, (col, lab, ptxt) in enumerate(cols):
        v = 100 * c[col].astype(str).str.lower().eq("true").mean()
        ax.add_patch(Rectangle((i - 0.32, 0), 0.64, v, facecolor=pal.STATE_SET["s2_o0"] if i == 0 else "#BDBDBD", edgecolor="none", zorder=2))
        ax.text(i, v + 2, f"{v:.0f}%", fontsize=fs.PT_SMALL, ha="center", va="bottom")
        if ptxt:
            ax.text(i, v + 12.5, ptxt.replace(" = ", f"{NL_}= "), fontsize=fs.PT_SMALL, ha="center", va="bottom", color=fs.TEXT2, linespacing=1.0)
        sd.add("G", "condition", condition=lab.replace(NL_, " "), solved_pct=v, histories=int(len(c)))
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 50, 100])
    ax.set_xticks(range(len(cols)), [lab for _, lab, _ in cols], linespacing=1.0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Trained histories solved (%)", labelpad=0.8)
    fs.clean(ax)
    cv.text(x0 + w - 0.8, y0 + 6.0, f"{len(c)} histories, state swapped in", size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- H: adaptation persists in every learner

def panel_decay(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "H", "Adaptation persists in every learner")
    s = cm.table("followup/fu4_state_summary.csv")
    d = cm.table("followup/fu4_state_dynamics.csv")
    delay = int(s["delay_ms"].max())
    curves = [("energy_adapt", fs.TEXT, 1.0), ("energy_v", "#7A7A7A", 0.7), ("energy_g", "#B5B5B5", 0.7)]
    pw, gap = (w - 10.4 - 3 * 2.4 - 1.0) / 4, 2.4
    top, ph = y0 + 11.8, h - 19.8
    for k, code in enumerate(models()):
        q = d[(d["model"] == code) & (d["context"] == 0) & (d["visibility"] == "hidden") & (d["delay_ms"] == delay) & (d["prefix_windows"] == 4)].sort_values("ms")
        cue = int(q.loc[q["segment"] == "cue", "ms"].max())
        dd = q[q["segment"] == "delay"]
        t = dd["ms"].to_numpy() - cue
        ax = cv.axes(x0 + 10.4 + k * (pw + gap), top, pw, ph)
        for col, colour, lw in curves:
            e = dd[col].to_numpy(float)
            y = np.clip(e / e[t == 20][0], 1e-12, None)
            ax.plot(t, y, color=colour, lw=lw, zorder=3)
            sd.add("H", col, learner=label(code), delay_ms=delay, ms_after_offset=t.tolist(), relative_energy=y.tolist())
        ax.set_yscale("log")
        ax.set_ylim(1e-12, 30)
        ax.set_xlim(0, delay)
        ax.set_xticks([0, delay], ["0", str(delay)])
        ticks = ax.get_xticklabels()
        ticks[0].set_horizontalalignment("left")
        ticks[-1].set_horizontalalignment("right")
        ax.set_yticks([1e-12, 1e-6, 1], ["10$^{-12}$", "10$^{-6}$", "1"] if k == 0 else ["", "", ""])
        ax.minorticks_off()
        ax.tick_params(labelsize=fs.PT_SMALL)
        fs.clean(ax)
        cv.text(x0 + 10.4 + k * (pw + gap) + pw / 2, top - 0.6, short(code), size=fs.PT_SMALL, ha="center", va="bottom")
    cv.text(x0 + 10.4 + (4 * pw + 3 * gap) / 2, y0 + h - 2.6, "ms after cue offset", size=fs.PT_SMALL + 0.5, ha="center", va="center")
    cv.text(x0 + 1.4, top + ph / 2, "Cue difference (energy)", size=fs.PT_SMALL + 0.5, ha="center", va="center", rotation=90)
    cv.key(x0 + 9.0, y0 + 5.6, [([dict(kind="hline", colour=c, lw=lw, w=2.0)], lab) for (_, c, lw), lab in
                                zip(curves, ("adaptation", "voltage", "synaptic current"))], size=fs.PT_SMALL, gap=1.2)


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("s2")
    for rel in ("plan_v2/names.csv", "plan_v2/fig2_delay_end_summary.csv", "plan_v2/fig2_delay_end_cells.csv", "plan_v2/fig2_raster.csv",
                "plan_v2/fig2_coverage.csv", "figure_data/fd_coverage_graded.csv", "figure_data/fd_soma_state.csv", "followup/fu4_state_summary.csv",
                "followup/fu4_state_dynamics.csv", "followup/fu9_drive_split.csv", "plan_v2/fig2_transplant.csv", "plan_v2/s2_context.csv", "plan_v2/tests.csv"):
        sd.source(cm.RESULTS / rel)
    for f in sorted(BAND_DIR.glob("*.png")) + sorted(BAND_DIR.glob("*.json")) + sorted(GAL_DIR.glob("*.png")) + sorted(GAL_DIR.glob("*.json")):
        sd.source(f)
    panel_band(cv, sd)
    panel_gallery(cv, sd)
    panel_coverage(cv, sd, XS[0], ROW3, U, R_H)
    panel_tau(cv, sd, XS[1], ROW3, U, R_H)
    panel_drive(cv, sd, XS[2], ROW3, fs.unit(2), R_H)
    panel_transplant(cv, sd, XS[0], ROW4, U, R_H)
    panel_context(cv, sd, XS[1], ROW4, U, R_H)
    panel_decay(cv, sd, XS[2], ROW4, fs.unit(2), R_H)
    cm.finish(cv, "s2", sd, kind="supplement")


if __name__ == "__main__":
    main()
