"""v4 Figure S7 (Cell), related to Figure 7 - Disease steals access, not memories, in detail (174 mm wide).

A  dark band: each cell's task-rate change under the uniform Kv4-loss lesion at 0.5x and 1x and under the Shal pattern of the
   same mean at 1x; the vulnerable cells (the 1% of memory-relevant cells with the largest change) by cell group
B  the healthy circuits match the fly in both brains: sparse Kenyon-cell coding broadened by APL removal, projection-neuron
   rates raised by removing the local neurons, the giant fibre's looming response
C  lesions after learning in every retained case and memory model (the circuit cases equal healthy)
D  learning under the lesion, run by run, with the design's power
E  relearning after the lesion, replicate by replicate (readout retrained)
F  lifting the lesion late removes access in every run
G  odour memory by delay in every case
H  the two Shal weightings of one mean loss
I  above 1x both weightings run away; and the cost of hyperexcitability, spikes per trial against bits of cue per decision
   (moved from Figure 7K)
J  no single superclass carries the Shal pattern's effect (descriptive)
K  the ageing operator's expression source (Davie 2018), and its effect: Kenyon-cell odor responses fall with dose
   (moved from Figure 7J)

Every number in a label is read from manuscript/results (plan_v2 via P12) or from the render metadata. Reader-facing names
only. Not shown (class c, RESULTS_INVENTORY_V4.md P15, P17, P18): the matched random lesions and the targeted-versus-random
contrast, the regime C dose trend and joint recovery, the per-dose regime B tests, the pattern-equals-uniform equivalence,
and the AFCA ageing direction; nor the Shal dose series below 0.5x (the susceptibility of E9).

Run after P12 and render/v4_s7_tiles.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\s7.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
from lib import figstyle as fs, palette as pal, qc  # noqa: E402

cm.use_v4()
TILES = cm.RENDERS / "v4" / "s7"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
TW, TH = META["tile_mm"]
U = fs.unit(1)
XS = [fs.unit_x(i) for i in range(4)]
ROW2, R_H = TH + 8.4, 46.0
ROW3 = ROW2 + R_H + 3.4
ROW4 = ROW3 + R_H + 3.4
R4_H = 62.0                                    # two panels hold a second plot each (from Fig. 7, 2026-10-23)
H = ROW4 + R4_H
WHITE = fs.ON_DARK
NL_ = chr(10)
READOUT, BRAIN = pal.STATE_SET["m0"], pal.STATE_SET["m0_joint"]                 # as Fig. 7
LEARNER_COL = {"readout": READOUT, "brain and readout": BRAIN}
LEARNER_LABEL = {"readout": "readout trained", "brain and readout": "brain + readout trained"}
DOSE_SHADE = {0.5: "#BDBDBD", 1.0: "#8C8C8C", 1.5: "#525252", 2.0: "#1A1A1A"}       # as Fig. 7
SHAL, SHAL_LIGHT = "#B4009E", "#E07BD3"                                         # as Fig. 7
MODELS = [("remote-dynamic-M0-head_only", "scene 1, readout"), ("remote-dynamic-M0-joint", "scene 1, brain + readout"),
          ("supp-M0-visibility", "scene 1, shared readout"), ("supp-S1-ring-brain", "scene 2, readout"), ("supp-S2-context-brain", "scene 3, readout")]
CASES = [("C01", [0.5, 1.0, 1.5, 2.0]), ("C04", [1.0, 2.0]), ("C13", [1.0, 2.0]), ("C16", [1.0, 2.0])]
CASE_COL = {"healthy": fs.TEXT, "C01": "#7B3294", "C04": "#E69F00", "C13": "#009E73", "C16": "#0072B2"}
SUPERCLASS_NAME = {"ascending_neuron": "ascending", "cb_intrinsic": "central brain intrinsic", "cb_sensory": "central brain sensory",
                   "descending_neuron": "descending", "ol_intrinsic": "optic lobe intrinsic", "ol_sensory": "optic lobe sensory",
                   "visual_projection": "visual projection", "vnc_intrinsic": "VNC intrinsic", "vnc_sensory": "VNC sensory"}
SHARE = LinearSegmentedColormap.from_list("share", ["#FFFFFF", "#BDBDBD", "#1A1A1A"])
BRAIN_MARK = {"the model of the disease study": dict(marker="o", fc=fs.TEXT), "scene 1, brain and readout trained": dict(marker="o", fc="white")}


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


def case_names():
    c = cm.table("plan_v2/fig7_cases.csv").set_index("code")["case"]
    return {k: str(v).split(" (")[0] for k, v in c.items()}


# ---------------------------------------------------------------- A: the disease painted on the brain, at both doses

def panel_band(cv, sd):
    pc = cm.table("plan_v2/s7_cells.csv")
    vu = cm.table("plan_v2/s7_vulnerable.csv")
    healthy = float(pc["rate_healthy_hz"].mean())
    tiles = [("uniform_0p5", "rate_uniform_dose0p5_hz", "Uniform Kv4 loss · 0.5×"), ("uniform_1", "rate_uniform_dose1_hz", "Uniform Kv4 loss · 1×"),
             ("shal_1", "rate_shal_dose1_hz", "Shal pattern · 1×")]
    for k, (key, col, name) in enumerate(tiles):
        x = k * (TW + fs.GUTTER)
        cv.tile(x, 0, np.asarray(Image.open(TILES / f"{key}.png").convert("RGB")), width=TW, height=TH, group="band", label=key)
        d = (pc[col] - pc["rate_healthy_hz"]).to_numpy(float)
        up, down = int((d > 0).sum()), int((d < 0).sum())
        tx = x + (4.4 if k == 0 else 1.2)
        tile_text(cv, tx, 1.0, name, size=fs.PT, va="top")
        tile_text(cv, tx, 3.9, f"{up:,} cells faster · {down:,} slower", va="top")
        tile_text(cv, tx, 6.5, f"mean rate {pc[col].mean():.2f} Hz (healthy {healthy:.2f})", va="top")
        sd.add("A", key, faster=up, slower=down, mean_rate_hz=float(pc[col].mean()), healthy_mean_rate_hz=healthy)
    x = 3 * (TW + fs.GUTTER)
    cv.tile(x, 0, np.asarray(Image.open(TILES / "vulnerable.png").convert("RGB")), width=TW, height=TH, group="band", label="vulnerable")
    tile_text(cv, x + 1.2, 1.0, "The vulnerable cells", size=fs.PT, va="top")
    tile_text(cv, x + 1.2, 3.9, f"{len(vu)}: the top 1% of memory cells", va="top")
    tx = vu["transmitter"].value_counts()
    mono = [(t, int(tx.get(t, 0))) for t in ("octopamine", "dopamine", "serotonin") if tx.get(t, 0)]
    tile_text(cv, x + 1.2, 6.5, ", ".join(f"{n} {t}" for t, n in mono), va="top")
    white_letter(cv, 1.2, 0.9, "A")
    sd.add("A", "vulnerable", cells=int(len(vu)), transmitters=tx.to_dict())
    # keys: the rate change (A1-A3) and the cell groups of the vulnerable cells (A4)
    ky = TH + 3.0
    u = np.linspace(0, 1, 64)
    bar = np.vstack([pal.SIGNED_ON_DARK(0.5 - 0.22 - 0.28 * u[::-1])[:, :3], np.zeros((10, 3)), pal.SIGNED_ON_DARK(0.5 + 0.22 + 0.28 * u)[:, :3]])
    ax = cv.axes(1.0, ky - 0.7, 14.0, 1.4)
    ax.imshow(bar[None], aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_axis_off()
    cv.text(16.0, ky, f"task-rate change (Hz, log): {fs.MINUS}{META['rate_max_hz']:g} to +{META['rate_max_hz']:g}", size=fs.PT_SMALL, va="center",
            color=fs.TEXT2)
    grp = vu["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other").value_counts()
    items = [([dict(kind="mark", s=7, fc=pal.on_white(pal.ATLAS_ON_DARK[g]), ec=pal.on_white(pal.ATLAS_ON_DARK[g]))], f"{pal.SHORT_GROUP[g]} {n}")
             for g, n in grp.items()]
    cv.key(x + 0.6, ky, items[:3], size=fs.PT_SMALL, gap=1.2)
    cv.key(x + 0.6, ky + 2.5, items[3:], size=fs.PT_SMALL, gap=1.2)
    sd.add("A", "vulnerable groups", **{g: int(n) for g, n in grp.items()})


# ---------------------------------------------------------------- B: healthy circuits match the fly, both brains

def panel_signatures(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "B", "Healthy circuits match the fly in both brains")
    s = cm.table("plan_v2/s7_signatures.csv")
    brains = list(BRAIN_MARK)
    cv.key(x0 + 4.0, y0 + 6.0, [([dict(kind="mark", s=8, marker="o", fc=BRAIN_MARK[b]["fc"], ec=fs.TEXT)], b) for b in brains], size=fs.PT_SMALL, gap=2.4)
    specs = [("KC fraction responding", ("intact", "APL removed"), ["intact", "no APL"], "KCs responding", (0, 0.7), [0, 0.3, 0.6], 15.5),
             ("KC pair overlap", ("intact", "APL removed"), ["intact", "no APL"], "KC overlap", (0, 1.0), [0, 0.5, 1.0], 15.0),
             ("PN mean rate (Hz)", ("intact", "local neurons removed"), ["intact", "no LNs"], "PN rate (Hz)", (0, 36), [0, 15, 30], 15.5),
             ("giant-fibre spikes", ("blank", "loom"), ["blank", "loom"], "Giant fiber", (0, 95), [0, 40, 80], 12.0)]
    x = x0 + 9.0
    top, ah = y0 + 13.4, h - 21.6
    rng = np.random.default_rng(707)
    for measure, conds, ticks, lab, lim, yt, aw in specs:
        ax = cv.axes(x, top, aw, ah)
        for i, c in enumerate(conds):
            for j, b in enumerate(brains):
                v = s[(s["brain"] == b) & (s["condition"] == c) & (s["measure"] == measure)]["value"].to_numpy(float)
                xx = i + (j - 0.5) * 0.36 + (rng.uniform(-0.06, 0.06, len(v)) if len(v) > 1 else 0.0)
                ax.scatter(xx, v, s=6, marker="o", facecolor=BRAIN_MARK[b]["fc"], edgecolor=fs.TEXT, linewidth=0.4, zorder=3)
                if len(v) > 1:
                    ax.plot([i + (j - 0.5) * 0.36 - 0.13, i + (j - 0.5) * 0.36 + 0.13], [v.mean()] * 2, color=fs.TEXT, lw=0.8, zorder=4)
                sd.add("B", measure, brain=b, condition=c, values=v.tolist())
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ax.set_ylim(*lim)
        ax.set_yticks(yt, [f"{t:g}" for t in yt])
        ax.set_xticks(range(len(conds)), ticks, linespacing=1.0)
        ax.tick_params(axis="x", length=0, labelsize=fs.PT_SMALL, pad=1.0)
        fs.clean(ax)
        cv.text(x + aw / 2, top - 0.8, lab, size=fs.PT_SMALL, ha="center", va="bottom")
        x += aw + 6.0


# ---------------------------------------------------------------- C: lesions after learning, every case and model

def panel_regime_a(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "C", "Lesions after learning: every case and memory model")
    a = cm.table("plan_v2/s7_regime_a.csv")
    names = case_names()
    blocks = [("healthy", [0.0])] + CASES
    lab_w, cw, ch, gap = 25.0, 4.5, 4.2, 1.4
    mx, my = x0 + lab_w, y0 + 14.4
    starts, x = [], 0.0
    for _, ds in blocks:
        starts.append(x)
        x += len(ds) * cw + gap
    total = x - gap
    ax = cv.axes(mx, my, total, ch * len(MODELS))
    for i, (code, label) in enumerate(MODELS):
        q = a[a["model"] == code]
        ref = float(q[q["case"] == "healthy"]["solved"].iloc[0])
        for (case, ds), x_ in zip(blocks, starts):
            for k, d in enumerate(ds):
                v = float(q[(q["case"] == case) & np.isclose(q["dose"], d)]["solved"].iloc[0])
                ax.add_patch(Rectangle((x_ + k * cw, i * ch), cw, ch, facecolor=SHARE(min(v / ref, 1.0)), edgecolor="white", linewidth=0.6))
                sd.add("C", "cell", model=label, case=names.get(case, case), dose=d, solved=v, healthy=ref)
        cv.text(mx - 0.8, my + (i + 0.5) * ch, label, size=fs.PT_SMALL, ha="right", va="center")
    ax.set_xlim(0, total)
    ax.set_ylim(ch * len(MODELS), 0)
    ax.set_axis_off()
    for (case, ds), x_ in zip(blocks, starts):
        if case == "healthy":
            cv.text(mx + x_ + cw / 2, my + ch * len(MODELS) + 1.6, "healthy", size=fs.PT_SMALL, ha="center", va="center")
            continue
        cv.text(mx + x_ + len(ds) * cw / 2, my - 1.4, names[case], size=fs.PT_SMALL, ha="center", va="center")
        for k, d in enumerate(ds):
            cv.text(mx + x_ + (k + 0.5) * cw, my + ch * len(MODELS) + 1.6, f"{d:g}", size=fs.PT_SMALL, ha="center", va="center")
    cv.text(mx + starts[1] + 2 * cw, my + ch * len(MODELS) + 4.4, "Dose", size=fs.PT_SMALL, ha="center", va="center")
    t = cm.table("plan_v2/tests.csv")
    eq = t[t["family"] == "F12.regimeA.equivalence"]
    cx = mx + (starts[2] + total) / 2
    cv.line([mx + starts[2] + 0.3, mx + total - 0.3], [my + ch * len(MODELS) + 3.1, my + ch * len(MODELS) + 3.1], fs.TEXT2, lw=0.4)
    cv.text(cx, my + ch * len(MODELS) + 3.7, f"TOST, Holm {fs.P_SYM} ≤{NL_}{fs.fmt_p(float(eq['p_adjusted'].max()))}", size=fs.PT_SMALL,
            ha="center", va="top", linespacing=1.0, color=fs.TEXT2)
    end = cv.text(x0 + 4.0, y0 + 6.0, "share of healthy histories solved", size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    kx = x0 + 4.0 + fs.text_width("share of healthy histories solved", fs.PT_SMALL) + 1.4
    cv.text(kx, y0 + 6.0, "0", size=fs.PT_SMALL, va="center")
    ax2 = cv.axes(kx + 1.6, y0 + 5.3, 10.0, 1.4)
    ax2.imshow(SHARE(np.linspace(0, 1, 64))[None, :, :3], aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    for sp in ax2.spines.values():
        sp.set_linewidth(0.3)
        sp.set_color("#D9D9D9")
    ax2.set_xticks([])
    ax2.set_yticks([])
    cv.text(kx + 12.2, y0 + 6.0, "1", size=fs.PT_SMALL, va="center")
    sd.add("C", "equivalence", tests=int(len(eq)), holm_max=float(eq["p_adjusted"].max()))


# ---------------------------------------------------------------- D: learning under the lesion, every run

def panel_regime_b(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "D", "Learning under the lesion")
    rb = cm.table("plan_v2/s7_regime_b.csv")
    pw = cm.table("plan_v2/s7_power.csv")
    n6 = int(rb[(rb["case"] == "C01")].groupby(["dose", "trained_parameters"]).size().min())
    p6 = pw[pw["n"] == n6].iloc[0]
    drop = 100 * float(p6["target_drop_histories"]) / 18
    ax = cv.axes(x0 + 9.0, y0 + 10.4, w - 10.6, h - 18.4)
    doses = [0.0, 0.5, 1.0, 1.5, 2.0]
    rng = np.random.default_rng(717)
    for j, (tp, off) in enumerate((("readout", -0.17), ("brain and readout", 0.17))):
        for i, d in enumerate(doses):
            q = rb[(rb["trained_parameters"] == tp) & np.isclose(rb["dose"], d) & rb["case"].isin(["C01", "healthy"])]
            v = q["trained_solved"].to_numpy(float)
            ax.scatter(i + off + rng.uniform(-0.07, 0.07, len(v)), v, s=5, facecolor=LEARNER_COL[tp], edgecolor="none", zorder=3)
            ax.plot([i + off - 0.12, i + off + 0.12], [np.median(v)] * 2, color=fs.TEXT, lw=0.8, zorder=4)
            sd.add("D", tp, dose=d, runs=int(len(v)), values=v.tolist())
    ax.set_xlim(-0.6, len(doses) - 0.4)
    ax.set_ylim(0, 19.5)
    ax.set_yticks([0, 9, 18])
    ax.set_xticks(range(len(doses)), ["healthy"] + [f"{d:g}" for d in doses[1:]])
    ax.tick_params(axis="x", length=0, labelsize=fs.PT_SMALL)
    ax.set_xlabel("Dose", labelpad=0.8)
    ax.set_ylabel("Trained histories solved", labelpad=0.8)
    fs.clean(ax)
    for i, tp in enumerate(LEARNER_COL):
        cv.key(x0 + 10.6, y0 + h - 15.4 + 2.4 * i, [([dict(kind="mark", s=6, fc=LEARNER_COL[tp], ec=LEARNER_COL[tp])], LEARNER_LABEL[tp])], size=fs.PT_SMALL)
    sd.add("D", "power", replicates=n6, power=float(p6["power"]), drop_pp=drop)


# ---------------------------------------------------------------- E: relearning after the lesion

def panel_regime_c(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "E", "Relearning after the lesion")
    r = cm.table("plan_v2/fig7_relearning.csv")
    r = r[r["trained_parameters"] == "readout"]
    ax = cv.axes(x0 + 9.0, y0 + 10.4, w - 10.6, h - 18.4)
    for q in r.itertuples():
        ax.plot([0, 1, 2], [q.solved_0, q.solved_50, q.solved_100], color=DOSE_SHADE[float(q.dose)], lw=0.7, marker="o", ms=1.8, zorder=3)
        sd.add("E", "replicate", dose=float(q.dose), replicate=int(q.replicate), solved=[int(q.solved_0), int(q.solved_50), int(q.solved_100)])
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0, 19.5)
    ax.set_yticks([0, 9, 18])
    ax.set_xticks([0, 1, 2], ["0", "50", "100"])
    ax.set_xlabel("Retraining (%)", labelpad=0.8)
    ax.set_ylabel("Trained histories solved", labelpad=0.8)
    fs.clean(ax)
    end = cv.text(x0 + 10.0, y0 + 12.0, "dose", size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    cv.key(x0 + 10.0 + fs.text_width("dose", fs.PT_SMALL) + 1.0, y0 + 12.0, [([dict(kind="hline", w=2.0, colour=c, lw=0.8)], f"{d:g}") for d, c in DOSE_SHADE.items()],
           size=fs.PT_SMALL, gap=0.9)


# ---------------------------------------------------------------- F: lifting the lesion late

def panel_restoration(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "F", "Lifting the lesion late")
    r = cm.table("plan_v2/fig7_restoration.csv")
    lost = int((r["trained_solved_restored"] < r["trained_solved_as_trained"]).sum())
    ax = cv.axes(x0 + 9.0, y0 + 10.4, w - 10.6, h - 18.4)
    rng = np.random.default_rng(727)
    for j, tp in enumerate(("readout", "brain and readout")):
        q = r[r["trained_parameters"] == tp]
        for row in q.itertuples():
            jit = rng.uniform(-0.05, 0.05)
            xs = [j * 2.4 + jit, j * 2.4 + 1 + jit]
            ys = [row.trained_solved_as_trained, row.trained_solved_restored]
            ax.plot(xs, ys, color=DOSE_SHADE[float(row.dose)], lw=0.6, zorder=2)
            ax.scatter(xs, ys, s=5, facecolor=LEARNER_COL[tp], edgecolor="none", zorder=3)
            sd.add("F", tp, dose=float(row.dose), replicate=int(row.replicate), as_trained=float(ys[0]), restored=float(ys[1]))
    ax.set_xlim(-0.5, 3.9)
    ax.set_ylim(0, 19.5)
    ax.set_yticks([0, 9, 18])
    ax.set_xticks([0, 1, 2.4, 3.4], ["in", "lifted", "in", "lifted"])
    ax.tick_params(axis="x", length=0, labelsize=fs.PT_SMALL)
    ax.set_ylabel("Trained histories solved", labelpad=0.8)
    fs.clean(ax)
    for j, tp in enumerate(("readout", "brain + readout")):
        cv.text(x0 + 9.0 + (w - 10.6) * (j * 2.4 + 0.5 + 0.5) / 4.4, y0 + h - 4.2, tp, size=fs.PT_SMALL, ha="center", va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- G: odour memory by delay

def panel_olfactory(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "G", "Odor memory by delay")
    o = cm.table("plan_v2/s7_olfactory.csv")
    o = o[o["readout"] == "mbon"]
    names = case_names()
    delays = sorted(int(d) for d in o["delay_ms"].unique())
    cv.text(x0 + 4.0, y0 + 6.0, "mushroom-body output readout, dose 1", size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    ax = cv.axes(x0 + 9.0, y0 + 10.4, w - 10.6, h - 18.4)
    for case in ("healthy", "C01", "C04", "C13", "C16"):
        q = o[(o["case"] == case) & (np.isclose(o["dose"], 0.0) if case == "healthy" else np.isclose(o["dose"], 1.0))]
        m = q.groupby("delay_ms")["accuracy"].mean()
        ax.plot(range(len(delays)), [m.loc[d] for d in delays], color=CASE_COL[case], lw=1.0 if case == "healthy" else 0.8, marker="o", ms=1.8,
                zorder=4 if case == "healthy" else 3)
        sd.add("G", names.get(case, case), delays=delays, accuracy=[float(m.loc[d]) for d in delays])
    ax.axhline(0.5, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
    ax.set_xlim(-0.4, len(delays) - 0.6)
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([0.5, 0.75, 1.0], ["0.5", "0.75", "1"])
    ax.set_xticks(range(len(delays)), [str(d) for d in delays])
    ax.tick_params(axis="x", labelsize=fs.PT_SMALL)
    ax.set_xlabel("Delay (ms)", labelpad=0.8)
    ax.set_ylabel("Odor-pair accuracy", labelpad=0.8)
    fs.clean(ax)
    ky = y0 + h - 8.0 - 11.0
    for i, case in enumerate(("healthy", "C01", "C04", "C13", "C16")):
        cv.key(x0 + 10.0, ky + 2.3 * i, [([dict(kind="hline", w=2.0, colour=CASE_COL[case], lw=0.9)], names.get(case, case))], size=fs.PT_SMALL)


# ---------------------------------------------------------------- H: two Shal weightings

def panel_weights(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "H", "Two Shal weightings")
    pc = cm.table("plan_v2/s7_cells.csv")
    ax = cv.axes(x0 + 11.6, y0 + 10.4, w - 13.2, h - 18.4)
    bins = np.linspace(0, 2.2, 23)
    for col, colour, filled in (("weight_global_mean", SHAL_LIGHT, True), ("weight_class_means", SHAL, False)):
        v = pc[col].to_numpy(float)
        cnt, edges = np.histogram(v, bins)
        if filled:
            ax.stairs(cnt, edges, fill=True, color=colour, alpha=0.55, lw=0, zorder=2)
        else:
            ax.stairs(cnt, edges, color=colour, lw=0.9, zorder=3)
        sd.add("H", col, mean=float(v.mean()), minimum=float(v.min()), maximum=float(v.max()), counts=cnt.tolist(), edges=edges.tolist())
    ax.set_yscale("log")
    ax.set_ylim(0.8, 3e5)
    ax.set_yticks([1, 100, 10000], ["1", "100", "10,000"])
    ax.minorticks_off()
    ax.set_xlim(0, 2.2)
    ax.set_xticks([0, 1, 2])
    ax.axvline(1.0, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
    ax.set_xlabel("Weight (mean 1)", labelpad=0.8)
    ax.set_ylabel("Cells", labelpad=0.8)
    fs.clean(ax)
    cv.key(x0 + 4.0, y0 + 6.0, [([dict(kind="hline", w=2.4, colour=SHAL, lw=0.9)], "class means"), ([dict(kind="box", size=1.8, fc=SHAL_LIGHT)], "global mean")],
           size=fs.PT_SMALL, gap=1.6)


# ---------------------------------------------------------------- I: both weightings run away above 1x

def panel_runaway(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "I", "Runaway above 1×, and its cost")
    r = cm.table("plan_v2/fig7_shal_runaway.csv")
    ax = cv.axes(x0 + 9.0, y0 + 10.4, w - 10.6, 19.0)
    for pat, colour in (("Shal-patterned (type, class means)", SHAL), ("Shal-patterned (type, global mean)", SHAL_LIGHT)):
        q = r[r["pattern"] == pat].sort_values("dose")
        ax.plot(q["dose"], 100 * q["share_above_100hz"], color=colour, lw=0.9, marker="o", ms=2.0, zorder=3)
        sd.add("I", pat, doses=q["dose"].tolist(), share_above_100hz=q["share_above_100hz"].tolist(), task_mean_hz=q["task_mean_hz"].tolist())
    g = r[(r["pattern"] == "Shal-patterned (type, global mean)") & (r["dose"] >= 1.5)]["share_above_100hz"]
    ax.set_xlim(0.35, 2.15)
    ax.set_ylim(-2, 52)
    ax.set_xticks([0.5, 1, 1.5, 2], ["0.5", "1", "1.5", "2"])
    ax.set_yticks([0, 25, 50])
    ax.set_xlabel("Dose", labelpad=0.8)
    ax.set_ylabel(f"Cells above{NL_}100 Hz (%)", labelpad=0.8, linespacing=1.0)
    fs.clean(ax)
    energy(cv, sd, x0, y0 + 37.2, w, h - 37.2)


def energy(cv, sd, x0, y0, w, h):
    """The cost of hyperexcitability under the uniform Kv4-type loss (moved from Figure 7K)."""
    e = cm.table("plan_v2/fig7_energy.csv")
    healthy = e[e["code"] == "healthy"].iloc[0]
    c1 = e[e["code"] == "C01"].sort_values("dose")
    lo, hi = c1["spikes_per_trial"].min() / healthy["spikes_per_trial"], c1["spikes_per_trial"].max() / healthy["spikes_per_trial"]
    ax = cv.axes(x0 + 9.0, y0 + 3.4, w - 10.6, h - 11.2)
    ax.scatter([healthy["spikes_per_trial"]], [healthy["cue_bits_per_decision"]], s=12, facecolor="white", edgecolor=fs.TEXT, linewidth=0.7, zorder=4)
    ax.annotate("healthy", (healthy["spikes_per_trial"], healthy["cue_bits_per_decision"]), xytext=(3, -1), textcoords="offset points",
                fontsize=fs.PT_SMALL, ha="left", va="top")
    ax.plot(c1["spikes_per_trial"], c1["cue_bits_per_decision"], color="#7B3294", lw=0.6, zorder=2)
    for _, r in c1.iterrows():
        ax.scatter([r["spikes_per_trial"]], [r["cue_bits_per_decision"]], s=8, facecolor=DOSE_SHADE[float(r["dose"])], edgecolor="#7B3294",
                   linewidth=0.5, zorder=3)
        sd.add("I", "energy", case=r["case"], dose=float(r["dose"]), spikes_per_trial=float(r["spikes_per_trial"]),
               bits_per_decision=float(r["cue_bits_per_decision"]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e4, 4e6)
    ax.set_ylim(0.005, 2)
    ax.set_xticks([1e4, 1e5, 1e6], ["10$^4$", "10$^5$", "10$^6$"])
    ax.set_yticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
    fs.minor(ax, x=False, y=False)
    ax.set_xlabel("Spikes per trial", labelpad=0.8)
    ax.set_ylabel(f"Bits of cue{NL_}per decision", labelpad=0.8, linespacing=1.0)
    fs.clean(ax)
    others = e[~e["code"].isin(["healthy", "C01"])]
    same = bool(np.allclose(others["spikes_per_trial"], healthy["spikes_per_trial"]) and np.allclose(others["cue_bits_per_decision"], healthy["cue_bits_per_decision"]))
    sd.add("I", "energy summary", spikes_ratio_range=[float(lo), float(hi)], bits_range=[float(c1["cue_bits_per_decision"].min()),
           float(c1["cue_bits_per_decision"].max())], circuit_cases_equal_healthy=same, healthy_bits=float(healthy["cue_bits_per_decision"]))


# ---------------------------------------------------------------- J: leave one superclass out

def panel_loo(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "J", "No single class carries it")
    lo = cm.table("plan_v2/s7_loo.csv")
    whole = lo[lo["uniform_superclass"] == "none"].iloc[0]
    lo = lo[lo["uniform_superclass"] != "none"]
    lo = lo.assign(name=lo["uniform_superclass"].map(SUPERCLASS_NAME)).sort_values("name").reset_index(drop=True)
    lab_w = 23.0
    ax = cv.axes(x0 + lab_w, y0 + 10.4, w - lab_w - 1.4, h - 18.4)
    for col, colour, off in (("trained_solved_remote-dynamic-M0-head_only", READOUT, -0.15), ("trained_solved_remote-dynamic-M0-joint", BRAIN, 0.15)):
        ax.scatter(lo[col], np.arange(len(lo)) + off, s=7, facecolor=colour, edgecolor="none", zorder=3)
        ax.plot([float(whole[col])] * 2, [-0.6, len(lo) - 0.4], color=colour, lw=0.6, ls=(0, (2.0, 1.4)), zorder=1)
        sd.add("J", col, superclasses=lo["name"].tolist(), solved=lo[col].tolist(), whole_pattern=float(whole[col]))
    ax.set_ylim(len(lo) - 0.5, -1.9)
    ax.set_yticks(range(len(lo)), lo["name"].tolist())
    ax.set_xlim(0, 18.5)
    ax.set_xticks([0, 9, 18])
    ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL, pad=1.0)
    ax.set_xlabel("Trained histories solved", labelpad=0.8)
    fs.clean(ax)
    ax.text(float(whole["trained_solved_remote-dynamic-M0-head_only"]), -1.2, "whole pattern", fontsize=fs.PT_SMALL, ha="center", va="center", color=fs.TEXT2)
    cv.text(x0 + 4.0, y0 + 6.0, "each class made uniform, 1×", size=fs.PT_SMALL, va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- K: the ageing operator's source

def panel_ageing(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "K", "Aging: source and effect")
    ex = cm.table("plan_v2/s7_ageing_expression.csv")
    ex = ex[ex["neuronal"].astype(str).str.lower().eq("true")].sort_values("log2f").reset_index(drop=True)
    pan = float(ex.loc[ex["unit"] == "pan-neuronal", "log2f"].iloc[0])
    units = ex[ex["unit"] != "pan-neuronal"].reset_index(drop=True)
    kc_col = pal.on_white(pal.ATLAS_ON_DARK["central brain intrinsic"])
    cv.text(x0 + 4.0, y0 + 6.0, f"K$^+$ channels, {len(units)} neuronal units", size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    ax = cv.axes(x0 + 9.0, y0 + 10.4, w - 10.6, 19.0)
    for i, r in units.iterrows():
        kc = r["unit"] == "G-KC"
        ax.plot([0, r["log2f"]], [i, i], color=kc_col if kc else "#BDBDBD", lw=0.8 if kc else 0.6, zorder=2)
        ax.scatter([r["log2f"]], [i], s=9 if kc else 4, facecolor=kc_col if kc else fs.TEXT2, edgecolor="none", zorder=3)
        if kc:
            ax.text(-0.04, i, f"γ Kenyon cells {fs.num(float(r['log2f']), 2, sign=True)}", fontsize=fs.PT_SMALL, ha="right", va="center")
        sd.add("K", "unit", unit=str(r["unit"]), log2f=float(r["log2f"]), kenyon_cells=bool(kc))
    ax.axvline(pan, color=fs.TEXT, lw=0.6, ls=(0, (2.0, 1.4)), zorder=1)
    ax.text(pan + 0.04, -0.9, f"pan-neuronal{NL_}{abs(pan) if round(pan, 2) == 0 else pan:.2f}", fontsize=fs.PT_SMALL, ha="left", va="bottom", linespacing=1.0)
    ax.set_ylim(-1.5, len(units) - 0.2)
    ax.set_yticks([])
    lim = max(0.6, float(np.abs(units["log2f"]).max()) + 0.05)
    ax.set_xlim(-lim, lim)
    ax.set_xticks([-0.5, 0, 0.5], [f"{fs.MINUS}0.5", "0", "0.5"])
    ax.set_xlabel("log$_2$ old / young", labelpad=0.8)
    fs.clean(ax, left=False)
    sd.add("K", "pan-neuronal", log2f=pan)
    ageing_effect(cv, sd, x0, y0 + 37.2, w, h - 37.2)


def ageing_effect(cv, sd, x0, y0, w, h):
    """The ageing operator lowers Kenyon-cell odor responses with the network stable (moved from Figure 7J)."""
    a = cm.table("plan_v2/fig7_aging.csv")
    stable = float(a["fraction_over_100hz"].max()) == 0.0
    ax = cv.axes(x0 + 11.0, y0 + 3.4, w - 12.6, h - 11.2)
    for brain, col in zip(list(dict.fromkeys(a["brain"])), (READOUT, BRAIN)):
        q = a[a["brain"] == brain].sort_values("dose")
        ax.plot(q["dose"], q["kc_ratio"], color=col, lw=0.8, marker="o", ms=2.0, zorder=3)
        for _, r in q.iterrows():
            sd.add("K", "operator", brain=brain, dose=float(r["dose"]), kc_ratio=float(r["kc_ratio"]), task_mean_hz=float(r["task_mean_hz"]),
                   fraction_over_100hz=float(r["fraction_over_100hz"]))
    ax.axhline(1.0, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
    ax.set_xlim(0.3, 1.7)
    ax.set_xticks([0.5, 1.0, 1.5], ["0.5", "1", "1.5"])
    ax.set_ylim(0.6, 1.16)
    ax.set_yticks([0.6, 0.8, 1.0])
    ax.set_xlabel("Operator dose", labelpad=0.8)
    ax.set_ylabel(f"Kenyon-cell odor{NL_}responses (× healthy)", labelpad=0.8, linespacing=1.0)
    fs.clean(ax)
    for k, (brain, col) in enumerate(zip(list(dict.fromkeys(a["brain"])), (READOUT, BRAIN))):
        lab = "disease-study brain" if "disease" in str(brain) else "scene-1 brain"
        cv.key(x0 + 17.0, y0 + 5.4 + 2.4 * k, [([dict(kind="hline", colour=col, lw=0.8, w=1.8)], lab)], size=fs.PT_SMALL)


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("s7")
    for rel in ("plan_v2/s7_cells.csv", "plan_v2/s7_vulnerable.csv", "plan_v2/s7_signatures.csv", "plan_v2/s7_regime_a.csv", "plan_v2/tests.csv",
                "plan_v2/fig7_cases.csv", "plan_v2/s7_regime_b.csv", "plan_v2/s7_power.csv", "plan_v2/fig7_relearning.csv", "plan_v2/fig7_restoration.csv",
                "plan_v2/s7_olfactory.csv", "plan_v2/fig7_shal_runaway.csv", "plan_v2/s7_loo.csv", "plan_v2/s7_ageing_expression.csv",
                "plan_v2/fig7_energy.csv", "plan_v2/fig7_aging.csv"):
        sd.source(cm.RESULTS / rel)
    for f in sorted(TILES.glob("*.png")) + sorted(TILES.glob("*.json")):
        sd.source(f)
    panel_band(cv, sd)
    panel_signatures(cv, sd, XS[0], ROW2, fs.unit(2), R_H)
    panel_regime_a(cv, sd, XS[2], ROW2, fs.unit(2), R_H)
    panel_regime_b(cv, sd, XS[0], ROW3, U, R_H)
    panel_regime_c(cv, sd, XS[1], ROW3, U, R_H)
    panel_restoration(cv, sd, XS[2], ROW3, U, R_H)
    panel_olfactory(cv, sd, XS[3], ROW3, U, R_H)
    panel_weights(cv, sd, XS[0], ROW4, U, R4_H)
    panel_runaway(cv, sd, XS[1], ROW4, U, R4_H)
    panel_loo(cv, sd, XS[2], ROW4, U, R4_H)
    panel_ageing(cv, sd, XS[3], ROW4, U, R4_H)
    cm.finish(cv, "s7", sd, kind="supplement")


if __name__ == "__main__":
    main()
