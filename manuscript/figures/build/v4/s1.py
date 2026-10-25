"""v4 Figure S1 (Cell), related to Figure 1 - The model, what training changes, and its validation (174 mm wide).

A  dark band, where training acts: each cell glowing by the mean |change| of its parameter group over the 24 planning maps
   (output gain, input gain, threshold; magma, log), and the cells of the 1% most consistently changed groups
B  the brain changes in proportion to training length (24 maps; Spearman with a BCa interval, permutation p, Holm)
C  mean |change| by cell group and parameter
D  every planning map's learning curve: depth reached vs updates, by difficulty tier
E  a ladder in every replicate set: updates per map (rows by optimal moves) in the original set and five replicates
F  stages entered with a deadlocking alternative cost more updates
G  wiring matters at the hardest map: the analysed lineage in six sets, and from a neutral start the real wiring, shuffled
   weights on the real wiring and a degree-preserving rewiring (one run each)
H  the model at a glance
I  bit-exact on every GPU: every analysed model replayed on the 11 GPUs of the three training hosts

Every number in a label is read from manuscript/results (main, extended, followup, figure_data, stats; plan_v2 via P12) or
from the render metadata. Reader-facing names only. Not shown (class c, RESULTS_INVENTORY_V4.md): zero-update transfer to
held-out maps (none solved), the X8 readout-only vs joint comparison (L4), the learner-success counts (L9) and the early
benchmark run outside the recompute tolerance.

Run after P12 and render/v4_s1_tiles.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\s1.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
from lib import figstyle as fs, palette as pal, qc  # noqa: E402

cm.use_v4()
TILES = cm.RENDERS / "v4" / "s1"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
BAND_H, TW, TITLE_MM = META["band_mm"], META["tile_mm"], META["title_mm"]
U = fs.unit(1)
XS = [fs.unit_x(i) for i in range(4)]
ROW2, R_H = BAND_H + 4.2, 46.0
ROW3 = ROW2 + R_H + 3.4
ROW4 = ROW3 + R_H + 3.4
H = ROW4 + R_H
WHITE = fs.ON_DARK
NL_ = chr(10)
PARAMS = [("post_log_gain", "output gain"), ("pre_log_gain", "input gain"), ("threshold_mv", "threshold")]
PCOL = {"post_log_gain": fs.TEXT, "pre_log_gain": "#9A9A9A", "threshold_mv": pal.ACCENT}
GROUP_ORDER = ["optic lobe intrinsic", "central brain intrinsic", "VNC intrinsic", "visual projection & centrifugal",
               "descending & ascending", "motor & efferent", "sensory", "other"]


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
    """Vertical key bar, first colour at the top."""
    ax = cv.axes(x, y, w, h)
    ax.imshow(np.asarray(colours)[:, None, :3] * np.ones((1, 2, 1)), aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_axis_off()


def sci(x):
    e = int(np.floor(np.log10(abs(x))))
    return f"{x / 10 ** e:.1f} × 10$^{{{e}}}$"


def num(v):
    return f"{v:g}".replace("-", fs.MINUS)


def pct(v):
    p = 100 * v
    return f"{p:.0f}%" if p >= 10 else f"{p:.1f}%" if p >= 1 else f"{p:.2g}%"


# ---------------------------------------------------------------- A: where training acts

def panel_band(cv, sd):
    cells = cm.table("figure_data/fd_cells.csv")
    groups = cm.table("figure_data/fd_group_change.csv")
    top = groups[groups["top_1pct"].astype(str).str.lower().eq("true")]
    names = {"post_log_gain": "Output gain", "pre_log_gain": "Input gain", "threshold_mv": "Threshold", "consistent": "Most consistent 1%"}
    for k, t in enumerate(META["tiles"]):
        x = k * (TW + fs.GUTTER)
        img = np.asarray(Image.open(TILES / f"{t['name']}.png").convert("RGB"))
        cv.tile(x, 0, img, width=TW, height=BAND_H, group="band", label=t["name"])
        tx = x + (4.4 if k == 0 else 1.2)
        tile_text(cv, tx, 1.1, names[t["name"]], va="top", size=fs.PT)
        if t["name"] == "consistent":
            line = f"{len(top)} groups, {int(top['n_cells'].sum()):,} cells"
        else:
            line = f"{pct(t['changed_located'] / t['cells_located'])} of cells changed"
        tile_text(cv, tx, 4.1, line, va="top")
        sd.add("A", "tile", **{k_: v for k_, v in t.items()}, line=line)
    white_letter(cv, 1.2, 0.9, "A")
    lo, hi = META["tiles"][0]["lo"], META["tiles"][0]["hi"]
    bx, by, bh = TW - 2.4, BAND_H - 19.0, 10.0
    gradient(cv, bx, by, 1.2, bh, pal.MAGNITUDE_ON_DARK(np.linspace(1.0, 0.22, 96)))
    tile_text(cv, bx - 0.4, by + 0.3, pct(hi), va="center", ha="right")
    tile_text(cv, bx - 0.4, by + bh - 0.3, pct(lo), va="center", ha="right")
    tile_text(cv, bx + 1.2, by - 0.9, f"mean |change|{NL_}(of range)", va="bottom", ha="right", linespacing=1.0)
    # the groups of the consistent 1%, keyed in the last tile
    grp = top["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other")
    present = [g for g in GROUP_ORDER if (grp == g).any()]
    x3 = 3 * (TW + fs.GUTTER)
    for i, g in enumerate(present):
        yy = BAND_H - 6.0 - 2.4 * (len(present) - 1 - i)
        cv.mark(x3 + TW - 1.6, yy, s=5, fc=pal.ATLAS_ON_DARK[g], ec=pal.ATLAS_ON_DARK[g], lw=0.3)
        tile_text(cv, x3 + TW - 2.6, yy, pal.SHORT_GROUP[g], va="center", ha="right")
    t = META["tiles"][-1]
    length = 100 / t["scale_um_per_px"] / (img.shape[1] / TW)
    cv.line([x3 + 1.4, x3 + 1.4 + length], [BAND_H - 1.8, BAND_H - 1.8], WHITE, lw=1.0, solid_capstyle="butt")
    sd.add("A", "consistent groups by cell group", **{g: int((grp == g).sum()) for g in present})


# ---------------------------------------------------------------- B: change scales with training length

def panel_length(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "B", "Change scales with training")
    f = cm.table("followup/fu7_brain_change.csv")
    t = cm.table("stats/statistical_tests.csv").set_index("test_id").loc["fu7.spearman.updates"]
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 15.6)
    ax.scatter(f["updates"], f["brain_change"], s=8, facecolor=pal.ACCENT, edgecolor="none", zorder=3)
    for _, r in f.iterrows():
        sd.add("B", "map", map_id=r["map_id"], updates=int(r["updates"]), brain_change=float(r["brain_change"]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([100, 1000], ["100", "1,000"])
    ax.minorticks_off()
    ax.set_yticks([40, 100, 200], ["40", "100", "200"])
    ax.set_xlabel("Training updates", labelpad=0.8)
    ax.set_ylabel("Total parameter change", labelpad=0.8)
    fs.clean(ax)
    ax.text(0.04, 0.96, f"{fs.RHO} = {float(t['effect']):.3f} [{float(t['effect_ci_low']):.3f}, {float(t['effect_ci_high']):.3f}]{NL_}"
            f"Holm {fs.P_SYM} = {fs.fmt_p(float(t['p_adjusted']))}", transform=ax.transAxes, fontsize=fs.PT_SMALL, ha="left", va="top", linespacing=1.0)
    cv.text(x0 + w - 0.8, y0 + 6.0, f"{len(f)} maps", size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)
    sd.add("B", "test", rho=float(t["effect"]), ci=[float(t["effect_ci_low"]), float(t["effect_ci_high"])], holm_p=float(t["p_adjusted"]),
           method="Spearman, BCa bootstrap interval, permutation p")


# ---------------------------------------------------------------- C: by cell group

def panel_groups(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "C", "Where it acts, by group")
    b = cm.table("extended/ed04/ed04b_parameter_change_anatomy.csv")
    s = b[b["stratum_type"] == "superclass"].assign(group=lambda d: d["stratum"].map(pal.SUPERCLASS_GROUP).fillna("other"))
    lab_w = 15.0
    ax = cv.axes(x0 + lab_w, y0 + 9.8, w - lab_w - 1.0, h - 18.0)
    rows = []
    for p, _ in PARAMS:
        q = s[s["parameter"] == p]
        agg = q.groupby("group").apply(lambda d: np.average(d["mean_normalized_abs_change"], weights=d["n_groups"]), include_groups=False)
        rows.append(agg.rename(p))
    m = pd.concat(rows, axis=1)
    order = [g for g in GROUP_ORDER if g in m.index]
    floor = 3e-5
    for i, g in enumerate(order):
        for j, (p, _) in enumerate(PARAMS):
            v = float(m.loc[g, p])
            ax.plot([max(v, floor)], [i + (j - 1) * 0.2], marker="o", ms=2.6, mfc=PCOL[p] if v > 0 else "white", mec=PCOL[p], mew=0.5, ls="none",
                    clip_on=False)
            sd.add("C", "group", group=g, parameter=p, mean_abs_change=v)
    ax.set_yticks(range(len(order)), [pal.SHORT_GROUP[g] for g in order])
    ax.set_ylim(len(order) - 0.5, -0.6)
    ax.set_xscale("log")
    ax.set_xlim(floor / 1.5, 3e-2)
    ax.set_xticks([1e-4, 1e-3, 1e-2], ["0.01%", "0.1%", "1%"])
    fs.minor(ax, x=False, y=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mean |change| (of range)", labelpad=0.8)
    fs.clean(ax)
    items = [([dict(kind="mark", s=6, fc=PCOL[p], ec=PCOL[p])], lab) for p, lab in PARAMS]
    cv.key(x0 + 4.0, y0 + 5.0, items[:2], size=fs.PT_SMALL, gap=1.2)
    cv.key(x0 + 4.0, y0 + 7.6, items[2:], size=fs.PT_SMALL, gap=1.2)


# ---------------------------------------------------------------- D: every learning curve

def tier_colour(t, n=8):
    g = 0.78 - 0.70 * t / (n - 1)
    return (g, g, g)


def panel_curves(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "D", "Every map learns, harder maps take longer")
    a = cm.table("extended/ed03/ed03a_learning_curves_all.csv")
    c = cm.table("main/fig2/fig2c_cost_vs_difficulty.csv").set_index("run_id")
    d = a[a["run_id"].astype(str).str.startswith("difficulty-") & a["condition"].eq("joint")]
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 30.0, h - 15.6)
    for run, q in d.groupby("run_id"):
        q = q.sort_values(["updates", "depth_fraction"])
        tier = int(c.loc[run, "tier"])
        x = q["updates"].to_numpy(float)
        y = q["depth_fraction"].to_numpy(float)
        keep = x >= 1
        ax.plot(x[keep], y[keep], color=tier_colour(tier), lw=0.6, zorder=2 + tier / 10, drawstyle="steps-post")
        ax.scatter([x[-1]], [y[-1]], s=5, facecolor=tier_colour(tier), edgecolor="none", zorder=3)
        sd.add("D", "map", run_id=run, tier=tier + 1, optimal_moves=int(c.loc[run, "optimal_moves"]), updates=x.tolist(), depth_fraction=y.tolist())
    ax.set_xscale("log")
    ax.set_xlim(3, 3000)
    ax.set_xticks([10, 100, 1000], ["10", "100", "1,000"])
    fs.minor(ax, x=False, y=False)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.5, 1], ["0", "50", "100"])
    ax.set_xlabel("Training updates", labelpad=0.8)
    ax.set_ylabel("Solution depth reached (%)", labelpad=0.8)
    fs.clean(ax)
    kx = x0 + w - 17.6
    cv.text(kx, y0 + 8.4, f"tier (optimal{NL_}moves)", size=fs.PT_SMALL, va="center", color=fs.TEXT2, linespacing=1.0)
    for t in range(8):
        yy = y0 + 12.6 + 2.6 * t
        cv.line([kx, kx + 2.4], [yy, yy], to_rgb(tier_colour(t)), lw=0.9)
        cv.text(kx + 3.0, yy, f"{t + 1} ({int(c[c['tier'] == t]['optimal_moves'].min())}–{int(c[c['tier'] == t]['optimal_moves'].max())})",
                size=fs.PT_SMALL, va="center")
    cv.text(x0 + 9.0, y0 + 6.0, f"{d['run_id'].nunique()} maps, brain and readout trained", size=fs.PT_SMALL, va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- E: a ladder in every set

def panel_sets(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "E", "A ladder in every set")
    x = cm.table("plan_v2/x10_maps.csv")
    order = x.drop_duplicates("map").sort_values(["optimal_moves", "map"])["map"].tolist()
    sets = sorted(x["replicate"].unique())
    m = x.pivot(index="map", columns="replicate", values="updates").loc[order, sets].to_numpy(float)
    lo, hi = np.log10(30), np.log10(3000)
    u = np.clip((np.log10(m) - lo) / (hi - lo), 0, 1)
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 18.0, h - 15.6)
    ax.imshow(pal.MAGNITUDE_ON_DARK(0.08 + 0.92 * u), aspect="auto", interpolation="nearest", extent=(-0.5, len(sets) - 0.5, len(order) - 0.5, -0.5))
    opt = x.drop_duplicates("map").set_index("map").loc[order, "optimal_moves"].to_numpy(int)
    ticks = [0, len(order) // 2, len(order) - 1]
    ax.set_yticks(ticks, [str(opt[i]) for i in ticks])
    ax.set_xticks(range(len(sets)), ["orig."] + [str(s) for s in sets[1:]])
    ax.tick_params(axis="x", length=0, labelsize=fs.PT_SMALL)
    ax.set_ylabel("Map (optimal moves)", labelpad=0.8)
    ax.set_xlabel("Replicate set", labelpad=0.8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    gradient(cv, x0 + w - 7.6, y0 + 9.4, 1.4, h - 17.6, pal.MAGNITUDE_ON_DARK(0.08 + 0.92 * np.linspace(1, 0, 96)))
    cv.text(x0 + w - 5.6, y0 + 9.6, f"{10 ** hi:,.0f}", size=fs.PT_SMALL, va="center")
    cv.text(x0 + w - 5.6, y0 + h - 8.4, f"{10 ** lo:,.0f}", size=fs.PT_SMALL, va="center")
    cv.text(x0 + w - 0.8, y0 + 6.0, "updates", size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)
    for i, mp in enumerate(order):
        sd.add("E", "map", map_id=mp, optimal_moves=int(opt[i]), updates_by_set=m[i].astype(int).tolist())


# ---------------------------------------------------------------- F: deadlocks cost updates

def panel_stages(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "F", "Deadlock risk costs updates")
    b = cm.table("extended/ed03/ed03b_stage_costs_all.csv")
    d = b[b["run_id"].astype(str).str.startswith("difficulty-") & b["condition"].eq("joint")]
    groups = [(False, "no"), (True, "yes")]
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 16.6)
    rng = np.random.default_rng(81)
    for i, (dead, lab) in enumerate(groups):
        q = d[(d["deadlock_alternatives_at_entry"] > 0) == dead]["updates_spent"].to_numpy(float)
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(q)), q + 1, s=2.5, facecolor="#9A9A9A", edgecolor="none", alpha=0.6, zorder=2)
        med = float(np.median(q))
        ax.plot([i - 0.3, i + 0.3], [med + 1, med + 1], color=fs.TEXT, lw=1.0, zorder=3)
        ax.text(i + 0.33, med + 1, f"{med:.0f}", fontsize=fs.PT_SMALL, ha="left", va="center")
        sd.add("F", "stages", deadlock_alternative_at_entry=dead, stages=int(len(q)), median_updates=med, updates=q.astype(int).tolist())
    ax.set_yscale("log")
    ax.set_ylim(0.7, 3000)
    ax.set_yticks([1, 11, 101, 1001], ["0", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1], [f"no{NL_}({int(((d['deadlock_alternatives_at_entry'] > 0) == False).sum())})",
                           f"yes{NL_}({int((d['deadlock_alternatives_at_entry'] > 0).sum())})"], linespacing=1.0)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Deadlocking move at the stage entry", labelpad=0.8)
    ax.set_ylabel("Updates per stage", labelpad=0.8)
    fs.clean(ax)
    cv.text(x0 + w - 0.8, y0 + 6.0, "median", size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- G: wiring at the hardest map

def panel_wiring(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "G", "Wiring matters at the hardest map")
    wd = cm.table("plan_v2/s1_wiring.csv")
    rows = ["analysed lineage, real wiring", "real wiring", "weights shuffled, wiring kept", "rewired, degrees kept"]
    labels = {"analysed lineage, real wiring": f"analyzed lineage,{NL_}six sets", "real wiring": f"neutral start,{NL_}real wiring",
              "weights shuffled, wiring kept": f"neutral start,{NL_}weights shuffled", "rewired, degrees kept": f"neutral start,{NL_}rewired (degrees kept)"}
    lab_w = 27.0
    ax = cv.axes(x0 + lab_w, y0 + 8.4, w - lab_w - 15.0, h - 16.6)
    for i, cond in enumerate(rows):
        q = wd[wd["condition"] == cond]
        solved = q["solved"].astype(str).str.lower().eq("true")
        if len(q) > 1:
            ax.add_patch(Rectangle((q["updates"].min(), i - 0.06), q["updates"].max() - q["updates"].min(), 0.12, facecolor=fs.GRID, edgecolor="none", zorder=2))
            ax.scatter(q["updates"], np.full(len(q), i), s=10, facecolor=fs.TEXT, edgecolor="none", zorder=3)
            out = f"all {len(q)}{NL_}learned"
        else:
            r = q.iloc[0]
            ok = bool(solved.iloc[0])
            ax.add_patch(Rectangle((0, i - 0.13), r["updates"], 0.26, facecolor="#BDBDBD" if ok else "#E0E0E0", edgecolor="none", zorder=1))
            ax.scatter([r["updates"]], [i], s=16 if ok else 22, marker="o" if ok else "x", facecolor=fs.TEXT if ok else None,
                       color=fs.TEXT, linewidth=0.8, zorder=3)
            out = "learned" if ok else f"stopped,{NL_}no progress"
        ax.text(2680, i, out, fontsize=fs.PT_SMALL, ha="left", va="center", clip_on=False, linespacing=1.0)
        for _, r in q.iterrows():
            sd.add("G", "run", condition=cond, run=int(r["run"]), status=r["status"], updates=int(r["updates"]), solved=bool(r["solved"]))
    ax.set_yticks(range(len(rows)), [labels[r] for r in rows], linespacing=1.0)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlim(0, 2600)
    ax.set_xticks([0, 1000, 2000], ["0", "1,000", "2,000"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Training updates (60-move map)", labelpad=0.8)
    fs.clean(ax)


# ---------------------------------------------------------------- H: the model at a glance

def panel_spec(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "H", "The model at a glance")
    s = cm.table("main/fig1/fig1a_model_specification.csv").set_index(["section", "quantity"])["value"]
    b = cm.table("main/fig1/fig1b_parameter_budget.csv")
    lin = b[b["model_kind"] == "connectome_linear"]

    def v(sec, q):
        return s.loc[(sec, q)]

    def n(sec, q):
        return float(v(sec, q))

    brain = lin[(lin["component"] == "brain")]
    readout = int(lin[(lin["component"] == "readout") & lin["parameter"].str.startswith("proposal")]["n_elements"].astype(int).sum())
    cols = [("Network", [(f"{int(n('network', 'neurons')):,}", "neurons (all of the CNS)"),
                         (f"{n('network', 'stored_connectome_entries') / 1e6:.2f} M", "connections (wiring fixed)"),
                         (f"{n('network', 'excitatory_entries') / 1e6:.2f} M / {n('network', 'inhibitory_entries') / 1e6:.2f} M", "excitatory / inhibitory"),
                         (f"{int(n('network', 'shared_parameter_groups')):,}", "cell-type groups"),
                         (f"{int(n('network', 'input_cells')):,}", "visual input cells"),
                         (f"{int(n('network', 'observed_cells')):,}", f"readout cells ({int(n('readout', 'populations'))} × {int(n('readout', 'cells_per_population'))})")]),
            ("Neurons", [(f"{num(n('lif', 'v_rest'))} / {num(n('lif', 'v_th'))} mV", "rest / threshold"),
                         (f"{n('lif', 'tau_m'):g} / {n('lif', 'tau_syn'):g} ms", "membrane / synaptic τ"),
                         (f"{n('lif', 'w_syn'):g} mV", "per synapse (cap {:g})".format(n("lif", "conn_cap"))),
                         (f"+{n('lif', 'adapt_jump'):g} mV, {n('lif', 'adapt_tau'):g} ms", "adaptation per spike, τ"),
                         (f"{n('lif', 't_ref'):g} / {n('lif', 'delay'):g} ms", "refractory / delay"),
                         (f"{n('simulation', 'time_step'):g} ms", "time step")]),
            ("Trained", [(f"{int(brain['n_elements'].astype(int).sum()):,}", "brain parameters"),
                         (f"{len(brain)} × {int(brain['n_elements'].astype(int).iloc[0]):,}", "gains in, out, threshold"),
                         (f"{readout:,}", "readout weights and biases"),
                         (f"{int(n('readout', 'feature_dim'))}", "readout features"),
                         (f"{int(n('readout', 'actions'))}", "actions"),
                         (f"{int(n('simulation', 'decision_window'))} ms", "decision window")])]
    cw = (w - 1.0) / len(cols)
    for k, (head, items) in enumerate(cols):
        cx = x0 + 1.0 + k * cw
        cv.text(cx, y0 + 6.4, head, size=fs.PT, va="center", fontweight="bold")
        for i, (val, lab) in enumerate(items):
            yy = y0 + 10.6 + 5.6 * i
            cv.text(cx, yy, val, size=fs.PT_SMALL, va="center", fontweight="bold")
            cv.text(cx, yy + 2.2, lab, size=fs.PT_SMALL, va="center", color=fs.TEXT2)
            sd.add("H", head, value=val, item=lab)


# ---------------------------------------------------------------- I: bit-exact on every GPU

def short_model(m):
    m = m.replace(" (readout shared by hidden- and visible-goal training)", ", shared readout")
    return m.replace("the analysed planning model, mixed vs double precision", "planning model, double precision")


def panel_replay(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "I", "Bit-exact on every GPU")
    r = cm.table("plan_v2/s1_replay.csv")
    hosts = list(dict.fromkeys(r["host"]))
    models = list(dict.fromkeys(r["model"]))
    models = [m for m in models if m.startswith("the analysed")] + [m for m in models if not m.startswith("the analysed")]
    cols = [(hst, g) for hst in hosts for g in sorted(r[r["host"] == hst]["gpu"].unique())]
    lab_w, cell = 36.0, 3.0
    gx0, gy0 = x0 + lab_w, y0 + 11.4
    for j, (hst, g) in enumerate(cols):
        hx = gx0 + j * cell + hosts.index(hst) * 1.6
        for i, m in enumerate(models):
            q = r[(r["host"] == hst) & (r["gpu"] == g) & (r["model"] == m)].iloc[0]
            ok = bool(str(q["passed"]).lower() == "true") and int(q["histories_identical"]) == int(q["histories"])
            cv.box(hx + 0.25, gy0 + i * cell + 0.25, cell - 0.5, cell - 0.5, fc=fs.TEXT if ok else "white", ec=fs.TEXT, lw=0.3)
    for i, m in enumerate(models):
        cv.text(gx0 - 1.0, gy0 + i * cell + cell / 2, short_model(m), size=fs.PT_SMALL, ha="right", va="center")
    for hst in hosts:
        js = [j for j, c in enumerate(cols) if c[0] == hst]
        cx = gx0 + (js[0] + js[-1] + 1) / 2 * cell + hosts.index(hst) * 1.6
        cv.text(cx, gy0 - 1.4, f"{hst}{NL_}({len(js)} GPU{'s' if len(js) > 1 else ''})", size=fs.PT_SMALL, ha="center", va="bottom", linespacing=1.0)
    hist = int(r["histories"].sum())
    same = int(r["histories_identical"].sum())
    dmax = float(r["max_abs_logit_difference"].max())
    v = cm.table("extended/ed02/ed02b_validation_checks.csv")
    v = v[(v["record"] == "validation") & (v["check_file"] == "validation.json")].set_index("item")["value"]
    yb = gy0 + len(models) * cell + 3.2
    restore = "exact" if str(v["full_state_restore_exact"]).lower() == "true" else "not exact"
    sd.add("I", "replay", replays=int(len(r)), histories=hist, identical=same, max_abs_logit_difference=dmax,
           gradient_recompute_max_abs=float(v["full_vs_recomputed_gradient_max_abs"]), full_state_restore_exact=str(v["full_state_restore_exact"]))


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("s1")
    for rel in ("figure_data/fd_cells.csv", "figure_data/fd_group_change.csv", "followup/fu7_brain_change.csv", "stats/statistical_tests.csv",
                "extended/ed04/ed04b_parameter_change_anatomy.csv", "extended/ed03/ed03a_learning_curves_all.csv", "extended/ed03/ed03b_stage_costs_all.csv",
                "main/fig2/fig2c_cost_vs_difficulty.csv", "plan_v2/x10_maps.csv", "plan_v2/s1_wiring.csv", "plan_v2/s1_replay.csv",
                "main/fig1/fig1a_model_specification.csv", "main/fig1/fig1b_parameter_budget.csv", "extended/ed02/ed02b_validation_checks.csv"):
        sd.source(cm.RESULTS / rel)
    for f in sorted(TILES.glob("*.png")) + sorted(TILES.glob("*.json")):
        sd.source(f)
    panel_band(cv, sd)
    panel_length(cv, sd, XS[0], ROW2, U, R_H)
    panel_groups(cv, sd, XS[1], ROW2, U, R_H)
    panel_curves(cv, sd, XS[2], ROW2, fs.unit(2), R_H)
    panel_sets(cv, sd, XS[0], ROW3, U, R_H)
    panel_stages(cv, sd, XS[1], ROW3, U, R_H)
    panel_wiring(cv, sd, XS[2], ROW3, fs.unit(2), R_H)
    panel_spec(cv, sd, XS[0], ROW4, fs.unit(2), R_H)
    panel_replay(cv, sd, XS[2], ROW4, fs.unit(2), R_H)
    cm.finish(cv, "s1", sd, kind="supplement")


if __name__ == "__main__":
    main()
