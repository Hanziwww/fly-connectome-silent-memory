"""Fig. 3 - Brain and readout learn as a pair; learning recruits the same cell types on every map (180 x 151 mm).

Row 1  a  swap test: initial or trained brain x initial or trained readout, 24 maps per pairing (tier colours); McNemar, Holm
       b  first-decision margin change split into readout, brain and interaction terms per map; Hodges-Lehmann, Wilcoxon, Holm
       c  training controls: what is trained, the graph, updates relative to joint training on the same map, solved maps
Row 2  d  readout geometry: principal angles, final vs initial readout (24 maps) and between maps (276 pairs)
       e  consistency of parameter change across maps: 24 x 24 Spearman matrices of |change| over all parameter groups
          (vs the stratified null; BH-adjusted family F3e.consistency)
       f  where learning acts: mean |change| by synaptic distance from the inputs and to the readout
Band   g  whole-CNS map: every located cell coloured by its group's consistency score (mean within-map percentile)
       h  one neuron of each of the eight most consistently changed type-level groups, with per-map percentiles

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\fig3.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.patches import Circle, Rectangle
from PIL import Image
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, palette as pal, qc  # noqa: E402

U = (fs.FULL - 3 * fs.GAP) / 4                 # 43.65-mm grid unit, four per row
XS = [k * (U + fs.GAP) for k in range(4)]
ROW1, ROW2, ROW_H = 0.0, 44.0, 41.0
AX_TOP, AX_H = 6.5, 27.0                       # row-2 chart axes: offset from the row top and height (mm)
BAND = 88.5
TILES = cm.RENDERS / "fig3"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
H = BAND + META["band_mm"]
TIER = [colormaps["viridis"](0.02 + 0.82 * t / 7) for t in range(8)]
VIR = colormaps["viridis"]
COST = colormaps["BrBG_r"]
PARAMS = [("post_log_gain", "output gain"), ("pre_log_gain", "input gain"), ("threshold_mv", "threshold")]
PCOL = {"post_log_gain": "#1A1A1A", "pre_log_gain": "#9A9A9A", "threshold_mv": pal.ACCENT}
MAPS = [f"D{t}-T{k}" for t in range(8) for k in range(3)]
RUNS = [f"difficulty-{m}" for m in MAPS]
WHITE = fs.ON_DARK
EDGE = "#ABABAB"
NT = {"acetylcholine": "ACh", "gaba": "GABA", "glutamate": "Glu"}
ROLE = {"observed": ", readout", "input": ", input", "input_and_observed": ", input and readout"}


def tier_of(m):
    return int(m.replace("difficulty-", "")[1])


def tile_text(cv, x, y, s, size=fs.PT_SMALL, **kw):
    t = cv.text(x, y, s, size=size, color=WHITE, **kw)
    t.set_path_effects(fs.dark_halo(0.8))
    return t


def cost_colour(ratio):
    """Updates relative to joint training: blue-green below 1 (cheaper), brown above (costlier); log2 range -3 to 4."""
    t = float(np.clip(np.log2(ratio), -3.0, 4.0))
    return COST(0.5 + 0.3 * (t / 4.0 if t >= 0 else t / 3.0))


def score_colour(v):
    m = META["map"]
    t = np.clip((np.asarray(v, float) - m["score_lo"]) / (m["score_hi"] - m["score_lo"]), 0, 1)
    return pal.MAGNITUDE_ON_DARK(m["cmap_lo"] + (1 - m["cmap_lo"]) * t)


def hbar(cv, x, y, w, h, colours, ticks, labels):
    """Horizontal colour bar (colours left to right) with tick labels below; ticks in 0-1."""
    ax = cv.axes(x, y, w, h)
    ax.imshow(np.asarray(colours)[None, :, :3], aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_yticks([])
    ax.set_xticks(ticks, labels)
    ax.tick_params(axis="x", length=1.0, pad=0.4, labelsize=fs.PT_SMALL)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    return ax


# ---------------------------------------------------------------- a: swap test

def panel_a(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "a")
    c = cm.table("main/fig3/fig3c_four_combination.csv")
    c = c[c["set"] == "difficulty"]
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    ax = bd.mm_axes(cv, x0, y0, w, h)
    lab_w, top, gap, ch = 8.6, 5.4, 0.5, 11.2
    cw = (w - lab_w - gap) / 2
    q, pitch = 1.8, 2.1
    names = {"old": "initial", "new": "trained"}
    for j, ro in enumerate(("old", "new")):
        ax.text(lab_w + j * (cw + gap) + cw / 2, top - 0.6, f"{names[ro]} readout", ha="center", va="bottom", fontsize=fs.PT_SMALL)
    for i, br in enumerate(("old", "new")):
        cy = top + i * (ch + gap)
        ax.text(lab_w - 0.9, cy + ch / 2, f"{names[br]}\nbrain", ha="right", va="center", fontsize=fs.PT_SMALL, linespacing=1.15)
        for j, ro in enumerate(("old", "new")):
            cx = lab_w + j * (cw + gap)
            own = (br, ro) == ("new", "new")
            ax.add_patch(Rectangle((cx, cy), cw, ch, facecolor="#F3F3F3", edgecolor=fs.TEXT if own else "none", linewidth=0.5))
            d = c[(c["brain"] == br) & (c["readout"] == ro)]
            assert len(d) == 24 and set(d["map_id"]) == set(MAPS)
            gx0 = cx + (cw - (8 * pitch - (pitch - q))) / 2
            n_ok = 0
            for _, r in d.iterrows():
                t, k = tier_of(r["map_id"]), int(r["map_id"][-1])
                ok = str(r["solved"]).lower() == "true"
                n_ok += ok
                ax.add_patch(Rectangle((gx0 + t * pitch, cy + 1.0 + k * pitch), q, q, facecolor=TIER[t] if ok else "white",
                                       edgecolor="none" if ok else EDGE, linewidth=0.3))
                sd.add("a", "map", brain=names[br], readout=names[ro], map_id=r["map_id"], solved=ok)
            ax.text(cx + cw / 2, cy + 1.0 + 3 * pitch + 0.6, f"{n_ok} of 24 solved", ha="center", va="top", fontsize=fs.PT_SMALL)
    pv = {k: float(tests.loc[f"fig3c.mcnemar.11_vs_{k}", "p_adjusted"]) for k in ("00", "01", "10")}
    y = top + 2 * ch + gap + 1.5
    ax.text(0.0, y, "Own trained pair vs each mixed pair:", ha="left", va="top", fontsize=fs.PT_SMALL)
    ax.text(0.0, y + 2.3, f"{fs.P_SYM} = {fs.fmt_p(max(pv.values()))} each (McNemar, Holm)", ha="left", va="top", fontsize=fs.PT_SMALL)
    for k, p in pv.items():
        sd.add("a", "McNemar", contrast=f"trained pair vs {k}", p_holm=p, test_id=f"fig3c.mcnemar.11_vs_{k}")
    ty = y0 + 39.0
    lab = "Difficulty tier D0"
    cv.text(x0, ty, lab, size=fs.PT_SMALL, va="center")
    tx = x0 + fs.text_width(lab, fs.PT_SMALL) + 0.7
    for t in range(8):
        cv.box(tx + t * 1.75, ty - 0.65, 1.5, 1.3, TIER[t])
    cv.text(tx + 8 * 1.75 + 0.45, ty, "D7", size=fs.PT_SMALL, va="center")


# ---------------------------------------------------------------- b: margin decomposition

def panel_b(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "b")
    d = cm.table("main/fig3/fig3d_margin_decomposition.csv")
    d = d[d["set"] == "difficulty"]
    assert len(d) == 24
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    lab, pcol = 9.8, 9.4
    ax = cv.axes(x0 + lab, y0 + 5.4, w - lab - pcol, 27.4)
    terms = [("readout_term", "readout", "fig3d.wilcoxon.readout"), ("brain_term", "brain", "fig3d.wilcoxon.brain"),
             ("interaction_term", "interaction", "fig3d.wilcoxon.interaction")]
    rng = np.random.default_rng(3)
    cols = [TIER[tier_of(m)] for m in d["map_id"]]
    for i, (col, name, tid) in enumerate(terms):
        vals = d[col].to_numpy(float)
        ax.scatter(vals, i - 0.1 + (rng.random(len(vals)) - 0.5) * 0.36, s=6, c=cols, linewidth=0, zorder=3)
        t = tests.loc[tid]
        fs.interval(ax, i + 0.33, float(t["effect"]), float(t["effect_ci_low"]), float(t["effect_ci_high"]), orient="h", ms=8)
        ax.text(1.04, i, fs.fmt_p(float(t["p_adjusted"])), transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=fs.PT_SMALL)
        for m, v in zip(d["map_id"], vals):
            sd.add("b", name, map_id=m, logit=float(v))
        sd.add("b", "Wilcoxon", term=name, hodges_lehmann=float(t["effect"]), ci_low=float(t["effect_ci_low"]),
               ci_high=float(t["effect_ci_high"]), p_holm=float(t["p_adjusted"]), test_id=tid)
    ax.axvline(0, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=1)
    ax.set_yticks(range(3), [t[1] for t in terms])
    ax.set_ylim(2.65, -0.6)
    ax.set_xlim(-0.6, 1.5)
    ax.set_xticks([-0.5, 0, 0.5, 1, 1.5], [fs.MINUS + "0.5", "0", "0.5", "1", "1.5"])
    fs.minor(ax, y=False)
    fs.clean(ax)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Change in first-decision margin (logit)", labelpad=1)
    ax.text(1.04, -0.6, "Holm " + fs.P_SYM, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=fs.PT_SMALL)
    cv.key(x0 + 0.6, y0 + 39.6, [([dict(kind="mark", s=6, fc=TIER[0], space=0.25), dict(kind="mark", s=6, fc=TIER[4], space=0.25),
                                   dict(kind="mark", s=6, fc=TIER[7])], "map"),
                                  ([dict(kind="hline", w=3.0, lw=0.8, colour=fs.TEXT, mark_fc=fs.TEXT, s=8)], "Hodges–Lehmann, 95% CI")],
           gap=2.0)


# ---------------------------------------------------------------- c: training controls

CONDITIONS = [("joint", "joint (reference)"), ("head_only", "readout only"), ("brain_only", "brain only"), ("frozen", "nothing (frozen)"),
              ("demonstration", "demonstration loss"), ("entry_ce", "entry-only loss"), ("alternating", "alternating updates"),
              ("no_recovery", "no recovery states"), ("neutral", "neutral initial gains"), ("rewired", "rewired graph"),
              ("weight_shuffle", "shuffled weights"), ("board_mlp", "board MLP (no brain)")]
BREAKS = (4, 8, 11)


def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    a = cm.table("main/fig3/fig3a_training_controls.csv")
    b = cm.table("main/fig3/fig3b_control_costs.csv")
    maps = ["D1-T0", "D3-T0", "D5-T0", "D7-T0"]
    ax = bd.mm_axes(cv, x0, y0, w, h)
    lab_x, bx, rx, gx = 19.8, 23.0, 29.0, 36.0
    mx0, mw = 41.0, 9.9
    sx = mx0 + 4 * mw + 1.3
    ry0, ry1, gap = 5.6, 36.8, 0.7
    rh = (ry1 - ry0 - gap * len(BREAKS)) / len(CONDITIONS)
    hy1, hy2 = 1.3, 3.9
    ax.text((bx + rx) / 2, hy1, "trained", ha="center", va="center", fontsize=fs.PT_SMALL)
    for xx, lab in ((bx, "brain"), (rx, "readout"), (gx, "graph")):
        ax.text(xx, hy2, lab, ha="center", va="center", fontsize=fs.PT_SMALL)
    ax.text(mx0 + 2 * mw, hy1, "updates relative to joint training on the same map", ha="center", va="center", fontsize=fs.PT_SMALL)
    for j, m in enumerate(maps):
        ax.text(mx0 + (j + 0.5) * mw, hy2, m.split("-")[0], ha="center", va="center", fontsize=fs.PT_SMALL)
    ax.text(sx, hy2, "solved", ha="left", va="center", fontsize=fs.PT_SMALL)
    for i, (cond, lab) in enumerate(CONDITIONS):
        yy = ry0 + i * rh + gap * sum(i >= k for k in BREAKS)
        yc = yy + rh / 2
        ax.text(lab_x, yc, lab, ha="right", va="center", fontsize=fs.PT_SMALL)
        rows = a[a["condition"] == cond]
        assert len(rows) == 4, cond
        r0 = rows.iloc[0]
        mlp = r0["model_kind"] == "board_mlp"
        for xx, col in ((bx, "train_core"), (rx, "train_head")):
            if mlp and col == "train_core":
                ax.plot([xx - 0.7, xx + 0.7], [yc, yc], color=EDGE, lw=0.5)
                continue
            on = str(r0[col]).lower() == "true"
            ax.add_patch(Rectangle((xx - 0.85, yc - 0.85), 1.7, 1.7, facecolor=pal.ACCENT if on else "white",
                                   edgecolor=pal.ACCENT if on else EDGE, linewidth=0.35))
        graph = "none" if mlp else {"real": "real", "rewired": "rewired", "weight_shuffle": "shuffled"}[r0["structure"]]
        ax.text(gx, yc, graph, ha="center", va="center", fontsize=fs.PT_SMALL)
        solved = run = 0
        for j, m in enumerate(maps):
            r = rows[rows["map_id"] == m].iloc[0]
            status, ok = r["status_group"], str(r["final_solved"]).lower() == "true"
            xx = mx0 + j * mw
            rr = b[(b["condition"] == cond) & (b["map_id"] == m)]["updates_ratio"]
            ratio = float(rr.iloc[0]) if len(rr) and np.isfinite(rr.iloc[0]) else None
            if status == "never_run":
                ax.plot([xx + mw * 0.4, xx + mw * 0.6], [yc, yc], color=EDGE, lw=0.5)
            else:
                run += 1
                solved += ok
                if cond == "joint":
                    fc, txt, ec = COST(0.5), "1", fs.TEXT
                elif status == "baseline":
                    fc, txt, ec = "#DCDCDC", "0", "none"
                elif ok and ratio is not None:
                    fc, txt, ec = cost_colour(ratio), f"{ratio:.2g}", "none"
                else:
                    fc, txt, ec = "white", ("stop" if status.startswith("stopped") else "fail"), EDGE
                ax.add_patch(Rectangle((xx + 0.15, yy + 0.15), mw - 0.3, rh - 0.3, facecolor=fc, edgecolor=ec, linewidth=0.35))
                ax.text(xx + mw / 2, yc, txt, ha="center", va="center", fontsize=fs.PT_SMALL)
            sd.add("c", "run", condition=cond, map_id=m, status=status, solved=ok, updates_ratio=ratio,
                   trained_brain=str(r0["train_core"]).lower() == "true", trained_readout=str(r0["train_head"]).lower() == "true",
                   graph=graph)
        ax.text(sx, yc, f"{solved} of {run}", ha="left", va="center", fontsize=fs.PT_SMALL)
    ky = y0 + 39.0
    cv.key(x0 + 1.0, ky, [([dict(kind="box", fc=pal.ACCENT, size=1.5)], "trained"),
                          ([dict(kind="box", fc="white", ec=EDGE, lw=0.35, size=1.5)], "fixed")], gap=1.8)
    v = np.linspace(-3, 4, 141)
    hbar(cv, x0 + mx0, ky - 0.6, 21.0, 1.2, [cost_colour(2 ** t) for t in v], [(t + 3) / 7 for t in (-2, 0, 2, 4)], ["0.25", "1", "4", "16"])
    end = cv.key(x0 + mx0 + 22.2, ky, [([], "× joint")], gap=2.0)
    cv.key(end + 2.2, ky, [([dict(kind="hline", w=1.6, lw=0.5, colour=EDGE)], "not run")])


# ---------------------------------------------------------------- d: readout geometry

def panel_d(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "d")
    e = cm.table("main/fig3/fig3e_readout_geometry.csv")
    pf = e[e["row_type"] == "pairwise_final"]
    vi = e[(e["row_type"] == "vs_initial") & e["model"].str.startswith("difficulty-")]
    assert len(pf) == 276 and len(vi) == 24
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="mark", s=5, fc="#A6A6A6")], "trained readouts of two maps (276 pairs)")])
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="mark", s=5, fc=TIER[0], space=0.2), dict(kind="mark", s=5, fc=TIER[4], space=0.2),
                                  dict(kind="mark", s=5, fc=TIER[7])], "trained vs initial readout (24 maps)")])
    ax = cv.axes(x0 + 7.4, y0 + AX_TOP, w - 7.6, AX_H)
    rng = np.random.default_rng(5)
    for k in range(3):
        col = f"angle_{k + 1}_deg"
        a = pf[col].to_numpy(float)
        ax.scatter(k - 0.18 + (rng.random(len(a)) - 0.5) * 0.24, a, s=1.5, color="#A6A6A6", linewidth=0, zorder=2)
        b = vi[col].to_numpy(float)
        ax.scatter(k + 0.18 + (rng.random(len(b)) - 0.5) * 0.16, b, s=5, c=[TIER[tier_of(m)] for m in vi["model"]], linewidth=0, zorder=3)
        for xv, med in ((k - 0.18, np.median(a)), (k + 0.18, np.median(b))):
            ax.plot([xv - 0.14, xv + 0.14], [med, med], color=fs.TEXT, lw=0.8, zorder=4)
        sd.add("d", "median", angle=k + 1, between_maps_deg=float(np.median(a)), vs_initial_deg=float(np.median(b)))
        for (m1, m2), v in zip(pf[["model", "other_model"]].itertuples(index=False), a):
            sd.add("d", "between maps", angle=k + 1, model=m1, other_model=m2, degrees=float(v))
        for m, v in zip(vi["model"], b):
            sd.add("d", "trained vs initial", angle=k + 1, model=m, degrees=float(v))
    ax.set_xticks(range(3), ["1st", "2nd", "3rd"])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 50)
    ax.set_yticks([0, 25, 50])
    fs.minor(ax, x=False)
    fs.clean(ax)
    ax.set_ylabel("Angle (degrees)", labelpad=1)
    ax.set_xlabel("Principal angle between readouts", labelpad=1)


# ---------------------------------------------------------------- e: consistency matrices

def panel_e(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "e")
    st4 = cm.table("supplementary/st4_parameter_change.csv")
    cons = cm.table("extended/ed04/ed04a_parameter_change_consistency.csv")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    d = st4[st4["run_id"].isin(RUNS)]
    strip, sgap, bgap = 1.0, 0.3, 1.6
    s = (w - 3 * (strip + sgap) - 2 * bgap) / 3
    top = y0 + AX_TOP
    tier_rgb = np.array([TIER[tier_of(m)][:3] for m in MAPS])
    ps = []
    for k, (p, lab) in enumerate(PARAMS):
        X = d.pivot(index="run_id", columns="group", values=f"delta_{p}").loc[RUNS].abs().to_numpy(float)
        C = np.corrcoef(rankdata(X, axis=1))
        off = C[~np.eye(24, dtype=bool)].mean()
        row = cons[(cons["parameter"] == p) & (cons["checkpoint"] == "final") & (cons["metric"] == "mean_pairwise_spearman")].iloc[0]
        assert abs(off - float(row["observed"])) < 1e-9, (p, off, row["observed"])
        tid = f"ed04a.mean_pairwise_spearman.{p}"
        ps.append(float(tests.loc[tid, "p_adjusted"]))
        C[np.eye(24, dtype=bool)] = np.nan
        xs = x0 + k * (strip + sgap + s + bgap)
        st = cv.axes(xs, top, strip, s)
        st.imshow(tier_rgb[:, None, :], aspect="auto", interpolation="none")
        st.set_axis_off()
        ax = cv.axes(xs + strip + sgap, top, s, s)
        ax.imshow(C, cmap=VIR, vmin=0, vmax=1, interpolation="none", aspect="auto")
        ax.set_axis_off()
        mid = xs + strip + sgap + s / 2
        cv.text(mid, top - 0.6, f"{lab}, {X.shape[1]:,} groups" if k == 0 else lab, size=fs.PT_SMALL, ha="center", va="bottom")
        cv.text(mid, top + s + 0.8, f"mean {off:.2f}, null {float(row['null_mean']):.2f}", size=fs.PT_SMALL, ha="center", va="top")
        iu = np.triu_indices(24, 1)
        for i, j in zip(*iu):
            sd.add("e", lab, map_a=MAPS[i], map_b=MAPS[j], spearman=float(C[i, j]))
        sd.add("e", "test", parameter=p, observed=float(off), null_mean=float(row["null_mean"]), null_q025=float(row["null_q025"]),
               null_q975=float(row["null_q975"]), p_adjusted=ps[-1], correction=cm.correction([tid]), test_id=tid, groups=int(X.shape[1]))
    ky = y0 + 38.8
    lab = f"Spearman {fs.RHO} of |change| between maps"
    cv.text(x0 + 1.0, ky, lab, size=fs.PT_SMALL, va="center")
    bx = x0 + 1.0 + fs.text_width(lab, fs.PT_SMALL) + 1.4
    hbar(cv, bx, ky - 0.6, 16.0, 1.2, VIR(np.linspace(0, 1, 128)), [0, 0.5, 1], ["0", "0.5", "1"])
    adj = cm.adjusted_label([f"ed04a.mean_pairwise_spearman.{p}" for p, _ in PARAMS])
    cv.text(bx + 17.4, ky, f"each {adj} = {fs.fmt_p(max(ps))} vs the stratified null", size=fs.PT_SMALL, va="center")


# ---------------------------------------------------------------- f: where learning acts

def panel_f(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "f")
    b = cm.table("extended/ed04/ed04b_parameter_change_anatomy.csv")
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="hline", colour=PCOL[p], w=2.6, lw=0.9)], lab) for p, lab in PARAMS], gap=1.6)
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="mark", s=5, fc="white", ec=fs.TEXT, lw=0.5)], "no path (all three parameters)")])
    ax_l, ax_w = x0 + 7.4, w - 9.0
    ax = cv.axes(ax_l, y0 + AX_TOP, ax_w, AX_H)
    specs = [("hops_from_input_class", ["0", "1", "2", ">=3"], 0.0, "from the inputs"),
             ("hops_to_observed_class", ["0", "1", "2"], 5.6, "to the readout")]
    lim = (-0.45, 9.0)
    ticks, labels = [], []
    for stype, order, off, glab in specs:
        xs = off + np.arange(len(order))
        xn = off + len(order)
        for p, _ in PARAMS:
            g = b[(b["stratum_type"] == stype) & (b["parameter"] == p)].set_index("stratum")
            v = [100 * float(g.loc[s_, "mean_normalized_abs_change"]) for s_ in order]
            ax.plot(xs, v, color=PCOL[p], lw=0.9, marker="o", ms=2.3, mew=0, zorder=3)
            vn = 100 * float(g.loc["unreachable", "mean_normalized_abs_change"])
            ax.plot([xn], [vn], ls="none", marker="o", ms=2.3, mfc="white", mec=fs.TEXT, mew=0.5, zorder=4, clip_on=False)
            for s_ in order + ["unreachable"]:
                sd.add("f", stype, parameter=p, stratum=s_, mean_abs_change_pct_of_range=100 * float(g.loc[s_, "mean_normalized_abs_change"]),
                       groups=int(g.loc[s_, "n_groups"]), cells=int(g.loc[s_, "n_cells"]))
        ticks += list(xs) + [xn]
        labels += [o.replace(">=", "≥") for o in order] + ["none"]
        mid = ax_l + ((off + xn) / 2 - lim[0]) / (lim[1] - lim[0]) * ax_w
        cv.text(mid, y0 + AX_TOP + AX_H + 2.9, glab, size=fs.PT_SMALL, ha="center", va="top")
    cv.text(ax_l + ax_w / 2, y0 + AX_TOP + AX_H + 5.0, "Synaptic steps", size=fs.PT, ha="center", va="top")
    ax.set_xticks(ticks, labels)
    ax.set_xlim(*lim)
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0, 0.4, 0.8])
    fs.minor(ax, x=False)
    fs.clean(ax)
    ax.set_ylabel("Mean |change| (% of range)", labelpad=1)


# ---------------------------------------------------------------- g, h: dark band

def map_percentiles():
    """Within-map percentile of combined normalized |change| for every group (24 maps x groups, tier order)."""
    st4 = cm.table("supplementary/st4_parameter_change.csv")
    d = st4[st4["run_id"].isin(RUNS)].sort_values(["run_id", "group"], kind="mergesort")
    n = int(d["group"].nunique())
    assert sorted(d["run_id"].unique()) == RUNS
    pct = rankdata(d["normalized_abs_change_sum"].to_numpy().reshape(len(RUNS), n), axis=1) / n
    g = cm.table("figure_data/fd_group_change.csv").set_index("group")
    assert np.allclose(pct.mean(0), g.loc[np.arange(n), "consistency_score"].to_numpy(), rtol=0, atol=1e-9)
    return pct, g


def scale_bar(cv, right, y, length_um, um_per_px, px_per_mm):
    length = length_um / um_per_px / px_per_mm
    cv.line([right - length, right], [y, y], WHITE, lw=1.0, solid_capstyle="butt")
    tile_text(cv, right, y - 0.5, f"{length_um} µm", ha="right", va="bottom")


BADGE_MM = 0.95                                # radius of the numbered badges on the cells shown in h


def panel_g(cv, sd, groups):
    mw, bh = META["map_mm"]
    m = META["map"]
    img = np.asarray(Image.open(TILES / "map.png").convert("RGB")).copy()
    px_per_mm = img.shape[1] / mw
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    for mk in m["marks"]:                      # badge plates: black discs under the numbered rings (annotation, not data)
        img[(xx + 0.5 - mk["x_px"]) ** 2 + (yy + 0.5 - mk["y_px"]) ** 2 <= (BADGE_MM * px_per_mm) ** 2] = 0
    cv.tile(0, BAND, img, width=mw, height=bh, group="fig3band", label="map")
    cv.letter(1.2, BAND + 0.9, "g")
    cv.fig.texts[-1].set_color(WHITE)
    # colour key in the black space left of the VNC
    kx, ky = 1.5, BAND + 23.0
    tile_text(cv, kx, ky, "Change percentile" + "\n" + "of the cell's group," + "\n" + "mean of 24 maps", va="top", linespacing=1.05)
    by, bhh = ky + 6.6, 13.0
    gax = cv.axes(kx + 0.1, by, 1.4, bhh)
    gax.imshow(score_colour(np.linspace(m["score_hi"], m["score_lo"], 256))[:, None, :3] * np.ones((1, 2, 1)), aspect="auto",
               extent=(0, 1, 0, 1), interpolation="bilinear")
    gax.set_axis_off()
    for v in (0.9, 0.6, 0.3):
        yv = by + (m["score_hi"] - v) / (m["score_hi"] - m["score_lo"]) * bhh
        cv.line([kx + 1.5, kx + 2.1], [yv, yv], WHITE, lw=0.4)
        tile_text(cv, kx + 2.5, yv, f"{v:.1f}", va="center")
    cv.box(kx + 0.1, by + bhh + 1.3, 1.4, 1.4, m["grey"])
    tile_text(cv, kx + 2.5, by + bhh + 2.0, "no type", va="center")
    # numbered badges centred on the cells shown in h
    ov = bd.mm_axes(cv, 0, BAND, mw, bh)
    for mk in m["marks"]:
        x, y = mk["x_px"] / px_per_mm, mk["y_px"] / px_per_mm
        ov.add_patch(Circle((x, y), BADGE_MM - 0.12, facecolor="none", edgecolor=WHITE, linewidth=0.4))
        tile_text(cv, x, BAND + y + 0.05, str(mk["rank"]), ha="center", va="center")
        sd.add("g", "featured cell", rank=mk["rank"], type=mk["type"], body_id=mk["body_id"], x_mm=round(x, 3), y_mm=round(y, 3))
    tile_text(cv, mw - 1.4, BAND + 37.0, f"{m['cells']:,}" + "\n" + "cells", ha="right", va="top", linespacing=1.05)
    ox, oy = mw - 12.6, bh + BAND - 6.4
    cv.line([ox, ox + 2.8], [oy, oy], WHITE, lw=0.5)
    tile_text(cv, ox + 3.2, oy, "L", va="center")
    cv.line([ox, ox], [oy, oy + 2.8], WHITE, lw=0.5)
    tile_text(cv, ox, oy + 3.2, "V", ha="center", va="top")
    scale_bar(cv, mw - 1.4, BAND + bh - 1.6, 100, m["scale_um_per_px"], px_per_mm)
    sd.add("g", "map", cells=m["cells"], typed_cells=m["typed_cells"], top1pct_cells=m["top1pct_cells"], score_lo=m["score_lo"],
           score_hi=m["score_hi"], um_per_px=m["scale_um_per_px"], scale_bar_um=100,
           unfold="brain frontal; VNC rotated 90 degrees about the neck (msa/cnsview.py)")
    lv = groups["group_level"].value_counts().to_dict()
    sd.add("g", "groups", type_level=int(lv.get("type", 0)), superclass_level=int(lv.get("superclass", 0)))


def panel_h(cv, sd, pct):
    mw, bh = META["map_mm"]
    tw, th = META["tile_mm"]
    for k, t in enumerate(META["tiles"]):
        r, c = divmod(k, 4)
        x, y = mw + fs.GUTTER + c * (tw + fs.GUTTER), BAND + r * (th + fs.GUTTER)
        img = np.asarray(Image.open(TILES / f"{t['name']}.png").convert("RGB"))
        cv.tile(x, y, img, width=tw, height=th, group="fig3band", label=t["name"])
        tx = x + 1.1
        if k == 0:
            cv.letter(x + 1.2, y + 0.9, "h")
            cv.fig.texts[-1].set_color(WHITE)
            tx = x + 4.2
        tile_text(cv, tx, y + 3.2, t["type"], size=fs.PT, va="baseline")
        tile_text(cv, tx + fs.text_width(t["type"], fs.PT) + 0.8, y + 3.2, f"#{t['rank']}", va="baseline")
        nt = NT.get(t["transmitter"], t["transmitter"])
        tile_text(cv, x + 1.1, y + th - 1.0, f"{nt}, {t['n_cells']} cells{ROLE.get(t['membership'], '')}", va="bottom")
        scale_bar(cv, x + tw - 1.2, y + th - 1.5, t["scale_bar_um"], t["scale_um_per_px"], img.shape[1] / tw)
        # per-map percentile strip (tiers separated)
        v = pct[:, t["parameter_group"]]
        sw, tg, sh = 0.46, 0.22, 1.0
        width = 24 * sw + 7 * tg
        sx0, sy0 = x + tw - 1.1 - width, y + 1.3
        ov = bd.mm_axes(cv, sx0, sy0, width, sh)
        for i, val in enumerate(v):
            ov.add_patch(Rectangle((i * sw + (i // 3) * tg, 0), sw, sh, facecolor=score_colour(val), edgecolor="none", linewidth=0.5))
        if k == 0:
            tile_text(cv, sx0 + width / 2, sy0 + sh + 0.5, "24 maps, D0 to D7", ha="center", va="top")
        for mp, val in zip(MAPS, v):
            sd.add("h", "per-map percentile", type=t["type"], map_id=mp, percentile=float(val))
        sd.add("h", "neuron", body_id=t["body_id"], type=t["type"], superclass=t["superclass"], group=t["group"], consistency_rank=t["rank"],
               consistency_score=t["score"], n_cells=t["n_cells"], membership=t["membership"], transmitter=t["transmitter"],
               skeleton=t["skeleton"], scale_bar_um=t["scale_bar_um"])


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("fig3")
    for rel in ("main/fig3/fig3a_training_controls.csv", "main/fig3/fig3b_control_costs.csv", "main/fig3/fig3c_four_combination.csv",
                "main/fig3/fig3d_margin_decomposition.csv", "main/fig3/fig3e_readout_geometry.csv", "supplementary/st4_parameter_change.csv",
                "extended/ed04/ed04a_parameter_change_consistency.csv", "extended/ed04/ed04b_parameter_change_anatomy.csv",
                "figure_data/fd_group_change.csv", "figure_data/fd_neuron_selection.csv", "figure_data/fd_cells.csv",
                "stats/statistical_tests.csv"):
        sd.source(cm.RESULTS / rel)
    panel_a(cv, sd, XS[0], ROW1, U, ROW_H)
    panel_b(cv, sd, XS[1], ROW1, U, ROW_H)
    panel_c(cv, sd, XS[2], ROW1, 2 * U + fs.GAP, ROW_H)
    panel_d(cv, sd, XS[0], ROW2, U, ROW_H)
    panel_e(cv, sd, XS[1], ROW2, 2 * U + fs.GAP, ROW_H)
    panel_f(cv, sd, XS[3], ROW2, U, ROW_H)
    pct, groups = map_percentiles()
    panel_g(cv, sd, groups)
    panel_h(cv, sd, pct)
    cm.finish(cv, "fig3", sd)


if __name__ == "__main__":
    main()
