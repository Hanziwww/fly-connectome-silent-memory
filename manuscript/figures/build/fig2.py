"""Fig. 2 - Whole-CNS training across task difficulty (180 x 152 mm), dense layout.

a  curriculum depth against parameter updates for all 24 maps, coloured by tier; board icons key the tiers
b  cost against certified optimal moves on log-log axes (updates; simulated neuron-ms): Theil-Sen slopes with CI, rho
c  stage barcodes: every curriculum stage of every map coloured by the updates it took; push entries and
   deadlocking alternatives marked
d  per-map push excess share and rho(stage updates, deadlocking alternatives) with Hodges-Lehmann estimates
e  what predicts cost: Spearman rho of updates with each task descriptor (Holm)
f  outcomes: every map solved and verified after reload; no zero-update transfer to held-out maps

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\fig2.py
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, palette as pal, qc  # noqa: E402

H = 152.0
ROW2 = 63.0
TIER = [colormaps["viridis"](0.02 + 0.82 * t / 7) for t in range(8)]
STAGE_CMAP = colormaps["magma_r"]
STAGE_NORM = LogNorm(1, 400)


def tier_of(map_id):
    return int(map_id[1])


# ---------------------------------------------------------------- a: learning curves

def panel_a(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "a")
    b = cm.table("main/fig2/fig2b_learning_curves.csv")
    boards_t = cm.table("figure_data/fd_boards.csv")
    ax = cv.axes(x0 + 9.5, y0 + 3.6, w - 33.0, h - 11.0)
    for run, g in b.groupby("run_id", sort=True):
        g = g.sort_values("assessment")
        m = g["map_id"].iloc[0]
        col = TIER[tier_of(m)]
        x = g["updates"].to_numpy()
        y = g["depth_fraction"].to_numpy()
        intr = bool(g["interrupted"].astype(str).str.lower().eq("true").any())
        ax.step(x, y, where="post", color=col, lw=0.65, alpha=0.95, ls=(0, (2.4, 1.0)) if intr else "-")
        p = g[g["stage_passed"].astype(str).str.lower().eq("true")]
        ax.scatter(p["updates"], p["depth_fraction"], s=1.6, color=col, linewidth=0, zorder=3)
        last = g.iloc[-1]
        ax.scatter([last["updates"]], [last["depth_fraction"]], s=9, facecolor=col, edgecolor="white", linewidth=0.3, zorder=4)
        for _, r in g.iterrows():
            sd.add("a", "assessment", run_id=run, map_id=m, updates=int(r["updates"]), depth_fraction=float(r["depth_fraction"]),
                   stage_passed=str(r["stage_passed"]).lower() == "true", interrupted=intr)
    ax.set_xscale("symlog", linthresh=10, linscale=0.35)
    ax.set_xlim(0, 2600)
    ax.set_xticks([0, 10, 100, 1000], ["0", "10", "100", "1,000"])
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel("Parameter updates", labelpad=1)
    ax.set_ylabel("Curriculum depth (fraction of solution)")
    fs.clean(ax)
    ax.text(0.99, 0.03, "24 maps; dots: stages passed;" + chr(10) + "dashed: interrupted and resumed runs", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=fs.PT_SMALL, linespacing=1.0)
    # tier key with the T0 board of each tier
    kx = x0 + w - 21.0
    cell = 0.40
    ax_b = bd.mm_axes(cv, kx, y0 + 3.6, 21.0, h - 11.0)
    pitch = (h - 11.0) / 8
    for t in range(8):
        yy = t * pitch
        brd = bd.Board.from_table(boards_t, f"D{t}-T0")
        bw_, bh_ = brd.size_mm(cell)
        bd.draw(ax_b, brd, 0.2 + (4.0 - bw_) / 2, yy + (pitch - bh_) / 2, cell, lw=0.15)
        ax_b.add_patch(Rectangle((5.0, yy + pitch / 2 - 0.35), 2.6, 0.7, facecolor=TIER[t], edgecolor="none"))
        lib = cm.table("main/fig1/fig1d_task_library.csv")
        lt = lib[(lib["tier"] == t) & lib["split"].eq("train")]
        ax_b.text(8.3, yy + pitch / 2, f"{t}   {int(lt['optimal_moves'].min())}{fs.MINUS}{int(lt['optimal_moves'].max())}",
                  ha="left", va="center", fontsize=fs.PT_SMALL)
    ax_b.text(8.3, -1.0, "tier, optimal moves", ha="left", va="bottom", fontsize=fs.PT_SMALL)


# ---------------------------------------------------------------- b: scaling

def panel_b(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "b")
    c = cm.table("main/fig2/fig2c_cost_vs_difficulty.csv")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    specs = [("updates", "Parameter updates", "fig2c.theilsen.log10_updates~log10_optimum", "fig2c.spearman.updates~optimum"),
             ("neuron_ms", "Simulated neuron-ms", "fig2c.theilsen.log10_neuron_ms~log10_optimum", "fig2c.spearman.neuron_ms~optimum")]
    ah = (h - 15.0) / 2
    for k, (col, lab, tid, rid) in enumerate(specs):
        ax = cv.axes(x0 + 10.0, y0 + 3.6 + k * (ah + 6.6), w - 11.0, ah)
        x = c["optimal_moves"].to_numpy(float)
        y = c[col].to_numpy(float)
        intr = c["interrupted"].astype(str).str.lower().eq("true").to_numpy()
        lx, ly = np.log10(x), np.log10(y)
        t = tests.loc[tid]
        slope, lo, hi = float(t["effect"]), float(t["effect_ci_low"]), float(t["effect_ci_high"])
        icept = float(np.median(ly - slope * lx))
        gx = np.linspace(lx.min() - 0.03, lx.max() + 0.03, 50)
        mx, my = float(np.median(lx)), float(np.median(ly))
        band_lo = np.minimum(my + lo * (gx - mx), my + hi * (gx - mx))
        band_hi = np.maximum(my + lo * (gx - mx), my + hi * (gx - mx))
        ax.fill_between(10 ** gx, 10 ** band_lo, 10 ** band_hi, color="#D9D9D9", lw=0, zorder=1)
        ax.plot(10 ** gx, 10 ** (icept + slope * gx), color=fs.TEXT, lw=0.7, zorder=2)
        for xi, yi, mi, ii in zip(x, y, c["map_id"], intr):
            colr = TIER[tier_of(mi)]
            ax.scatter([xi], [yi], s=10, facecolor="white" if ii else colr, edgecolor=colr, linewidth=0.7, zorder=3)
            sd.add("b", col, map_id=mi, optimal_moves=float(xi), value=float(yi), interrupted=bool(ii))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(8, 70)
        ax.set_xticks([10, 20, 40], ["10", "20", "40"])
        fs.minor(ax)
        fs.clean(ax)
        ax.set_ylabel(lab)
        if k == 1:
            ax.set_xlabel("Certified optimal moves", labelpad=1)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        rho = float(tests.loc[rid, "effect"])
        ax.text(0.03, 0.97, f"slope {slope:.2f} [{lo:.2f}, {hi:.2f}]", transform=ax.transAxes, ha="left", va="top", fontsize=fs.PT_SMALL)
        ax.text(0.03, 0.83, f"{fs.RHO} = {rho:.2f}, {fs.p_text(float(tests.loc[rid, 'p_adjusted']))}", transform=ax.transAxes, ha="left",
                va="top", fontsize=fs.PT_SMALL)
        sd.add("b", "fit", measure=col, theil_sen_slope=slope, ci_low=lo, ci_high=hi, intercept=icept, rho=rho, test_id=tid)


# ---------------------------------------------------------------- c: stage barcodes

def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    d = cm.table("main/fig2/fig2d_stage_costs.csv")
    boards_t = cm.table("figure_data/fd_boards.csv")
    maps = sorted(d["map_id"].unique(), key=lambda m: (tier_of(m), m))
    L = int(d["stage"].max())
    icon_w, lab_w, right_w = 4.2, 12.6, 15.0
    gx0 = lab_w + icon_w + 1.0
    cw = (w - gx0 - right_w) / L
    top = 6.2
    rh = (h - top - 1.0) / len(maps)
    ax = bd.mm_axes(cv, x0, y0, w, h)
    for s in (1, 10, 20, 30, 40, 50, 60):
        ax.text(gx0 + (s - 0.5) * cw, top - 0.8, str(s), ha="center", va="bottom", fontsize=fs.PT_SMALL)
    ax.text(gx0, top - 3.3, "curriculum stage (moves before the goal)", ha="left", va="bottom", fontsize=fs.PT_SMALL)
    ax.text(w - right_w + 1.0, top - 0.8, "updates", ha="left", va="bottom", fontsize=fs.PT_SMALL)
    for i, m in enumerate(maps):
        yy = top + i * rh
        g = d[d["map_id"] == m].sort_values("stage")
        ax.add_patch(Rectangle((lab_w - 1.6, yy + 0.25), 1.0, rh - 0.5, facecolor=TIER[tier_of(m)], edgecolor="none"))
        ax.text(lab_w - 2.2, yy + rh / 2, m, ha="right", va="center", fontsize=fs.PT_SMALL)
        brd = bd.Board.from_table(boards_t, m)
        cell = min(icon_w, rh - 0.3) / 10
        bw_, bh_ = brd.size_mm(cell)
        bd.draw(ax, brd, lab_w + (icon_w - bw_) / 2, yy + (rh - bh_) / 2, cell, lw=0.1)
        for _, r in g.iterrows():
            s = int(r["stage"])
            u = int(r["updates_spent"])
            xx = gx0 + (s - 1) * cw
            fc = STAGE_CMAP(0.06 + 0.88 * STAGE_NORM(u)) if u > 0 else "#ECECEC"
            ax.add_patch(Rectangle((xx + 0.05, yy + 0.18), cw - 0.1, rh - 0.36, facecolor=fc, edgecolor="none"))
            if str(r["push_entry"]).lower() == "true":
                ax.add_patch(Rectangle((xx + 0.05, yy + rh - 0.42), cw - 0.1, 0.32, facecolor=fs.TEXT, edgecolor="none"))
            dl = r["deadlock_alternatives_at_entry"]
            if dl == dl and dl > 0:
                ax.add_patch(plt_dot(xx + cw / 2, yy + rh * 0.42, 0.22 + 0.1 * min(dl, 3)))
            sd.add("c", "stage", map_id=m, stage=s, updates=u, push_entry=str(r["push_entry"]).lower() == "true",
                   deadlock_alternatives=None if dl != dl else float(dl))
        tot = int(g["updates_spent"].sum())
        bx = w - right_w + 1.0
        ax.add_patch(Rectangle((bx, yy + 0.4), 7.5 * np.log10(max(tot, 1)) / np.log10(2500), rh - 0.8, facecolor=TIER[tier_of(m)], edgecolor="none"))
        ax.text(bx + 7.5 * np.log10(max(tot, 1)) / np.log10(2500) + 0.5, yy + rh / 2, f"{tot:,}", ha="left", va="center", fontsize=fs.PT_SMALL)
    # key, in the empty area above the short rows
    ky = top + 1.6 * rh
    kx0 = gx0 + 27 * cw
    kax = cv.axes(x0 + kx0, y0 + ky - 0.7, 16.0, 1.4)
    kax.imshow(STAGE_CMAP(0.06 + 0.88 * STAGE_NORM(np.logspace(0, np.log10(400), 128)))[None, :, :3], aspect="auto", extent=(0, 1, 0, 1))
    kax.set_yticks([])
    kax.set_xticks([0, (np.log10(10)) / np.log10(400), np.log10(100) / np.log10(400), 1], ["1", "10", "100", "400"])
    kax.tick_params(axis="x", length=1.0, pad=0.4, labelsize=fs.PT_SMALL)
    for side in ("top", "right", "left"):
        kax.spines[side].set_visible(False)
    cv.text(x0 + kx0 + 17.0, y0 + ky, "updates in the stage (grey: none)", size=fs.PT_SMALL, va="center")
    kx = x0 + kx0
    cv.box(kx, y0 + ky + 4.5, 1.4, 0.35, fs.TEXT)
    cv.text(kx + 2.0, y0 + ky + 4.4, "stage entry pushes a box", size=fs.PT_SMALL, va="center")
    cv.mark(kx + 0.7, y0 + ky + 7.0, s=4.0, fc="white", ec=fs.TEXT, lw=0.4)
    cv.text(kx + 2.0, y0 + ky + 7.0, "deadlocking alternatives at the stage entry", size=fs.PT_SMALL, va="center")


def plt_dot(x, y, r):
    from matplotlib.patches import Circle
    return Circle((x, y), r, facecolor="white", edgecolor=fs.TEXT, linewidth=0.25, zorder=4)


# ---------------------------------------------------------------- d: per-map stage statistics

def per_map(d):
    rows = []
    for m, g in d.groupby("map_id"):
        u = g["updates_spent"].to_numpy(float)
        push = g["push_entry"].astype(str).str.lower().eq("true").to_numpy()
        share = (u[push].sum() / u.sum() if u.sum() > 0 else 0.0) - push.mean()
        dl = g["deadlock_alternatives_at_entry"].to_numpy(float)
        rho = spearmanr(u, dl).statistic if np.nanstd(dl) > 0 else np.nan
        rows.append((m, share, rho))
    return rows


def panel_d(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "d")
    d = cm.table("main/fig2/fig2d_stage_costs.csv")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    rows = per_map(d)
    shares = np.array([r[1] for r in rows])
    rhos = np.array([r[2] for r in rows if np.isfinite(r[2])])
    assert int((shares < 0).sum()) == int(tests.loc["fig2d.sign.push_excess_share", "n_negative"])
    assert int((rhos > 0).sum()) == int(tests.loc["fig2d.sign.rho_deadlock", "n_positive"])
    specs = [(shares, "Push excess share", "fig2d.wilcoxon.push_excess_share", "fig2d.sign.push_excess_share", (-0.35, 0.2)),
             (rhos, f"{fs.RHO}(updates, deadlocks)", "fig2d.wilcoxon.rho_deadlock", "fig2d.sign.rho_deadlock", (-0.6, 1.0))]
    aw = (w - 4.0) / 2
    for k, (vals, lab, hl_id, sign_id, lim) in enumerate(specs):
        ax = cv.axes(x0 + 6.0 + k * (aw + 3.0), y0 + 4.0, aw - 5.0, h - 10.0)
        pairs = sorted(((r[1] if k == 0 else r[2]), tier_of(r[0])) for r in rows if k == 0 or np.isfinite(r[2]))
        order = np.array([p[0] for p in pairs])
        cols = [TIER[p[1]] for p in pairs]
        ax.scatter(np.arange(len(order)), order, s=8, c=cols, linewidth=0, zorder=3)
        t = tests.loc[hl_id]
        ax.axhspan(float(t["effect_ci_low"]), float(t["effect_ci_high"]), color="#D9D9D9", lw=0, zorder=0)
        ax.axhline(float(t["effect"]), color=fs.TEXT, lw=0.7, zorder=1)
        ax.axhline(0, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=1)
        ax.set_ylim(*lim)
        ax.set_xlim(-1, len(order))
        ax.set_xticks([])
        fs.clean(ax, bottom=False)
        ax.set_ylabel(lab, labelpad=1)
        s_ = tests.loc[sign_id]
        neg, pos = int(s_["n_negative"]), int(s_["n_positive"])
        ax.text(0.03, 0.97, f"{neg if k == 0 else pos} of {neg + pos} {'below' if k == 0 else 'above'} 0", transform=ax.transAxes, ha="left",
                va="top", fontsize=fs.PT_SMALL)
        ax.text(0.03, 0.86, f"HL {fs.num(float(t['effect']), 2)}, {fs.p_text(float(t['p_adjusted']))}", transform=ax.transAxes, ha="left",
                va="top", fontsize=fs.PT_SMALL)
        ax.text(0.5, -0.04, "maps, sorted", transform=ax.transAxes, ha="center", va="top", fontsize=fs.PT_SMALL)
        for v in vals:
            sd.add("d", lab, value=float(v))
        sd.add("d", "tests", measure=lab, hl=float(t["effect"]), ci_low=float(t["effect_ci_low"]), ci_high=float(t["effect_ci_high"]),
               p_holm=float(t["p_adjusted"]), test_id=hl_id)


# ---------------------------------------------------------------- e: descriptors

def panel_e(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "e")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    desc = [("optimum", "optimal moves"), ("tier", "tier"), ("board_size", "board size"), ("floor_cells", "floor cells"),
            ("boxes", "boxes"), ("witness_pushes", "pushes in solution"), ("deadlocking_alternatives_total", "deadlocking alternatives")]
    ax = cv.axes(x0 + 26.0, y0 + 4.0, w - 28.0, h - 9.0)
    for i, (k, lab) in enumerate(desc):
        t = tests.loc[f"fig2c.spearman.updates~{k}"]
        sig = str(t["significant_adjusted"]).lower() == "true"
        fs.interval(ax, i, float(t["effect"]), float(t["effect_ci_low"]), float(t["effect_ci_high"]), orient="h",
                    fc=fs.TEXT if sig else "white", ms=10)
        sd.add("e", "spearman", descriptor=k, rho=float(t["effect"]), ci_low=float(t["effect_ci_low"]), ci_high=float(t["effect_ci_high"]),
               p_holm=float(t["p_adjusted"]), test_id=f"fig2c.spearman.updates~{k}")
    ax.set_yticks(range(len(desc)), [d_[1] for d_ in desc])
    ax.set_ylim(len(desc) - 0.5, -0.5)
    ax.set_xlim(-0.4, 1.0)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.axvline(0, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)))
    fs.clean(ax)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"Spearman {fs.RHO} with updates", labelpad=1)
    ax.text(-0.38, -0.45, "open: not significant", ha="left", va="top", fontsize=fs.PT_SMALL)


# ---------------------------------------------------------------- f: outcomes and transfer

def panel_f(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "f")
    a = cm.table("main/fig2/fig2a_acquisition_outcomes.csv")
    e = cm.table("main/fig2/fig2e_zero_update_transfer.csv")
    ax = bd.mm_axes(cv, x0, y0, w, h)
    s = 1.55
    n = int(a["reload_solved_count"].iloc[0])
    ci = (float(a["reload_solved_ci_low"].iloc[0]), float(a["reload_solved_ci_high"].iloc[0]))
    ax.text(3.6, 1.0, f"trained maps solved after reload: {n} of {len(a)} (95% CI {ci[0]:.2f}{fs.MINUS}{ci[1]:.2f})", ha="left", va="top",
            fontsize=fs.PT_SMALL)
    for i, (_, r) in enumerate(a.sort_values(["tier", "map_id"]).iterrows()):
        ok = str(r["reload_verified"]).lower() == "true"
        ax.add_patch(Rectangle((3.6 + i * (s + 0.25), 4.0), s, s, facecolor=TIER[int(r["tier"])] if ok else "white", edgecolor="none"))
        sd.add("f", "trained map", map_id=r["map_id"], reload_verified=ok)
    y = 8.2
    ax.text(3.6, y, "zero-update transfer (new board, no further training):", ha="left", va="top", fontsize=fs.PT_SMALL)
    rec = e[e["row_type"] == "record"]
    width = {sp: int(rec[rec["split"] == sp].groupby("model").size().max()) for sp in ("heldout", "new_start", "transform")}
    for j, (mdl, g) in enumerate(rec.groupby("model", sort=True)):
        yy = y + 4.8 + j * 2.3
        ax.text(3.6, yy + s / 2, mdl.replace("difficulty-", ""), ha="left", va="center", fontsize=fs.PT_SMALL)
        xx = 13.0
        for split, lab in (("heldout", "held-out"), ("new_start", "new start"), ("transform", "rotated")):
            gg = g[g["split"] == split].sort_values("task")
            for k, (_, r) in enumerate(gg.iterrows()):
                ok = str(r["solved"]).lower() == "true"
                ax.add_patch(Rectangle((xx + k * (s * 0.8 + 0.2), yy + 0.1), s * 0.8, s * 0.8, facecolor=fs.TEXT if ok else "white",
                                       edgecolor="#8C8C8C", linewidth=0.3))
                sd.add("f", "transfer", model=mdl, split=split, task=r["task"], solved=ok)
            if j == 0:
                ax.text(xx, y + 4.4, lab, ha="left", va="bottom", fontsize=fs.PT_SMALL)
            xx += width[split] * (s * 0.8 + 0.2) + 2.4
    tot = rec.groupby("split")["solved"].apply(lambda v: (int(v.astype(str).str.lower().eq("true").sum()), len(v)))
    ax.text(3.6, h - 0.3, f"solved: held-out {tot['heldout'][0]} of {tot['heldout'][1]}, new starts {tot['new_start'][0]} of "
            f"{tot['new_start'][1]}, rotations {tot['transform'][0]} of {tot['transform'][1]}", ha="left", va="bottom", fontsize=fs.PT_SMALL)


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("fig2")
    for rel in ("main/fig2/fig2a_acquisition_outcomes.csv", "main/fig2/fig2b_learning_curves.csv", "main/fig2/fig2c_cost_vs_difficulty.csv",
                "main/fig2/fig2d_stage_costs.csv", "main/fig2/fig2e_zero_update_transfer.csv", "figure_data/fd_boards.csv",
                "main/fig1/fig1d_task_library.csv", "stats/statistical_tests.csv"):
        sd.source(cm.RESULTS / rel)
    panel_a(cv, sd, 0.0, 0.0, 118.0, ROW2 - 3.0)
    panel_b(cv, sd, 121.0, 0.0, 59.0, ROW2 - 3.0)
    panel_c(cv, sd, 0.0, ROW2, 118.0, H - ROW2)
    panel_d(cv, sd, 121.0, ROW2, 59.0, 30.0)
    panel_e(cv, sd, 121.0, ROW2 + 32.0, 59.0, 28.0)
    panel_f(cv, sd, 121.0, ROW2 + 62.0, 59.0, H - ROW2 - 62.0)
    cm.finish(cv, "fig2", sd)


if __name__ == "__main__":
    main()
