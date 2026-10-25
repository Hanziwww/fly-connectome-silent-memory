"""Extended Data Fig. 3 - Every learning curve, stage costs, repairs and difficulty descriptors (180 x 112 mm).

a  depth reached vs updates for every training condition on D1, D3, D5 and D7 (tier colours), each over the joint-training
   curves of the same maps (grey); x marks a run that stopped unsolved
b  updates spent per curriculum stage of the 24 difficulty maps, by deadlocking alternatives at the stage entry and push entry
c  repairs during training per map: failed lessons, retention failures and regressions of earlier suffixes or margins
d  Spearman correlations between the eight difficulty descriptors of the 24 maps

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\ed03.py
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, palette as pal, qc  # noqa: E402

ROW1, R1H = 0.0, 62.0
ROW2, R2H = 66.0, 46.0
H = ROW2 + R2H
TIER = [colormaps["viridis"](0.02 + 0.82 * t / 7) for t in range(8)]
MAPS = ["D1", "D3", "D5", "D7"]
CONDITIONS = [("joint", "joint (difficulty runs)"), ("head_only", "readout only"), ("brain_only", "brain only"), ("frozen", "nothing (frozen)"),
              ("demonstration", "demonstration loss"), ("entry_ce", "entry-only loss"), ("alternating", "alternating updates"),
              ("no_recovery", "no recovery states"), ("neutral", "neutral initial gains"), ("rewired", "rewired graph"),
              ("weight_shuffle", "shuffled weights"), ("board_mlp", "board MLP (no brain)")]
EDGE = "#ABABAB"


def curve(d):
    d = d.sort_values("assessment")
    return d["updates"].to_numpy(float), d["depth_fraction"].to_numpy(float)


def run_ids(cond, m):
    if cond == "joint":
        return [f"difficulty-{m}-T0"]
    return [f"control-{m}-{cond}", f"remote-{m}-{cond}"]


# ---------------------------------------------------------------- a: learning curves for every condition

def panel_a(cv, sd):
    a = cm.table("extended/ed03/ed03a_learning_curves_all.csv")
    cv.letter(0.0, ROW1, "a")
    cv.key(3.6, ROW1 + 1.5, [([dict(kind="hline", colour=TIER[int(m[1])], lw=0.9, w=2.6)], m) for m in MAPS] +
           [([dict(kind="hline", colour="#C8C8C8", lw=0.9, w=2.6)], "joint training, same map"),
            ([dict(kind="mark", s=8, marker="x", fc=fs.TEXT)], "stopped unsolved")], gap=1.8)
    cw, ch = fs.COL, 22.4
    for k, (cond, lab) in enumerate(CONDITIONS):
        r, c = divmod(k, 6)
        x0, y0 = fs.col_x(c), ROW1 + 6.0 + r * (ch + 6.4)
        ax = cv.axes(x0 + 6.4, y0 + 2.6, cw - 6.6, ch - 2.6)
        cv.text(x0 + 6.4 + (cw - 6.6) / 2, y0 + 1.6, lab, size=fs.PT_SMALL, ha="center", va="center")
        for m in MAPS:
            col = TIER[int(m[1])]
            if cond != "joint":
                ref = a[a["run_id"] == f"difficulty-{m}-T0"]
                if len(ref):
                    x, y = curve(ref)
                    ax.plot(np.maximum(x, 1), y, color="#C8C8C8", lw=0.6, drawstyle="steps-post", zorder=1)
            for rid in run_ids(cond, m):
                d = a[a["run_id"] == rid]
                if d.empty:
                    continue
                x, y = curve(d)
                solved = bool(d["solved"].astype(str).str.lower().eq("true").any())
                ax.plot(np.maximum(x, 1), y, color=col, lw=0.8, drawstyle="steps-post", zorder=3)
                if not solved:
                    ax.plot([max(x[-1], 1)], [y[-1]], marker="x", ms=2.6, mew=0.6, color=fs.TEXT, ls="none", zorder=4, clip_on=False)
                sd.add("a", "run", condition=cond, map=m, run_id=rid, updates=int(x[-1]), final_depth_fraction=float(y[-1]), solved=solved,
                       assessments=int(len(d)))
        ax.set_xscale("log")
        ax.set_xlim(1, 2e4)
        ax.set_ylim(0, 1.04)
        ax.set_xticks([1, 100, 10000], ["1", "100", "10$^{4}$"])
        ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        fs.minor(ax, y=False)
        fs.clean(ax)
        if c == 0:
            ax.set_ylabel("Depth", labelpad=1)
        else:
            ax.tick_params(axis="y", labelleft=False)
        if r == 1:
            ax.set_xlabel("Updates", labelpad=1)
        if cond == "frozen":
            ax.text(0.5, 0.5, "0 updates;" + "\n" + "no map solved", transform=ax.transAxes, ha="center", va="center", fontsize=fs.PT_SMALL,
                    linespacing=1.05)


# ---------------------------------------------------------------- b: stage costs

def panel_b(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "b")
    b = cm.table("extended/ed03/ed03b_stage_costs_all.csv")
    b = b[b["run_id"].str.startswith("difficulty-")].copy()
    b["push"] = b["push_entry"].astype(str).str.lower().eq("true")
    b["dl"] = b["deadlock_alternatives_at_entry"].fillna(0).astype(int)
    assert set(b["dl"].unique()) <= {0, 1}
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="mark", s=4, fc=fs.TEXT)], "push entry"), ([dict(kind="mark", s=4, fc="white", ec="#7A7A7A", lw=0.4)],
                                                                                  "other entry"), ([dict(kind="hline", w=2.4, lw=0.9, colour=pal.ACCENT)], "median")], gap=1.6)
    ax = cv.axes(x0 + 8.4, y0 + 6.8, w - 8.8, h - 14.6)
    rng = np.random.default_rng(9)
    for k in range(2):
        for push, off in ((True, -0.18), (False, 0.18)):
            v = b.loc[(b["dl"] == k) & (b["push"] == push), "updates_spent"].to_numpy(float)
            if not len(v):
                continue
            vv = np.maximum(v, 0.5)
            ax.scatter(k + off + (rng.random(len(v)) - 0.5) * 0.24, vv, s=1.8, facecolor=fs.TEXT if push else "white",
                       edgecolor=fs.TEXT if push else "#7A7A7A", linewidth=0.3, zorder=2)
            ax.plot([k + off - 0.14, k + off + 0.14], [max(np.median(v), 0.5)] * 2, color=pal.ACCENT, lw=0.9, zorder=3)
            sd.add("b", "stages", deadlock_alternatives=k, push_entry=push, stages=int(len(v)),
                   median_updates=float(np.median(v)), zero_update_share=float((v == 0).mean()))
    ax.set_yscale("log")
    ax.set_ylim(0.4, 2000)
    ax.set_yticks([0.5, 1, 10, 100, 1000], ["0", "1", "10", "100", "1,000"])
    fs.minor(ax, x=False)
    ax.set_xticks(range(2), ["none", "one"])
    ax.set_xlim(-0.6, 1.6)
    fs.clean(ax)
    ax.set_ylabel("Updates per stage", labelpad=1)
    ax.set_xlabel("Deadlocking alternatives at the stage entry", labelpad=1)
    cv.text(x0 + 8.4, y0 + h - 1.6, f"{len(b):,} stages of 24 maps", size=fs.PT_SMALL, va="center")


# ---------------------------------------------------------------- c: repairs per map

REPAIRS = [("lesson_failures", "failed lessons", "#4D4D4D"), ("retention_failures", "retention failures", "#9A9A9A"),
           ("regressions_suffix", "suffix regressions", pal.ACCENT), ("regressions_margin", "margin regressions", "#F2A3C0")]


def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    b = cm.table("extended/ed03/ed03b_stage_costs_all.csv")
    b = b[b["run_id"].str.startswith("difficulty-")]
    tot = b.groupby("map_id")[[k for k, _, _ in REPAIRS]].sum()
    maps = sorted(tot.index, key=lambda m: (int(m[1]), m))
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="box", fc=c, size=1.3)], lab) for _, lab, c in REPAIRS[:2]], gap=1.6)
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="box", fc=c, size=1.3)], lab) for _, lab, c in REPAIRS[2:]], gap=1.6)
    ax = cv.axes(x0 + 8.4, y0 + 6.8, w - 8.8, h - 14.6)
    for i, m in enumerate(maps):
        bottom = 0
        for k, lab, col in REPAIRS:
            v = float(tot.loc[m, k])
            ax.bar(i, v, bottom=bottom, width=0.72, color=col, linewidth=0)
            bottom += v
        sd.add("c", "map", map_id=m, **{k: int(tot.loc[m, k]) for k, _, _ in REPAIRS})
    for i, m in enumerate(maps):
        ax.add_patch(bd.Rectangle((i - 0.36, -0.06), 0.72, 0.035, transform=ax.get_xaxis_transform(), facecolor=TIER[int(m[1])], edgecolor="none",
                                  linewidth=0.5, clip_on=False))
    ax.set_xlim(-0.7, len(maps) - 0.3)
    ax.set_xticks([])
    fs.minor(ax, x=False)
    fs.clean(ax)
    ax.set_ylabel("Events per map", labelpad=1)
    ax.set_xlabel("24 maps, D0 to D7 (tier strip)", labelpad=6.5)


# ---------------------------------------------------------------- d: descriptor correlations

DESC = [("optimal_moves", "optimal moves"), ("tier", "tier"), ("board_size", "board size"), ("floor_cells", "floor cells"), ("boxes", "boxes"),
        ("shortest_dag_states", "solution-DAG states"), ("witness_pushes", "witness pushes"), ("deadlocking_alternatives_total", "deadlocking alternatives")]


def panel_d(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "d")
    c = cm.table("extended/ed03/ed03c_difficulty_descriptors.csv")
    M = c.pivot(index="descriptor_a", columns="descriptor_b", values="spearman_rho")
    keys = [k for k, _ in DESC]
    A = M.loc[keys, keys].to_numpy(float)
    n = len(keys)
    lab_w = 21.0
    s = min(w - lab_w - 0.6, h - 9.0)
    cell = s / n
    ax = bd.mm_axes(cv, x0, y0, w, h)
    cmap = colormaps["Greys"]
    top = 3.4
    for i in range(n):
        ax.text(lab_w - 0.6, top + (i + 0.5) * cell, DESC[i][1], ha="right", va="center", fontsize=fs.PT_SMALL)
        for j in range(n):
            if j > i:
                continue
            v = A[i, j]
            ax.add_patch(bd.Rectangle((lab_w + j * cell, top + i * cell), cell - 0.15, cell - 0.15, facecolor=cmap(0.04 + 0.5 * abs(v)),
                                      edgecolor="none", linewidth=0.5))
            if i != j:
                ax.text(lab_w + (j + 0.5) * cell, top + (i + 0.5) * cell, f"{v:.2f}".replace("0.", ".").replace("-", fs.MINUS), ha="center",
                        va="center", fontsize=fs.PT_SMALL)
                sd.add("d", "rho", a=keys[i], b=keys[j], spearman=float(v), maps=int(c["n"].iloc[0]))
    ax.text(lab_w, top + n * cell + 1.6, f"Spearman {fs.RHO} over {int(c['n'].iloc[0])} maps", ha="left", va="top", fontsize=fs.PT_SMALL)


def main():
    cv = fs.Canvas(fs.FULL, H, ppi_min=300)
    sd = qc.SourceData("ed03")
    for rel in ("extended/ed03/ed03a_learning_curves_all.csv", "extended/ed03/ed03b_stage_costs_all.csv",
                "extended/ed03/ed03c_difficulty_descriptors.csv"):
        sd.source(cm.RESULTS / rel)
    panel_a(cv, sd)
    panel_b(cv, sd, fs.col_x(0), ROW2, fs.span(2), R2H)
    panel_c(cv, sd, fs.col_x(2), ROW2, fs.span(2), R2H)
    panel_d(cv, sd, fs.col_x(4), ROW2, fs.span(2), R2H)
    cm.finish(cv, "ed03", sd, kind="extended")


if __name__ == "__main__":
    main()
