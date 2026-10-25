"""v4 Figure S3 (Cell), related to Figure 3 - No circuit is special, in every scene and under every connection rule (174 mm).

A  dark band: the scene-2 cue ensemble (top 1% of cells by |cue difference|) with its internal connections beside three of its
   20 matched random sets drawn by the same rule (as Fig. 3A for scene 1)
B  what the ensembles are made of: cell groups of the scene-1 and scene-2 ensembles (the matched sets share them by design)
   against all cells
C  silencing the ensemble or cutting its synapses changes no more trained histories than a matched set: histories solved
   relative to the matched mean, by the ensemble's scene and the scene tested
D  removing whole populations is an exact no-op: histories solved, network rate and the giant fibre's looming response
E  the healthy-circuit signatures of the disease study hold under every connection rule

Every number in a label is read from manuscript/results (plan_v2 via P12, followup, figure_data) or from the render metadata.
Reader-facing names only. Not shown (class c): the matched random-cell removals and the descending-neuron control (D5), the
mushroom-body signature at same-type gain 1 (D9), fu5's type-matched wiring null (it shares 41% of the ensemble's cells). The sign
coupling by tuning rank (fu8) is left out: it is strongest in the most tuned cells in four of five state sets but not in scene 3,
context 1.

Run after P12 and render/v4_s3_tiles.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\s3.py
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
TILES = cm.RENDERS / "v4" / "s3"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
BAND_H = META["band_mm"]
TW = META["tile_mm"][0]
U = fs.unit(1)
XS = [fs.unit_x(i) for i in range(4)]
ROW2, R_H = BAND_H + 4.2, 46.0
ROW3 = ROW2 + R_H + 3.4
H = ROW3 + R_H
WHITE = fs.ON_DARK
NL_ = chr(10)
TARGET, MATCHED = fs.TEXT, "#B5B5B5"
COMBOS = [("scene 1", "scene 1"), ("scene 1", "scene 2"), ("scene 2", "scene 2"), ("scene 2", "scene 1")]
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


def scene_no(s):
    return s.split()[-1]


# ---------------------------------------------------------------- A: the scene-2 ensemble and its matched sets

def panel_band(cv, sd):
    wiring = cm.table("plan_v2/fig3_set_wiring.csv")
    wiring = wiring[wiring["ensemble_scene"] == "scene 2"].set_index("set")
    titles = {"ensemble": "Cue ensemble, scene 2", "matched 1": "Matched set 1", "matched 2": "Matched set 2", "matched 3": "Matched set 3"}
    xs = [k * (TW + fs.GUTTER) for k in range(4)]
    for x, t in zip(xs, META["sets"]):
        img = np.asarray(Image.open(TILES / f"{t['name']}.png").convert("RGB"))
        cv.tile(x, 0, img, width=TW, height=BAND_H, group="band", label=t["name"])
        tile_text(cv, x + (4.4 if x == 0 else 1.2), 1.0, titles[t["set"]], va="top", size=fs.PT)
        n = int(wiring.loc[t["set"], "internal_connections"])
        tile_text(cv, x + 1.0, BAND_H - 8.6, f"{t['cells']:,} cells", va="center")
        tile_text(cv, x + 1.0, BAND_H - 6.2, f"{n:,}", va="center")
        tile_text(cv, x + 1.0, BAND_H - 3.8, "connections", va="center")
        sd.add("A", "set", set=t["set"], cells=int(t["cells"]), located=int(t["located"]), internal_connections=n,
               depolarized_more_by_cue0=int(t["cue0"]), by_cue1=int(t["cue1"]), no_difference=int(t["no_difference"]))
    white_letter(cv, 1.0, 0.8, "A")
    kx, ky = xs[0] + TW - 11.6, BAND_H - 11.0
    for i, (c, lab) in enumerate(((pal.CUE_ON_DARK[0], "cue 0"), (pal.CUE_ON_DARK[1], "cue 1"))):
        cv.mark(kx + 0.7, ky + 2.4 * i, s=6.0, fc=c, ec=c, lw=0.3)
        tile_text(cv, kx + 1.6, ky + 2.4 * i, lab, va="center")
    tile_text(cv, kx, ky - 5.0, "depolarized", va="center")
    tile_text(cv, kx, ky - 2.6, "more by", va="center")
    t = META["sets"][-1]
    length = 100 / t["scale_um_per_px"] / (np.asarray(Image.open(TILES / f"{t['name']}.png")).shape[1] / TW)
    cv.line([xs[3] + TW - 1.4 - length, xs[3] + TW - 1.4], [BAND_H - 1.8, BAND_H - 1.8], WHITE, lw=1.0, solid_capstyle="butt")
    e = wiring.loc["ensemble", "internal_connections"]
    m = wiring.drop(index="ensemble")["internal_connections"]
    sd.add("A", "view", scale_bar_um=100, um_per_px=t["scale_um_per_px"], wiring_ratio=float(e / m.mean()))


# ---------------------------------------------------------------- B: what the ensembles are made of

def panel_composition(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "B", "Made of the same groups")
    cells = cm.table("figure_data/fd_cells.csv")
    grp = cells.set_index("cell")["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other")
    s1 = cm.table("plan_v2/fig3_band_sets.csv")
    s2 = cm.table("plan_v2/s3_band_sets.csv")
    bars = [("ensemble, scene 1", grp.loc[s1[s1["set"] == "ensemble"]["cell"]]), ("ensemble, scene 2", grp.loc[s2[s2["set"] == "ensemble"]["cell"]]),
            ("all cells", grp)]
    lab_w = 14.0
    ax = cv.axes(x0 + lab_w, y0 + 8.4, w - lab_w - 1.0, h - 21.6)
    for i, (lab, g) in enumerate(bars):
        share = g.value_counts(normalize=True)
        left = 0.0
        for k in GROUP_ORDER:
            v = float(share.get(k, 0.0))
            if v > 0:
                ax.add_patch(Rectangle((left, i - 0.32), v, 0.64, facecolor=pal.on_white(pal.ATLAS_ON_DARK[k]), edgecolor="white", linewidth=0.3))
                if v >= 0.12:
                    ax.text(left + v / 2, i, f"{100 * v:.0f}", fontsize=fs.PT_SMALL, ha="center", va="center", color=fs.TEXT)
            left += v
            sd.add("B", "composition", set=lab, group=k, share=v)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(bars) - 0.5, -0.5)
    ax.set_yticks(range(len(bars)), [b[0].replace(", ", f",{NL_}") for b in bars], linespacing=1.0)
    ax.set_xticks([0, 0.5, 1], ["0", "50", "100% of cells"])
    ax.tick_params(axis="y", length=0)
    fs.clean(ax)
    ax.spines["left"].set_visible(False)
    present = [k for k in GROUP_ORDER if any(float(g.value_counts(normalize=True).get(k, 0)) > 0.005 for _, g in bars)]
    rows = [present[:3], present[3:6], present[6:]]
    for r, items in enumerate(rows):
        if items:
            cv.key(x0 + 1.0, y0 + h - 7.8 + 2.6 * r, [([dict(kind="box", fc=pal.on_white(pal.ATLAS_ON_DARK[k]))], pal.SHORT_GROUP[k]) for k in items],
                   size=fs.PT_SMALL, gap=1.2)
    cv.text(x0 + w - 0.8, y0 + 6.0, "matched sets: same groups", size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- C: histories solved relative to matched sets

def panel_solved(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "C", "No more histories change than with a matched set")
    d = cm.table("plan_v2/fig3_ensemble_sets.csv")
    halves = [("x6", "silencing it at the decision"), ("x7", "cutting its internal synapses")]
    hw = (w - 9.0 - 3.0) / 2
    rng = np.random.default_rng(301)
    for j, (kind, lab) in enumerate(halves):
        ax = cv.axes(x0 + 9.0 + j * (hw + 3.0), y0 + 10.4, hw, h - 19.6)
        q0 = d[d["kind"] == kind]
        for i, (es, ts) in enumerate(COMBOS):
            q = q0[(q0["ensemble_scene"] == es) & (q0["test_scene"] == ts)]
            m = q[q["set"] != "ensemble"]["solved"].to_numpy(float)
            e = float(q.loc[q["set"] == "ensemble", "solved"].iloc[0])
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(m)), m - m.mean(), s=4, facecolor=MATCHED, edgecolor="none", zorder=2)
            ax.scatter([i], [e - m.mean()], s=14, facecolor=TARGET, edgecolor=TARGET, zorder=3)
            sd.add("C", kind, ensemble_scene=es, test_scene=ts, ensemble_solved=e, matched_mean_solved=float(m.mean()), histories=int(q["histories"].iloc[0]))
        ax.axhline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=0)
        ax.set_ylim(-5.5, 5.5)
        ax.set_yticks([-4, 0, 4], [f"{fs.MINUS}4", "0", "+4"] if j == 0 else ["", "", ""])
        ax.set_xlim(-0.6, len(COMBOS) - 0.4)
        ax.set_xticks(range(len(COMBOS)), [f"{scene_no(es)}→{scene_no(ts)}" for es, ts in COMBOS])
        ax.tick_params(axis="x", length=0)
        if j == 0:
            ax.set_ylabel(f"Histories solved,{NL_}relative to matched", labelpad=0.8, linespacing=1.0)
        fs.clean(ax)
        cv.text(x0 + 9.0 + j * (hw + 3.0) + hw / 2, y0 + 9.6, lab, size=fs.PT_SMALL, ha="center", va="bottom")
    cv.text(x0 + 9.0 + (2 * hw + 3.0) / 2, y0 + h - 2.4, "Ensemble's scene → tested in (18 histories each)", size=fs.PT_SMALL + 0.5, ha="center", va="center")
    cv.key(x0 + w - 36.0, y0 + 6.0, [([dict(kind="mark", s=10, fc=TARGET, ec=TARGET)], "ensemble"),
                                     ([dict(kind="mark", s=6, fc=MATCHED, ec=MATCHED)], "20 matched sets")], size=fs.PT_SMALL, gap=1.4)


# ---------------------------------------------------------------- D: whole populations, exact no-op

def panel_noop(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "D", "Removing whole populations changes nothing")
    r = cm.table("plan_v2/fig3_removals.csv")
    rows = [("healthy", None, None), ("Kenyon cells, half", "Kenyon cells", 0.5), ("Kenyon cells, all", "Kenyon cells", 1.0),
            (f"antennal-lobe local{NL_}neurons, half", "GABAergic antennal-lobe local neurons", 0.5),
            (f"antennal-lobe local{NL_}neurons, all", "GABAergic antennal-lobe local neurons", 1.0)]
    measures = [("solved", f"Histories{NL_}solved", (100, 130)), ("task_mean_hz", f"Task{NL_}rate (Hz)", (0.30, 0.42)),
                ("giant_fibre_spikes", f"Giant fiber{NL_}spikes, looming", (60, 100))]
    lab_w = 31.0
    mw = (w - lab_w - 1.0 - 2 * 2.6) / 3
    top, ph = y0 + 12.4, h - 19.6
    vals = {}
    for i, (lab, pop, frac) in enumerate(rows):
        q = r[r["kind"] == "healthy"] if pop is None else r[(r["population"] == pop) & (r["fraction_removed"] == frac) & (r["kind"] == "whole population")]
        vals[lab] = q.iloc[0]
        extra = "" if pop is None else f" ({int(q.iloc[0]['cells']):,})"
        cv.text(x0 + lab_w - 1.0, top + ph * (i + 0.5) / len(rows), lab + extra, size=fs.PT_SMALL, ha="right", va="center", linespacing=1.0)
    for k, (col, lab, lim) in enumerate(measures):
        ax = cv.axes(x0 + lab_w + k * (mw + 2.6), top, mw, ph)
        ref = float(vals["healthy"][col])
        ax.axvline(ref, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
        for i, (rl, _, _) in enumerate(rows):
            v = float(vals[rl][col])
            ax.scatter([v], [i], s=12, facecolor=TARGET if i else "white", edgecolor=TARGET, linewidth=0.6, zorder=3)
            sd.add("D", col, condition=rl, value=v)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks([])
        ax.set_xlim(*lim)
        ax.tick_params(axis="x", labelsize=fs.PT_SMALL)
        fs.clean(ax, left=False)
        ax.spines["left"].set_visible(False)
        cv.text(x0 + lab_w + k * (mw + 2.6) + mw / 2, top - 0.6, lab, size=fs.PT_SMALL, ha="center", va="bottom", linespacing=1.0)
    same = all(float(vals[rl][c]) == float(vals["healthy"][c]) for rl, _, _ in rows for c, _, _ in measures)
    sd.add("D", "summary", all_identical_to_healthy=bool(same))


# ---------------------------------------------------------------- E: signatures under every connection rule

def panel_rules(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "E", "Healthy-circuit signatures hold under every connection rule")
    v = cm.table("plan_v2/s3_variants.csv")
    order = ["as built", "presynaptic signs", "same-type gain 0.05", "same-type gain 0.2", "same-type gain 1", "synapse cap 30", "synapse cap 120",
             "no synapse cap"]
    v = v.set_index("variant").loc[order]
    sigs = [("stable network", f"stable{NL_}network"), ("antennal-lobe inhibition", f"antennal{NL_}lobe{NL_}inhibition"),
            ("looming escape", f"looming{NL_}escape"), ("monoamine no-op", f"monoamine{NL_}no-op")]
    lab_w, cw, bar_w = 24.0, 11.0, w - 24.0 - 4 * 11.0 - 3.6
    top = y0 + 14.4
    rh = (y0 + h - 3.6 - top) / len(order)
    for k, (_, head) in enumerate(sigs):
        cv.text(x0 + lab_w + k * cw + cw / 2, top - 0.6, head, size=fs.PT_SMALL, ha="center", va="bottom", linespacing=1.0)
    cv.text(x0 + lab_w + 4 * cw + 1.0 + bar_w / 2, top - 0.6, f"healthy rate{NL_}in the task (Hz)", size=fs.PT_SMALL, ha="center", va="bottom", linespacing=1.0)
    ax = cv.axes(x0 + lab_w + 4 * cw + 1.0, top, bar_w, rh * len(order))
    for i, var in enumerate(order):
        yc = top + rh * (i + 0.5)
        cv.text(x0 + lab_w - 1.0, yc, var, size=fs.PT_SMALL, ha="right", va="center")
        for k, (sname, _) in enumerate(sigs):
            ok = str(v.loc[var, sname]).lower() == "true"
            cv.box(x0 + lab_w + k * cw + cw / 2 - 1.1, yc - 1.1, 2.2, 2.2, fc=fs.TEXT if ok else "white", ec=fs.TEXT, lw=0.3)
            sd.add("E", "signature", variant=var, signature=sname, reproduced=ok)
        hz = float(v.loc[var, "task_hz"])
        ax.add_patch(Rectangle((0, i - 0.3), hz, 0.6, facecolor=fs.TEXT if var == "as built" else "#BDBDBD", edgecolor="none"))
        sd.add("E", "healthy rate", variant=var, task_hz=hz)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0, 0.5)
    ax.set_xticks([0, 0.25, 0.5], ["0", "0.25", "0.5"])
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=fs.PT_SMALL)
    fs.clean(ax, left=False)
    ax.spines["left"].set_visible(False)
    cv.key(x0 + 4.0, y0 + 6.0, [([dict(kind="box", fc=fs.TEXT)], "signature reproduced")], size=fs.PT_SMALL)


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("s3")
    for rel in ("plan_v2/fig3_set_wiring.csv", "plan_v2/s3_band_sets.csv", "plan_v2/fig3_band_sets.csv", "figure_data/fd_cells.csv",
                "plan_v2/fig3_ensemble_sets.csv", "plan_v2/fig3_removals.csv", "plan_v2/s3_variants.csv"):
        sd.source(cm.RESULTS / rel)
    for f in sorted(TILES.glob("*.png")) + sorted(TILES.glob("*.json")):
        sd.source(f)
    panel_band(cv, sd)
    panel_composition(cv, sd, XS[0], ROW2, U, R_H)
    panel_solved(cv, sd, XS[1], ROW2, fs.unit(3), R_H)
    panel_noop(cv, sd, XS[0], ROW3, fs.unit(2), R_H)
    panel_rules(cv, sd, XS[2], ROW3, fs.unit(2), R_H)
    cm.finish(cv, "s3", sd, kind="supplement")


if __name__ == "__main__":
    main()
