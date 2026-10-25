"""Fig. 6 - A recurrently wired, sign-consistent ensemble next to the readout holds the cue state (180 x 145 mm).

Band   a  the M0 cue ensemble (1% of cells with the largest |cue 1 - cue 0| voltage difference) and its connections
       b  one neuron of each of the eight most cue-responsive cell types, with their mean |dv| in the six state sets
Row 1  c  cells over 1 mV by synaptic distance from the inputs and to the readout (six state sets)
       d  connections within the ensemble vs matched random sets, at four ensemble sizes (six state sets)
       e  sign motif: excitatory share of same- and opposite-preference connections; difference vs preference permutation
       f  M0 wiring between the two preference groups: connections, density and excitatory share
Row 2  g  superclass and transmitter enrichment of the ensemble (odds ratios with 95% CI; ED8b, BH-adjusted family F8.enrichment)
       h  inhibitory transmitters in the larger and the smaller preference group of the ensemble (six state sets)

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\fig6.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, palette as pal, qc  # noqa: E402

U = (fs.FULL - 3 * fs.GAP) / 4
XS = [k * (U + fs.GAP) for k in range(4)]
TILES = cm.RENDERS / "fig6"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
BAND_H = META["band_mm"]
ROW1, ROW2, ROW_H = BAND_H + 4.0, BAND_H + 48.0, 41.0
H = ROW2 + ROW_H
AX_TOP, AX_H = 6.5, 27.0
SETS = [("m0", "M0 readout-only (hidden)", "M0"), ("m0_joint", "M0 joint (hidden)", "M0 joint"),
        ("m0vis_visible", "M0 shared readout (visible)", "M0, goal visible"), ("s1", "S1 readout-only (hidden)", "S1"),
        ("s2_o0", "S2 readout-only, orientation 0 (hidden)", "S2, 0°"), ("s2_o180", "S2 readout-only, orientation 180 (hidden)", "S2, 180°")]
WHITE = fs.ON_DARK
EDGE = "#ABABAB"
NT = {"acetylcholine": "ACh", "gaba": "GABA", "glutamate": "Glu"}
INHIBITORY = ("gaba", "glutamate", "histamine")                # negative weights in the model connectome
EXC_C, INH_C = "#4D4D4D", "#CFCFCF"
TOP_FRACTION = 0.01


def style(key):
    return dict(color=pal.STATE_SET[key], lw=1.0 if key == "m0" else 0.6, ls=(0, (2.2, 1.2)) if key == "m0_joint" else "-")


def tile_text(cv, x, y, s, size=fs.PT_SMALL, **kw):
    t = cv.text(x, y, s, size=size, color=WHITE, **kw)
    t.set_path_effects(fs.dark_halo(0.8))
    return t


def scale_bar(cv, right, y, length_um, um_per_px, px_per_mm):
    length = length_um / um_per_px / px_per_mm
    cv.line([right - length, right], [y, y], WHITE, lw=1.0, solid_capstyle="butt")
    tile_text(cv, right, y - 0.5, f"{length_um} µm", ha="right", va="bottom")


def ensemble_cells(dv):
    k = int(np.ceil(TOP_FRACTION * len(dv)))
    return np.sort(np.argsort(-np.abs(dv), kind="mergesort")[:k])


def state_table():
    s = cm.table("figure_data/fd_soma_state.csv")
    c = cm.table("figure_data/fd_cells.csv")
    assert np.array_equal(s["cell"].to_numpy(), np.arange(len(s))) and np.array_equal(c["cell"].to_numpy(), np.arange(len(c)))
    return s, c


def mag_colour(v, lo=0.1, hi=50.0):
    t = np.clip((np.log10(np.maximum(np.asarray(v, float), 1e-9)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)), 0, 1)
    return pal.MAGNITUDE_ON_DARK(t)


# ---------------------------------------------------------------- a: the cue ensemble

def panel_a(cv, sd):
    mw, bh = META["map_mm"]
    m = META["map"]
    ens = cm.table("figure_data/fd_ensemble.csv")
    e1 = ens[(ens["state_key"] == "m0") & np.isclose(ens["top_fraction"], TOP_FRACTION)].iloc[0]
    assert m["cells"] == int(e1["cells"]) and m["connections"] == int(e1["observed_connections"])
    img = np.asarray(Image.open(TILES / "ensemble.png").convert("RGB"))
    cv.tile(0, 0, img, width=mw, height=bh, group="fig6band", label="ensemble")
    cv.letter(1.2, 0.9, "a")
    cv.fig.texts[-1].set_color(WHITE)
    px_per_mm = img.shape[1] / mw
    kx, ky = 1.5, 26.0
    floor_mv = np.floor(m["min_abs_dv_mv"] * 10) / 10              # rounded down so the bound holds for every cell
    tile_text(cv, kx, ky, "Top 1% by |Δv|" + "\n" + f"({m['cells']:,} cells," + "\n" + f"|Δv| ≥ {floor_mv:.1f} mV)", va="top",
              linespacing=1.05)
    tile_text(cv, kx, ky + 7.6, "Preferred cue", va="center")
    for i, (k, n) in enumerate(((0, m["cue0"]), (1, m["cue1"]))):
        yy = ky + 10.0 + i * 2.4
        cv.mark(kx + 0.8, yy, s=7.0, fc=pal.CUE_ON_DARK[k], ec="#000000", lw=0.3)
        tile_text(cv, kx + 2.0, yy, f"cue {k}: {n:,}", va="center")
    yy = ky + 15.6
    cv.line([kx, kx + 1.6], [yy, yy], "#C9D7E6", lw=0.5)
    tile_text(cv, kx + 2.0, yy, f"{m['connections']:,}" + "\n" + "connections", va="top", linespacing=1.05)
    yy -= 0.9
    ox, oy = mw - 12.6, bh - 6.4
    cv.line([ox, ox + 2.8], [oy, oy], WHITE, lw=0.5)
    tile_text(cv, ox + 3.2, oy, "L", va="center")
    cv.line([ox, ox], [oy, oy + 2.8], WHITE, lw=0.5)
    tile_text(cv, ox, oy + 3.2, "V", ha="center", va="top")
    scale_bar(cv, mw - 1.4, bh - 1.6, 100, m["scale_um_per_px"], px_per_mm)
    sd.add("a", "ensemble", state_set="M0 readout-only (hidden)", cells=m["cells"], located_cells=m["located"], cue0_cells=m["cue0"],
           cue1_cells=m["cue1"], connections=m["connections"], drawn_connections=m["drawn_connections"], min_abs_dv_mv=m["min_abs_dv_mv"],
           um_per_px=m["scale_um_per_px"], scale_bar_um=100, unfold="brain frontal; VNC rotated 90 degrees about the neck (msa/cnsview.py)")


# ---------------------------------------------------------------- b: cue-responsive cell types

def panel_b(cv, sd):
    mw, bh = META["map_mm"]
    tw, th = META["tile_mm"]
    atlas = cm.table("extended/ed08/ed08a_type_atlas.csv")
    atlas = atlas[atlas["record"] == "cue_difference_by_type"]
    piv = atlas.pivot_table(index="type", columns="phase_or_state_set", values="mean_abs_voltage_cue_difference")
    size = atlas.groupby("type")["cells"].first()
    for k, t in enumerate(META["tiles"]):
        r, c = divmod(k, 4)
        x, y = mw + fs.GUTTER + c * (tw + fs.GUTTER), r * (th + fs.GUTTER)
        img = np.asarray(Image.open(TILES / f"{t['name']}.png").convert("RGB"))
        cv.tile(x, y, img, width=tw, height=th, group="fig6band", label=t["name"])
        tx = x + 1.1
        if k == 0:
            cv.letter(x + 1.2, y + 0.9, "b")
            cv.fig.texts[-1].set_color(WHITE)
            tx = x + 4.2
        tile_text(cv, tx, y + 3.2, t["type"], size=fs.PT, va="baseline")
        tile_text(cv, tx + fs.text_width(t["type"], fs.PT) + 0.8, y + 3.2, f"#{t['rank']}", va="baseline")
        vals = [float(piv.loc[t["type"], label]) for _, label, _ in SETS]
        assert abs(vals[0] - t["type_mean_abs_dv_mv"]) < 1e-6
        n = int(size.loc[t["type"]])
        nt = NT.get(t["transmitter"], t["transmitter"])
        tile_text(cv, x + 1.1, y + th - 1.0, f"{nt}, {n} cells, {vals[0]:.0f} mV", va="bottom")
        scale_bar(cv, x + tw - 1.2, y + th - 1.5, t["scale_bar_um"], t["scale_um_per_px"], img.shape[1] / tw)
        sq, gp = 1.5, 0.25
        width = 6 * sq + 5 * gp
        sx0, sy0 = x + tw - 1.1 - width, y + 1.2
        ov = bd.mm_axes(cv, sx0, sy0, width, sq)
        for i, v in enumerate(vals):
            ov.add_patch(Rectangle((i * (sq + gp), 0), sq, sq, facecolor=mag_colour(v), edgecolor="none", linewidth=0.5))
        if k == 0:
            gax = cv.axes(sx0, sy0 + sq + 0.7, width, 0.7)
            gax.imshow(mag_colour(np.logspace(-1, np.log10(50), 128))[None, :, :3], aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
            gax.set_axis_off()
            tile_text(cv, sx0, sy0 + sq + 1.9, "0.1", va="top")
            tile_text(cv, sx0 + width, sy0 + sq + 1.9, "50 mV", ha="right", va="top")
        for (key, label, short), v in zip(SETS, vals):
            sd.add("b", "type mean |dv|", type=t["type"], state_set=label, mean_abs_dv_mv=v)
        sd.add("b", "neuron", body_id=t["body_id"], type=t["type"], rank=t["rank"], superclass=t["superclass"], role=t["role"],
               transmitter=t["transmitter"], type_cells=n, cell_dv_mv=t["cell_dv_mv"], skeleton=t["skeleton"], scale_bar_um=t["scale_bar_um"])


# ---------------------------------------------------------------- c: distance from the inputs and to the readout

def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    d = cm.table("figure_data/fd_cue_by_distance.csv")
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="mark", s=5, fc="white", ec=fs.TEXT, lw=0.5)], "no path")])
    ax_l, ax_w = x0 + 7.4, w - 8.4
    ax = cv.axes(ax_l, y0 + AX_TOP, ax_w, AX_H)
    specs = [("from_input", 0.0, "steps from the inputs"), ("to_readout", 5.6, "steps to the readout")]
    order = ["0", "1", "2", ">=3"]
    lim = (-0.45, 10.0)
    ticks, labels = [], []
    for direction, off, glab in specs:
        xs = off + np.arange(len(order))
        xn = off + len(order)
        for key, label, short in SETS:
            g = d[(d["state_key"] == key) & (d["direction"] == direction)].set_index("hop_class")
            v = [100 * float(g.loc[c_, "fraction_over_1_mv"]) for c_ in order]
            st = style(key)
            ax.plot(xs, v, color=st["color"], lw=st["lw"], ls=st["ls"], marker="o", ms=2.0, mew=0, zorder=3 if key == "m0" else 2)
            ax.plot([xn], [100 * float(g.loc["none", "fraction_over_1_mv"])], ls="none", marker="o", ms=2.0, mfc="white", mec=st["color"],
                    mew=0.5, zorder=3, clip_on=False)
            for c_ in order + ["none"]:
                sd.add("c", direction, state_set=label, hop_class=c_, cells=int(g.loc[c_, "cells"]),
                       percent_over_1_mv=100 * float(g.loc[c_, "fraction_over_1_mv"]), median_abs_dv_mv=float(g.loc[c_, "median_abs_dv_mv"]))
        ticks += list(xs) + [xn]
        labels += [o.replace(">=", "≥") for o in order] + ["none"]
        mid = ax_l + ((off + xn) / 2 - lim[0]) / (lim[1] - lim[0]) * ax_w
        cv.text(mid, y0 + AX_TOP + AX_H + 2.9, glab, size=fs.PT_SMALL, ha="center", va="top")
    ax.set_xticks(ticks, labels)
    ax.set_xlim(*lim)
    ax.set_ylim(0, 52)
    ax.set_yticks([0, 25, 50])
    fs.minor(ax, x=False)
    fs.clean(ax)
    ax.set_ylabel("Cells over 1 mV (%)", labelpad=1)


# ---------------------------------------------------------------- d: ensemble wiring vs matched sets

def panel_d(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "d")
    e = cm.table("figure_data/fd_ensemble.csv")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    ps = [float(tests.loc[f"fig_circuit.ensemble.{k}", "p_adjusted"]) for k, _, _ in SETS]
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="mark", s=6, fc=fs.TEXT)], "1%, tested"), ([dict(kind="mark", s=6, fc="white", ec=fs.TEXT, lw=0.5)],
                                                                                      "0.5, 2 and 5%")], gap=1.6)
    cv.text(x0 + 3.6, y0 + 3.9, f"1%: each {fs.P_SYM} = {fs.fmt_p(max(ps))} (Holm)", size=fs.PT_SMALL, va="center")
    ax = cv.axes(x0 + 7.4, y0 + AX_TOP, w - 7.8, AX_H)
    for i, (key, label, short) in enumerate(SETS):
        g = e[e["state_key"] == key].sort_values("top_fraction")
        x = 100 * g["top_fraction"].to_numpy() * 10 ** (0.018 * (i - 2.5))
        r = g["ratio"].to_numpy()
        lo = (g["observed_connections"] / g["null_q975"]).to_numpy()
        hi = (g["observed_connections"] / g["null_q025"]).to_numpy()
        st = style(key)
        ax.plot(x, r, color=st["color"], lw=st["lw"], ls=st["ls"], zorder=2)
        for xi, ri, l_, h_, f in zip(x, r, lo, hi, g["top_fraction"]):
            tested = np.isclose(f, TOP_FRACTION)
            ax.plot([xi, xi], [l_, h_], color=st["color"], lw=0.5, zorder=3)
            ax.plot([xi], [ri], marker="o", ms=2.4, mfc=st["color"] if tested else "white", mec=st["color"], mew=0.5, ls="none", zorder=4)
        for (_, rr), l_, h_ in zip(g.iterrows(), lo, hi):
            sd.add("d", "ensemble", state_set=label, top_fraction=float(rr["top_fraction"]), cells=int(rr["cells"]),
                   observed=int(rr["observed_connections"]), null_mean=float(rr["null_mean"]), ratio=float(rr["ratio"]), ratio_low=float(l_),
                   ratio_high=float(h_), p_holm=float(tests.loc[rr["test_id"], "p_adjusted"]) if isinstance(rr["test_id"], str) else None)
    ax.axhline(1, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5], ["0.5", "1", "2", "5"])
    ax.xaxis.set_minor_locator(__import__("matplotlib.ticker", fromlist=["NullLocator"]).NullLocator())
    ax.set_xlim(0.4, 6.2)
    ax.set_ylim(0.8, 3.8)
    ax.set_yticks([1, 2, 3])
    fs.minor(ax, x=False)
    fs.clean(ax)
    ax.set_ylabel("Connections / matched sets", labelpad=1)
    ax.set_xlabel("Ensemble size (% of cells)", labelpad=1)


# ---------------------------------------------------------------- e: sign motif

def panel_e(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "e")
    s = cm.table("figure_data/fd_sign_motif.csv").set_index("state_key")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    ps = [float(tests.loc[f"fig_circuit.sign.{k}", "p_adjusted"]) for k, _, _ in SETS]
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="box", fc="#DADADA", size=1.4)], "95% of preference shuffles")])
    cv.text(x0 + 3.6, y0 + 3.9, f"each {fs.P_SYM} = {fs.fmt_p(max(ps))} (Holm)", size=fs.PT_SMALL, va="center")
    a1w = 13.6
    ax1 = cv.axes(x0 + 7.4, y0 + AX_TOP, a1w, AX_H)
    for key, label, short in SETS:
        r = s.loc[key]
        st = style(key)
        ax1.plot([0, 1], [r["excitatory_share_same"], r["excitatory_share_opposite"]], color=st["color"], lw=st["lw"], ls=st["ls"], marker="o",
                 ms=2.2, mew=0, zorder=3 if key == "m0" else 2)
        sd.add("e", "sign motif", state_set=label, connections=int(r["connections"]), excitatory_share_same=float(r["excitatory_share_same"]),
               excitatory_share_opposite=float(r["excitatory_share_opposite"]), difference=float(r["difference"]), null_q025=float(r["null_q025"]),
               null_q975=float(r["null_q975"]), p_holm=float(tests.loc[r["test_id"], "p_adjusted"]), test_id=r["test_id"])
    ax1.set_xticks([0, 1], ["same", "opposite"])
    ax1.set_xlim(-0.35, 1.35)
    ax1.set_ylim(0.45, 0.85)
    ax1.set_yticks([0.5, 0.6, 0.7, 0.8])
    fs.clean(ax1)
    ax1.set_ylabel("Excitatory share", labelpad=1)
    ax1.set_xlabel("Cue preference", labelpad=1)
    ax2 = cv.axes(x0 + 7.4 + a1w + 7.4, y0 + AX_TOP, w - 7.4 - a1w - 7.4 - 0.4, AX_H)
    for i, (key, label, short) in enumerate(SETS):
        r = s.loc[key]
        ax2.add_patch(Rectangle((i - 0.32, r["null_q025"]), 0.64, r["null_q975"] - r["null_q025"], facecolor="#DADADA", edgecolor="none",
                                linewidth=0.5, zorder=1))
        ax2.plot([i], [r["difference"]], marker="o", ms=2.6, mfc=pal.STATE_SET[key], mec=pal.STATE_SET[key], mew=0.5, ls="none", zorder=3)
    ax2.axhline(0, color=fs.AXIS, lw=fs.LW_HAIR, zorder=0)
    ax2.set_xlim(-0.7, 5.7)
    ax2.set_xticks([])
    ax2.set_ylim(-0.1, 0.35)
    ax2.set_yticks([0, 0.1, 0.2, 0.3])
    fs.clean(ax2, bottom=False)
    ax2.set_ylabel("Same − opposite", labelpad=1)
    ax2.set_xlabel("State set", labelpad=3.6)


# ---------------------------------------------------------------- f: wiring between the preference groups (M0)

def panel_f(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "f")
    s, cells = state_table()
    dv = s["dv_m0_mv"].to_numpy(np.float64)
    top = ensemble_cells(dv)
    edges = cm.table("figure_data/fd_ensemble_edges.csv")
    pre, post, wgt = edges["pre_cell"].to_numpy(), edges["post_cell"].to_numpy(), edges["weight"].to_numpy()
    assert set(np.unique(np.concatenate([pre, post]))).issubset(set(top.tolist()))
    pref = (dv > 0).astype(int)
    n_group = [int((pref[top] == g).sum()) for g in (0, 1)]
    motif = cm.table("figure_data/fd_sign_motif.csv").set_index("state_key").loc["m0"]
    same = pref[pre] == pref[post]
    assert int((same & (wgt > 0)).sum()) == int(motif["same_excitatory"]) and int((~same & (wgt < 0)).sum()) == int(motif["opposite_inhibitory"])
    ax = bd.mm_axes(cv, x0, y0, w, h)
    lab_w, top_y, gap = 12.6, 6.2, 0.6
    cw = (w - lab_w - gap) / 2
    ch = (33.6 - top_y - gap) / 2
    for j, g in enumerate((0, 1)):
        cx = lab_w + j * (cw + gap) + cw / 2
        lab = f"to cue-{g} cells"
        x_start = cx - (1.3 + 0.5 + fs.text_width(lab, fs.PT_SMALL)) / 2
        ax.add_patch(Rectangle((x_start, top_y - 3.3), 1.3, 1.3, facecolor=pal.CUE[g], edgecolor="none", linewidth=0.5))
        ax.text(x_start + 1.8, top_y - 2.65, lab, ha="left", va="center", fontsize=fs.PT_SMALL)
    for i, gp in enumerate((0, 1)):
        cy = top_y + i * (ch + gap)
        ax.add_patch(Rectangle((0.0, cy + ch / 2 - 3.0), 1.3, 1.3, facecolor=pal.CUE[gp], edgecolor="none", linewidth=0.5))
        ax.text(1.8, cy + ch / 2 - 2.35, f"from cue-{gp}", ha="left", va="center", fontsize=fs.PT_SMALL)
        ax.text(lab_w - 0.8, cy + ch / 2 + 0.2, f"{n_group[gp]:,} cells", ha="right", va="center", fontsize=fs.PT_SMALL)
        for j, gq in enumerate((0, 1)):
            cx = lab_w + j * (cw + gap)
            m = (pref[pre] == gp) & (pref[post] == gq)
            n = int(m.sum())
            exc = float((wgt[m] > 0).mean())
            dens = n / (n_group[gp] * n_group[gq])
            ax.add_patch(Rectangle((cx, cy), cw, ch, facecolor="#F3F3F3", edgecolor="none", linewidth=0.5))
            ax.text(cx + cw / 2, cy + 2.3, f"{n:,}", ha="center", va="center", fontsize=fs.PT)
            bx, bw, by = cx + 1.0, cw - 2.0, cy + 4.2
            ax.add_patch(Rectangle((bx, by), bw * exc, 1.6, facecolor=EXC_C, edgecolor="none", linewidth=0.5))
            ax.add_patch(Rectangle((bx + bw * exc, by), bw * (1 - exc), 1.6, facecolor=INH_C, edgecolor="none", linewidth=0.5))
            ax.text(cx + cw / 2, cy + 7.4, f"{100 * exc:.0f}% excitatory", ha="center", va="center", fontsize=fs.PT_SMALL)
            ax.text(cx + cw / 2, cy + 9.6, f"{100 * dens:.1f}% of pairs", ha="center", va="center", fontsize=fs.PT_SMALL)
            sd.add("f", "block", sender=f"cue {gp} preferring", receiver=f"cue {gq} preferring", connections=n, excitatory_share=exc,
                   density=dens, sender_cells=n_group[gp], receiver_cells=n_group[gq])
    cv.key(x0 + 0.6, y0 + 37.4, [([dict(kind="box", fc=EXC_C, size=1.4)], "excitatory (ACh)"),
                                  ([dict(kind="box", fc=INH_C, size=1.4)], "inhibitory (GABA, Glu, HA)")], gap=1.6)


# ---------------------------------------------------------------- g: enrichment

CATS = [("superclass", "vnc_intrinsic", "VNC\nintrinsic"), ("superclass", "vnc_motor", "VNC\nmotor"), ("superclass", "descending_neuron", "DN"),
        ("superclass", "ascending_neuron", "AN"), ("superclass", "cb_intrinsic", "brain\nintrinsic"), ("superclass", "vnc_sensory", "VNC\nsensory"),
        ("superclass", "visual_projection", "VPN"), ("superclass", "ol_intrinsic", "OL\nintrinsic"),
        ("nt", "gaba", "GABA"), ("nt", "acetylcholine", "ACh"), ("nt", "glutamate", "Glu"), ("nt", "octopamine", "OA"), ("nt", "serotonin", "5-HT"),
        ("nt", "dopamine", "DA"), ("nt", "histamine", "HA")]
FLOOR = 3e-4


def panel_g(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "g")
    b = cm.table("extended/ed08/ed08b_atlas_enrichment.csv")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    b = b.assign(p_adjusted=b["test_id"].map(tests["p_adjusted"]))
    adj = cm.adjusted_label(b["test_id"])
    ax_l, ax_w = x0 + 9.2, w - 9.6
    ax = cv.axes(ax_l, y0 + AX_TOP, ax_w, AX_H)
    n_sc = sum(1 for c in CATS if c[0] == "superclass")
    split = 0.7                                          # gap between the superclass and transmitter groups (category units)
    pos = [ci + (split if ci >= n_sc else 0.0) for ci in range(len(CATS))]
    lim = (-0.6, pos[-1] + 0.6)
    for ci, (var, cat, lab) in enumerate(CATS):
        for i, (key, label, short) in enumerate(SETS):
            r = b[(b["state_set"] == label) & (b["variable"] == var) & (b["category"] == cat)].iloc[0]
            x = pos[ci] + (i - 2.5) * 0.12
            col = pal.STATE_SET[key]
            sig = float(r["p_adjusted"]) < 0.05
            orr = float(r["odds_ratio"])
            lo, hi = max(float(r["ci_low"]), FLOOR), max(float(r["ci_high"]), FLOOR)
            ax.plot([x, x], [lo, hi], color=col, lw=0.5, zorder=2)
            if orr > 0:
                ax.plot([x], [orr], marker="o", ms=2.3, mfc=col if sig else "white", mec=col, mew=0.5, ls="none", zorder=3)
            else:
                ax.plot([x], [FLOOR], marker="v", ms=2.6, mfc=col if sig else "white", mec=col, mew=0.5, ls="none", zorder=3, clip_on=False)
            sd.add("g", var, category=cat, state_set=label, odds_ratio=orr, ci_low=float(r["ci_low"]), ci_high=float(r["ci_high"]),
                   top_in_category=int(r["top_in_category"]), expected_in_top=float(r["expected_in_top"]), cells_in_category=int(r["cells_in_category"]),
                   p_adjusted=float(r["p_adjusted"]), correction=cm.correction([r["test_id"]]), test_id=r["test_id"])
    ax.axhline(1, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 1.4, 40)
    ax.set_yticks([FLOOR, 1e-3, 1e-2, 1e-1, 1, 10], ["0", "0.001", "0.01", "0.1", "1", "10"])
    fs.minor(ax, x=False)
    ax.set_xlim(*lim)
    ax.set_xticks(pos, [""] * len(CATS))
    ax.tick_params(axis="x", length=0)
    fs.clean(ax)
    ax.set_ylabel("Odds ratio", labelpad=1)
    xm = lambda v: ax_l + (v - lim[0]) / (lim[1] - lim[0]) * ax_w  # noqa: E731
    for ci, (_, _, lab) in enumerate(CATS):              # category labels placed line by line so two-line labels align
        for li, line in enumerate(lab.split("\n")):
            cv.text(xm(pos[ci]), y0 + AX_TOP + AX_H + 1.2 + li * 2.0, line, size=fs.PT_SMALL, ha="center", va="top")
    cv.text(xm((pos[0] + pos[n_sc - 1]) / 2), y0 + 3.9, "Superclass", size=fs.PT, ha="center", va="center")
    cv.text(xm((pos[n_sc] + pos[-1]) / 2), y0 + 3.9, "Transmitter", size=fs.PT, ha="center", va="center")
    ol = b[(b["state_set"] == SETS[0][1]) & (b["category"] == "ol_intrinsic")].iloc[0]
    ax.text(pos[n_sc - 1] + 0.42, 4.4e-4, f"M0: {int(ol['top_in_category'])} of {int(ol['cells_in_category']):,}" + "\n" +
            f"({ol['expected_in_top']:.0f} expected)", ha="left", va="bottom", fontsize=fs.PT_SMALL, linespacing=1.05)
    cv.key(x0 + 0.6, y0 + 40.0, [([dict(kind="mark", s=6, fc=fs.TEXT)], f"{adj} < 0.05"),
                                  ([dict(kind="mark", s=6, fc="white", ec=fs.TEXT, lw=0.5)], "not significant"),
                                  ([dict(kind="mark", s=7, marker="v", fc=fs.TEXT)], "none in the ensemble"),
                                  ([dict(kind="vline", h=2.0, lw=0.5)], "95% CI")], gap=2.2)


# ---------------------------------------------------------------- h: inhibitory transmitters by preference group

def panel_h(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "h")
    s, cells = state_table()
    nt = cells["transmitter"].fillna("unknown").to_numpy()
    inh = np.isin(nt, INHIBITORY)
    cv.text(x0 + 3.6, y0 + 1.5, "GABA, Glu or histamine", size=fs.PT_SMALL, va="center")
    cv.text(x0 + 3.6, y0 + 3.9, "(inhibitory in the model)", size=fs.PT_SMALL, va="center")
    ax = cv.axes(x0 + 7.4, y0 + AX_TOP, w - 8.4, AX_H)
    for key, label, short in SETS:
        dv = s[f"dv_{key}_mv"].to_numpy(np.float64)
        top = ensemble_cells(dv)
        groups = [top[dv[top] < 0], top[dv[top] > 0]]
        big, small = sorted(groups, key=len, reverse=True)
        v = [100 * float(inh[big].mean()), 100 * float(inh[small].mean())]
        st = style(key)
        ax.plot([0, 1], v, color=st["color"], lw=st["lw"], ls=st["ls"], marker="o", ms=2.2, mew=0, zorder=3 if key == "m0" else 2)
        sd.add("h", "inhibitory share", state_set=label, larger_group_cells=len(big), smaller_group_cells=len(small),
               larger_group_inhibitory_pct=v[0], smaller_group_inhibitory_pct=v[1],
               larger_group_prefers=f"cue {int(dv[big[0]] > 0)}", smaller_group_prefers=f"cue {int(dv[small[0]] > 0)}")
    ax.set_xticks([0, 1], ["larger", "smaller"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0, 70)
    ax.set_yticks([0, 30, 60])
    fs.minor(ax, x=False)
    fs.clean(ax)
    ax.set_ylabel("Inhibitory cells (%)", labelpad=1)
    ax.set_xlabel("Preference group in the ensemble", labelpad=1)


def legend_row(cv, y):
    items = [([dict(kind="hline", colour=pal.STATE_SET[k], lw=style(k)["lw"], ls=style(k)["ls"], w=3.0)], short) for k, _, short in SETS]
    end = cv.key(XS[0] + 7.4, y, items, gap=2.4)
    return end


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("fig6")
    for rel in ("figure_data/fd_cells.csv", "figure_data/fd_soma_state.csv", "figure_data/fd_cue_by_distance.csv", "figure_data/fd_ensemble.csv",
                "figure_data/fd_sign_motif.csv", "figure_data/fd_ensemble_edges.csv", "figure_data/fd_neuron_selection.csv",
                "extended/ed08/ed08a_type_atlas.csv", "extended/ed08/ed08b_atlas_enrichment.csv", "stats/statistical_tests.csv"):
        sd.source(cm.RESULTS / rel)
    panel_a(cv, sd)
    panel_b(cv, sd)
    panel_c(cv, sd, XS[0], ROW1, cm_u(1), ROW_H)
    panel_d(cv, sd, XS[1], ROW1, cm_u(1), ROW_H)
    panel_e(cv, sd, XS[2], ROW1, cm_u(1), ROW_H)
    panel_f(cv, sd, XS[3], ROW1, cm_u(1), ROW_H)
    legend_row(cv, ROW1 + 39.9)
    panel_g(cv, sd, XS[0], ROW2, cm_u(3), ROW_H)
    panel_h(cv, sd, XS[3], ROW2, cm_u(1), ROW_H)
    cm.finish(cv, "fig6", sd)


def cm_u(n):
    return n * U + (n - 1) * fs.GAP


if __name__ == "__main__":
    main()
