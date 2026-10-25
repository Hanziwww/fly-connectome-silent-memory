"""Extended Data Fig. 6 - The full intervention factorial (180 x 150 mm).

a  every intervention condition (rows) in every model unit (columns): net change in solved histories relative to the
   unperturbed baseline, as a share of the histories (brown: lost, blue-green: gained; grey: not run)
b  component interpolations towards the other cue's state vs norm-matched scrambled changes of the same size: histories solved
   with the cue change minus with the scramble, per unit, window, component and dose (1 of 144 tests significant after Holm, marked)
c  norm audit of the state edits: largest relative norm error and largest |direction cosine| with the true difference
d  every condition with a margin: mean change of the first-decision margin vs net change in solved histories (Spearman rho, descriptive)
e  decision-window edits in the readout-only, joint, untrained-readout and reset models: the reset model, which carries no state, is
   unchanged by every edit

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\ed06.py
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, palette as pal, qc  # noqa: E402

U = (fs.FULL - 3 * fs.GAP) / 4
XS = [k * (U + fs.GAP) for k in range(4)]
H = 150.0
UNITS = [("M0", "M0"), ("M0joint", "M0 joint"), ("M0vis.hidden", "shared, hidden"), ("M0vis.visible", "shared, visible"), ("S1", "S1"), ("S2", "S2")]
SHORT = {"M0": "M0", "M0joint": "joint", "M0vis.hidden": "hidden", "M0vis.visible": "visible", "S1": "S1", "S2": "S2"}
B_LAB = {"M0": "M0", "M0joint": "joint", "M0vis.hidden": "shared,\nhidden", "M0vis.visible": "shared,\nvisible", "S1": "S1", "S2": "S2"}
CMAP = colormaps["BrBG"]
NORM = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
NA = "#E6E6E6"
WHOLE = ["clear_all_state", "swap_all_state", "neural_clear", "opposite_cue_swap", "reset_each_window", "full_reset"]
FAMILIES = [("controls", ["sham", "restore", "same_cue_swap"]), ("whole state", WHOLE),
            ("clear one component", ["clear_voltage", "clear_current", "clear_adaptation", "clear_resources", "clear_delay_buffer"]),
            ("swap one component", ["swap_voltage", "swap_current", "swap_adaptation", "swap_resources", "swap_delay_buffer"]),
            ("interpolate towards the other cue", ["interpolate_cue"]), ("norm-matched scramble", ["norm_matched_scramble"]),
            ("suppress a candidate group", ["group_suppression"]), ("route through a group", ["route_network", "route_readout", "route_both"]),
            ("delay content", ["timing_blank", "timing_neutral_stationary", "timing_neutral_walk_0", "timing_neutral_walk_1"])]
KEYS = ["source", "intervention", "component", "operation", "window", "fraction", "group", "side", "strength", "channel", "input_kind"]


def cond_key(r):
    return "|".join(str(r[k]) for k in KEYS)


# ---------------------------------------------------------------- a: factorial barcode

def panel_a(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "a")
    a = cm.table("extended/ed06/ed06a_intervention_factorial.csv")
    a = a[a["unit"].isin([u for u, _ in UNITS])].copy()
    assert a["intervention"].isin([i for _, ints in FAMILIES for i in ints]).all(), "an intervention has no family"
    a["key"] = a.apply(cond_key, axis=1)
    a["value"] = a["net_change"] / a["histories"]
    lab_w = 33.0
    cw = (w - lab_w - 1.0) / len(UNITS)
    top = 7.4
    rows = []
    for fam, ints in FAMILIES:
        sub = a[a["intervention"].isin(ints)]
        order = sub.drop_duplicates("key").copy()
        order["io"] = order["intervention"].map(ints.index)
        order = order.sort_values(["io", "component", "window", "fraction", "group", "side", "strength", "channel", "input_kind"], na_position="first")
        rows.append((fam, list(order["key"])))
    n_rows = sum(len(k) for _, k in rows)
    gap = 1.2
    rh = (h - top - 7.0 - gap * (len(rows) - 1)) / n_rows
    ax = bd.mm_axes(cv, x0, y0, w, h)
    for j, (u, lab) in enumerate(UNITS):
        ax.text(lab_w + (j + 0.5) * cw, top - 0.8, SHORT[u], ha="center", va="bottom", fontsize=fs.PT_SMALL)
    ax.plot([lab_w + 2 * cw + 0.6, lab_w + 4 * cw - 0.8], [top - 3.3] * 2, color=fs.TEXT, lw=0.35)
    ax.text(lab_w + 3 * cw, top - 3.6, "shared readout", ha="center", va="bottom", fontsize=fs.PT_SMALL)
    y = top
    pivot = a.pivot_table(index="key", columns="unit", values="value", aggfunc="first")
    for fam, keys in rows:
        y_start = y
        for k in keys:
            for j, (u, _) in enumerate(UNITS):
                v = pivot.loc[k, u] if u in pivot.columns and k in pivot.index else np.nan
                col = NA if not np.isfinite(v) else CMAP(NORM(v))
                ax.add_patch(bd.Rectangle((lab_w + j * cw, y), cw - 0.15, rh, facecolor=col, edgecolor="none", linewidth=0.5))
            y += rh
        ax.plot([lab_w - 0.8, lab_w - 0.8], [y_start + 0.1, y - 0.1], color=fs.TEXT, lw=0.4)
        ax.text(lab_w - 1.4, (y_start + y) / 2, f"{fam} ({len(keys)})", ha="right", va="center", fontsize=fs.PT_SMALL)
        y += gap
    for r in a.itertuples():
        sd.add("a", "condition", unit=r.unit, intervention=r.intervention, component=r.component, window=r.window, fraction=r.fraction, group=r.group,
               side=r.side, strength=r.strength, channel=r.channel, input_kind=r.input_kind, histories=int(r.histories),
               baseline_solved=int(r.baseline_solved), solved=int(r.solved), lost=int(r.lost), gained=int(r.gained), net_change=int(r.net_change))
    kx = x0 + lab_w
    k_ax = cv.axes(kx, y0 + h - 3.6, w - lab_w - 1.0, 1.2)
    vals = np.linspace(-1, 1, 128)
    k_ax.imshow(CMAP(NORM(vals))[None, :, :3], aspect="auto", extent=(-1, 1, 0, 1), interpolation="bilinear")
    k_ax.set_yticks([])
    k_ax.set_xticks([-1, 0, 1], [fs.MINUS + "1", "0", "1"])
    k_ax.tick_params(axis="x", length=1.0, pad=0.4, labelsize=fs.PT_SMALL)
    for side in ("top", "right", "left"):
        k_ax.spines[side].set_visible(False)
    cv.text(x0 + lab_w - 2.6, y0 + h - 4.4, "net change / histories", size=fs.PT_SMALL, ha="right", va="center")
    cv.text(x0 + lab_w - 2.6, y0 + h - 2.2, "grey: not run", size=fs.PT_SMALL, ha="right", va="center")


# ---------------------------------------------------------------- b: cue vs scramble

def panel_b(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "b")
    d = cm.table("main/fig6/fig6d_component_interventions.csv").copy()
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    d["diff"] = (d["cue_solved"] - d["scrambled_solved"]) / d["histories"]
    comps = ["voltage", "current", "adaptation", "continuous_combined"]
    comp_lab = {"voltage": "v", "current": "g", "adaptation": "adapt.", "continuous_combined": "all"}
    windows = sorted(d["window"].unique(), key=lambda s: ["write", "hold", "read"].index(s) if s in ("write", "hold", "read") else 9)
    fracs = sorted(d["fraction"].unique())
    cv.text(x0 + 3.6, y0 + 1.5, "solved with the cue change minus with the scramble (share of histories); doses 0.25, 0.5, 1", size=fs.PT_SMALL,
            va="center")
    lab_w = 16.0
    cell = (w - lab_w - 1.0) / (len(comps) * len(fracs) + (len(comps) - 1) * 0.5)
    ch = 3.0
    ax = bd.mm_axes(cv, x0, y0, w, h)
    top = 7.6
    for c_i, comp in enumerate(comps):
        xs = lab_w + (c_i * (len(fracs) + 0.5)) * cell
        ax.text(xs + len(fracs) * cell / 2, top - 0.6, comp_lab[comp], ha="center", va="bottom", fontsize=fs.PT_SMALL)
    y = top
    n_sig = 0
    for u, ulab in UNITS:
        du = d[d["unit"] == u]
        if du.empty:
            continue
        ax.text(0.0, y + len(windows) * ch / 2, B_LAB[u], ha="left", va="center", fontsize=fs.PT_SMALL, linespacing=1.1)
        for wi, win in enumerate(windows):
            ax.text(lab_w - 0.6, y + (wi + 0.5) * ch, win, ha="right", va="center", fontsize=fs.PT_SMALL)
            for c_i, comp in enumerate(comps):
                for f_i, fr in enumerate(fracs):
                    r = du[(du["window"] == win) & (du["component"] == comp) & (du["fraction"] == fr)]
                    xx = lab_w + (c_i * (len(fracs) + 0.5) + f_i) * cell
                    if r.empty:
                        ax.add_patch(bd.Rectangle((xx, y + wi * ch), cell - 0.12, ch - 0.12, facecolor=NA, edgecolor="none", linewidth=0.5))
                        continue
                    r = r.iloc[0]
                    ax.add_patch(bd.Rectangle((xx, y + wi * ch), cell - 0.12, ch - 0.12, facecolor=CMAP(NORM(float(r["diff"]))), edgecolor="none",
                                              linewidth=0.5))
                    tid = r["test_id"] if isinstance(r["test_id"], str) else None
                    p = float(tests.loc[tid, "p_adjusted"]) if tid else None
                    if p is not None and p < 0.05:
                        n_sig += 1
                        ax.add_patch(bd.Rectangle((xx - 0.05, y + wi * ch - 0.05), cell - 0.02, ch - 0.02, facecolor="none", edgecolor=fs.TEXT,
                                                  linewidth=0.7))
                    sd.add("b", "cell", unit=u, window=win, component=comp, fraction=float(fr), histories=int(r["histories"]),
                           cue_solved=int(r["cue_solved"]), scrambled_solved=int(r["scrambled_solved"]), p_holm=p, test_id=tid)
        y += len(windows) * ch + 1.4
    n_tests = int(d["test_id"].notna().sum())
    cv.text(x0 + lab_w, y0 + y + 1.6, f"colour as in a; outlined: Holm {fs.P_SYM} < 0.05 ({n_sig} of {n_tests} tests; doses 0.5 and 1 tested)",
            size=fs.PT_SMALL, va="center")


# ---------------------------------------------------------------- c: norm audit

def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    b = cm.table("extended/ed06/ed06b_norm_audit.csv")
    models = [("remote-dynamic-M0-head_only", "M0"), ("remote-dynamic-M0-joint", "joint"), ("supp-M0-visibility", "shared"),
              ("supp-S1-ring-brain", "S1"), ("supp-S2-context-brain", "S2")]
    fields = [("v", "voltage"), ("g", "current"), ("adapt", "adaptation"), ("g_slow", "slow current"), ("g_slow_cls", "slow, class")]
    fcol = {"v": "#1A1A1A", "g": "#7A7A7A", "adapt": pal.ACCENT, "g_slow": "#56B4E9", "g_slow_cls": "#009E73"}
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="mark", s=6, fc=fcol[f], ec=fcol[f])], lab) for f, lab in fields[:3]], gap=1.2)
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="mark", s=6, fc=fcol[f], ec=fcol[f])], lab) for f, lab in fields[3:]] +
           [([dict(kind="mark", s=6, fc="white", ec=fs.TEXT, lw=0.5)], "zero")], gap=1.2)
    floor = 1e-11
    ah = (h - 7.6 - 13.0) / 2
    for k, (col, lab) in enumerate((("max_relative_norm_error", "Norm error"), ("max_absolute_direction_cosine", "|cos| with cue diff."))):
        ax = cv.axes(x0 + 9.4, y0 + 7.6 + k * (ah + 3.0), w - 9.8, ah)
        for i, (m, mlab) in enumerate(models):
            for j, (f, flab) in enumerate(fields):
                r = b[(b["model"] == m) & (b["field"] == f)]
                if r.empty:
                    continue
                v = float(r[col].iloc[0])
                ax.plot([i + (j - 2) * 0.12], [max(v, floor)], marker="o", ms=2.2, mfc=fcol[f] if v > 0 else "white", mec=fcol[f], mew=0.5,
                        ls="none", clip_on=False)
                if k == 0:
                    sd.add("c", "audit", model=m, field=f, entries=int(r["entries"].iloc[0]), invalid=int(r["invalid"].iloc[0]),
                           max_relative_norm_error=float(r["max_relative_norm_error"].iloc[0]),
                           max_absolute_direction_cosine=float(r["max_absolute_direction_cosine"].iloc[0]),
                           max_zero_boundary_fraction=float(r["max_zero_boundary_fraction"].iloc[0]))
        ax.set_yscale("log")
        ax.set_ylim(floor / 2, 0.1)
        ax.set_yticks([floor, 1e-8, 1e-5, 1e-2], ["0", "10$^{−8}$", "10$^{−5}$", "10$^{−2}$"])
        fs.minor(ax, x=False)
        ax.set_xticks(range(len(models)), [m for _, m in models] if k == 1 else [""] * len(models))
        ax.set_xlim(-0.6, len(models) - 0.4)
        fs.clean(ax)
        ax.set_ylabel(lab, labelpad=1)
    cv.text(x0 + 9.4, y0 + h - 1.6, f"{int(b['entries'].sum()):,} edits; {int(b['invalid'].sum())} invalid", size=fs.PT_SMALL, va="center")


# ---------------------------------------------------------------- d: margin change vs outcome

D_FAMILIES = [("whole state", WHOLE, "#1A1A1A"),
              ("one component", ["clear_voltage", "clear_current", "clear_adaptation", "clear_resources", "clear_delay_buffer", "swap_voltage",
                                 "swap_current", "swap_adaptation", "swap_resources", "swap_delay_buffer"], "#7A7A7A"),
              ("interpolate", ["interpolate_cue"], pal.ACCENT), ("scramble", ["norm_matched_scramble"], "#F2A3C0"),
              ("group, route", ["group_suppression", "route_network", "route_readout", "route_both"], "#56B4E9"),
              ("controls", ["sham", "restore", "same_cue_swap"], "#009E73")]


def panel_d(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "d")
    a = cm.table("extended/ed06/ed06a_intervention_factorial.csv")
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="mark", s=5, fc=c, ec=c)], lab) for lab, _, c in D_FAMILIES[:3]], gap=1.2)
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="mark", s=5, fc=c, ec=c)], lab) for lab, _, c in D_FAMILIES[3:]], gap=1.2)
    ax = cv.axes(x0 + 9.0, y0 + 7.6, w - 9.4, h - 7.6 - 8.6)
    rng = np.random.default_rng(13)
    placed = 0
    for lab, ints, col in D_FAMILIES:
        d = a[a["intervention"].isin(ints) & a["margin_mean_change"].notna()]
        placed += len(d)
        x = d["margin_mean_change"].to_numpy(float)
        y = (d["net_change"] / d["histories"]).to_numpy(float) + (rng.random(len(d)) - 0.5) * 0.02
        ax.scatter(x, y, s=3.0, color=col, linewidth=0, alpha=0.9, zorder=3 if lab != "controls" else 4)
        sd.add("d", "family", family=lab, conditions=int(len(d)), median_margin_change=float(np.median(x)),
               median_net_change_share=float(np.median(d["net_change"] / d["histories"])))
    fin = a[a["margin_mean_change"].notna()]
    assert placed == len(fin), (placed, len(fin))
    rho = float(spearmanr(fin["margin_mean_change"], fin["net_change"] / fin["histories"]).statistic)
    sd.add("d", "spearman", conditions=int(len(fin)), rho=rho, note="descriptive; conditions share histories and are not independent")
    ax.text(0.97, 0.03, f"{fs.RHO} = {rho:.2f}\n{len(fin):,} conditions", transform=ax.transAxes, ha="right", va="bottom", fontsize=fs.PT_SMALL,
            linespacing=1.15)
    ax.axhline(0, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=1)
    ax.axvline(0, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=1)
    fs.clean(ax)
    ax.set_xlabel("Margin change (logit)", labelpad=1)
    ax.set_ylabel("Net change / histories", labelpad=1)


# ---------------------------------------------------------------- e: decision-window edits need carried state

E_UNITS = [("M0-head_only", "M0"), ("M0-joint", "joint"), ("M0-carry_untrained", "untrained"), ("M0-reset", "reset")]
E_ROWS = [("restore", "restore"), ("clear_voltage", "clear v"), ("clear_current", "clear g"), ("clear_adaptation", "clear adapt."),
          ("clear_delay_buffer", "clear delay"), ("full_reset", "full reset"), ("opposite_cue_swap", "cue swap")]


def panel_e(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "e")
    a = cm.table("extended/ed06/ed06a_intervention_factorial.csv")
    b = a[a["source"] == "remote_batch1"]
    assert len(b) == len(E_UNITS) * len(E_ROWS) and set(b["unit"]) == {u for u, _ in E_UNITS}
    hist = sorted(set(b["histories"].astype(int)))
    cv.text(x0 + 3.6, y0 + 1.5, f"decision window; net change of {hist[0]} histories", size=fs.PT_SMALL, va="center")
    lab_w, top = 11.0, 6.4
    cw = (w - lab_w - 0.4) / len(E_UNITS)
    rh = (h - top - 0.4) / len(E_ROWS)
    ax = bd.mm_axes(cv, x0, y0, w, h)
    for i, (iv, ilab) in enumerate(E_ROWS):
        ax.text(lab_w - 0.6, top + (i + 0.5) * rh, ilab, ha="right", va="center", fontsize=fs.PT_SMALL)
    for j, (u, lab) in enumerate(E_UNITS):
        ax.text(lab_w + (j + 0.5) * cw, top - 0.6, lab, ha="center", va="bottom", fontsize=fs.PT_SMALL)
        for i, (iv, _) in enumerate(E_ROWS):
            r = b[(b["unit"] == u) & (b["intervention"] == iv)].iloc[0]
            n = int(r["net_change"])
            ax.add_patch(bd.Rectangle((lab_w + j * cw, top + i * rh), cw - 0.15, rh - 0.15, facecolor=CMAP(NORM(n / r["histories"])),
                                      edgecolor="none", linewidth=0.5))
            ax.text(lab_w + (j + 0.5) * cw - 0.075, top + (i + 0.5) * rh - 0.075, (fs.MINUS if n < 0 else "+" if n > 0 else "") + str(abs(n)),
                    ha="center", va="center", fontsize=fs.PT_SMALL)
            sd.add("e", "condition", unit=u, model=r["model"], intervention=iv, histories=int(r["histories"]), baseline_solved=int(r["baseline_solved"]),
                   solved=int(r["solved"]), lost=int(r["lost"]), gained=int(r["gained"]), net_change=n,
                   margin_mean_change=float(r["margin_mean_change"]))


def main():
    cv = fs.Canvas(fs.FULL, H, ppi_min=300)
    sd = qc.SourceData("ed06")
    for rel in ("extended/ed06/ed06a_intervention_factorial.csv", "extended/ed06/ed06b_norm_audit.csv", "main/fig6/fig6d_component_interventions.csv",
                "stats/statistical_tests.csv"):
        sd.source(cm.RESULTS / rel)
    panel_a(cv, sd, XS[0], 0.0, 2 * U + fs.GAP, H)
    panel_b(cv, sd, XS[2], 0.0, 2 * U + fs.GAP, 73.0)
    panel_c(cv, sd, XS[2], 75.5, U, H - 75.5)
    panel_d(cv, sd, XS[3], 75.5, U, 42.0)
    panel_e(cv, sd, XS[3], 119.5, U, H - 119.5)
    cm.finish(cv, "ed06", sd, kind="extended")


if __name__ == "__main__":
    main()
