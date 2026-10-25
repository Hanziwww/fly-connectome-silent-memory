"""Extended Data Fig. 9 - Decoding and dynamics at every horizon: the carried state is input-driven and its code is time-specific
(180 x 130 mm).

Row 1 a  time generalisation of the cue decoder within the decision window (5 models): decoders trained at one time on histories
         at trained timings, tested at every time on held-out histories; chance-centred colour map, with the share of cells
         outside the 95% binomial band around chance and the mean same-time accuracy
Row 2 b  state-prediction R2 for every model, split and horizon: persistence, autonomous affine dynamics, input and phase
      c  eigenvalues of the fitted affine dynamics (1-ms step) with circles of equal decay time
      d  largest singular gain of the affine propagator by horizon (transient amplification, then decay)
Row 3 e  demixed reconstruction on held-out data: time, cue x time (and, for S2, context x time and cue x context x time) components
      f  decoding board variables of each T1 map from brain states before and after training, and from the board itself
         (decoders fitted on other maps; accuracy minus the majority baseline; 8 maps)
      g  histories solved when the cue decoder was right vs wrong, per model and held-out split (95% Clopper-Pearson CI)

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\ed09.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import NullLocator
from scipy.stats import beta, binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, palette as pal, qc  # noqa: E402

U = (fs.FULL - 3 * fs.GAP) / 4
XS = [k * (U + fs.GAP) for k in range(4)]
ROW2, ROW_H = 46.0, 41.0
ROW3 = ROW2 + ROW_H + 2.0
H = ROW3 + ROW_H
AX_TOP, AX_H = 6.5, 27.0
MODELS = [("remote-dynamic-M0-head_only", "M0 readout-only", "M0", "m0"), ("remote-dynamic-M0-joint", "M0 joint", "joint", "m0_joint"),
          ("supp-M0-visibility", "M0 shared readout", "shared", "m0vis_visible"), ("supp-S1-ring-brain", "S1 readout-only", "S1", "s1"),
          ("supp-S2-context-brain", "S2 readout-only", "S2", "s2_o0")]
ACC = colormaps["PiYG"]
ACC_LO, ACC_HI = 0.3, 0.7
METHODS = [("persistence", "persistence"), ("affine", "affine"), ("input_phase", "input + phase")]
R2_CMAP = LinearSegmentedColormap.from_list("r2", [colormaps["BrBG"](t) for t in np.linspace(0.12, 0.88, 64)])
R2_NORM = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=1.0)
NA = "#E6E6E6"


def model_style(key):
    return dict(color=pal.STATE_SET[key], lw=1.0 if key == "m0" else 0.6, ls=(0, (2.2, 1.2)) if key == "m0_joint" else "-")


def model_key(cv, x, y):
    return cv.key(x, y, [([dict(kind="hline", colour=pal.STATE_SET[k], lw=model_style(k)["lw"], ls=model_style(k)["ls"], w=2.6)], short)
                         for _, _, short, k in MODELS], gap=1.6)


def clopper(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return float(lo), float(hi)


# ---------------------------------------------------------------- a: time generalisation

def panel_a(cv, sd):
    cv.letter(0, 0, "a")
    g = cm.table("figure_data/fd_time_generalisation.csv")
    td = cm.table("extended/ed09/ed09a_decoding_all.csv")
    td = td[td["record_type"] == "time_decoder"].set_index(["model", "method"])["value"]
    cv.text(3.6, 1.5, "cue decoder trained at one time (trained timings), tested at every time (held-out histories)", size=fs.PT_SMALL, va="center")
    kx = fs.FULL - 42.0
    cax = cv.axes(kx, 0.9, 20.0, 1.2)
    cax.imshow(ACC(np.linspace(0, 1, 128))[None, :, :3], aspect="auto", extent=(ACC_LO, ACC_HI, 0, 1), interpolation="bilinear")
    cax.set_axis_off()
    cv.text(kx - 0.6, 1.5, f"accuracy {ACC_LO:g}", size=fs.PT_SMALL, ha="right", va="center")
    cv.text(kx + 20.6, 1.5, f"{ACC_HI:g} (white: chance, 0.5)", size=fs.PT_SMALL, ha="left", va="center")
    bw = (fs.FULL - 4 * fs.GAP) / 5
    side = bw - 5.6
    for k, (model, long, short, key) in enumerate(MODELS):
        d = g[g["model"] == model]
        acc = d.pivot_table(index="train_ms", columns="test_ms", values="accuracy").to_numpy()
        assert acc.shape == (40, 40)
        n = int(d["n_test_histories"].iloc[0])
        lo, hi = binom.ppf(0.025, n, 0.5) / n, binom.ppf(0.975, n, 0.5) / n
        out = float(np.mean((acc < lo) | (acc > hi)))
        same = float(np.mean(np.diag(acc)))
        assert abs(same - float(td.loc[(model, "same_time_mean_accuracy")])) < 1e-6
        assert abs(float(acc.mean()) - float(td.loc[(model, "all_time_mean_accuracy")])) < 1e-6
        x0 = k * (bw + fs.GAP)
        ax = cv.axes(x0 + 5.2, 6.4, side, side)
        ax.imshow(acc, cmap=ACC, vmin=ACC_LO, vmax=ACC_HI, origin="lower", extent=(-0.5, 39.5, -0.5, 39.5), interpolation="nearest")
        ax.plot([-0.5, 39.5], [-0.5, 39.5], color=fs.TEXT, lw=fs.LW_HAIR, ls=(0, (2, 1.5)))
        ax.set_xticks([0, 10, 20, 30], ["0", "10", "20", "30"])
        ax.set_yticks([0, 10, 20, 30], ["0", "10", "20", "30"] if k == 0 else [""] * 4)
        ax.tick_params(length=1.2, pad=0.6, labelsize=fs.PT_SMALL)
        ax.set_xlabel("Test time (ms)", labelpad=1)
        if k == 0:
            ax.set_ylabel("Train time (ms)", labelpad=1)
        cv.text(x0 + 5.2 + side / 2, 4.6, long, size=fs.PT_SMALL, ha="center", va="center")
        cv.text(x0 + 5.2, 6.4 + side + 5.9, f"same-time accuracy {same:.2f}", size=fs.PT_SMALL, va="center")
        cv.text(x0 + 5.2, 6.4 + side + 8.1, f"{100 * out:.1f}% of cells beyond chance band", size=fs.PT_SMALL, va="center")
        sd.add("a", "summary", model=model, train_histories=int(d["n_train_histories"].iloc[0]), test_histories=n, band_low=float(lo),
               band_high=float(hi), share_outside_band=out, same_time_mean_accuracy=same, all_time_mean_accuracy=float(acc.mean()),
               min_accuracy=float(acc.min()), max_accuracy=float(acc.max()))


# ---------------------------------------------------------------- b: R2 at every horizon

def panel_b(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "b")
    b = cm.table("extended/ed09/ed09b_dynamics_all.csv")
    horizons = sorted(set(b["horizon_ms"].astype(int)))
    assert horizons == [1, 20, 40, 80]
    splits = [("input_conditioned_dynamics", "held_time", "timing"), ("input_conditioned_dynamics", "held_input", "input"),
              ("m1_dynamics", "heldout_histories", "histories")]
    cv.text(x0 + 3.6, y0 + 1.5, "state prediction R² by method and horizon (ms); rows: model, held-out split", size=fs.PT_SMALL, va="center")
    lab_m, lab_s = 15.0, 11.0
    gap = 1.0
    cs = (w - lab_m - lab_s - 2 * gap - 0.4) / (len(METHODS) * len(horizons))
    hdr = 8.0
    mgap = 0.6
    rh = (h - hdr - (len(MODELS) - 1) * mgap - 0.2) / (len(MODELS) * len(splits))
    ax = bd.mm_axes(cv, x0, y0, w, h)
    cx0 = lab_m + lab_s
    for m_i, (meth, mlab) in enumerate(METHODS):
        gx = cx0 + m_i * (len(horizons) * cs + gap)
        ax.text(gx + len(horizons) * cs / 2, hdr - 3.0, mlab, ha="center", va="bottom", fontsize=fs.PT_SMALL)
        ax.plot([gx + 0.2, gx + len(horizons) * cs - 0.35], [hdr - 2.75] * 2, color=fs.TEXT, lw=0.35)
        for h_i, hz in enumerate(horizons):
            ax.text(gx + (h_i + 0.5) * cs, hdr - 0.4, str(hz), ha="center", va="bottom", fontsize=fs.PT_SMALL)
    y = hdr
    for model, long, short, key in MODELS:
        ax.text(0.0, y + rh * 1.5, short, ha="left", va="center", fontsize=fs.PT_SMALL)
        for rec, split, slab in splits:
            ax.text(lab_m + lab_s - 0.6, y + rh / 2, slab, ha="right", va="center", fontsize=fs.PT_SMALL)
            for m_i, (meth, _) in enumerate(METHODS):
                gx = cx0 + m_i * (len(horizons) * cs + gap)
                for h_i, hz in enumerate(horizons):
                    r = b[(b["record_type"] == rec) & (b["model"] == model) & (b["split"] == split) & (b["method"] == meth) & (b["horizon_ms"] == hz)]
                    if r.empty:
                        ax.add_patch(bd.Rectangle((gx + h_i * cs, y + 0.05), cs - 0.12, rh - 0.1, facecolor=NA, edgecolor="none", linewidth=0.5))
                        continue
                    v = float(r["value"].iloc[0])
                    ax.add_patch(bd.Rectangle((gx + h_i * cs, y + 0.05), cs - 0.12, rh - 0.1, facecolor=R2_CMAP(R2_NORM(v)), edgecolor="none",
                                              linewidth=0.5))
                    ax.text(gx + (h_i + 0.5) * cs - 0.06, y + rh / 2, f"{v:.2f}".replace("-", fs.MINUS), ha="center", va="center", fontsize=fs.PT_SMALL)
                    raw = r["details"].iloc[0]
                    det = json.loads(raw) if isinstance(raw, str) else {}
                    sd.add("b", "r2", model=model, split=split, method=meth, horizon_ms=hz, r2=v, histories=det.get("histories"),
                           windows=det.get("windows"), state_pairs=det.get("state_pairs"))
            y += rh
        y += mgap


# ---------------------------------------------------------------- c: eigenvalues

def affine():
    a = cm.table("extended/ed09/ed09a_decoding_all.csv")
    a = a[a["record_type"] == "affine_dynamics"].set_index("model")
    return {m: (float(a.loc[m, "value"]), json.loads(a.loc[m, "details"])) for m, *_ in MODELS}


def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    af = affine()
    model_key(cv, x0 + 3.6, y0 + 1.5)
    cv.text(x0 + 3.6, y0 + 3.9, "arcs: decay time 10, 20, 50 ms; black: unit circle", size=fs.PT_SMALL, va="center")
    ax = cv.axes(x0 + 9.0, y0 + AX_TOP, w - 9.4, AX_H)
    th = np.linspace(-0.45, 0.45, 200)
    for tau in (10, 20, 50):
        r = np.exp(-1.0 / tau)
        ax.plot(r * np.cos(th), r * np.sin(th), color=fs.GRID, lw=fs.LW_THIN, zorder=1)
    ax.plot(np.cos(th), np.sin(th), color=fs.AXIS, lw=fs.LW_THIN, zorder=1)
    r2s = []
    for model, long, short, key in MODELS:
        r2, d = af[model]
        r2s.append(r2)
        re, im = np.array(d["eigenvalue_real"]), np.array(d["eigenvalue_imag"])
        ax.plot(re, im, marker="o", ms=2.4, mfc="white" if key == "m0_joint" else pal.STATE_SET[key], mec=pal.STATE_SET[key], mew=0.6, ls="none",
                zorder=3 if key == "m0" else 2)
        for i, (a_, b_) in enumerate(zip(re, im)):
            sd.add("c", "eigenvalue", model=model, index=i, real=float(a_), imag=float(b_), decay_ms=float(d["stable_decay_ms"][i]), heldout_r2_1ms=r2)
    ax.set_xlim(0.86, 1.005)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xticks([0.9, 0.95, 1.0], ["0.9", "0.95", "1"])
    ax.set_yticks([-0.2, 0, 0.2], [fs.MINUS + "0.2", "0", "0.2"])
    fs.clean(ax)
    ax.set_xlabel("Real part", labelpad=1)
    ax.set_ylabel("Imaginary part", labelpad=1)


# ---------------------------------------------------------------- d: transient amplification

def panel_d(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "d")
    af = affine()
    model_key(cv, x0 + 3.6, y0 + 1.5)
    cv.text(x0 + 3.6, y0 + 3.9, "largest singular value of the propagator", size=fs.PT_SMALL, va="center")
    ax = cv.axes(x0 + 9.0, y0 + AX_TOP, w - 9.4, AX_H)
    for model, long, short, key in MODELS[::-1]:
        _, d = af[model]
        hz, gain = d["propagation_horizons_ms"], d["max_singular_gain"]
        ax.plot(hz, gain, marker="o", ms=2.0, mew=0, **model_style(key), zorder=3 if key == "m0" else 2)
        for a_, b_ in zip(hz, gain):
            sd.add("d", "gain", model=model, horizon_ms=int(a_), max_singular_gain=float(b_))
    ax.axhline(1.0, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 20, 40, 80, 160], ["1", "20", "40", "80", "160"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.8, 200)
    ax.set_ylim(0.05, 3.0)
    ax.set_yticks([0.1, 1], ["0.1", "1"])
    fs.clean(ax)
    ax.set_xlabel("Horizon (ms)", labelpad=1)
    ax.set_ylabel("Gain", labelpad=1)


# ---------------------------------------------------------------- e: demixed reconstruction

def panel_e(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "e")
    a = cm.table("extended/ed09/ed09a_decoding_all.csv")
    a = a[a["record_type"] == "demixed_reconstruction"]
    factors = [("factor:time", "time", fs.TEXT, "o"), ("factor:cue_time", "cue × time", pal.CUE[1], "o"),
               ("factor:context_time", "context × time", "#7A7A7A", "s"), ("factor:cue_context_time", "cue × context × time", pal.CUE[0], "s")]
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="mark", s=6, fc=c, ec=c, marker=mk)], lab) for _, lab, c, mk in factors[:2]], gap=1.6)
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="mark", s=6, fc=c, ec=c, marker=mk)], lab) for _, lab, c, mk in factors[2:]], gap=1.6)
    ax = cv.axes(x0 + 9.0, y0 + AX_TOP, w - 9.4, AX_H)
    for i, (model, long, short, key) in enumerate(MODELS):
        for f_i, (fac, lab, col, mk) in enumerate(factors):
            r = a[(a["model"] == model) & (a["method"] == fac)]
            if r.empty:
                continue
            v = float(r["value"].iloc[0])
            ax.plot([v], [i + (f_i - 1.5) * 0.16], marker=mk, ms=2.4, mfc=col, mec=col, ls="none", zorder=3)
            sd.add("e", "demixed", model=model, factor=fac, heldout_r2=v, components=json.loads(r["details"].iloc[0]).get("components"))
    ax.axvline(0, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=1)
    ax.set_yticks(range(len(MODELS)), [m[2] for m in MODELS])
    ax.set_ylim(len(MODELS) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
    ax.set_xlim(-0.4, 0.6)
    ax.set_xticks([-0.2, 0, 0.2, 0.4], [fs.MINUS + "0.2", "0", "0.2", "0.4"])
    fs.clean(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Held-out R²", labelpad=1)


# ---------------------------------------------------------------- f: static cross-map decoding

def panel_f(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "f")
    a = cm.table("extended/ed09/ed09a_decoding_all.csv")
    s = a[a["record_type"] == "static_cross_family_decoding"].copy()
    det = s["details"].apply(json.loads)
    s["majority"] = det.apply(lambda d: d["majority_baseline"])
    s["states"] = det.apply(lambda d: d["states"])
    s["correct"] = det.apply(lambda d: d["correct"])
    s[["phase", "source", "target"]] = s["method"].str.split(":", expand=True)
    s["excess"] = s["value"] - s["majority"]
    assert s["model"].nunique() == 8
    targets = [("push", "push"), ("solvable_up", "solvable up"), ("solvable_down", "solvable down"), ("solvable_left", "solvable left"),
               ("solvable_right", "solvable right")]
    conds = [("features", "before", "brain state, before training", "white", fs.TEXT), ("features", "after", "brain state, after training", fs.TEXT,
                                                                                       fs.TEXT),
             ("boards", "after", "board input", "#BDBDBD", "#BDBDBD")]
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="mark", s=6, fc=fc, ec=ec, lw=0.5)], lab) for _, _, lab, fc, ec in conds], gap=1.8)
    cv.text(x0 + 3.6, y0 + 3.9, "decoders fitted on the other maps, tested on each T1 map (8 maps; bars: mean)", size=fs.PT_SMALL, va="center")
    ax = cv.axes(x0 + 9.0, y0 + AX_TOP, w - 9.4, AX_H)
    rng = np.random.default_rng(29)
    for t_i, (t, tlab) in enumerate(targets):
        for c_i, (src, ph, lab, fc, ec) in enumerate(conds):
            d = s[(s["target"] == t) & (s["source"] == src) & (s["phase"] == ph)].sort_values("model")
            assert len(d) == 8
            xc = t_i + (c_i - 1) * 0.26
            xx = xc + (rng.random(len(d)) - 0.5) * 0.12
            ax.scatter(xx, d["excess"], s=5.0, facecolor=fc, edgecolor=ec, linewidth=0.5, zorder=3)
            ax.plot([xc - 0.1, xc + 0.1], [d["excess"].mean()] * 2, color=fs.TEXT if src == "features" else "#7A7A7A", lw=1.0, zorder=4)
            for r in d.itertuples():
                sd.add("f", "decoding", map=r.model, target=t, source=src, phase=ph, accuracy=float(r.value), majority_baseline=float(r.majority),
                       states=int(r.states), correct=int(r.correct))
    same = s[s["source"] == "boards"].pivot_table(index=["model", "target"], columns="phase", values="value")
    assert np.allclose(same["before"], same["after"])
    ax.axhline(0, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=1)
    ax.set_xticks(range(len(targets)), [tl for _, tl in targets])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(targets) - 0.4)
    ax.set_ylim(-0.75, 0.6)
    ax.set_yticks([-0.5, 0, 0.5], [fs.MINUS + "0.5", "0", "0.5"])
    fs.clean(ax)
    ax.set_ylabel("Accuracy − majority", labelpad=1)


# ---------------------------------------------------------------- g: decoder vs behaviour

def panel_g(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "g")
    a = cm.table("extended/ed09/ed09a_decoding_all.csv")
    x = a[a["record_type"] == "decoder_behaviour_cross"]
    rows = [json.loads(d) for d in x["details"]]
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="mark", s=6, fc=fs.TEXT, ec=fs.TEXT)], "decoder right"),
                                ([dict(kind="mark", s=6, fc="white", ec=fs.TEXT, lw=0.5)], "decoder wrong")], gap=1.6)
    cv.text(x0 + 3.6, y0 + 3.9, "histories solved (95% CI)", size=fs.PT_SMALL, va="center")
    lines = []
    for model, long, short, key in MODELS:
        for vis in (False, True):
            for split, slab in (("held_time", "timing"), ("held_input", "input")):
                rr = [r for r in rows if r["model"] == model and r["split"] == split and bool(r["visible"]) == vis]
                if not rr:
                    continue
                lines.append((f"{short}{', visible' if vis else ''}, {slab}", rr, model, split, vis))
    lab_w = max(fs.text_width(l[0], fs.PT_SMALL) for l in lines) + 1.0
    ax = cv.axes(x0 + lab_w, y0 + AX_TOP, w - lab_w - 0.6, AX_H)
    for i, (lab, rr, model, split, vis) in enumerate(lines):
        for correct, fc, dy in ((1, fs.TEXT, -0.15), (0, "white", 0.15)):
            sub = [r for r in rr if int(r["correct"]) == correct]
            n = sum(r["histories"] for r in sub)
            k = sum(r["histories"] for r in sub if r["solved"])
            if n == 0:
                continue
            lo, hi = clopper(k, n)
            ax.plot([lo, hi], [i + dy] * 2, color=fs.TEXT, lw=0.5, solid_capstyle="butt", zorder=2)
            ax.plot([k / n], [i + dy], marker="o", ms=2.4, mfc=fc, mec=fs.TEXT, mew=0.5, ls="none", zorder=3)
            sd.add("g", "cross", model=model, split=split, visible=vis, decoder_correct=bool(correct), histories=n, solved=k, ci_low=lo, ci_high=hi)
    ax.set_yticks(range(len(lines)), [l[0] for l in lines])
    ax.set_ylim(len(lines) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    fs.clean(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share solved", labelpad=1)


def cm_u(n):
    return n * U + (n - 1) * fs.GAP


def main():
    cv = fs.Canvas(fs.FULL, H, ppi_min=300)
    sd = qc.SourceData("ed09")
    for rel in ("extended/ed09/ed09a_decoding_all.csv", "extended/ed09/ed09b_dynamics_all.csv", "figure_data/fd_time_generalisation.csv"):
        sd.source(cm.RESULTS / rel)
    panel_a(cv, sd)
    panel_b(cv, sd, XS[0], ROW2, cm_u(2), ROW_H)
    panel_c(cv, sd, XS[2], ROW2, cm_u(1), ROW_H)
    panel_d(cv, sd, XS[3], ROW2, cm_u(1), ROW_H)
    panel_e(cv, sd, XS[0], ROW3, cm_u(1), ROW_H)
    panel_f(cv, sd, XS[1], ROW3, cm_u(2), ROW_H)
    panel_g(cv, sd, XS[3], ROW3, cm_u(1), ROW_H)
    cm.finish(cv, "ed09", sd, kind="extended")


if __name__ == "__main__":
    main()
