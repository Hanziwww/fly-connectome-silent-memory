"""Fig. 7 - Use of the cue memory is tuned to the trained timings (180 x 149 mm).

Row 1  a  every cue pair of 21 learner configurations at the 9 trained and 26 untrained timings (split squares: cue 0 | cue 1
          solved), cue pairs with both vs neither cue solved at untrained timings, sign test (Holm)
       b  dense delay scan of the M0 readout-only model (20-ms steps): behaviour and cue decoding per cue, trained timings outlined,
          and the share of cue pairs with both cues solved or decoded at each delay
Row 2  c  first-decision margin of every history at trained and untrained timings for 12 learners
       d  cue decoding on held-out timings and held-out inputs (accuracy with 95% CI; cue-pair tests, Holm)
       e  share of the cue-difference energy in the 3-dimensional readout subspace per timing vs a random 3-of-512 subspace
Row 3  f  state prediction from persistence, autonomous affine dynamics or input and phase, by horizon (held-out timings, held-out inputs)
       g  matched-duration control: delays filled with neutral input vs blank delays of the same length (discordant pairs, Holm)

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\fig7.py
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, palette as pal, qc  # noqa: E402

U = (fs.FULL - 3 * fs.GAP) / 4
XS = [k * (U + fs.GAP) for k in range(4)]
ROW1, R1H = 0.0, 59.0
ROW2, ROW3, ROW_H = 63.0, 108.0, 41.0
H = ROW3 + ROW_H
AX_TOP, AX_H = 6.5, 27.0
A_W = 105.0
EDGE = "#ABABAB"
TRAINED = [(d, p) for d in (0, 40, 160) for p in (0, 1, 4)]
DELAYS, PREFIXES = (0, 20, 40, 80, 160, 320, 640), (0, 1, 2, 4, 8)
UNTRAINED = [(d, p) for d in DELAYS for p in PREFIXES if (d, p) not in TRAINED]
ROWS = [("M0", "remote-dynamic-M0-head_only", False, "readout-only"), ("M0", "remote-dynamic-M0-joint", False, "joint"),
        ("M0", "supp-M0-visibility", False, "shared, goal hidden"), ("M0", "supp-M0-visibility", True, "shared, goal visible"),
        ("M0", "remote-dynamic-M0-carry_untrained", False, "untrained readout"), ("M0", "remote-dynamic-M0-reset", False, "state reset (memoryless)"),
        ("M0", "remote-dynamic-M0-gru", False, "GRU"),
        ("S1", "supp-S1-ring-brain", False, "readout-only"), ("S1", "supp-origin-S1-ring-p05-fresh", False, "new readout"),
        ("S1", "supp-origin-S1-ring-neutral-real", False, "neutral gains, real graph"),
        ("S1", "supp-origin-S1-ring-neutral-rewired", False, "neutral gains, rewired"), ("S1", "supp-transfer-S1-ring", False, "M0 readout, no updates"),
        ("S1", "supp-S1-ring-gru", False, "GRU"), ("S1", "supp-origin-S1-ring-leaky", False, "fixed decaying trace"),
        ("S2", "supp-S2-context-brain", False, "readout-only"), ("S2", "supp-origin-S2-context-p05-fresh", False, "new readout"),
        ("S2", "supp-origin-S2-context-neutral-real", False, "neutral gains, real graph"),
        ("S2", "supp-origin-S2-context-neutral-rewired", False, "neutral gains, rewired"), ("S2", "supp-transfer-S2-context", False, "M0 readout, no updates"),
        ("S2", "supp-S2-context-gru", False, "GRU"), ("S2", "supp-origin-S2-context-leaky", False, "fixed decaying trace")]
CONFIGS = [("remote-dynamic-M0-head_only", False, "M0", "#1A1A1A"), ("remote-dynamic-M0-joint", False, "M0 joint", "#8C8C8C"),
           ("supp-M0-visibility", False, "shared, goal hidden", "#9CCDEE"), ("supp-M0-visibility", True, "shared, goal visible", "#2C8FD0"),
           ("supp-S1-ring-brain", False, "S1", "#8E24AA"), ("supp-S2-context-brain", False, "S2", "#00897B")]
METHODS = [("persistence", "persistence", "#BDBDBD"), ("affine", "autonomous affine", "#7A7A7A"), ("input_phase", "input and phase", "#1A1A1A")]


def truthy(s):
    return s.astype(str).str.lower().eq("true")


def split_square(ax, x, y, s, solved0, solved1, edge=EDGE, lw=0.3):
    """A cue pair: left half lit in the cue-0 colour if cue 0 was solved, right half in the cue-1 colour if cue 1 was."""
    if solved0:
        ax.add_patch(Rectangle((x, y), s / 2, s, facecolor=pal.CUE[0], edgecolor="none", linewidth=0.5, zorder=2))
    if solved1:
        ax.add_patch(Rectangle((x + s / 2, y), s / 2, s, facecolor=pal.CUE[1], edgecolor="none", linewidth=0.5, zorder=2))
    ax.add_patch(Rectangle((x, y), s, s, facecolor="none", edgecolor=edge, linewidth=lw, zorder=3))


def p_compact(p):
    """P as plain decimals down to 0.0001 (rows 1.7 mm apart cannot carry superscripts), scientific below."""
    return f"{p:#.2g}" if 1e-4 <= p < 1e-3 else fs.fmt_p(p)


def clopper(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return float(lo), float(hi)


# ---------------------------------------------------------------- a: cue-pair raster

def panel_a(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "a")
    e = cm.table("extended/ed05/ed05a_memory_behaviour_all.csv")
    e = e.assign(ok=truthy(e["solved"]), vis=truthy(e["visible"]))
    cls = cm.table("extended/ed05/ed05c_cue_pair_classes.csv").assign(vis=lambda d: truthy(d["visible"]))
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    ax = bd.mm_axes(cv, x0, y0, w, h)
    lab_r, pitch, sq = 23.6, 1.7, 1.44
    tx0 = lab_r + 0.9
    ux0 = tx0 + len(TRAINED) * pitch + 1.4
    cx = ux0 + len(UNTRAINED) * pitch + 1.2
    px = w - 0.2
    top = 5.2
    ax.text(tx0 + len(TRAINED) * pitch / 2, top - 1.0, "trained", ha="center", va="bottom", fontsize=fs.PT_SMALL)
    ax.text(ux0 + len(UNTRAINED) * pitch / 2, top - 1.0, "untrained timings (delay, then prefix)", ha="center", va="bottom", fontsize=fs.PT_SMALL)
    ax.text(cx, top - 1.0, "both:none", ha="left", va="bottom", fontsize=fs.PT_SMALL)
    ax.text(px, top - 1.0, "Holm " + fs.P_SYM, ha="right", va="bottom", fontsize=fs.PT_SMALL)
    y = top
    fam_rows = {}
    last_family = None
    for family, model, vis, label in ROWS:
        if last_family is not None and family != last_family:
            y += 1.0
        last_family = family
        d = e[(e["model"] == model) & (e["vis"] == vis)]
        contexts = sorted(d["context"].unique())
        assert set(map(tuple, d.loc[d["split"] == "train", ["delay_ms", "prefix_windows"]].drop_duplicates().to_numpy())) == set(TRAINED), model
        y_first = y
        for ctx in contexts:
            dc = d[d["context"] == ctx]
            ok = {(r.delay_ms, r.prefix_windows, r.cue): r.ok for r in dc.itertuples()}
            for j, t in enumerate(TRAINED + UNTRAINED):
                xx = (tx0 + j * pitch) if j < len(TRAINED) else (ux0 + (j - len(TRAINED)) * pitch)
                split_square(ax, xx, y, sq, ok[(t[0], t[1], 0)], ok[(t[0], t[1], 1)])
            if len(contexts) > 1 and family == "S2" and model == ROWS[14][1]:
                ax.text(tx0 - 0.5, y + sq / 2, f"{180 * ctx}°", ha="right", va="center", fontsize=fs.PT_SMALL)
            y += pitch
        yc = (y_first + y - pitch + sq) / 2
        ax.text(lab_r - (5.6 if (len(contexts) > 1 and model == ROWS[14][1]) else 0.0), yc, label, ha="right", va="center", fontsize=fs.PT_SMALL)
        r = cls[(cls["model"] == model) & (cls["vis"] == vis) & (cls["split"] == "untrained")].iloc[0]
        tid = r["test_id"]
        ax.text(cx, yc, f"{int(r['both_solved'])}:{int(r['none_solved'])}", ha="left", va="center", fontsize=fs.PT_SMALL)
        more_none = int(r["both_solved"]) <= int(r["none_solved"])        # grey: no excess of pairs with both cues solved
        ax.text(px, yc, p_compact(float(tests.loc[tid, "p_adjusted"])), ha="right", va="center", fontsize=fs.PT_SMALL,
                color="#8C8C8C" if more_none else fs.TEXT)
        fam_rows.setdefault(family, []).append((y_first, y - pitch + sq))
        tr = cls[(cls["model"] == model) & (cls["vis"] == vis) & (cls["split"] == "train")].iloc[0]
        sd.add("a", "learner", family=family, model=model, goal_visible=vis, label=label, trained_both=int(tr["both_solved"]),
               trained_one=int(tr["one_solved"]), trained_none=int(tr["none_solved"]), untrained_both=int(r["both_solved"]),
               untrained_one=int(r["one_solved"]), untrained_none=int(r["none_solved"]), p_holm=float(tests.loc[tid, "p_adjusted"]), test_id=tid)
        for hrow in d.itertuples():
            sd.add("a", "history", model=model, goal_visible=vis, context=int(hrow.context), delay_ms=int(hrow.delay_ms),
                   prefix_windows=int(hrow.prefix_windows), cue=int(hrow.cue), split=hrow.split, solved=bool(hrow.ok))
    for family, spans in fam_rows.items():
        y_a, y_b = spans[0][0], spans[-1][1]
        ax.plot([1.6, 1.6], [y_a, y_b], color=fs.TEXT, lw=0.5)
        ax.text(0.9, (y_a + y_b) / 2, family, ha="center", va="center", rotation=90, fontsize=fs.PT)
    ky = y0 + h - 1.6
    kx = x0 + 3.0
    k_ax = bd.mm_axes(cv, kx, ky - 0.8, 40, 1.6)
    split_square(k_ax, 0.0, 0.05, 1.48, True, False)
    cv.text(kx + 2.0, ky, "cue 0 solved", size=fs.PT_SMALL, va="center")
    split_square(k_ax, 13.0, 0.05, 1.48, False, True)
    cv.text(kx + 15.0, ky, "cue 1 solved", size=fs.PT_SMALL, va="center")
    cv.text(kx + 27.5, ky, "one square per cue pair; grey " + fs.P_SYM + ": no excess of both over none", size=fs.PT_SMALL, va="center")


# ---------------------------------------------------------------- b: dense delay scan

def panel_b(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "b")
    d = cm.table("extended/ed05/ed05d_dense_delay_scan.csv")
    d = d.assign(ok=truthy(d["solved"]), dec=truthy(d["decoded_correct"]), tr=truthy(d["train_time"]))
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    delays = sorted(d["delay_ms"].unique())
    ax = bd.mm_axes(cv, x0, y0, w, h)
    lab_w = 7.6
    pitch = (w - lab_w - 0.4) / len(delays)
    sq = pitch * 0.84
    strips = [("ok", "Behaviour, M0 readout-only (20-ms delay steps)", 3.6), ("dec", "Cue decoder, same histories", 18.4)]
    for key, title, top in strips:
        ax.text(lab_w, top - 0.8, title, ha="left", va="bottom", fontsize=fs.PT_SMALL)
        for i, p in enumerate(PREFIXES):
            yy = top + i * pitch
            ax.text(lab_w - 0.8, yy + sq / 2, str(p), ha="right", va="center", fontsize=fs.PT_SMALL)
            for j, dl in enumerate(delays):
                g = d[(d["delay_ms"] == dl) & (d["prefix_windows"] == p)].set_index("cue")
                trained = bool(g["tr"].all())
                split_square(ax, lab_w + j * pitch, yy, sq, bool(g.loc[0, key]), bool(g.loc[1, key]), edge=fs.TEXT if trained else EDGE,
                             lw=0.6 if trained else 0.3)
        ax.text(1.0, top + 2.5 * pitch, "prefix", ha="center", va="center", rotation=90, fontsize=fs.PT_SMALL)
    yl = 18.4 + 5 * pitch + 0.4
    for j, dl in enumerate(delays):
        if dl % 160 == 0:
            ax.text(lab_w + j * pitch + sq / 2, yl, str(dl), ha="center", va="top", fontsize=fs.PT_SMALL)
    ax.text(lab_w + (len(delays) * pitch) / 2, yl + 2.3, "Delay after the cue (ms); outlined: trained timings", ha="center", va="top",
            fontsize=fs.PT_SMALL)
    for dl in delays:
        for p in PREFIXES:
            g = d[(d["delay_ms"] == dl) & (d["prefix_windows"] == p)].set_index("cue")
            sd.add("b", "history pair", delay_ms=int(dl), prefix_windows=int(p), trained=bool(g["tr"].all()), cue0_solved=bool(g.loc[0, "ok"]),
                   cue1_solved=bool(g.loc[1, "ok"]), cue0_decoded=bool(g.loc[0, "dec"]), cue1_decoded=bool(g.loc[1, "dec"]))
    # summary: cue pairs with both, one or neither cue solved (decoded) at trained and untrained timings
    pairs = d.groupby(["delay_ms", "prefix_windows"]).agg(tr=("tr", "all"), ok=("ok", "sum"), dec=("dec", "sum"))
    tests_by = {"ok": tests.loc["ed05d.behaviour_pairs.untrained"], "dec": tests.loc["ed05d.decoder_pairs.untrained"]}
    tests_tr = {"ok": tests.loc["ed05d.behaviour_pairs.train"], "dec": tests.loc["ed05d.decoder_pairs.train"]}
    bx0, bw, by0, bh = lab_w + 20.0, w - lab_w - 20.0 - 9.4, 38.6, 3.0
    shades = ("#8C8C8C", "#D4D4D4", "white")
    rows = [("ok", True, "solved, trained"), ("ok", False, "solved, untrained"), ("dec", True, "decoded, trained"), ("dec", False, "decoded, untrained")]
    ax.text(bx0, by0 - 1.0, "both", ha="left", va="bottom", fontsize=fs.PT_SMALL)
    ax.text(bx0 + bw / 2, by0 - 1.0, "one", ha="center", va="bottom", fontsize=fs.PT_SMALL)
    ax.text(bx0 + bw, by0 - 1.0, "neither", ha="right", va="bottom", fontsize=fs.PT_SMALL)
    ax.text(w - 0.2, by0 - 1.0, "Holm " + fs.P_SYM, ha="right", va="bottom", fontsize=fs.PT_SMALL)
    for i, (key, trained, lab) in enumerate(rows):
        yy = by0 + i * (bh + 0.9) + (1.6 if i >= 2 else 0.0)
        sub = pairs[pairs["tr"] == trained][key]
        counts = [int((sub == 2).sum()), int((sub == 1).sum()), int((sub == 0).sum())]
        n = sum(counts)
        ax.text(bx0 - 0.8, yy + bh / 2, f"{lab} ({n})", ha="right", va="center", fontsize=fs.PT_SMALL)
        xx = bx0
        for c, shade in zip(counts, shades):
            wd = bw * c / n
            ax.add_patch(Rectangle((xx, yy), wd, bh, facecolor=shade, edgecolor=EDGE if shade == "white" else "none", linewidth=0.3))
            if c and wd > 3.0:
                ax.text(xx + wd / 2, yy + bh / 2, str(c), ha="center", va="center", fontsize=fs.PT_SMALL)
            xx += wd
        t = (tests_tr if trained else tests_by)[key]
        ax.text(w - 0.2, yy + bh / 2, fs.fmt_p(float(t["p_adjusted"])), ha="right", va="center", fontsize=fs.PT_SMALL)
        assert counts[0] == int(t["statistic"]) and counts[0] + counts[2] == int(t["n"]), (key, trained, counts, t["statistic"], t["n"])
        sd.add("b", "pair classes", measure="behaviour" if key == "ok" else "cue decoder", trained=trained, both=counts[0], one=counts[1],
               neither=counts[2], p_holm=float(t["p_adjusted"]), test_id=t.name)


# ---------------------------------------------------------------- c: first-decision margins

C_LEARNERS = [("M0", "remote-dynamic-M0-head_only", False, "readout\nonly"), ("M0", "remote-dynamic-M0-joint", False, "joint"),
              ("M0", "supp-M0-visibility", False, "goal\nhidden"), ("M0", "supp-M0-visibility", True, "goal\nvisible"),
              ("M0", "remote-dynamic-M0-reset", False, "state\nreset"), ("M0", "remote-dynamic-M0-gru", False, "GRU"),
              ("S1", "supp-S1-ring-brain", False, "readout\nonly"), ("S1", "supp-S1-ring-gru", False, "GRU"),
              ("S1", "supp-origin-S1-ring-leaky", False, "fixed\ntrace"),
              ("S2", "supp-S2-context-brain", False, "readout\nonly"), ("S2", "supp-S2-context-gru", False, "GRU"),
              ("S2", "supp-origin-S2-context-leaky", False, "fixed\ntrace")]
CLIP = 8.0


def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    e = cm.table("extended/ed05/ed05a_memory_behaviour_all.csv")
    e = e.assign(vis=truthy(e["visible"]))
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="mark", s=4, fc=fs.TEXT)], "trained timings"),
                                ([dict(kind="mark", s=4, fc="white", ec="#7A7A7A", lw=0.4)], "untrained timings"),
                                ([dict(kind="hline", w=2.4, lw=0.8)], "median")], gap=1.8)
    cv.text(x0 + 3.6, y0 + 3.9, "shared readout: one readout trained with the goal hidden and visible", size=fs.PT_SMALL, va="center")
    ax_l, ax_w = x0 + 8.4, w - 8.6
    ax = cv.axes(ax_l, y0 + AX_TOP, ax_w, AX_H)
    pos, fam_x = [], {}
    xpos = 0.0
    last = None
    for family, model, vis, lab in C_LEARNERS:
        if last is not None and family != last:
            xpos += 0.6
        last = family
        pos.append(xpos)
        fam_x.setdefault(family, []).append(xpos)
        xpos += 1.0
    rng = np.random.default_rng(7)
    for xi, (family, model, vis, lab) in zip(pos, C_LEARNERS):
        d = e[(e["model"] == model) & (e["vis"] == vis)]
        for split, off, filled in (("train", -0.2, True), ("untrained", 0.2, False)):
            m = d.loc[d["split"] == split, "first_decision_margin"].to_numpy(float)
            mc = np.clip(m, -CLIP, CLIP)
            xx = xi + off + (rng.random(len(mc)) - 0.5) * 0.24
            ax.scatter(xx, mc, s=2.2, facecolor=fs.TEXT if filled else "white", edgecolor=fs.TEXT if filled else "#7A7A7A", linewidth=0.3,
                       zorder=3 if filled else 2)
            med = float(np.median(m))
            ax.plot([xi + off - 0.16, xi + off + 0.16], [med, med], color=fs.TEXT, lw=0.9, zorder=4)
            sd.add("c", split, family=family, model=model, goal_visible=vis, histories=int(len(m)), median_margin=med,
                   share_correct=float((m > 0).mean()), clipped=int((np.abs(m) > CLIP).sum()))
            for v in m:
                sd.add("c", "history margin", model=model, goal_visible=vis, split=split, margin=float(v))
    ax.axhline(0, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=1)
    ax.set_xlim(-0.6, xpos - 0.4)
    ax.set_ylim(-CLIP - 0.5, CLIP + 0.5)
    ax.set_yticks([-8, -4, 0, 4, 8], [fs.MINUS + "8", fs.MINUS + "4", "0", "4", "8"])
    ax.set_xticks(pos, [""] * len(pos))
    ax.tick_params(axis="x", length=0)
    fs.minor(ax, x=False)
    fs.clean(ax)
    ax.set_ylabel("First-decision margin (logit)", labelpad=1)
    xm = lambda v: ax_l + (v + 0.6) / (xpos - 0.4 + 0.6) * ax_w  # noqa: E731
    for xi, (_, _, _, lab) in zip(pos, C_LEARNERS):
        for li, line in enumerate(lab.split("\n")):
            cv.text(xm(xi), y0 + AX_TOP + AX_H + 1.2 + li * 2.0, line, size=fs.PT_SMALL, ha="center", va="top")
    for family, xs_ in fam_x.items():
        cv.text(xm((xs_[0] + xs_[-1]) / 2), y0 + AX_TOP + 0.4, family, size=fs.PT, ha="center", va="bottom")


# ---------------------------------------------------------------- d: held-out decoding

def panel_d(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "d")
    f = cm.table("main/fig6/fig6f_decoding_dynamics.csv")
    f = f[f["record_type"] == "held_history_cue_decoding"].assign(ok=lambda x: truthy(x["correct"]), vis=lambda x: truthy(x["visible"]))
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="mark", s=6, fc=fs.TEXT)], "held-out timings"),
                                ([dict(kind="mark", s=6, fc="white", ec=fs.TEXT, lw=0.5)], "held-out inputs")], gap=1.6)
    ps = []
    ax = cv.axes(x0 + 7.4, y0 + AX_TOP, w - 7.8, AX_H)
    for i, (model, vis, lab, col) in enumerate(CONFIGS):
        for split, off, filled in (("held_time", -0.15, True), ("held_input", 0.15, False)):
            g = f[(f["model"] == model) & (f["vis"] == vis) & (f["split"] == split)]
            k, n = int(g["ok"].sum()), int(len(g))
            lo, hi = clopper(k, n)
            ax.plot([i + off, i + off], [lo, hi], color=col, lw=0.6, zorder=2)
            ax.plot([i + off], [k / n], marker="o", ms=2.6, mfc=col if filled else "white", mec=col, mew=0.6, ls="none", zorder=3)
            tid = f"fig6f.decoder_pairs.{model}{'.visible' if vis else ''}.{split}"
            ps.append(float(tests.loc[tid, "p_adjusted"]))
            sd.add("d", split, model=model, goal_visible=vis, correct=k, histories=n, accuracy=k / n, ci_low=lo, ci_high=hi,
                   pairs_both_correct=int(tests.loc[tid, "statistic"]), pairs_decided=int(tests.loc[tid, "n"]), p_holm=ps[-1], test_id=tid)
    ax.axhline(0.5, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=1)
    ax.set_xlim(-0.6, len(CONFIGS) - 0.4)
    ax.set_xticks([])
    ax.set_ylim(0.2, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    fs.clean(ax, bottom=True)
    ax.set_ylabel("Cue decoded (fraction)", labelpad=1)
    ax.set_xlabel(f"cue pairs: each Holm {fs.P_SYM} ≥ {np.floor(min(ps) * 100) / 100:.2f}", labelpad=2.0)


# ---------------------------------------------------------------- e: readout energy fraction

E_CONFIGS = [("remote-dynamic-M0-head_only", False, 0, "#1A1A1A"), ("remote-dynamic-M0-joint", False, 0, "#8C8C8C"),
             ("supp-M0-visibility", False, 0, "#9CCDEE"), ("supp-M0-visibility", True, 0, "#2C8FD0"),
             ("supp-S1-ring-brain", False, 0, "#8E24AA"), ("supp-S2-context-brain", False, 0, "#00897B"), ("supp-S2-context-brain", False, 1, "#7FCDC3")]


def panel_e(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "e")
    g = cm.table("main/fig6/fig6c_readout_geometry.csv")
    g = g[g["phase"] == "after"].assign(vis=lambda x: truthy(x["visible"]))
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    iso = float(g["isotropic_reference"].iloc[0])
    assert np.allclose(g["isotropic_reference"], iso) and abs(iso - 3 / 512) < 1e-9
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="mark", s=5, fc=fs.TEXT)], "trained"), ([dict(kind="mark", s=5, fc="white", ec=fs.TEXT, lw=0.5)],
                                                                                    "untrained timings")], gap=1.6)
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="hline", w=2.6, lw=0.5, ls=(0, (2, 1.5)))], "random 3 of 512 dimensions")])
    ax = cv.axes(x0 + 8.4, y0 + AX_TOP, w - 8.8, AX_H)
    rng = np.random.default_rng(11)
    for i, (model, vis, ctx, col) in enumerate(E_CONFIGS):
        d = g[(g["model"] == model) & (g["vis"] == vis) & (g["context"] == ctx)]
        for split, filled in (("untrained", False), ("train", True)):
            v = d.loc[d["split"] == split, "readout_energy_fraction"].to_numpy(float)
            xx = i + (rng.random(len(v)) - 0.5) * 0.5
            ax.scatter(xx, v, s=3.2, facecolor=col if filled else "white", edgecolor=col, linewidth=0.4, zorder=3 if filled else 2)
            for vv in v:
                sd.add("e", split, model=model, goal_visible=vis, context=int(ctx), energy_fraction=float(vv))
    ax.axhline(iso, color=fs.AXIS, lw=0.5, ls=(0, (2, 1.5)), zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 0.08)
    ax.set_yticks([1e-3, 1e-2], ["0.001", "0.01"])
    fs.minor(ax, x=False)
    ax.set_xlim(-0.6, len(E_CONFIGS) - 0.4)
    ax.set_xticks([])
    fs.clean(ax, bottom=True)
    ax.set_ylabel("Cue energy in the readout", labelpad=1)
    sig = tests.loc["fig6c.sign.supp-S2-context-brain"]
    ax.set_xlabel(f"S2 below random ({fs.P_SYM} = {fs.fmt_p(float(sig['p_adjusted']))}); others n.s.", labelpad=2.0, fontsize=fs.PT_SMALL)
    for tid in [t for t in tests.index if t.startswith("fig6c.sign.")]:
        sd.add("e", "sign test", test_id=tid, timings_above=float(tests.loc[tid, "statistic"]), timings=int(tests.loc[tid, "n"]),
               p_holm=float(tests.loc[tid, "p_adjusted"]))


# ---------------------------------------------------------------- f: state prediction by horizon

def panel_f(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "f")
    f = cm.table("main/fig6/fig6f_decoding_dynamics.csv")
    f = f[f["record_type"] == "input_conditioned_dynamics_r2"]
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="hline", colour=c, w=2.6, lw=0.9)], lab) for _, lab, c in METHODS], gap=1.8)
    cv.text(x0 + 3.6, y0 + 3.9, "thick: M0 readout-only; thin: M0 joint, shared readout, S1, S2", size=fs.PT_SMALL, va="center")
    aw = (w - 9.0 - 6.0) / 2
    for k, (split, title) in enumerate((("held_time", "held-out timings"), ("held_input", "held-out inputs"))):
        ax = cv.axes(x0 + 8.4 + k * (aw + 6.0), y0 + AX_TOP + 2.4, aw, AX_H - 2.4)
        for meth, lab, col in METHODS:
            for model in sorted(f["model"].unique()):
                g = f[(f["split"] == split) & (f["method"] == meth) & (f["model"] == model)].sort_values("horizon_ms")
                thick = model == "remote-dynamic-M0-head_only"
                ax.plot(g["horizon_ms"], g["value"], color=col, lw=1.0 if thick else 0.45, marker="o" if thick else None, ms=2.0, mew=0,
                        zorder=3 if thick else 2)
                for r in g.itertuples():
                    sd.add("f", split, model=model, method=meth, horizon_ms=int(r.horizon_ms), r2=float(r.value))
        ax.axhline(0, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=1)
        ax.set_xscale("log")
        ax.set_xticks([1, 20, 40, 80], ["1", "20", "40", "80"])
        ax.xaxis.set_minor_locator(__import__("matplotlib.ticker", fromlist=["NullLocator"]).NullLocator())
        ax.set_xlim(0.8, 100)
        ax.set_ylim(-2.0, 1.05)
        ax.set_yticks([-2, -1, 0, 1], [fs.MINUS + "2", fs.MINUS + "1", "0", "1"])
        fs.clean(ax)
        ax.set_title(title, fontsize=fs.PT_SMALL, pad=1.5)
        ax.set_xlabel("Prediction horizon (ms)", labelpad=1)
        if k == 0:
            ax.set_ylabel("State prediction R²", labelpad=1)


# ---------------------------------------------------------------- g: matched-duration control

G_FAMILIES = [("M0", "M0"), ("M0joint", "M0 joint"), ("M0vis.hidden", "shared, goal hidden"), ("M0vis.visible", "shared, goal visible"),
              ("S1", "S1"), ("S2", "S2")]
G_INPUTS = [("neutral_stationary", "stationary"), ("neutral_walk_0", "walk 0"), ("neutral_walk_1", "walk 1")]


def panel_g(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "g")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    ax = bd.mm_axes(cv, x0, y0, w, h)
    lab_w = 25.0
    cw = (w - lab_w - 0.4) / len(G_INPUTS)
    top = 9.4
    rh = (h - top - 4.6) / len(G_FAMILIES)
    scale = 0.55                                       # mm per discordant history
    cv.text(x0 + 3.6, y0 + 1.5, "Delay of the same total length, filled with neutral input or left blank:", size=fs.PT_SMALL, va="center")
    cv.text(x0 + 3.6, y0 + 3.9, "histories solved in one condition only", size=fs.PT_SMALL, va="center")
    for j, (inp, lab) in enumerate(G_INPUTS):
        cxm = lab_w + (j + 0.5) * cw
        ax.text(cxm, top - 2.6, f"neutral {lab}", ha="center", va="bottom", fontsize=fs.PT_SMALL)
        ax.text(cxm - 0.6, top - 0.5, "blank", ha="right", va="bottom", fontsize=fs.PT_SMALL)
        ax.text(cxm + 0.6, top - 0.5, "neutral", ha="left", va="bottom", fontsize=fs.PT_SMALL)
    ps = []
    for i, (fam, flab) in enumerate(G_FAMILIES):
        yy = top + i * rh
        ax.text(lab_w - 1.0, yy + rh / 2, flab, ha="right", va="center", fontsize=fs.PT_SMALL)
        for j, (inp, lab) in enumerate(G_INPUTS):
            t = tests.loc[f"fig5c.mcnemar.{fam}.{inp}"]
            a, b = int(t["a_only"]), int(t["b_only"])
            ps.append(float(t["p_adjusted"]))
            cxm = lab_w + (j + 0.5) * cw
            ax.add_patch(Rectangle((cxm, yy + rh * 0.2), a * scale, rh * 0.6, facecolor="#7A7A7A", edgecolor="none", linewidth=0.5))
            ax.add_patch(Rectangle((cxm - b * scale, yy + rh * 0.2), b * scale, rh * 0.6, facecolor="#C8C8C8", edgecolor="none", linewidth=0.5))
            ax.plot([cxm, cxm], [yy + 0.25, yy + rh - 0.25], color=fs.TEXT, lw=0.35)
            ax.text(cxm + a * scale + 0.5, yy + rh / 2, str(a), ha="left", va="center", fontsize=fs.PT_SMALL)
            ax.text(cxm - b * scale - 0.5, yy + rh / 2, str(b), ha="right", va="center", fontsize=fs.PT_SMALL)
            sd.add("g", "McNemar", family=fam, neutral_input=inp, neutral_only=a, blank_only=b, both=int(t["both"]), neither=int(t["neither"]),
                   p_holm=float(t["p_adjusted"]), test_id=f"fig5c.mcnemar.{fam}.{inp}")
    cv.text(x0 + lab_w, y0 + h - 1.8, f"all 18 comparisons: Holm {fs.P_SYM} ≥ {np.floor(min(ps) * 100) / 100:.2f} (McNemar, pooled over durations)",
            size=fs.PT_SMALL, va="center")


def legend_configs(cv, x, y):
    items = [([dict(kind="mark", s=6, fc=c, ec=c)], lab) for _, _, lab, c in CONFIGS[:4]] + \
            [([dict(kind="mark", s=6, fc="#8E24AA")], "S1"), ([dict(kind="mark", s=6, fc="#00897B")], "S2 (0°)"),
             ([dict(kind="mark", s=6, fc="#7FCDC3")], "S2 (180°)")]
    return cv.key(x, y, items, gap=1.6)


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("fig7")
    for rel in ("extended/ed05/ed05a_memory_behaviour_all.csv", "extended/ed05/ed05c_cue_pair_classes.csv", "extended/ed05/ed05d_dense_delay_scan.csv",
                "main/fig6/fig6c_readout_geometry.csv", "main/fig6/fig6f_decoding_dynamics.csv", "main/fig5/fig5c_matched_duration.csv",
                "stats/statistical_tests.csv"):
        sd.source(cm.RESULTS / rel)
    panel_a(cv, sd, 0.0, ROW1, A_W, R1H)
    panel_b(cv, sd, A_W + fs.GAP, ROW1, fs.FULL - A_W - fs.GAP, R1H)
    panel_c(cv, sd, XS[0], ROW2, 2 * U + fs.GAP, ROW_H)
    panel_d(cv, sd, XS[2], ROW2, U, ROW_H)
    panel_e(cv, sd, XS[3], ROW2, U, ROW_H)
    legend_configs(cv, XS[2] + 0.6, ROW2 + ROW_H - 1.0)
    panel_f(cv, sd, XS[0], ROW3, 2 * U + fs.GAP, ROW_H)
    panel_g(cv, sd, XS[2], ROW3, 2 * U + fs.GAP, ROW_H)
    cm.finish(cv, "fig7", sd)


if __name__ == "__main__":
    main()
