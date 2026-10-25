"""Video S4 (Cell; related to Figure 7) - lost access, lost repair: one memory trial in each brain state, and the trace moved
between brain states.

Scene 1, readout trained, 160-ms delay, 4 input windows, lesion dose 1 (the lesion link, amendment 0015). Act 1, the regimes
of Figure 7B in five columns: healthy; the lesion after learning; learning under the lesion; relearned under the lesion;
restored late. Act 2, Figure 7J's two-by-two for the readout trained healthy: written and read healthy; written healthy, read
lesioned; written lesioned, read healthy; written and read lesioned (the full state moved at the end of the delay, marked by a
white frame). Each column shows the cue-1 history's spikes ms by ms (cell-group colours, 4-ms afterglow), the brain state as
two chips (gray healthy, purple lesioned; act 1: learned and tested, act 2: written and read; until the move a moved trace shows
its writing brain's spikes), and both cues' histories on the corridor board with the cued target outlined, ending in a green
(solved) or red (failed) mark. Two live charts per column: the cells spiking per ms in the cue-1 history (on the video's own
time axis, phases banded), and per cue the readout's probability of a correct move at each decision (its softmax over the legal
moves, summed over the moves on a shortest solution toward the cued target; the game's own rules and distances).

Data: manuscript/results/plan_v2/video_lesion_spikes.csv, video_lesion_decisions.csv (P12, gated against the study table and
the readout jobs); the frozen memory-task boards (studies/brain-sokoban-v1/tasks/memory.json, read only); the board rules
(sokoban/lib/game.py, read only); soma positions from fd_cells.csv.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_video_s4.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import palette as pal  # noqa: E402
from render import project as rp, scenes as sc, video as vd  # noqa: E402

sys.path.insert(0, str(rp.ROOT / "sokoban" / "lib"))
import game  # noqa: E402  (pure numpy board logic from sokoban/lib, read only)

OUT = FIG / "v4" / "videos"
PLAN = rp.RESULTS / "plan_v2"
MEMORY = rp.ROOT / "studies" / "brain-sokoban-v1" / "tasks" / "memory.json"
ACTS = [("Four regimes, one rule", "learned", "tested",
         [("healthy", "H", "H"), ("lesion after learning", "H", "L"), ("learning under the lesion", "L", "L"),
          ("relearned under the lesion", "L", "L"), ("restored late", "L", "H")]),
        ("The trace moved between brain states", "written", "read",
         [("healthy", "H", "H"), ("written healthy, read lesioned", "H", "L"), ("written lesioned, read healthy", "L", "H"),
          ("lesion after learning", "L", "L")])]
HEADS = {"healthy": "healthy", "lesion after learning": "lesion after learning", "learning under the lesion": "learning under the lesion",
         "relearned under the lesion": "relearned under the lesion", "restored late": "restored late",
         "written healthy, read lesioned": "written healthy, read lesioned", "written lesioned, read healthy": "written lesioned, read healthy"}
STATE_COL = {"H": (200, 200, 200), "L": (123, 50, 148)}
STAGE = (330, 390)
Y_STAGE, Y_SPARK, Y_BOARD, Y_P = 256, 686, 752, 928   # tops of the CNS panel, the activity sparkline, the boards and the readout chart
DELAY_END, INPUT_END = 200, 360
GLOW_TAU, GLOW_MIN = 4.0, 0.05
CELL, CROP = 22, (1, 2, 8, 7)                     # board cell (px) and the crop (col0, row0, col1, row1) around the corridor
FRAMES_PER_MS = {"cue": 2.0, "delay": 1.0, "input": 1.0, "decision": 0.5}
OK, NO = (120, 200, 120), (230, 90, 90)
DELTAS = (-10, 10, -1, 1)
CUE_COL = {0: (110, 170, 255), 1: (214, 160, 60)}
PHASE_COLOUR = {"cue": (214, 160, 60), "delay": (90, 90, 90), "input": (92, 200, 255), "decision": (236, 236, 236)}


def phase(t):
    return "cue" if t <= 40 else "delay" if t <= DELAY_END else "input" if t <= INPUT_END else "decision"


def source(act, name, a, b, t):
    """Whose spikes a column shows at ms t: its own, except that a trace moved between brain states (act 2) shows its writing
    brain's spikes until the move at the end of the delay (the receiving brain's own writing is discarded by the move)."""
    if act == 0 or a == b or t > DELAY_END:
        return name
    return "healthy" if a == "H" else "lesion after learning"


def board_img(walls, goal, player, box, show_goal_fill):
    c0, r0, c1, r1 = CROP
    img = Image.new("RGB", ((c1 - c0) * CELL, (r1 - r0) * CELL))
    d = ImageDraw.Draw(img)
    for r in range(r0, r1):
        for c in range(c0, c1):
            i = r * 10 + c
            x, y = (c - c0) * CELL, (r - r0) * CELL
            d.rectangle([x, y, x + CELL - 1, y + CELL - 1], fill=(70, 70, 70) if i in walls else (22, 22, 22))
            if i == goal:
                m = CELL // 5
                d.rectangle([x + m, y + m, x + CELL - 1 - m, y + CELL - 1 - m], outline=(150, 150, 150) if not show_goal_fill else (236, 236, 236),
                            width=2)
    for i, col in ((box, (214, 160, 60)),):
        r, c = divmod(i, 10)
        m = CELL // 8
        d.rectangle([(c - c0) * CELL + m, (r - r0) * CELL + m, (c - c0 + 1) * CELL - 1 - m, (r - r0 + 1) * CELL - 1 - m], fill=col)
    r, c = divmod(player, 10)
    m = CELL // 5
    d.ellipse([(c - c0) * CELL + m, (r - r0) * CELL + m, (c - c0 + 1) * CELL - 1 - m, (r - r0 + 1) * CELL - 1 - m], fill=(236, 236, 236))
    return img


def moves(dec):
    """(player, box) before every decision and after the last one, with the history's status."""
    states = [(int(r.player), int(r.box)) for r in dec.itertuples()]
    p, b = states[-1]
    a = int(dec.iloc[-1]["action"])
    q = p + DELTAS[a]
    states.append((q, b + DELTAS[a] if q == b else b))
    return states, str(dec.iloc[-1]["status"])


def p_correct(dec, level, dist):
    """Per decision, the readout's probability of a move on a shortest solution toward the cued target (softmax over the legal
    moves); NaN where the box can no longer reach the target. Gate: each chosen move is the readout's legal argmax."""
    out = []
    for r in dec.itertuples():
        s_ = (int(r.player), (int(r.box),))
        legal = game.legal(level, s_)
        q = np.array([getattr(r, f"logit_{a}") for a in range(4)], float)
        if not legal[int(r.action)] or int(np.where(legal, q, -np.inf).argmax()) != int(r.action):
            raise RuntimeError(f"decision {r.step} of {r.combination}, cue {r.cue}: the move is not the readout's legal choice")
        if s_ not in dist:
            out.append(np.nan)
            continue
        pr = np.exp(q - q[legal].max()) * legal
        out.append(float(pr[game.optimal(level, s_, dist)].sum() / pr.sum()))
    return np.array(out)


def main():
    t0 = time.time()
    task = next(t for t in json.loads(MEMORY.read_text(encoding="utf-8")) if t["id"] == "M0")
    walls = set(task["levels"][0]["walls"])
    goals = [task["levels"][c]["goals"][0] for c in (0, 1)]
    levels = [game.restore(task["levels"][c]) for c in (0, 1)]
    dists = [game.distances(lv) for lv in levels]
    sp = pd.read_csv(PLAN / "video_lesion_spikes.csv")
    decs = pd.read_csv(PLAN / "video_lesion_decisions.csv")
    fd = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False)
    s, xyz = sc.somata(fd)
    pos = pd.Series(np.arange(len(s)), index=s["cell"].to_numpy())
    group = s["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other").to_numpy()
    colour = rp.srgb_to_linear(np.array([to_rgb(pal.ATLAS_ON_DARK[g]) for g in group]))
    stage = vd.Stage(xyz, *STAGE)
    px, py = stage.project(xyz)
    dens = sc.density_weights(stage.view, xyz, power=0.45)
    decay_g = np.exp(-1.0 / GLOW_TAU)
    hist = {}
    for name in {c for _, _, _, cols in ACTS for c, _, _ in cols}:
        q = sp[(sp["combination"] == name) & (sp["cue"] == 1)]
        by = {t: pos.reindex(g["cell"]).dropna().astype(int).to_numpy() for t, g in q.groupby("ms")}
        dc = {c: decs[(decs["combination"] == name) & (decs["cue"] == c)].sort_values("step") for c in (0, 1)}
        dd = {c: moves(dc[c]) for c in (0, 1)}
        hist[name] = dict(by=by, moves=dd, end=INPUT_END + 40 * (len(dd[1][0]) - 1),   # no spikes after the shown history ends
                          count=q.groupby("ms").size(), p={c: p_correct(dc[c], levels[c], dists[c]) for c in (0, 1)})
    n_dec = max(len(v["moves"][c][0]) - 1 for v in hist.values() for c in (0, 1))
    last_ms = INPUT_END + 40 * n_dec

    # live metrics on the video's own time axis (frames), so the short cue, delay and input are not squeezed by the decisions
    ms = np.arange(1, last_ms + 1)
    fpm = np.array([FRAMES_PER_MS[phase(t)] for t in ms])
    xpos = np.cumsum(fpm) - fpm / 2
    spans = [(1, 40), (41, DELAY_END), (DELAY_END + 1, INPUT_END), (INPUT_END + 1, last_ms)]
    bands = [(xpos[a - 1] - fpm[a - 1] / 2, xpos[b - 1] + fpm[b - 1] / 2, tuple(v // 7 for v in PHASE_COLOUR[phase(a)])) for a, b in spans]
    top_a = 10.0 ** np.ceil(np.log10(max(float(v["count"].max()) for v in hist.values())))

    def activity(act, name, a, b):
        v = np.array([hist[source(act, name, a, b, t)]["count"].get(t, 0) for t in ms], float)
        v[ms > hist[name]["end"]] = np.nan
        return v

    def kilo(v):
        return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"

    enc = vd.Encoder(OUT / "Video_S4.mp4")
    blank = Image.new("RGB", (vd.W, vd.H))
    poster, shown = None, {}
    for act, (title, top, bottom, cols) in enumerate(ACTS):
        glow = {name: np.zeros(len(s)) for name, _, _ in cols}
        width = (vd.W - 120) // len(cols)
        charts = {}
        for j, (name, a_, b_) in enumerate(cols):
            x = 60 + j * width
            cs = vd.Chart((x + 40, Y_SPARK, x + 300, Y_SPARK + 50), (xpos[0] - 1, xpos[-1] + 1), (0, top_a), "cells spiking per ms" if j == 0 else "",
                          [0, 100, top_a], log="symlog", bands=bands, ytick_fmt=kilo)
            cp = vd.Chart((x + 40, Y_P, x + 300, Y_P + 100), (0.5, n_dec + 0.5), (0, 1), "p(correct move)" if j == 0 else "", [0, 0.5, 1],
                          xticks=[(1, "1"), (n_dec, str(n_dec))], xlabel="decision")
            charts[name] = (cs, cp, activity(act, name, a_, b_))
        shown[title] = {name: {phase(a): float(np.mean(charts[name][2][a - 1:b])) for a, b in spans[:3]} for name, _, _ in cols}
        acc = 0.0
        for t in range(1, last_ms + 1):
            ph = phase(t)
            for name, a_, b_ in cols:
                g = glow[name] * decay_g
                on = hist[source(act, name, a_, b_, t)]["by"].get(t) if t <= hist[name]["end"] else None
                if on is not None:
                    g[on] += 1.0
                glow[name] = g
            acc += FRAMES_PER_MS[ph]
            reps = int(acc)
            acc -= reps
            if reps == 0 and t not in (1, DELAY_END, last_ms):
                continue
            img = blank.copy()
            d = ImageDraw.Draw(img)
            vd.text(d, (60, 40), f"Lost access, lost repair: {title.lower()}", 36, bold=True)
            vd.text(d, (60, 92), f"Scene 1, readout trained, 160-ms delay, lesion dose 1; {t} ms ({ph})", 22, fill=vd.MUTED)
            k = (t - INPUT_END - 1) // 40 if t > INPUT_END else -1
            for j, (name, a, b) in enumerate(cols):
                x = 60 + j * width
                lines = HEADS[name].split(", ") if act else [HEADS[name]]
                for li, line in enumerate(lines):
                    vd.text(d, (x, 140 + 26 * li), line, 22, bold=True)
                for li, (lab, st) in enumerate(((top, a), (bottom, b))):
                    yy = 200 + 26 * li
                    d.rectangle([x, yy + 4, x + 18, yy + 20], fill=STATE_COL[st])
                    vd.text(d, (x + 26, yy), f"{lab} {'healthy' if st == 'H' else 'lesioned'}", 20, fill=vd.MUTED)
                g = glow[name]
                i = np.flatnonzero(g > GLOW_MIN)
                frame = stage.frame(px[i], py[i], colour[i], np.minimum(g[i], 1.5) * dens[i])
                img.paste(frame, (x, Y_STAGE))
                if act == 1 and a != b and DELAY_END <= t < DELAY_END + 12:
                    d.rectangle([x - 3, Y_STAGE - 3, x + STAGE[0] + 2, Y_STAGE + STAGE[1] + 2], outline=(255, 255, 255), width=4)
                for c in (0, 1):
                    states, status = hist[name]["moves"][c]
                    if ph == "decision":
                        si = min(k + 1, len(states) - 1)
                    else:
                        si = 0
                    p_, b_ = states[si] if ph == "decision" else ((44 if ((t - DELAY_END - 1) // 40) % 2 == 0 else 54) if ph == "input" else 44, 34)
                    bimg = board_img(walls, goals[c], p_, b_, ph == "cue")
                    bx = x + c * (bimg.width + 12)
                    img.paste(bimg, (bx, Y_BOARD))
                    vd.text(d, (bx, Y_BOARD + bimg.height + 4), f"cue {c}", 18, fill=CUE_COL[c])
                    if ph == "decision" and k + 1 >= len(states) - 1:
                        d.ellipse([bx + bimg.width - 22, Y_BOARD + bimg.height + 4, bx + bimg.width - 4, Y_BOARD + bimg.height + 22],
                                  fill=OK if status == "solved" else NO)
                cs, cp, av = charts[name]
                cs.frame(d, xpos[t - 1])
                cs.series(d, xpos, av, (200, 200, 200), xpos[t - 1], width=2, label_fmt=None)
                cp.frame(d, k + 1 if ph == "decision" else None)
                for c in (0, 1):
                    pv = hist[name]["p"][c]
                    cp.series(d, np.arange(1, len(pv) + 1), pv, CUE_COL[c], k + 1 if ph == "decision" else 0, width=2, label_fmt=None)
            enc.add(img, max(reps, 1) + (40 if t == 1 else 0) + (60 if t == last_ms else 0))
            if act == 0 and t == INPUT_END + 40 * 3:
                poster = img
    frames = enc.close()
    poster.save(OUT / "Video_S4_poster.png", optimize=False)
    meta = dict(video="Video_S4.mp4", frames=frames, fps=vd.FPS, seconds=frames / vd.FPS, acts=[a[0] for a in ACTS], decisions_shown=n_dec,
                frames_per_ms=FRAMES_PER_MS, stage_px=list(STAGE),
                p_correct={name: {f"cue {c}": [None if np.isnan(v) else round(float(v), 4) for v in hist[name]["p"][c]] for c in (0, 1)}
                           for name in sorted(hist)},
                cells_spiking_per_ms_mean_shown=shown)
    (OUT / "Video_S4.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    print(f"Video S4: {frames} frames ({frames / vd.FPS:.1f} s), {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
