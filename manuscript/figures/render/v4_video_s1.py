"""Video S1 (Cell; related to Figure 1) - the trained connectome solves the hardest planning map, move by move.

Left: the 10 x 10 map with the agent, the three boxes and their goals, and the readout's choice among the four moves (a
softmax of its logits; the chosen move white, illegal moves dimmed). Right: the unfolded CNS on black, brain to the left as
in Figure 1B; every cell that spikes in the move's 40-ms run glows in its cell-group colour (Figure 1A), brighter with more
spikes. Each move is one fresh 40-ms simulation of the whole CNS from the board (the planning task has no memory between
moves); the video shows two moves per second. Below the CNS, two live charts: the cells spiking in each move, stacked by cell
group, and the readout's confidence (its softmax probability of the move it makes), with the moves that push a box marked.

Data: manuscript/results/plan_v2/video_planning_moves.csv, video_planning_cells.csv, video_planning_map.csv (P12, gated
against the record's matched logits); soma positions from figure_data/fd_cells.csv.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_video_s1.py
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

OUT = FIG / "v4" / "videos"
PLAN = rp.RESULTS / "plan_v2"
ACTIONS = ("up", "down", "left", "right")
FRAMES_PER_MOVE, FADE, HOLD = 15, 4, 60            # two moves per second at 30 fps; crossfade frames; title and end cards
STAGE = (900, 620)                                  # the CNS panel, landscape (brain left)
RIGHT = 980                                         # left edge of the brain and chart column
BOARD_XY, CELL = (110, 230), 58


def activity_frames(stage, s, xyz, cells, n_moves):
    group = s["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other").to_numpy()
    colour = rp.srgb_to_linear(np.array([to_rgb(pal.ATLAS_ON_DARK[g]) for g in group]))
    px, py = stage.project(xyz)
    dens = sc.density_weights(stage.view, xyz, power=0.45)
    pos = pd.Series(np.arange(len(s)), index=s["cell"].to_numpy())
    out, counts = [], []
    for m in range(n_moves):
        c = cells[cells["move"] == m]
        counts.append(int(len(c)))
        c = c[c["cell"].isin(pos.index)]
        i = pos.loc[c["cell"]].to_numpy()
        n = c["spikes"].to_numpy(float)
        w = (0.6 + 0.8 * np.log1p(n) / np.log1p(20)) * dens[i]
        out.append(stage.frame(px[i], py[i], colour[i], w))
    return out, counts


def panel(img, mv, m, n_moves, walls, goals, spiking, title, sub):
    d = ImageDraw.Draw(img)
    vd.text(d, (70, 60), title, 38, bold=True)
    vd.text(d, (70, 112), sub, 24, fill=vd.MUTED)
    row = mv.iloc[min(m, n_moves - 1)]
    player = int(row["player"]) if m < n_moves else int(mv.attrs["final_player"])
    boxes = [int(b) for b in str(row["boxes"]).split(";")] if m < n_moves else mv.attrs["final_boxes"]
    vd.board(d, *BOARD_XY, CELL, walls, goals, boxes, player)
    x0, y0 = BOARD_XY[0], BOARD_XY[1] + 10 * CELL + 34
    if m < n_moves:
        vd.text(d, (x0, y0), f"move {m + 1} of {n_moves}", 28, bold=True)
        q = np.array([row[f"logit_{a}"] for a in range(4)], float)
        legal = np.array([bool(row[f"legal_{a}"]) for a in range(4)])
        p = np.exp(q - q[legal].max()) * legal
        p /= p.sum()
        for a in range(4):
            y = y0 + 56 + a * 40
            chosen = a == int(row["action"])
            col = vd.INK if chosen else (vd.MUTED if legal[a] else vd.DIM)
            vd.text(d, (x0, y), ACTIONS[a], 24, fill=col, bold=chosen)
            vd.bar(d, x0 + 110, y + 4, 420, 22, float(p[a]), (236, 236, 236) if chosen else (110, 110, 110))
        vd.text(d, (x0 + 545, y0 + 56), "readout's choice", 20, fill=vd.MUTED)
    else:
        vd.text(d, (x0, y0), f"solved in {n_moves} moves, the certified optimum", 28, bold=True)


def legend(img, x, y, width):
    """The cell-group key, flowed into rows no wider than width."""
    d = ImageDraw.Draw(img)
    cx, cy = x, y
    for g in [g for g in pal.ATLAS_ON_DARK if g != "other"]:
        w = 22 + vd.font(19).getlength(g) + 30
        if cx + w - 30 > x + width:
            cx, cy = x, cy + 28
        col = tuple(int(round(255 * v)) for v in to_rgb(pal.ATLAS_ON_DARK[g]))
        d.ellipse([cx, cy + 4, cx + 14, cy + 18], fill=col)
        vd.text(d, (cx + 22, cy), g, 19, fill=vd.MUTED)
        cx += w


def main():
    t0 = time.time()
    mv = pd.read_csv(PLAN / "video_planning_moves.csv")
    cells = pd.read_csv(PLAN / "video_planning_cells.csv")
    mp = pd.read_csv(PLAN / "video_planning_map.csv")
    n_moves = len(mv)
    walls, goals = set(mp.loc[mp["wall"], "cell"]), set(mp.loc[mp["goal"], "cell"])
    # the board after the last move: replay the actions on the start board (the table holds the state before each move)
    player, boxes = int(mv.iloc[-1]["player"]), [int(b) for b in str(mv.iloc[-1]["boxes"]).split(";")]
    step = {0: -10, 1: 10, 2: -1, 3: 1}[int(mv.iloc[-1]["action"])]
    if player + step in boxes:
        boxes = [b + step if b == player + step else b for b in boxes]
    mv.attrs["final_player"], mv.attrs["final_boxes"] = player + step, boxes
    if set(boxes) != goals:
        raise RuntimeError("the last move does not leave every box on a goal")
    fd = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False)
    s, xyz = sc.somata(fd)
    stage = vd.Stage(xyz, *STAGE, rotate=True)
    acts, counts = activity_frames(stage, s, xyz, cells, n_moves)
    print(f"activity rendered {time.time() - t0:.0f} s", flush=True)
    title = "A whole-CNS connectome plans, move by move"
    sub = f"The hardest map of the ladder; each move is one 40-ms run of all {len(fd):,} neurons"
    enc = vd.Encoder(OUT / "Video_S1.mp4")
    blank = Image.new("RGB", (vd.W, vd.H))
    dark = stage.frame([], [], np.zeros((0, 3)), [])

    # live metrics: cells spiking per move by cell group, and the readout's confidence in its chosen move (pushes marked)
    groups = [g for g in pal.ATLAS_ON_DARK if g != "other"] + ["other"]
    gcol = [vd.rgb8(to_rgb(pal.ATLAS_ON_DARK[g])) for g in groups]
    cell_group = fd.set_index("cell")["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other")
    per = cells.assign(group=cells["cell"].map(cell_group)).groupby(["move", "group"]).size().unstack(fill_value=0)
    stacks = [[int(per.loc[m_].get(g, 0)) if m_ in per.index else 0 for g in groups] for m_ in range(n_moves)]
    if [sum(st) for st in stacks] != counts:
        raise RuntimeError("cells per move by group do not add up to the cells per move")
    conf, pushes = [], []
    for m_ in range(n_moves):
        row = mv.iloc[m_]
        q = np.array([row[f"logit_{a}"] for a in range(4)], float)
        legal = np.array([bool(row[f"legal_{a}"]) for a in range(4)])
        pr = np.exp(q - q[legal].max()) * legal
        conf.append(float(pr[int(row["action"])] / pr.sum()))
        nxt = mv.attrs["final_boxes"] if m_ == n_moves - 1 else [int(b) for b in str(mv.iloc[m_ + 1]["boxes"]).split(";")]
        pushes.append(sorted(nxt) != sorted(int(b) for b in str(row["boxes"]).split(";")))
    top = 1000 * int(np.ceil(max(counts) / 1000))
    ca = vd.Chart((RIGHT + 70, 800, RIGHT + 430, 1000), (-0.8, n_moves - 0.2), (0, top), "cells spiking per move", [0, top // 2, top],
                  xticks=[(0, "1"), (n_moves - 1, str(n_moves))], ytick_fmt="{:,.0f}", xlabel="move")
    cb = vd.Chart((RIGHT + 520, 800, RIGHT + 850, 1000), (-0.8, n_moves - 0.2), (0, 1), "readout confidence", [0, 0.5, 1],
                  xticks=[(0, "1"), (n_moves - 1, str(n_moves))], xlabel="move")

    def compose(cns, m):
        img = blank.copy()
        img.paste(cns, (RIGHT, 30))
        panel(img, mv, m, n_moves, walls, goals, counts[m] if m < n_moves else 0, title, sub)
        legend(img, RIGHT + 20, 30 + STAGE[1] + 14, STAGE[0] - 40)
        d = ImageDraw.Draw(img)
        x_now = min(m, n_moves - 1)
        ca.frame(d, x_now)
        ca.bars(d, range(n_moves), stacks, gcol, x_now, 4)
        cb.frame(d, x_now)
        cb.series(d, range(n_moves), conf, vd.INK, x_now, width=2, label_fmt="{:.2f}")
        for m_ in range(x_now + 1):
            if pushes[m_]:
                px_, py_ = cb.px(m_, conf[m_])
                d.ellipse([px_ - 4, py_ - 4, px_ + 4, py_ + 4], fill=(214, 160, 60))
        cb.keys(d, [("box push", (214, 160, 60), "dot")])
        return img

    enc.add(compose(dark, 0), HOLD)
    for m in range(n_moves):
        prev = acts[m - 1] if m else dark
        for f in range(FRAMES_PER_MOVE):
            cns = Image.blend(prev, acts[m], (f + 1) / FADE) if f < FADE - 1 else acts[m]
            enc.add(compose(cns, m))
    enc.add(compose(Image.blend(acts[-1], dark, 0.6), n_moves), HOLD)
    frames = enc.close()
    poster = compose(acts[n_moves // 2], n_moves // 2)
    poster.save(OUT / "Video_S1_poster.png", optimize=False)
    meta = dict(video="Video_S1.mp4", frames=frames, fps=vd.FPS, seconds=frames / vd.FPS, moves=n_moves, cells_spiking_per_move=counts,
                stage_px=list(STAGE), um_per_px=stage.view.um_per_px, readout_confidence=conf, pushes=int(sum(pushes)))
    (OUT / "Video_S1.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    print(f"Video S1: {frames} frames ({frames / vd.FPS:.1f} s), {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
