"""Flat Sokoban boards drawn as vector patches in millimetre axes (the 10 x 10 canvas of sokoban/lib/game.py).

A board is drawn cropped to its outer wall ring, at a fixed cell size, so larger rooms look larger. Cells: walls dark,
floor light, goals as small rings, boxes as rounded squares, the player as a disc. Moves are drawn as arrows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from . import figstyle as fs
from . import palette as pal

DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}       # action -> (d row, d col): up, down, left, right


def mm_axes(cv, left: float, top: float, width: float, height: float):
    """Axes whose data coordinates are millimetres from its top-left corner (x right, y down), no frame."""
    ax = cv.axes(left, top, width, height)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    return ax


class Board:
    """Cells of one board from figure_data/fd_boards.csv (or explicit walls/goals/boxes/player)."""

    def __init__(self, walls, goals, boxes, player):
        self.walls = set(int(c) for c in walls)
        self.goals = set(int(c) for c in goals)
        self.boxes = set(int(c) for c in boxes)
        self.player = None if player is None else int(player)
        floor = [c for c in range(100) if c not in self.walls]
        rows = [c // 10 for c in floor]
        cols = [c % 10 for c in floor]
        self.r0, self.r1 = max(min(rows) - 1, 0), min(max(rows) + 1, 9)
        self.c0, self.c1 = max(min(cols) - 1, 0), min(max(cols) + 1, 9)

    @classmethod
    def from_table(cls, df: pd.DataFrame, board_id: str) -> "Board":
        b = df[df["board_id"] == board_id]
        if len(b) != 100:
            raise KeyError(f"board {board_id} not found")
        pick = lambda col: b.loc[b[col].astype(str).str.lower() == "true", "cell"].tolist()  # noqa: E731
        players = pick("player_start")
        return cls(pick("wall"), pick("goal"), pick("box_start"), players[0] if players else None)

    @property
    def shape(self):
        """(rows, cols) of the cropped board including its outer wall ring."""
        return self.r1 - self.r0 + 1, self.c1 - self.c0 + 1

    def size_mm(self, cell: float):
        r, c = self.shape
        return c * cell, r * cell

    def xy(self, c: int, cell: float, x: float, y: float):
        """Centre (mm) of canvas cell c when the cropped board's top-left corner is at (x, y)."""
        return x + (c % 10 - self.c0 + 0.5) * cell, y + (c // 10 - self.r0 + 0.5) * cell


def draw(ax, board: Board, x: float, y: float, cell: float, show_goals: bool = True, boxes=None, player=None,
         goal_colour: str = pal.BOARD["goal"], lw: float = 0.25, dim: bool = False):
    """Draw a board with its top-left corner at (x, y) mm. boxes/player override the start state."""
    boxes = board.boxes if boxes is None else set(boxes)
    player = board.player if player is None else player
    floor_c, wall_c = pal.BOARD["floor"], pal.BOARD["wall"]
    for r in range(board.r0, board.r1 + 1):
        for c in range(board.c0, board.c1 + 1):
            k = r * 10 + c
            px, py = x + (c - board.c0) * cell, y + (r - board.r0) * cell
            ax.add_patch(Rectangle((px, py), cell, cell, facecolor=wall_c if k in board.walls else floor_c, edgecolor="none",
                                   linewidth=0, zorder=1))
    if show_goals:
        for g in board.goals:
            cx, cy = board.xy(g, cell, x, y)
            ax.add_patch(Circle((cx, cy), 0.26 * cell, facecolor="none", edgecolor=goal_colour, linewidth=max(lw, 0.25), zorder=3))
            ax.add_patch(Circle((cx, cy), 0.08 * cell, facecolor=goal_colour, edgecolor="none", zorder=3))
    for bx in boxes:
        cx, cy = board.xy(bx, cell, x, y)
        s = 0.74 * cell
        ax.add_patch(FancyBboxPatch((cx - s / 2, cy - s / 2), s, s, boxstyle=f"round,pad=0,rounding_size={0.12 * cell}",
                                    facecolor=pal.BOARD["box"], edgecolor=pal.BOARD["box_edge"] if lw >= 0.25 else "none",
                                    linewidth=lw if lw >= 0.25 else 0, zorder=4))
    if player is not None:
        cx, cy = board.xy(player, cell, x, y)
        ax.add_patch(Circle((cx, cy), 0.3 * cell, facecolor=pal.BOARD["player"], edgecolor="none", zorder=5))
    if dim:
        h, w = board.size_mm(cell)[1], board.size_mm(cell)[0]
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", alpha=0.55, edgecolor="none", zorder=6))


def arrow(ax, board: Board, start: int, action: int, x: float, y: float, cell: float, colour: str = fs.TEXT,
          lw: float = 0.6, length: float = 0.9, head: float = 2.2, z: float = 7):
    """An arrow from the centre of cell start one step in the direction of action (up, down, left, right)."""
    dr, dc = DELTAS[int(action)]
    cx, cy = board.xy(start, cell, x, y)
    ax.add_patch(FancyArrowPatch((cx, cy), (cx + dc * cell * length, cy + dr * cell * length), arrowstyle=f"-|>,head_length={head * 0.4},head_width={head * 0.25}",
                                 mutation_scale=4, color=colour, linewidth=lw, shrinkA=0, shrinkB=0, zorder=z))


def path(ax, board: Board, cells, x: float, y: float, cell: float, colour: str = pal.BOARD["path"], lw: float = 0.5, z: float = 2):
    """A polyline through the centres of successive player cells."""
    pts = np.array([board.xy(c, cell, x, y) for c in cells])
    ax.plot(pts[:, 0], pts[:, 1], color=colour, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=z)
