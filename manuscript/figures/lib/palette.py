"""Colours, fixed across figures. Hues follow Okabe-Ito (colour-blind safe); categories drawn on black tiles are
brightened copies. Every set here is checked by build/check_palette.py (CIE76 >= 10 normally, under protan / deutan /
tritan simulation and in greyscale where stated; against black for tile colours)."""
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb

# ---- cues (cue 0 / cue 1) and the signed difference cue 1 - cue 0
CUE = {0: "#0072B2", 1: "#D55E00"}                 # charts on white
CUE_ON_DARK = {0: "#5DA5E8", 1: "#F08A4B"}         # glyphs on black tiles
MAGNITUDE_ON_DARK = colormaps["magma"]

# ---- cells on black tiles
TILE = {
    "background": "#000000",
    "somata": "#404040",                           # every soma, 25% grey
    "shell": "#9AA4AE",                            # neuropil surface glow (drawn faint)
    "input": "#F2F2F2",                            # 3,200 visual input cells
}
# Superclasses in seven display groups + grey, one key for every anatomy tile (cells, arbors, readout pools).
SUPERCLASS_GROUP = {
    "ol_intrinsic": "optic lobe intrinsic",
    "cb_intrinsic": "central brain intrinsic",
    "vnc_intrinsic": "VNC intrinsic",
    "visual_projection": "visual projection & centrifugal", "visual_projection_tbc": "visual projection & centrifugal",
    "visual_centrifugal": "visual projection & centrifugal",
    "ol_sensory": "sensory", "cb_sensory": "sensory", "vnc_sensory": "sensory", "sensory_ascending": "sensory",
    "sensory_descending": "sensory", "cb_sensory_tbc": "sensory", "vnc_sensory_tbc": "sensory", "sensory_ascending_tbc": "sensory",
    "descending_neuron": "descending & ascending", "ascending_neuron": "descending & ascending",
    "efferent_descending": "descending & ascending", "efferent_ascending": "descending & ascending",
    "vnc_motor": "motor & efferent", "cb_motor": "motor & efferent", "vnc_efferent": "motor & efferent", "cb_efferent": "motor & efferent",
    "cb_endocrine": "motor & efferent", "vnc_endocrine": "motor & efferent", "ENS": "motor & efferent",
}
ATLAS_ON_DARK = {                                   # checked: CIE76 >= 10.4 under every simulation, with input white and grey
    "optic lobe intrinsic": "#6C7BFF",
    "central brain intrinsic": "#36D399",
    "VNC intrinsic": "#B48CFF",
    "visual projection & centrifugal": "#5CC8FF",
    "sensory": "#FFD45C",
    "descending & ascending": "#FF7B54",
    "motor & efferent": "#FF5FB0",
    "other": "#8C8C8C",
}
READOUT_CLASS = SUPERCLASS_GROUP                    # readout cells use the same groups and colours
READOUT_ON_DARK = ATLAS_ON_DARK


# ---- learners and conditions (charts on white)
LEARNER = {
    "connectome readout-only": "#1A1A1A",
    "connectome joint": "#6E6E6E",
    "visible goal": "#56B4E9",
    "reset state": "#BDBDBD",
    "untrained readout": "#DADADA",
    "GRU": "#CC79A7",
    "fixed trace": "#E69F00",
    "neutral, real graph": "#009E73",
    "neutral, rewired graph": "#9E2A10",
}
OUTCOME = {"solved": "#1A1A1A", "partial": "#9A9A9A", "unsolved": "#FFFFFF"}
MEMORY_CLASS = {"both": "#1A1A1A", "one": "#BDBDBD", "none": "#FFFFFF"}

# ---- boards
BOARD = {"wall": "#3A3A3A", "floor": "#F4F1EA", "goal": "#1A1A1A", "box": "#B8875A", "box_edge": "#5C4330",
         "player": "#1A1A1A", "path": "#7A7A7A"}


def diverging_on_white(lo: str = CUE[0], hi: str = CUE[1], name: str = "cue_white"):
    """White-centred diverging map between the two cue hues, for charts."""
    return LinearSegmentedColormap.from_list(name, [lo, "#FFFFFF", hi], N=256)


# ---- CIELAB helpers for maps built from the cue hues
_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def _lab(rgb):
    rgb = np.asarray(rgb, float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _rgb(lab):
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * _WHITE
    lin = xyz @ np.linalg.inv(_M).T
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def dark_diverging(lo: str, hi: str, l_end: float = 74.0, l_mid: float = 6.0, n: int = 256, name: str = "cue_dark"):
    """Dark-centred diverging map: black-ish centre, L* rising linearly to l_end at both ends, hue and chroma of the
    two end colours (chroma grows with L*, so the halves match in lightness and saturation)."""
    ends = _lab(np.array([to_rgb(lo), to_rgb(hi)]))
    half = n // 2
    t = np.linspace(1, 0, half)[:, None]
    rows = []
    for k, e in ((0, ends[0]), (1, ends[1])):
        lab = np.column_stack([l_mid + (l_end - l_mid) * t[:, 0], e[1] * t[:, 0], e[2] * t[:, 0]])
        rows.append(lab if k == 0 else lab[::-1])
    return ListedColormap(np.clip(_rgb(np.vstack(rows)), 0, 1), name=name)


SIGNED_ON_DARK = dark_diverging(CUE_ON_DARK[0], CUE_ON_DARK[1])   # negative (cue 0 higher) blue, positive orange


# ---- charts and schematics on white
ACCENT = "#D81B60"                 # trained parameters and learning progress
FIXED = "#9E9E9E"                  # fixed connectome structure
SHORT_GROUP = {"optic lobe intrinsic": "optic lobe", "central brain intrinsic": "central brain", "VNC intrinsic": "VNC",
               "visual projection & centrifugal": "visual proj.", "descending & ascending": "desc./asc.",
               "motor & efferent": "motor/eff.", "sensory": "sensory", "other": "other"}


def on_white(hex_colour: str, max_l: float = 58.0) -> str:
    """The same hue at CIE L* <= max_l, so a tile colour keeps its identity on a white page with enough contrast."""
    from matplotlib.colors import to_hex
    lab = _lab(np.array(to_rgb(hex_colour)))
    if lab[0] > max_l:
        scale = max_l / lab[0]
        lab = np.array([max_l, lab[1] * (0.6 + 0.4 * scale), lab[2] * (0.6 + 0.4 * scale)])
    return to_hex(np.clip(_rgb(lab), 0, 1))


# ---- distinct state sets (memory models) and history-level changes, charts on white
STATE_SET = {
    "m0": "#1A1A1A",            # M0 readout-only (hidden goal)
    "m0_joint": "#8C8C8C",      # M0 joint
    "m0vis_visible": "#56B4E9", # shared readout, goal visible
    "s1": "#8E24AA",            # S1 readout-only
    "s2_o0": "#00897B",         # S2 orientation 0
    "s2_o180": "#7FCDC3",       # S2 orientation 180
}
MARGIN_CMAP = "PuOr"            # margin change: orange = towards the wrong push, purple = towards the correct push
