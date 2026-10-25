"""Extended Data Fig. 10 - Run registry: every run of the study, including failed, stopped and never-run entries (180 x 78 mm).

a  every registered run (one square each) by layer and module, coloured by outcome group (ST1)
b  runs per outcome group, and verification after reload
c  training cost of every run that took parameter updates: updates vs training time (log axes), by outcome group

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\ed10.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, qc  # noqa: E402

U = (fs.FULL - 3 * fs.GAP) / 4
XS = [k * (U + fs.GAP) for k in range(4)]
H = 78.0
AX_TOP = 6.5
STATUS = [("acquired", "learned and solved", "#1A1A1A", "#1A1A1A"), ("baseline", "untrained baseline", "#9E9E9E", "#9E9E9E"),
          ("stopped_rule", "stopped by a rule", "#E69F00", "#E69F00"), ("stopped_user", "stopped by hand", "#CC79A7", "#CC79A7"),
          ("failed", "failed (numerics, code)", "#D55E00", "#D55E00"), ("check", "check passed", "#009E73", "#009E73"),
          ("evaluation", "evaluation complete", "#56B4E9", "#56B4E9"), ("never_run", "registered, never run", "white", "#9E9E9E")]
LAYERS = [("formal_lineage", "formal lineage"), ("local_static", "local, static task"), ("remote_batch1", "remote batch 1"),
          ("remote_supplement", "remote supplement"), ("mechanisms_v1", "mechanisms v1"), ("mechanisms_supplement", "mechanisms supplement"),
          ("registered_not_run", "registered, not run")]


def registry():
    s = cm.table("supplementary/st1_run_registry.csv").copy()
    e = cm.table("extended/ed10/ed10_run_registry_summary.csv")
    assert int(e["n_runs"].sum()) == len(s)
    chk = s.groupby(["layer", "module", "status", "status_group"]).size().rename("n").reset_index()
    m = chk.merge(e, on=["layer", "module", "status", "status_group"], how="outer", validate="one_to_one")
    assert (m["n"] == m["n_runs"]).all()
    assert set(s["status_group"]) <= {k for k, *_ in STATUS} and set(s["layer"]) == {k for k, _ in LAYERS}
    return s


# ---------------------------------------------------------------- a: the registry

def panel_a(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "a")
    s = registry()
    order = {k: i for i, (k, *_) in enumerate(STATUS)}
    rows = []
    for layer, llab in LAYERS:
        mods = sorted(s[s["layer"] == layer]["module"].unique(), key=lambda m: (-int((s["module"] == m).sum()), m))
        rows.append((layer, llab, mods))
    n_rows = sum(len(m) for *_, m in rows)
    lab_l = max(fs.text_width(l, fs.PT_SMALL) for _, l, _ in rows) + 2.2
    lab_m = max(fs.text_width(m.replace("_", " "), fs.PT_SMALL) for *_, mods in rows for m in mods) + 1.6
    top, lgap = 3.4, 1.0
    rh = (h - top - lgap * (len(rows) - 1)) / n_rows
    widest = int(s.groupby(["layer", "module"]).size().max())
    pitch = min(rh, (w - lab_l - lab_m - 4.5) / widest)
    sq = pitch - 0.35
    ax = bd.mm_axes(cv, x0, y0, w, h)
    y = top
    for layer, llab, mods in rows:
        y_start = y
        for mod in mods:
            d = s[(s["layer"] == layer) & (s["module"] == mod)].copy()
            d["o"] = d["status_group"].map(order)
            d = d.sort_values(["o", "run_id"], kind="mergesort")
            ax.text(lab_l + lab_m - 0.8, y + rh / 2, mod.replace("_", " "), ha="right", va="center", fontsize=fs.PT_SMALL)
            for i, r in enumerate(d.itertuples()):
                st = STATUS[order[r.status_group]]
                ax.add_patch(bd.Rectangle((lab_l + lab_m + i * pitch, y + (rh - sq) / 2), sq, sq, facecolor=st[2], edgecolor=st[3], linewidth=0.35))
                sd.add("a", "run", layer=layer, module=mod, run_id=r.run_id, task=r.task, condition=r.condition, status=r.status,
                       status_group=r.status_group, updates=None if np.isnan(r.updates) else float(r.updates),
                       training_seconds=None if np.isnan(r.training_seconds) else float(r.training_seconds),
                       verification=r.verification_status if isinstance(r.verification_status, str) else "", interrupted=bool(r.interrupted),
                       hardware=r.hardware if isinstance(r.hardware, str) else "")
            ax.text(lab_l + lab_m + len(d) * pitch + 0.6, y + rh / 2, str(len(d)), ha="left", va="center", fontsize=fs.PT_SMALL)
            y += rh
        ax.plot([lab_l - 0.8, lab_l - 0.8], [y_start + 0.25, y - 0.25], color=fs.TEXT, lw=0.4)
        ax.text(lab_l - 1.4, (y_start + y) / 2, llab, ha="right", va="center", fontsize=fs.PT_SMALL)
        y += lgap
    cv.text(x0 + 3.6, y0 + 1.0, f"{len(s)} runs, one square each (ST1)", size=fs.PT_SMALL, va="center")


# ---------------------------------------------------------------- b: outcome groups

def panel_b(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "b")
    s = registry()
    counts = s["status_group"].value_counts()
    lab_w = max(fs.text_width(lab, fs.PT_SMALL) for _, lab, *_ in STATUS) + 1.4
    ax = cv.axes(x0 + lab_w, y0 + 3.4, w - lab_w - 6.0, 26.0)
    for i, (k, lab, fc, ec) in enumerate(STATUS):
        n = int(counts.get(k, 0))
        ax.barh([i], [n], height=0.62, color=fc, edgecolor=ec, linewidth=0.35)
        ax.text(n + 1.5, i, str(n), ha="left", va="center", fontsize=fs.PT_SMALL)
        ver = int(((s["status_group"] == k) & (s["verification_status"] == "verified")).sum())
        sd.add("b", "group", status_group=k, label=lab, runs=n, verified_after_reload=ver)
    ax.set_yticks(range(len(STATUS)), [lab for _, lab, *_ in STATUS])
    ax.set_ylim(len(STATUS) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
    ax.set_xlim(0, 95)
    ax.set_xticks([0, 40, 80], ["0", "40", "80"])
    fs.clean(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Runs", labelpad=1)
    acq = s[s["status_group"] == "acquired"]
    stop = s[s["status_group"] == "stopped_rule"]
    n_ver = int((acq["verification_status"] == "verified").sum())
    n_int = int(s["interrupted"].astype(str).str.lower().eq("true").sum())
    cv.text(x0, y0 + 36.4, f"verified after reload: {n_ver} of {len(acq)} learned runs,", size=fs.PT_SMALL, va="center")
    cv.text(x0, y0 + 38.6, f"{int((stop['verification_status'] == 'verified').sum())} of {len(stop)} stopped; {n_int} interrupted and resumed",
            size=fs.PT_SMALL, va="center")


# ---------------------------------------------------------------- c: training cost

def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    s = registry()
    t = s[(s["updates"] > 0) & (s["training_seconds"] > 0)]
    ax = cv.axes(x0 + 9.0, y0 + 4.4, w - 9.4, h - 4.4 - 7.4)
    for k, lab, fc, ec in STATUS:
        d = t[t["status_group"] == k]
        if d.empty:
            continue
        ax.scatter(d["updates"], d["training_seconds"] / 3600, s=6.0, facecolor=fc, edgecolor=ec, linewidth=0.4, zorder=3)
        for r in d.itertuples():
            sd.add("c", "training run", run_id=r.run_id, layer=r.layer, module=r.module, status_group=k, updates=float(r.updates),
                   training_hours=float(r.training_seconds) / 3600)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5, 3e5)
    ax.set_ylim(3e-3, 30)
    ax.set_xticks([10, 1e3, 1e5], ["10", "10$^{3}$", "10$^{5}$"])
    ax.set_yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
    fs.minor(ax)
    fs.clean(ax)
    ax.set_xlabel("Updates", labelpad=1)
    ax.set_ylabel("Training time (h)", labelpad=1)
    cv.text(x0 + 3.6, y0 + 1.0, f"training cost of the {len(t)} runs with parameter updates; colours as in b", size=fs.PT_SMALL, va="center")


def cm_u(n):
    return n * U + (n - 1) * fs.GAP


def main():
    cv = fs.Canvas(fs.FULL, H, ppi_min=300)
    sd = qc.SourceData("ed10")
    for rel in ("extended/ed10/ed10_run_registry_summary.csv", "supplementary/st1_run_registry.csv"):
        sd.source(cm.RESULTS / rel)
    panel_a(cv, sd, XS[0], 0.0, cm_u(3), H)
    panel_b(cv, sd, XS[3], 0.0, U, 41.0)
    panel_c(cv, sd, XS[2] - 2.0, 44.0, 2 * U + fs.GAP + 2.0, H - 44.0)
    cm.finish(cv, "ed10", sd, kind="extended")


if __name__ == "__main__":
    main()
