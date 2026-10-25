"""Extended Data Fig. 2 - Numerical validation and compute reconstruction (180 x 120 mm).

a  precision and batch calibration: training time of each candidate on the two benchmark tasks, outcome per task, the selection
b  validation checks of the three precision configurations, and the sparse-layout speed probes
c  gradient recompute audit: largest recompute error of every audited run against its tolerance
d  compute reconstruction: forward calls rebuilt from saved records vs the saved counts (exact match; interruptions rebuilt as
   uninterrupted equivalents)
e  lifetime forward calls of every run by module

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\ed02.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, palette as pal, qc  # noqa: E402

U = (fs.FULL - 3 * fs.GAP) / 4
XS = [k * (U + fs.GAP) for k in range(4)]
ROW1, ROW2, ROW_H = 0.0, 62.0, 58.0
H = ROW2 + ROW_H
EDGE = "#ABABAB"
MODULE = {"benchmark": ("benchmark", "#7A7A7A"), "controls": ("controls", pal.ACCENT), "difficulty": ("difficulty ladder", "#1A1A1A")}


def module_of(run_id):
    return "benchmark" if run_id.startswith("benchmark") else ("difficulty" if run_id.startswith("difficulty") else "controls")


# ---------------------------------------------------------------- a: precision calibration

def panel_a(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "a")
    t = cm.table("extended/ed02/ed02a_precision_calibration.csv")
    cand = t[t["record"] == "candidate"].set_index("id")
    runs = t[t["record"] == "run"]
    sel = json.loads(t["backend_record"].dropna().iloc[0])
    cv.text(x0 + 3.6, y0 + 1.5, "Benchmark tasks C0 and C1 per candidate", size=fs.PT_SMALL, va="center")
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="box", fc="#4D4D4D", size=1.3)], "C0"), ([dict(kind="box", fc="#BDBDBD", size=1.3)], "C1"),
                                ([dict(kind="mark", s=7, marker="x", fc=fs.TEXT)], "numerical error")], gap=1.6)
    ax = cv.axes(x0 + 13.0, y0 + 7.4, w - 13.6, h - 19.0)
    order = ["fp64-b16", "mixed-b16", "fp32-b16", "fp32-b8"]
    for i, cid in enumerate(order):
        c = cand.loc[cid]
        left = 0.0
        for task, col in (("C0", "#4D4D4D"), ("C1", "#BDBDBD")):
            r = runs[runs["id"].str.endswith(f"-{cid}-{task}")]
            assert len(r) == 1, (cid, task)
            r = r.iloc[0]
            secs = float(r["training_seconds"])
            ax.barh(i, secs, left=left, height=0.56, color=col, linewidth=0)
            if r["status"] == "numerical_error":
                ax.plot([left + secs + 25], [i], marker="x", ms=3.0, mew=0.6, color=fs.TEXT, ls="none")
            sd.add("a", "run", candidate=cid, task=task, status=r["status"], updates=float(r["updates"]), training_seconds=secs)
            left += secs
        assert abs(left - float(c["training_seconds"])) < 1e-6, (cid, left, c["training_seconds"])
        if cid == sel["id"]:
            ax.text(left + 60, i, "selected", ha="left", va="center", fontsize=fs.PT_SMALL)
        sd.add("a", "candidate", candidate=cid, precision=c["precision"], batch=int(c["batch"]), eligible=str(c["eligible"]),
               total_training_seconds=float(c["training_seconds"]), selected=cid == sel["id"])
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xlim(0, 1050)
    ax.set_xticks([0, 500, 1000])
    ax.tick_params(axis="y", length=0)
    fs.clean(ax)
    ax.set_xlabel("Training time (s)", labelpad=1)
    cv.text(x0 + 13.0, y0 + h - 2.4, f"rule: {sel['selection']}", size=fs.PT_SMALL, va="center")


# ---------------------------------------------------------------- b: validation checks

CHECKS = [("status", "suite status"), ("full_vs_recomputed_gradient_max_abs", "gradient, full vs recomputed"),
          ("independent_batch_probe_equivalent", "independent batch probe"), ("full_state_restore_exact", "full state restore exact"),
          ("frozen_cache_invalidation", "frozen cache invalidated"), ("transient_intervention_restored", "transient intervention restored"),
          ("scientific_recording_shape", "recording shape"), ("peak_allocated_gib", "peak memory (GiB)")]


def panel_b(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "b")
    t = cm.table("extended/ed02/ed02b_validation_checks.csv")
    v = t[t["record"] == "validation"]
    files = [("validation.json", "default"), ("validation-float64.json", "float64"), ("validation-mixed.json", "mixed")]
    ax = bd.mm_axes(cv, x0, y0, w, h)
    lab_w, top = 22.6, 6.0
    cw = (w - lab_w - 0.2) / len(files)
    rh = 2.45
    for j, (f, lab) in enumerate(files):
        ax.text(lab_w + (j + 0.5) * cw, top - 0.8, lab, ha="center", va="bottom", fontsize=fs.PT_SMALL)
    for i, (item, lab) in enumerate(CHECKS):
        yy = top + i * rh
        if i % 2 == 0:
            ax.add_patch(bd.Rectangle((0, yy), w, rh, facecolor="#F4F4F4", edgecolor="none", linewidth=0.5, zorder=0))
        ax.text(lab_w - 0.8, yy + rh / 2, lab, ha="right", va="center", fontsize=fs.PT_SMALL)
        for j, (f, _) in enumerate(files):
            r = v[(v["check_file"] == f) & (v["item"] == item)]
            val = r["value"].iloc[0] if len(r) else None
            if val is None:
                txt = "n.r."
            elif str(val).lower() in ("true", "passed"):
                txt = "pass"
            elif item == "full_vs_recomputed_gradient_max_abs":
                txt = f"{float(val):.1e}".replace("e-0", "e−").replace("e-", "e−")
            elif item == "peak_allocated_gib":
                txt = f"{float(val):.2f}"
            elif item == "scientific_recording_shape":
                txt = "all cells"
            else:
                txt = str(val)
            ax.text(lab_w + (j + 0.5) * cw, yy + rh / 2, txt, ha="center", va="center", fontsize=fs.PT_SMALL)
            sd.add("b", "check", file=f, item=item, value=None if val is None else str(val))
    # speed probes (sparse CSR layouts), seconds per call
    sp = t[t["record"] == "speed_probe"]
    yy = top + len(CHECKS) * rh + 2.6
    ax.text(0.0, yy, "Speed probes (s per call, 4 repeats)", ha="left", va="center", fontsize=fs.PT_SMALL)
    sx = cv.axes(x0 + lab_w, y0 + yy + 1.8, w - lab_w - 0.6, 11.6)
    labels = []
    for k, r in enumerate(sp.itertuples()):
        d = json.loads(r.value)
        secs = d["seconds"]
        lab = f"{'col' if d.get('column_major') else 'row'}-major, {'det.' if d.get('deterministic') else 'non-det.'}" if "column_major" in d \
            else "segmented"
        labels.append(lab)
        sx.scatter(secs, [k] * len(secs), s=4, color=fs.TEXT, linewidth=0)
        sd.add("b", "speed probe", file=r.check_file, probe=r.item, layout=lab, seconds=";".join(f"{s:.4f}" for s in secs),
               max_repeat_difference=float(max(d.get("forward_vs_gradient", 0), d.get("repeated_gradient", 0), d.get("repeated_inference", 0))))
    sx.set_xscale("log")
    sx.set_xlim(0.04, 4)
    sx.set_xticks([0.1, 1], ["0.1", "1"])
    fs.minor(sx, y=False)
    sx.set_yticks(range(len(labels)), labels)
    sx.set_ylim(len(labels) - 0.5, -0.5)
    sx.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
    fs.clean(sx)


# ---------------------------------------------------------------- c: recompute audit

def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    t = cm.table("extended/ed02/ed02b_validation_checks.csv")
    a = t[t["record"] == "recompute_audit"].copy()
    a["module"] = a["run_id"].map(module_of)
    a = a.sort_values(["module", "run_id"], key=lambda s: s.map({"benchmark": 0, "controls": 1, "difficulty": 2}) if s.name == "module" else s)
    floor = 1e-8
    cv.text(x0 + 3.6, y0 + 1.5, "Largest gradient recompute error per run (dot) and its tolerance (dash)", size=fs.PT_SMALL, va="center")
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="mark", s=6, fc=MODULE[k][1], ec=MODULE[k][1])], MODULE[k][0]) for k in MODULE] +
           [([dict(kind="mark", s=6, fc="white", ec=fs.TEXT, lw=0.5)], "outside tolerance")], gap=1.6)
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 9.4, h - 17.0)
    xs = np.arange(len(a))
    for x, r in zip(xs, a.itertuples()):
        col = MODULE[r.module][1]
        err = max(float(r.max_recompute_error), floor)
        tol = max(float(r.tolerance), floor)
        ok = str(r.within_tolerance).lower() == "true"
        ax.plot([x - 0.35, x + 0.35], [tol, tol], color=EDGE, lw=0.6, zorder=1)
        ax.plot([x], [err], marker="o", ms=2.4, mfc=col if ok else "white", mec=col if ok else fs.TEXT, mew=0.5, ls="none", zorder=3, clip_on=False)
        sd.add("c", "audit", run_id=r.run_id, module=r.module, max_recompute_error=float(r.max_recompute_error), tolerance=float(r.tolerance),
               updates_checked=int(r.updates_checked), duplicates_removed=int(r.duplicates_removed), within_tolerance=ok)
    ax.set_yscale("log")
    ax.set_ylim(floor / 1.5, 3e-3)
    ax.set_yticks([floor, 1e-6, 1e-5, 1e-4, 1e-3], ["0", "10$^{−6}$", "10$^{−5}$", "10$^{−4}$", "10$^{−3}$"])
    fs.minor(ax, x=False)
    ax.set_xlim(-0.8, len(a) - 0.2)
    ax.set_xticks([])
    fs.clean(ax)
    ax.set_ylabel("Error (absolute)", labelpad=1)
    for mod in MODULE:
        idx = np.flatnonzero((a["module"] == mod).to_numpy())
        if len(idx):
            xm = x0 + 9.0 + (idx.mean() + 0.8) / (len(a)) * (w - 9.4)
            cv.text(xm, y0 + h - 7.6, f"{MODULE[mod][0]} ({len(idx)})", size=fs.PT_SMALL, ha="center", va="top")
    n_ok = int(a["within_tolerance"].astype(str).str.lower().eq("true").sum())
    cv.text(x0 + 9.0, y0 + h - 2.6, f"{n_ok} of {len(a)} runs within tolerance; the exception is an early fp32 benchmark audited at zero tolerance",
            size=fs.PT_SMALL, va="center")


# ---------------------------------------------------------------- d, e: compute reconstruction

def panel_d(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "d")
    c = cm.table("meta/compute_reconstruction.csv")
    c = c[c["compute_basis"] != "not_applicable"].copy()
    c["module_key"] = c["module"].map({"benchmark": "benchmark", "controls": "controls", "difficulty": "difficulty"})
    cv.text(x0 + 3.6, y0 + 1.5, "Forward calls rebuilt from saved records vs saved counts", size=fs.PT_SMALL, va="center")
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="mark", s=6, fc=MODULE[k][1], ec=MODULE[k][1])], MODULE[k][0]) for k in MODULE] +
           [([dict(kind="mark", s=8, marker="D", fc="white", ec=fs.TEXT, lw=0.5)], "interrupted, rebuilt as uninterrupted")], gap=1.6)
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.0, h - 14.6)
    for r in c.itertuples():
        col = MODULE[r.module_key][1]
        saved = r.saved_forward_calls
        rebuilt = r.rebuilt_forward_calls
        if not np.isfinite(saved):
            continue
        inter = str(r.interrupted).lower() == "true"
        if inter:
            ax.plot([saved, saved], [r.post_resume_rebuilt_calls, rebuilt], color=EDGE, lw=0.5, zorder=1)
            ax.plot([saved], [rebuilt], marker="D", ms=2.8, mfc="white", mec=fs.TEXT, mew=0.5, ls="none", zorder=3)
        ax.plot([saved], [r.post_resume_rebuilt_calls if inter else rebuilt], marker="o", ms=2.4, mfc=col, mec=col, mew=0, ls="none", zorder=4)
        sd.add("d", "run", run_id=r.run_id, module=r.module, saved_forward_calls=float(saved), rebuilt_forward_calls=float(rebuilt),
               post_resume_rebuilt_calls=None if not np.isfinite(r.post_resume_rebuilt_calls) else float(r.post_resume_rebuilt_calls),
               match=str(r.match), interrupted=inter, lifetime_low=float(r.lifetime_calls_low), lifetime_high=float(r.lifetime_calls_high))
    lim = (30, 1e5)
    ax.plot(lim, lim, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    fs.minor(ax)
    fs.clean(ax)
    ax.set_xlabel("Saved forward calls", labelpad=1)
    ax.set_ylabel("Rebuilt forward calls", labelpad=1)
    n_match = int(c["match"].astype(str).str.lower().eq("true").sum())
    ax.text(0.03, 0.97, f"{n_match} of {int(c['saved_forward_calls'].notna().sum())} runs: exact match", transform=ax.transAxes, ha="left", va="top",
            fontsize=fs.PT_SMALL)


def panel_e(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "e")
    c = cm.table("meta/compute_reconstruction.csv")
    c = c[c["compute_basis"] != "not_applicable"].copy()
    c["key"] = c["module"].map({"benchmark": 0, "controls": 1, "difficulty": 2})
    c = c.sort_values(["key", "lifetime_calls_high"])
    cv.text(x0 + 3.6, y0 + 1.5, "Lifetime forward calls per run (dot; bar: low-high bound for interrupted runs)", size=fs.PT_SMALL, va="center")
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 9.4, h - 17.0)
    xs = np.arange(len(c))
    for x, r in zip(xs, c.itertuples()):
        col = MODULE[r.module if r.module in MODULE else "controls"][1]
        ax.plot([x, x], [r.lifetime_calls_low, r.lifetime_calls_high], color=col, lw=0.8)
        ax.plot([x], [r.lifetime_calls_high], marker="o", ms=2.2, mfc=col, mec=col, mew=0, ls="none")
        sd.add("e", "run", run_id=r.run_id, module=r.module, status=r.status, lifetime_low=float(r.lifetime_calls_low),
               lifetime_high=float(r.lifetime_calls_high), neuron_ms_per_call=float(r.neuron_ms_per_call))
    ax.set_yscale("log")
    ax.set_ylim(2, 1e5)
    fs.minor(ax, x=False)
    ax.set_xlim(-0.8, len(c) - 0.2)
    ax.set_xticks([])
    fs.clean(ax)
    ax.set_ylabel("Forward calls", labelpad=1)
    for k, mod in enumerate(("benchmark", "controls", "difficulty")):
        idx = np.flatnonzero((c["key"] == k).to_numpy())
        if len(idx):
            xm = x0 + 9.0 + (idx.mean() + 0.8) / len(c) * (w - 9.4)
            cv.text(xm, y0 + h - 7.6, f"{MODULE[mod][0]} ({len(idx)})", size=fs.PT_SMALL, ha="center", va="top")
    per = c["neuron_ms_per_call"].astype(float)
    cv.text(x0 + 9.0, y0 + h - 2.6, f"one call = {per.max():,.0f} neuron-ms (batch 16); {per.min():,.0f} at batch 8", size=fs.PT_SMALL, va="center")


def main():
    cv = fs.Canvas(fs.FULL, H, ppi_min=300)
    sd = qc.SourceData("ed02")
    for rel in ("extended/ed02/ed02a_precision_calibration.csv", "extended/ed02/ed02b_validation_checks.csv", "meta/compute_reconstruction.csv"):
        sd.source(cm.RESULTS / rel)
    panel_a(cv, sd, XS[0], ROW1, U, ROW_H)
    panel_b(cv, sd, XS[1], ROW1, U, ROW_H)
    panel_c(cv, sd, XS[2], ROW1, 2 * U + fs.GAP, ROW_H)
    panel_d(cv, sd, XS[0], ROW2, 2 * U + fs.GAP, ROW_H)
    panel_e(cv, sd, XS[2], ROW2, 2 * U + fs.GAP, ROW_H)
    cm.finish(cv, "ed02", sd, kind="extended")


if __name__ == "__main__":
    main()
