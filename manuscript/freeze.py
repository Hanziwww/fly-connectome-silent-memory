"""Freeze the manuscript outputs.

1. Run the unit tests of the analysis and of the figure library.
2. Rebuild every figure render, the palette check and every figure of the chosen set in fresh processes, twice, and require
   both passes to be byte-identical (every file under the set's output folders, QC issues zero).
3. Read the analysis determinism check (manuscript/results/qa/determinism.csv, written by run_all.py --determinism).
4. Write manuscript/FREEZE_MANIFEST.csv (SHA-256 of every deliverable: analysis tables, figure-ready tables, renders,
   figures, source data, QC reports) and manuscript/FREEZE.md (summary: code version, environment, checks).

Sets: v1, the Nature-format figures (7 main, ED1-10; freeze-v1 and freeze-v1.1); v4, the Cell figures (7 main, S1-S7;
freeze-v2). A v4 freeze does not rebuild the v1 figures, which stay pinned by the freeze-v1.1 tag.

Not rebuilt: the renderer pilot (renders/pilot; one-off ChimeraX comparison used by no figure) and the downloaded
MaleCNS skeletons (inputs, verified by assets/skeletons/receipt.csv).

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\freeze.py [--set v4] [--passes 2]
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
FIG = MS / "figures"
RESULTS = MS / "results"
TESTS = ["manuscript/analysis/tests/run_tests.py", "manuscript/figures/tests/run_tests.py"]
V4_FIGURES = [f"fig{i}" for i in range(1, 8)] + [f"s{i}" for i in range(1, 8)]
SETS = {
    "v1": dict(title="Manuscript freeze", renders=["fig1_tiles", "fig3_tiles", "fig4_tiles", "fig5_tiles", "fig6_tiles", "ed04_tiles", "ed07_tiles",
                                                   "ed08_tiles"],
               figures=[f"fig{i}" for i in range(1, 8)] + [f"ed{i:02d}" for i in range(1, 11)], build_dir="",
               outputs=[FIG / "main", FIG / "extended", FIG / "renders", FIG / "qc", FIG / "source_data"], exclude=[FIG / "renders" / "pilot"],
               qc_dir=FIG / "qc", geometry="width 180 mm, height at most 170 mm", kinds="main and Extended Data", superseded=None),
    # renders in dependency order: S3 imports the Fig. 3 renderer, S6 and S7 read the Fig. 6 and Fig. 7 scales, the light Fig. 4I
    # tiles read the exemplar bodies of the dark access tiles; the four supplemental videos last (they read only results tables)
    "v4": dict(title="Manuscript freeze v3 (Cell figures and videos)",
               renders=["v4_trial_tiles", "v4_fig1_tiles", "v4_fig1_glance", "v4_fig2_tiles", "v4_fig3_tiles", "v4_fig3_maps", "v4_fig4_tiles",
                        "v4_fig4_access_light", "v4_fig5_tiles", "v4_fig6_tiles", "v4_fig7_tiles", "v4_icons"] + [f"v4_s{i}_tiles" for i in range(1, 8)]
                       + [f"v4_video_s{i}" for i in range(1, 5)],
               figures=V4_FIGURES, build_dir="v4/", outputs=[FIG / "v4", FIG / "renders" / "v4"], exclude=[FIG / "v4" / "preview"],
               not_rebuilt="the design previews of the revision-3 board (`figures/v4/preview`, built against a preview results root)",
               qc_dir=FIG / "v4" / "qc", geometry="width 174 mm, height at most 225 mm", kinds="main and Supplemental",
               superseded="the Nature-format figures (`figures/main`, `figures/extended`), pinned by the freeze-v1.1 tag"),
}
PACKAGES = ["numpy", "scipy", "pandas", "matplotlib", "numba", "pillow", "pyarrow"]
NOT_SELF_CHECKED = {"meta/output_manifest.csv": "analysis output manifest (lists the hashes checked above)",
                    "qa/determinism.csv": "record of the analysis determinism check"}
PASSES_WORD = {1: "once", 2: "twice"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files(roots, exclude):
    out = []
    for r in roots:
        for p in sorted(r.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and not any(p.is_relative_to(e) for e in exclude):
                out.append(p)
    return out


def hashes(roots, exclude=()):
    return {p.relative_to(ROOT).as_posix(): (p.stat().st_size, sha256(p)) for p in files(roots, exclude)}


def run(rel):
    """One script in a fresh interpreter with the launcher's environment (this process was started by launch.py)."""
    t = time.perf_counter()
    r = subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / rel)], cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"{rel} failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    issues = [int(n) for n in re.findall(r"^\[\w+\] (\d+) QC issue", r.stdout, flags=re.M)]
    return dict(script=rel, seconds=round(time.perf_counter() - t, 1), qc_issues=sum(issues), output=(r.stdout + r.stderr)[-400:])


def rebuild(s):
    log = [run(f"manuscript/figures/render/{name}.py") for name in s["renders"]]
    log += [run(f"manuscript/figures/build/{name}.py") for name in ["check_palette"] + [s["build_dir"] + f for f in s["figures"]]]
    bad = [x for x in log if x["qc_issues"]]
    if bad:
        raise SystemExit(f"QC issues: {[(x['script'], x['qc_issues']) for x in bad]}")
    return log


def version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout.strip() if r.returncode == 0 else None


def group(path: str) -> str:
    for prefix, name in (("manuscript/results/figure_data/", "figure-ready table"), ("manuscript/results/", "analysis table"),
                         ("manuscript/figures/v4/videos/", "supplemental video"),
                         ("manuscript/figures/v4/main/", "main figure"), ("manuscript/figures/v4/supplement/", "Supplemental figure"),
                         ("manuscript/figures/v4/source_data/", "source data"), ("manuscript/figures/v4/qc/", "QC record"),
                         ("manuscript/figures/main/", "main figure"), ("manuscript/figures/extended/", "Extended Data figure"),
                         ("manuscript/figures/renders/", "render"), ("manuscript/figures/source_data/", "source data"),
                         ("manuscript/figures/qc/", "QC record")):
        if path.startswith(prefix):
            return name
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=sorted(SETS), default="v1")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--analysis-runs", type=int, default=2, choices=(1, 2),
                        help="2: require the two-run determinism record; 1: one clean run, not repeated (the user's quick freeze)")
    args = parser.parse_args()
    s = SETS[args.set]
    t0 = time.perf_counter()
    # Code state at the start, before this run writes anything; the freeze's own records do not count as changes.
    head = git("rev-parse", "HEAD")
    dirty = [ln for ln in (git("status", "--porcelain", "--untracked-files=no") or "").splitlines()
             if not ln[3:].strip().startswith(("manuscript/FREEZE.md", "manuscript/FREEZE_MANIFEST.csv"))]
    for rel in TESTS:
        run(rel)
        print(f"tests passed: {rel}", flush=True)
    if args.analysis_runs == 2:
        det = pd.read_csv(RESULTS / "qa" / "determinism.csv")
        if not det["identical"].astype(str).str.lower().eq("true").all():
            raise SystemExit("analysis determinism check did not pass")
    else:
        det = pd.DataFrame(dict(file=[], identical=[]))
    before = hashes(s["outputs"], s["exclude"])
    passes, logs = [], []
    for i in range(args.passes):
        logs.append(rebuild(s))
        passes.append(hashes(s["outputs"], s["exclude"]))
        print(f"pass {i + 1}: {len(passes[-1])} files, {sum(x['seconds'] for x in logs[-1]):.0f} s", flush=True)
    keys = sorted(set().union(*passes))
    unstable = [k for k in keys if len({p.get(k) for p in passes}) != 1]
    if unstable:
        raise SystemExit(f"rebuild not byte-identical: {unstable[:20]}")
    changed = sorted(k for k in keys if before.get(k) != passes[-1].get(k))
    figs = hashes(s["outputs"], s["exclude"])
    res = hashes([RESULTS])
    det_ok = dict(zip(det["file"], det["identical"].astype(str).str.lower().eq("true")))
    rows = []
    for path, (size, digest) in sorted({**res, **figs}.items()):
        if path.startswith("manuscript/results/"):
            rel = path[len("manuscript/results/"):]
            if args.analysis_runs == 1:
                check, ok = "analysis run once from clean", None
            elif rel in det_ok:
                check, ok = "analysis run twice from clean", bool(det_ok[rel])
            else:
                check, ok = NOT_SELF_CHECKED[rel], None
        else:
            check, ok = f"rebuilt {PASSES_WORD.get(args.passes, f'{args.passes} times')}", True
        rows.append(dict(path=path, group=group(path), bytes=size, sha256=digest, reproducible=ok, check=check))
    man = pd.DataFrame(rows)
    man.to_csv(MS / "FREEZE_MANIFEST.csv", index=False, lineterminator="\n")
    qc = {f: json.loads((s["qc_dir"] / f"{f}_qc.json").read_text(encoding="utf-8")) for f in s["figures"]}
    versions = {p: version(p) for p in PACKAGES}
    counts = man.groupby("group").agg(files=("path", "size"), megabytes=("bytes", lambda b: round(b.sum() / 1e6, 1)))
    lines = [f"# {s['title']}", "",
             f"- Code: commit `{head}`" + (f" (tracked files modified at the start: {', '.join(ln[3:] for ln in dirty)})" if dirty else " (clean)") + ".",
             f"- Environment: Python {sys.version.split()[0]}; " + ", ".join(f"{k} {v}" for k, v in versions.items()) + ".",
             f"- Unit tests: {', '.join(TESTS)} passed.",
             (f"- Analysis: two complete runs from a clean state, {int(sum(det_ok.values()))} of {len(det_ok)} files byte-identical "
              "(`results/qa/determinism.csv`)." if args.analysis_runs == 2 else
              f"- Analysis: one complete run from a clean state ({len(res)} files); the two-run determinism check was not repeated for this "
              "freeze (the user asked for a quick freeze)."),
             f"- Figures: {len(s['renders'])} render scripts, the palette check and {len(s['figures'])} {s['kinds']} figures rebuilt "
             f"{PASSES_WORD.get(args.passes, f'{args.passes} times')} in fresh processes; "
             f"all {len(keys)} output files byte-identical across passes; {len(changed)} differed from the files on disk before the freeze"
             + (f" ({', '.join(changed[:8])}{', ...' if len(changed) > 8 else ''})" if changed else "") + ".",
             f"- QC: every figure has 0 issues (sizes below; {s['geometry']}).",
             "- Not rebuilt: `figures/renders/pilot` (renderer pilot, used by no figure) and the downloaded MaleCNS skeletons "
             "(inputs; SHA-256 in `figures/assets/skeletons/receipt.csv`)" + (f"; {s['not_rebuilt']}" if s.get("not_rebuilt") else "")
             + (f"; superseded: {s['superseded']}" if s["superseded"] else "") + ".",
             "", "| Figure | Size (mm) | Tiles | QC issues |", "|---|---|---:|---:|"]
    for f in s["figures"]:
        q = qc[f]
        lines.append(f"| {f} | {q['size_mm'][0]:g} × {q['size_mm'][1]:g} | {len(q.get('tiles', []))} | {len(q['issues'])} |")
    lines += ["", "| Deliverables | Files | MB |", "|---|---:|---:|"]
    lines += [f"| {g} | {int(r.files)} | {r.megabytes} |" for g, r in counts.iterrows()]
    lines += ["", "Every file, with its SHA-256, is listed in `FREEZE_MANIFEST.csv`.", ""]
    (MS / "FREEZE.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"freeze: {len(man)} files in the manifest; {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
