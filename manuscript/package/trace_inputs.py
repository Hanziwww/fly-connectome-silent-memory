r"""Records the files that the Cell figures' renders, the figure builds and the writing scripts read, so that the deposit holds
every input of every published item (the analysis lists its own inputs in manuscript/results/meta/input_manifest.csv).

Each script runs in a fresh interpreter with the launcher's environment (local/ left off the path) and an audit hook that
logs every file it opens under the project root. The renders, videos and graphical abstract rebuild in place: their folders
are copied first, and a copy is put back if a single byte differs after the run. The figure builds write to
manuscript/figures/v4/trace-rebuild, which is compared with the frozen figures and then removed. The QC reports there name
their own folder, so they are compared with that name replaced.

Writes manuscript/package/traced_inputs.csv: every data file read (untracked, outside the frozen outputs and the
interpreter), with its size, SHA-256 and the first script that read it.

  .\MIGRATION\python\python.exe -B -X utf8 MIGRATION\launch.py --script manuscript\package\trace_inputs.py
"""
import csv
import hashlib
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "manuscript" / "figures"
OUT_CSV = Path(__file__).resolve().parent / "traced_inputs.csv"
BUILD_OUT = FIG / "v4" / "trace-rebuild"          # common.finish() needs the folder under manuscript/figures
IN_PLACE = [FIG / "renders" / "v4", FIG / "v4" / "videos", FIG / "v4" / "ga"]
RENDERS = (["v4_trial_tiles", "v4_fig1_tiles", "v4_fig1_glance", "v4_fig2_tiles", "v4_fig3_tiles", "v4_fig3_maps", "v4_fig4_tiles",
            "v4_fig4_access_light", "v4_fig5_tiles", "v4_fig6_tiles", "v4_fig7_tiles", "v4_icons"]
           + [f"v4_s{i}_tiles" for i in range(1, 8)] + [f"v4_video_s{i}" for i in range(1, 5)] + ["v4_ga_tiles"])
SCRIPTS = ([f"manuscript/figures/render/{n}.py" for n in RENDERS]
           + ["manuscript/figures/build/check_palette.py"] + [f"manuscript/figures/build/v4/fig{i}.py" for i in range(1, 8)]
           + [f"manuscript/figures/build/v4/s{i}.py" for i in range(1, 8)] + ["manuscript/figures/build/v4/graphical_abstract.py"]
           + [f"manuscript/writing/{n}.py" for n in ("build_ledger", "build_macros", "build_table_s1", "build_tables_s2_s4", "build_data_s1")])
NOT_DATA = re.compile(r"^(MIGRATION/python/|flybrain/\.venv|flybrain/flybrain/|manuscript/results/|manuscript/figures/(renders|v4)/)"
                      r"|__pycache__|\.pyc?$")
_WRITE = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree(root, rename=None):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            data = p.read_bytes()
            if rename and p.suffix == ".json":
                data = data.replace(*rename)
            out[p.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
    return out


def child(log, script):
    """Inside the child interpreter: log every file opened under ROOT, then run the script as __main__."""
    seen = {}
    busy = [False]

    def hook(event, args):
        if busy[0] or event != "open":
            return
        path, mode, flags = args
        if path is None or isinstance(path, int):
            return
        busy[0] = True
        try:
            p = Path(os.fsdecode(path)).resolve()
            if p == ROOT or ROOT in p.parents:
                writing = any(c in mode for c in "wax+") if isinstance(mode, str) else bool(isinstance(flags, int) and flags & _WRITE)
                key = p.relative_to(ROOT).as_posix()
                seen[key] = seen.get(key, "") + ("w" if writing else "r")
        except (TypeError, ValueError, OSError):
            pass
        finally:
            busy[0] = False

    target = (ROOT / script).resolve()
    sys.argv = [str(target)]
    sys.path.insert(0, str(target.parent))
    sys.addaudithook(hook)
    try:
        runpy.run_path(str(target), run_name="__main__")
    finally:
        busy[0] = True
        Path(log).write_text("".join(json.dumps({"path": k, "ops": "".join(sorted(set(v)))}) + "\n" for k, v in sorted(seen.items())),
                             encoding="utf-8")


def env():
    e = os.environ.copy()
    e["PYTHONPATH"] = os.pathsep.join(p for p in e.get("PYTHONPATH", "").split(os.pathsep) if p and "local" not in Path(p).parts)
    return e


def main():
    work = Path(tempfile.mkdtemp(prefix="trace-inputs-"))
    backup = work / "backup"
    for d in IN_PLACE:
        shutil.copytree(d, backup / d.relative_to(FIG))
    before = {d: tree(backup / d.relative_to(FIG)) for d in IN_PLACE}
    reads = {}
    try:
        for script in SCRIPTS:
            log = work / (Path(script).stem + ".jsonl")
            e = env()
            if "/build/" in script:
                e["FLYBRAIN_V4_OUT"] = str(BUILD_OUT)
            r = subprocess.run([sys.executable, "-B", "-X", "utf8", __file__, "--child", str(log), script], cwd=ROOT, env=e,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(f"{script}: exit {r.returncode}", flush=True)
            if r.returncode != 0:
                raise SystemExit(r.stdout[-3000:] + r.stderr[-3000:])
            for line in log.read_text(encoding="utf-8").splitlines():
                x = json.loads(line)
                if "w" not in x["ops"]:
                    reads.setdefault(x["path"], script)
        for d in IN_PLACE:
            after = tree(d)
            changed = sorted(k for k in after if before[d].get(k) != after[k]) + sorted(set(before[d]) - set(after))
            if changed:
                shutil.rmtree(d)
                shutil.copytree(backup / d.relative_to(FIG), d)
                raise SystemExit(f"{d.relative_to(ROOT)}: {len(changed)} files differ after the rebuild (copy restored): {changed[:5]}")
        rename = (b"v4/trace-rebuild/", b"v4/")
        for sub in ("main", "supplement", "qc", "source_data"):
            a, b = tree(BUILD_OUT / sub, rename), tree(FIG / "v4" / sub)
            if a != b:
                raise SystemExit(f"figures/v4/{sub}: the rebuild differs from the frozen files: "
                                 f"{sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))[:5]}")
        print("every render, video, figure, source table and QC report rebuilt byte for byte")
    finally:
        if BUILD_OUT.exists():
            shutil.rmtree(BUILD_OUT)
        shutil.rmtree(work, ignore_errors=True)
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.splitlines())
    rows = [dict(path=p, bytes=(ROOT / p).stat().st_size, sha256=sha256(ROOT / p), read_by=s) for p, s in sorted(reads.items())
            if p not in tracked and not NOT_DATA.search(p) and (ROOT / p).is_file()]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256", "read_by"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{OUT_CSV.relative_to(ROOT)}: {len(rows)} data files")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2], sys.argv[3])
    else:
        main()
