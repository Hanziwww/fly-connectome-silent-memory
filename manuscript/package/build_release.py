r"""Assembles the release packages of the Cell paper in release/ (git-ignored) from the committed code, the frozen outputs and
the study's records:

  code     1_code/: the code release, the committed files that CODE_INCLUDE lists (the workspace README unchanged, since
           the analysis reads it) with the release's guide, license, citation file, notices and pinned requirements, and a
           zip of it; every text file is scanned for credentials and connection details
  data     2_zenodo/: the deposit as three Zenodo records, each a folder to upload as it is
             A_records_core      every file the analysis (its input manifest) and the renders, figure builds and writing
                                 scripts (traced_inputs.csv) read; every file of the records that Tables S2 and S3 cite,
                                 but for the bulky ones of record B; the studies' and the paper's result tables; the
                                 figures, renders, videos, source data and supplemental tables
             B_records_extended  the verification replays, per-update behavior features and dense recordings that no
                                 analysis reads
             C_code              the code release
           Each holds its zip archives (parts of at most PART_GB), MANIFEST.csv (every file with its archive, size and
           SHA-256), README.md, verify_deposit.py and zenodo.json (the metadata for the upload form). Records and code of
           experiments that the paper does not report are left out, and a few mixed tables and records are trimmed to what
           it reports, by the workspace's private list (manuscript/package/private, not released; MANIFEST.csv marks the
           trimmed files). Archives are reused when their members and the rules are unchanged (release/_work).
  paper    3_manuscript/: the Cell submission (cover letter, manuscript without figure pages, the figures as separate files,
           graphical abstract, Document S1, Tables S2-S4, Data S1, Videos S1-S4, the form's text), the bioRxiv preprint
           with its supplementary files, and the LaTeX source, compiled once on its own as a check
  figures  4_figures_and_tables/: the original figures (PDF, SVG, JPG; TIFF at 600 ppi for the main figures), the
           graphical abstract, Tables S1-S4, Data S1 and the videos, with INDEX.md
  check    the checks of every part (figure sizes and image resolution, the graphical abstract, the videos, Document S1's
           size, the front matter's lengths, the text checks, the marked fields left), and release/README.md

  .\MIGRATION\python\python.exe -B -X utf8 MIGRATION\launch.py --script manuscript\package\build_release.py [part ...]

Parts run in the order code, paper, figures, data, check; no part means all. DOIs and the affiliation, once known, go in
manuscript/package/release.json.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "manuscript" / "writing"))
sys.path.insert(0, str(ROOT / "manuscript" / "figures"))
import check_tex  # noqa: E402
import xlsx  # noqa: E402
from lib.names import DEV_LABELS, british  # noqa: E402

sys.path.insert(0, str(HERE / "private"))
try:                                     # what the paper does not report: private to the workspace, never released
    import exclusions as PRIVATE  # noqa: E402
    PRIVATE.SOURCE_VERSION = hashlib.sha256((HERE / "private" / "exclusions.py").read_bytes()).hexdigest()
except ImportError:
    PRIVATE = None

REL = ROOT / "release"
WORK = REL / "_work"
PAPER = ROOT / "manuscript" / "paper"
FIG = ROOT / "manuscript" / "figures"
SUPP = PAPER / "build" / "supplement"
CONF = json.loads((HERE / "release.json").read_text(encoding="utf-8"))
TITLE = "Brain state locks silent memories in a trained fly connectome"
FLYBRAIN_COMMIT = "c7878b9ee273564186acf8cbdef96289335efa5e"      # docs/workspace.md: the upstream checkout the study used
DATE = (1980, 1, 1, 0, 0, 0)                                        # zip member dates: rebuilds are byte-identical
PART_GB = 8.0
STORED = {".npz", ".npy", ".pt", ".parquet", ".png", ".jpg", ".jpeg", ".mp4", ".pdf", ".tif", ".tiff", ".zip", ".gz", ".xlsx",
          ".feather", ".pyd"}
TEXT = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".tsv", ".tex", ".bib", ".bst", ".sty", ".toml", ".yaml", ".yml",
        ".cfg", ".ini", ".sh", ".ps1", ".bat", ".log", ".err", ".out", ".js", ".mjs", ".html", ".css", ".cff", ".svg", ""}
FORBIDDEN = re.compile(r"(^|/)(local/|runtime/|\.venv[^/]*/|ssh\.txt$|[^/]*known-hosts[^/]*$|\.cluster\.json$|[^/]*\.env$|"
                       r"remote(-[^/]*)?\.toml$)")

# ---------------------------------------------------------------- the code release
CODE_INCLUDE = (".gitattributes", "README.md", "method.json", "research.py", "workflow.py", "Start Fly Sokoban.bat", "docs/",
                "data/external/receipt.csv", "sokoban/", "studykit/", "studies/", "tools/__init__.py", "tools/remote/",
                "MIGRATION/launch.py", "MIGRATION/setup_linux.py", "MIGRATION/requirements-linux.txt",
                "MIGRATION/environment-windows.json", "MIGRATION/environment-original.json", "MIGRATION/linux-version-sources.json",
                "manuscript/analysis/", "manuscript/figures/", "manuscript/writing/", "manuscript/paper/", "manuscript/package/",
                "manuscript/freeze.py", "manuscript/FREEZE.md", "manuscript/FREEZE_MANIFEST.csv")
CODE_EXCLUDE = re.compile(r"^studies/[^/]+/results/|^manuscript/figures/design/|^manuscript/package/private/")  # results: the
#                                                                                       deposit; design: mock-ups; private
SECRET = re.compile(r"BEGIN [A-Z ]*PRIVATE KEY|ssh-(?:rsa|ed25519|ecdsa)\s+AAAA[0-9A-Za-z+/]{20}|"
                    r"(?:password|passwd|passphrase|api[_-]?key|secret[_-]?key|access[_-]?token)[\"']?\s*[:=]\s*[\"']?[^\s\"',;)\]}]{4,}",
                    re.I)
CONNECTION = re.compile(r"\b(?!127\.0\.0\.1)(?:\d{1,3}\.){3}\d{1,3}:\d{2,5}\b|ssh\s+-p\s*\d{2,5}|" + "seeta" + r"cloud|"   # split: the
                        r"\bconnect\.[a-z0-9-]+\.[a-z]|"                                                            # scan must not
                        r"\b[\w.+-]+@(?!hanziwww\.com)[\w-]+(?:\.[\w-]+)*\.(?:com|cn|net|org|io|edu)\b", re.I)     # match itself
# a pattern runs only on text that holds one of its cheap triggers (the deposit's records are tens of GB)
SECRET_HINTS = ("pass", "secret", "api_key", "api-key", "apikey", "token", "private key", "ssh-rsa", "ssh-ed25519", "ssh-ecdsa")
CONNECTION_HINTS = ("@", "ssh", "connect.", "seeta" + "cloud")
COLON_DIGIT = re.compile(r":\d")                     # a literal first: fast even on number-heavy JSON
IP_BEFORE = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}$")


def ip_port(text):
    """Whether the text may hold an IPv4 address with a port (then CONNECTION decides)."""
    return any(i > 6 and text[i - 1].isdigit() and IP_BEFORE.search(text, i - 15 if i > 15 else 0, i)
               for i in (m.start() for m in COLON_DIGIT.finditer(text)))


def in_code(path):
    return ((path in CODE_INCLUDE or path.startswith(tuple(p for p in CODE_INCLUDE if p.endswith("/")))) and not CODE_EXCLUDE.search(path)
            and not (PRIVATE and any(rx.search(path) for rx in PRIVATE.CODE)))


def unreported(path):
    return bool(PRIVATE and any(rx.search(path) for rx in PRIVATE.DATA))


def trims(path):
    return bool(PRIVATE and PRIVATE.trims(path))


# ---------------------------------------------------------------- the deposit
NOT_DATA = re.compile(r"^(MIGRATION/python/|flybrain/(?!cache/|data/morphology/skeletons/))|__pycache__/|\.pyc$|\.partial$|"
                      r"^manuscript/figures/v4/(preview|design|trace-rebuild)/")
OUTPUTS = ("manuscript/figures/v4", "manuscript/figures/renders/v4", "manuscript/paper/build/supplement")
GROUPS = {  # (record, group): what the README says the archive holds
    ("A", "runs"): "training runs and evaluations: configurations, models, logs and results",
    ("A", "recordings"): "the recordings of every state variable of the trained models that the analysis reads",
    ("A", "results"): "the paper's analysis tables and the studies' result tables",
    ("A", "figures"): "figures, renders, videos, the graphical abstract, source data and Tables S1–S4 and Data S1",
    ("A", "assets"): "files derived from the MaleCNS v1.0 connectome (connection matrix, skeletons, display geometry) and the "
                     "method's development catalog",
    ("B", "verification"): "the verification replays of the training runs",
    ("B", "behavior-features"): "the behavior features of every training update",
    ("B", "recordings"): "the dense recordings of the mechanism experiments that no analysis reads",
}


# ---------------------------------------------------------------- small helpers
def run(cmd, **kw):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)
    if r.returncode != 0:
        raise SystemExit(f"{' '.join(map(str, cmd))} failed ({r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r.stdout


def py(script, *args):
    return run([sys.executable, "-B", "-X", "utf8", str(ROOT / script), *args])


def git(*args):
    return run(["git", *args]).strip()


def head():
    return dict(commit=git("rev-parse", "HEAD"), date=git("log", "-1", "--format=%cs"),
                dirty=[x for x in git("status", "--porcelain").splitlines() if x and not x[3:].startswith(("release/", "grep.exe"))])


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def size(n):
    for unit, k in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= k:
            return f"{n / k:.1f} {unit}"
    return f"{n} B"


def doi(key, what="DOI to be added"):
    return f"https://doi.org/{CONF[key]}" if CONF.get(key) else f"[{what}]"


def fill(template, **values):
    text = (HERE / "templates" / template).read_text(encoding="utf-8")
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    left = re.findall(r"\{\{\w+\}\}", text)
    if left:
        raise SystemExit(f"{template}: unfilled {left}")
    for word, us in british(text):
        raise SystemExit(f"{template}: British spelling {word!r} (American: {us})")
    return text


def reset(folder):
    folder = folder.resolve()
    if REL.resolve() not in folder.parents:
        raise SystemExit(f"refusing to clear {folder}")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    return folder


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.write_bytes if isinstance(data, bytes) else lambda t: path.write_text(t, encoding="utf-8", newline="\n"))(data)


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def zip_info(arc):
    info = zipfile.ZipInfo(arc, DATE)
    info.create_system = 3
    info.external_attr = 0o644 << 16
    info.compress_type = zipfile.ZIP_STORED if Path(arc).suffix.lower() in STORED else zipfile.ZIP_DEFLATED
    return info


def zip_tree(folder, out, top):
    """A deterministic zip of a folder, under the top-level name top."""
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    with zipfile.ZipFile(out, "w", compresslevel=6) as z:
        for p in files:
            z.writestr(zip_info(f"{top}/{p.relative_to(folder).as_posix()}"), p.read_bytes())
    return out


def scan(text, path, found, limit=3):
    # Author-approved publication emails are metadata, not remote-host credentials.
    for author in CONF.get("authors", []):
        email = author.get("email", "")
        if email:
            text = text.replace(email, "AUTHOR_EMAIL")
    low = text.lower()
    for rx, kind, hit in ((SECRET, "credential", any(h in low for h in SECRET_HINTS)),
                          (CONNECTION, "connection detail", any(h in low for h in CONNECTION_HINTS) or ip_port(text))):
        if not hit:
            continue
        for m in rx.finditer(text):
            if len([f for f in found if f[0] == path]) >= limit:
                return
            found.append((path, kind, m.group(0)[:80]))


def tex_text(tex, ledger):
    """TeX of the paper -> plain text, marked fields kept as [...]."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    tex = re.sub(r"\\tofill\{([^{}]*)\}", r"[\1]", tex)
    return check_tex.convert(tex, check_tex.Doc(), "release", ledger, False)


def unwrap(text):
    """Plain text with the source's line breaks removed: one line per paragraph, heading and list item (for web forms)."""
    out = []
    for para in re.split(r"\n[ \t]*\n", re.sub(r"[ \t]+", " ", text)):
        cur = []
        for line in (x.strip() for x in para.split("\n")):
            if line.startswith("- ") and cur:
                out.append(" ".join(cur))
                cur = []
            if line:
                cur.append(line)
        if cur:
            out.append(" ".join(cur))
    return "\n\n".join(out)


def ledger():
    return {r["id"]: r for r in csv.DictReader(open(ROOT / "manuscript" / "writing" / "number_ledger.csv", encoding="utf-8"))}


# ---------------------------------------------------------------- part 1: code
def committed():
    """path -> bytes of every file of HEAD, so the release is exactly a commit."""
    tar = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        return {m.name: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}


def code_extras(h, led):
    readme = fill("code_readme.md", CITATION=citation(), NEURONS=led["L01"]["display"], DATA_DOI=doi("doi_records_core"),
                  CODE_DOI=doi("doi_code"), VERSION=CONF["version"], COMMIT=h["commit"][:12], DATE=h["date"],
                  FLYBRAIN_COMMIT=FLYBRAIN_COMMIT)
    env = json.loads((ROOT / "MIGRATION" / "environment-windows.json").read_text(encoding="utf-8"))
    pins = [f"{d['name']}=={d['version'].split('+')[0]}" for d in env["distributions"] if d["name"] != "flybrain"]
    req = ("# The analysis, figures and writing scripts: Python 3.12.13, as recorded in MIGRATION/environment-windows.json.\n"
           "# FlyBrain 0.1.0 is cloned into ./flybrain (README). PyTorch ran as the CUDA 13.2 build (2.14.0+cu132); the analysis\n"
           "# uses the CPU only.\n" + "\n".join(pins) + "\n")
    def cff_authors(indent, include_email=False):
        lines = []
        for author in CONF["authors"]:
            lines.extend([indent + '- family-names: ' + json.dumps(author["family_name"]),
                          indent + '  given-names: ' + json.dumps(author["given_names"])])
            if include_email and author.get("email"):
                lines.append(indent + '  email: ' + json.dumps(author["email"]))
        return lines
    cff = "\n".join([
        "cff-version: 1.2.0",
        'message: "If you use this code, cite the article, and FlyBrain and the MaleCNS connectome that it builds on."',
        "type: software", f'title: "Code of: {TITLE}"', f'version: "{CONF["version"]}"', f'date-released: "{h["date"]}"',
        "license: MIT", "authors:", *cff_authors("  ", True),
        *([f'doi: "{CONF["doi_code"]}"'] if CONF.get("doi_code") else []),
        "keywords:", *[f'  - "{k}"' for k in ("connectome", "Drosophila", "working memory", "spike-frequency adaptation",
                                             "activity-silent memory", "state-dependent retrieval")],
        "preferred-citation:", "  type: article", f'  title: "{TITLE}"', "  authors:", *cff_authors("    "), "  year: 2026", *([f'  doi: "{CONF["doi_article"]}"'] if CONF.get("doi_article") else []),
        ""])
    mit = ("MIT License\n\nCopyright (c) 2026 Han Wang\n\n"
           "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated\n"
           "documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the\n"
           "rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit\n"
           "persons to whom the Software is furnished to do so, subject to the following conditions:\n\n"
           "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the\n"
           "Software.\n\n"
           "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE\n"
           "WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR\n"
           "COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR\n"
           "OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n")
    ignore = "\n".join(["# Data of the Zenodo records, unpacked into this folder, and local outputs", "__pycache__/", "*.pyc",
                        "/flybrain/", "/runs/", "/archive/", "/studies/**/runs/", "/studies/**/results/",
                        "/studies/brain-sokoban-v1/mechanisms*/**/*.npz", "/manuscript/results/", "/manuscript/figures/v4/",
                        "/manuscript/figures/renders/", "/manuscript/figures/assets/skeletons/*",
                        "!/manuscript/figures/assets/skeletons/receipt.csv", "/manuscript/paper/build/", "/sokoban/artifacts/**/*.bin",
                        "/release/", "", "# Machine-specific settings and credentials, never committed", "/local/", "/runtime/",
                        "/ssh.txt", "*known-hosts*", "/remote.toml", "/remote-*.toml", ""])
    zen = zenodo("C", h)
    return {".github/README.md": readme, "START_HERE.md": readme, "LICENSE": mit, "CITATION.cff": cff,
            "THIRD_PARTY_NOTICES.md": fill("third_party.md", FLYBRAIN_COMMIT=FLYBRAIN_COMMIT),
            "requirements-analysis.txt": req, ".gitignore": ignore, ".zenodo.json": json.dumps(zen["metadata"], indent=1) + "\n"}


def build_code():
    h, led = head(), ledger()
    name = f"{CONF['name']}-{CONF['version']}"
    out = reset(REL / "1_code")
    tree = out / name
    files = {p: b for p, b in committed().items() if in_code(p)}
    files.update({p: t.encode("utf-8") for p, t in code_extras(h, led).items()})
    found = []
    for p, b in sorted(files.items()):
        write(tree / p, b)
        if Path(p).suffix.lower() in TEXT:
            scan(b.decode("utf-8", errors="replace"), p, found)
    rows = [dict(path=p, bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for p, b in sorted(files.items())]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["path", "bytes", "sha256"], lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    write(tree / "RELEASE_MANIFEST.csv", buf.getvalue())
    archive = zip_tree(tree, out / f"{name}.zip", name)
    secrets = [f for f in found if f[1] == "credential"]
    report = {"commit": h["commit"], "uncommitted": h["dirty"], "files": len(rows) + 1, "zip": archive.name,
              "zip_bytes": archive.stat().st_size, "findings": found}
    write(WORK / "code.json", json.dumps(report, indent=1))
    print(f"code: {len(rows) + 1} files, {archive.name} {size(archive.stat().st_size)}; {len(found)} scan findings"
          + (f"; {len(h['dirty'])} uncommitted changes NOT included" if h["dirty"] else ""))
    for f in found:
        print(f"  {f[1]}: {f[0]}: {f[2]}")
    if secrets:
        raise SystemExit("code: credential-like text found; nothing to publish until it is resolved")
    return report


def citation():
    return f"https://doi.org/{CONF['doi_article']}" if CONF.get("doi_article") else "prepared for submission"


# ---------------------------------------------------------------- part 4 of the deposit metadata: Zenodo
def zenodo(record, h, archives=None):
    common = dict(creators=CONF["creators"],
                  access_right=("restricted" if record in ("A", "B") else "open"), language="eng", publication_date=h["date"],
                  keywords=["connectome", "Drosophila", "MaleCNS", "spiking neural network", "working memory",
                            "spike-frequency adaptation", "activity-silent memory", "state-dependent retrieval"])
    if record in ("A", "B"):
        common["access_conditions"] = "Files are restricted during peer review. Editors and reviewers may obtain access from Chunjie Tian (tianchunjie@gdmu.edu.cn)."
    if record == "C":
        related = [{"identifier": "https://github.com/HEREISCB/flybrain", "relation": "requires", "scheme": "url"}]
    else:
        related = [{"identifier": "https://male-cns.janelia.org/download/", "relation": "isDerivedFrom", "scheme": "url"}]
    links = {"A": [("doi_records_extended", "isSupplementedBy"), ("doi_code", "isCompiledBy")],
             "B": [("doi_records_core", "isSupplementTo"), ("doi_code", "isCompiledBy")],
             "C": [("doi_records_core", "compiles"), ("doi_records_extended", "compiles")]}[record]
    for key, rel in [("doi_article", "isSupplementTo"), ("doi_biorxiv", "isSupplementTo"), *links]:
        if CONF.get(key):
            related.append({"identifier": CONF[key], "relation": rel, "scheme": "doi"})
    meta = {
        "A": dict(upload_type="dataset", license="cc-by-4.0",
                  title=f"Simulation records, trained models and result tables of \"{TITLE}\" (part 1 of 2)"),
        "B": dict(upload_type="dataset", license="cc-by-4.0",
                  title=f"Simulation records of \"{TITLE}\" (part 2 of 2): verification replays, behavior features and dense recordings"),
        "C": dict(upload_type="software", license="MIT", title=f"Code of \"{TITLE}\""),
    }[record]
    desc = {"A": "Every file that the paper's analysis, renders, figure builds and writing scripts read; the models, configurations, "
                 "logs and results of every training run and evaluation in Table S3; and the published result tables, figures, "
                 "videos, figure source data and supplemental tables. Unpack the archives into the code release; MANIFEST.csv "
                 "lists every file with its SHA-256, and verify_deposit.py checks and unpacks them. See README.md.",
            "B": "The verification replays, behavior features of every training update and dense recordings that no analysis "
                 "of the paper reads, completing the run folders of part 1. See README.md.",
            "C": f"Code of the article, release {CONF['version']}: training and testing of the whole-CNS model, the declared "
                 "protocols, the analysis, the code of every figure, video and supplemental table, and the LaTeX source. "
                 "See START_HERE.md."}[record]
    return {"metadata": {**common, **meta, "description": f"<p>{desc}</p><p>Funding: {CONF['funding_statement']}</p>", "related_identifiers": related,
                         "version": CONF["version"] if record == "C" else "1.0",
                         **({"notes": "Archives: " + ", ".join(archives)} if archives else {})}}


# ---------------------------------------------------------------- part 2: data
def cited_records():
    """Every path that Tables S2 and S3 give as a record (files or folders)."""
    out = set()
    for book, sheets in (("Table_S2.xlsx", ("Tests",)), ("Table_S3.xlsx", ("Training runs", "Evaluations"))):
        wb = xlsx.read(SUPP / book)
        for sheet in sheets:
            rows = wb[sheet]
            i = rows[0].index("Record")
            for r in rows[1:]:
                for part in re.split(r";\s*|,\s+", str(r[i] if i < len(r) and r[i] else "")):
                    part = part.strip().rstrip("/")
                    if part:
                        out.add(part if "/" in part else f"studies/brain-pathology-v1/runs/{part}")
    return sorted(out)


def walk(path):
    p = ROOT / path
    if p.is_file():
        yield path
    elif p.is_dir():
        for d, dirs, files in os.walk(p):
            dirs[:] = sorted(x for x in dirs if not FORBIDDEN.search(f"{x}/") and x != "__pycache__")
            for f in sorted(files):
                yield (Path(d) / f).relative_to(ROOT).as_posix()


def inventory():
    """path -> set of reasons, for every file of the deposit."""
    tracked = set(git("ls-files").splitlines())
    read = defaultdict(set)
    for r in csv.DictReader(open(ROOT / "manuscript" / "results" / "meta" / "input_manifest.csv", encoding="utf-8")):
        read[r["path"]].add("analysis input")
    for r in csv.DictReader(open(HERE / "traced_inputs.csv", encoding="utf-8")):
        read[r["path"]].add("figure or writing input")
    files = defaultdict(set, {p: set(v) for p, v in read.items()})
    for rec in cited_records():
        for p in walk(rec):
            files[p].add("cited record")
    for base in ["manuscript/results", "runs/linear-joint", *OUTPUTS, *[f"studies/{s.name}/results" for s in (ROOT / "studies").iterdir()]]:
        for p in walk(base):
            files[p].add("result or output")
    keep, dropped = {}, defaultdict(int)
    for p, why in files.items():
        if FORBIDDEN.search(p):
            dropped["forbidden"] += 1
        elif NOT_DATA.search(p):
            dropped["not data"] += 1
        elif unreported(p):
            dropped["not reported"] += 1
        elif p in tracked and in_code(p):
            dropped["in the code release"] += 1
        elif not (ROOT / p).is_file():
            dropped["missing"] += 1
        else:
            keep[p] = why
    missing = [p for p, why in files.items() if not (ROOT / p).is_file() and not FORBIDDEN.search(p) and not NOT_DATA.search(p)]
    return keep, set(read), dict(dropped), missing


def place(path, read):
    """(record, group) of a deposit file."""
    name = path.rsplit("/", 1)[-1]
    if path not in read and (name == "verification.json" or "/behavior-features/" in path
                             or (path.startswith("studies/brain-sokoban-v1/mechanisms") and name.endswith(".npz"))):
        return ("B", "verification" if name == "verification.json" else "behavior-features" if "/behavior-features/" in path
                else "recordings")
    if path.startswith("manuscript/results/") or re.match(r"studies/[^/]+/results/", path):
        return ("A", "results")
    if path.startswith(OUTPUTS):
        return ("A", "figures")
    if path.startswith(("flybrain/", "sokoban/", "manuscript/figures/assets/", "archive/")):
        return ("A", "assets")
    if path.startswith("studies/brain-sokoban-v1/mechanisms"):
        return ("A", "recordings")
    return ("A", "runs")


def write_archive(task):
    """One zip of the deposit (a worker process): members in order, fixed dates, SHA-256 and a scan of every text member."""
    out, members, sidecar, rules = task
    tmp = Path(out + ".partial")
    rows, found = [], []
    with zipfile.ZipFile(tmp, "w", compresslevel=6, allowZip64=True) as z:
        for arc, src in members:
            info = zip_info(arc)
            if trims(arc):
                data = PRIVATE.trim(arc, Path(src).read_bytes())
                if data is not None:                       # a mixed table or record, trimmed to what the paper reports
                    z.writestr(info, data)
                    scan(data.decode("latin-1"), arc, found)
                    rows.append([arc, len(data), hashlib.sha256(data).hexdigest(), "yes"])
                    continue
            h, n, tail = hashlib.sha256(), 0, ""
            text = info.compress_type == zipfile.ZIP_DEFLATED
            with open(src, "rb") as f, z.open(info, "w", force_zip64=os.path.getsize(src) > (1 << 30)) as dst:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
                    n += len(chunk)
                    dst.write(chunk)
                    if text:
                        s = chunk.decode("latin-1")
                        scan(tail + s, arc, found)
                        tail = s[-200:]
            rows.append([arc, n, h.hexdigest(), ""])
    os.replace(tmp, out)
    Path(sidecar).write_text(json.dumps(dict(members=[[a, os.path.getsize(s), os.stat(s).st_mtime_ns] for a, s in members],
                                             rules=rules, rows=rows, found=found)), encoding="utf-8")
    return out, rows, found


def build_data(workers):
    h = head()
    keep, read, dropped, missing = inventory()
    # identical files stored once: hard links (one inode) keep the first path of the record that needs them
    by_inode = defaultdict(list)
    for p in keep:
        st = os.stat(ROOT / p)
        by_inode[(st.st_dev, st.st_ino)].append(p)
    same_as = {}
    for paths in by_inode.values():
        if len(paths) > 1:
            core = sorted(p for p in paths if place(p, read)[0] == "A")
            canon = core[0] if core else sorted(paths)[0]
            same_as.update({p: canon for p in paths if p != canon})
    entries = defaultdict(list)
    for p in sorted(keep):
        if p not in same_as:
            entries[place(p, read)].append(p)
    plans = []
    for (record, group), paths in sorted(entries.items()):
        parts, cur, est, last = [], [], 0.0, None
        for p in paths:
            b = (ROOT / p).stat().st_size
            e = b if Path(p).suffix.lower() in STORED else b / 3.3
            folder = p.rsplit("/", 1)[0]
            if cur and est + e > PART_GB * 1e9 and folder != last:
                parts.append(cur)
                cur, est = [], 0.0
            cur.append(p)
            est += e
            last = folder
        if cur:
            parts.append(cur)
        prefix = {"A": "core", "B": "extended"}[record]
        for k, part in enumerate(parts, 1):
            plans.append((record, group, f"{prefix}-{group}" + (f"-{k}" if len(parts) > 1 else "") + ".zip", part))
    folders = {"A": REL / "2_zenodo" / "A_records_core", "B": REL / "2_zenodo" / "B_records_extended"}
    for f in folders.values():
        f.mkdir(parents=True, exist_ok=True)
    side = WORK / "data"
    side.mkdir(parents=True, exist_ok=True)
    planned = {(r, n) for r, _, n, _ in plans}
    for r, f in folders.items():                       # archives of an earlier plan that this plan no longer has
        for z in f.glob("*.zip"):
            if (r, z.name) not in planned:
                z.unlink()
    tasks, results = [], {}
    for record, group, name, part in plans:
        out = folders[record] / name
        sc = side / f"{record}-{name}.json"
        members = [[p, os.path.getsize(ROOT / p), os.stat(ROOT / p).st_mtime_ns] for p in part]
        rules = PRIVATE.SOURCE_VERSION if any(trims(p) for p in part) else None
        if out.exists() and sc.exists():
            old = json.loads(sc.read_text(encoding="utf-8"))
            if old["members"] == members and old.get("rules") == rules and all(len(r) == 4 for r in old["rows"]):
                results[(record, name)] = (old["rows"], old["found"])
                continue
        tasks.append((str(out), [(p, str(ROOT / p)) for p in part], str(sc), rules))
    print(f"data: {len(keep)} files ({sum((ROOT / p).stat().st_size for p in keep) / 1e9:.1f} GB with copies), "
          f"{len(plans)} archives, {len(tasks)} to write; left out: {dropped}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for out, rows, found in ex.map(write_archive, tasks):
            p = Path(out)
            results[("A" if p.parent == folders["A"] else "B", p.name)] = (rows, found)
            print(f"  {p.parent.name}/{p.name}: {len(rows)} files, {size(p.stat().st_size)}", flush=True)
    summary = {}
    for record, folder in folders.items():
        manifest, table, findings = [], [], []
        for rec, group, name, part in plans:
            if rec != record:
                continue
            rows, found = results[(rec, name)]
            findings += found
            manifest += [dict(archive=name, path=a, bytes=n, sha256=s, subset=t, same_as="") for a, n, s, t in rows]
            table.append((name, len(rows), (folder / name).stat().st_size, GROUPS[(rec, group)]))
        stored = {m["path"]: m for m in manifest}
        all_stored = {**stored}
        for other in folders:                          # a linked copy may be stored in the other record
            if other != record and (WORK / f"manifest-{other}.json").exists():
                all_stored.update(json.loads((WORK / f"manifest-{other}.json").read_text(encoding="utf-8")))
        for p, canon in sorted(same_as.items()):
            if place(p, read)[0] == record:
                c = all_stored.get(canon) or stored.get(canon)
                if c is None:
                    continue
                manifest.append(dict(archive=c["archive"], path=p, bytes=c["bytes"], sha256=c["sha256"], subset=c.get("subset", ""), same_as=canon))
        write(WORK / f"manifest-{record}.json", json.dumps(stored))
        manifest.sort(key=lambda m: m["path"])
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=["archive", "path", "bytes", "sha256", "subset", "same_as"], lineterminator="\n")
        w.writeheader()
        w.writerows(manifest)
        write(folder / "MANIFEST.csv", buf.getvalue())
        rows_md = "\n".join([f"| `{n}` | {k:,} | {size(b)} | {what} |" for n, k, b, what in table])
        archive_table = "| Archive | Files | Size | Contents |\n|---|---:|---:|---|\n" + rows_md
        n_sub = sum(1 for m in manifest if m["subset"] and not m["same_as"])
        subsets = (f"{n_sub} intermediate tables and records are deposited as subsets, limited to the conditions that the paper "
                   "reports (`subset` = yes in `MANIFEST.csv`). Their size and SHA-256 differ from the analysis's input manifest "
                   "(`manuscript/results/meta/input_manifest.csv`), which records the study's complete tables; the analysis "
                   "reproduces its outputs from the subsets with the declared family sizes and generator state in "
                   "`manuscript/analysis/declared/` of the code release." if n_sub else
                   "No file of this part is a subset (`subset` in `MANIFEST.csv`).")
        readme = (fill("data_core_readme.md", CITATION=citation(), CODE_DOI=doi("doi_code"), EXTENDED_DOI=doi("doi_records_extended"),
                       ARCHIVE_TABLE=archive_table, SUBSETS=subsets) if record == "A" else
                  fill("data_extended_readme.md", CITATION=citation(), CODE_DOI=doi("doi_code"), CORE_DOI=doi("doi_records_core"),
                       ARCHIVE_TABLE=archive_table, SUBSETS=subsets))
        write(folder / "README.md", readme)
        copy(HERE / "verify_deposit.py", folder / "verify_deposit.py")
        write(folder / "zenodo.json", json.dumps(zenodo(record, h, [t[0] for t in table]), indent=1, ensure_ascii=False) + "\n")
        total = sum(b for _, _, b, _ in table) + sum(p.stat().st_size for p in folder.iterdir() if p.suffix != ".zip")
        summary[record] = dict(files=len(manifest), archives=len(table), bytes=total, findings=findings)
        if total > 50e9:
            raise SystemExit(f"record {record}: {size(total)}, above Zenodo's 50 GB")
    write(WORK / "data.json", json.dumps(dict(summary=summary, dropped=dropped, missing=missing[:200], n_missing=len(missing)), indent=1))
    for record, s in summary.items():
        print(f"record {record}: {s['files']:,} files in {s['archives']} archives, {size(s['bytes'])}; {len(s['findings'])} scan findings")
        for f in s["findings"][:20]:
            print(f"  {f[1]}: {f[0]}: {f[2]}")
    if missing:
        print(f"  {len(missing)} listed inputs or records are missing on disk, e.g. {missing[:5]}")
    build_code_record(h)
    return summary


def build_code_record(h):
    code_zip = REL / "1_code" / f"{CONF['name']}-{CONF['version']}.zip"
    folder = REL / "2_zenodo" / "C_code"
    if not code_zip.exists():
        print("record C: build the code part first")
        return
    reset(folder)
    copy(code_zip, folder / code_zip.name)
    write(folder / "README.md", fill("code_record_readme.md", CITATION=citation(), VERSION=CONF["version"], COMMIT=h["commit"][:12],
                                     DATE=h["date"], ZIP=code_zip.name, DATA_DOI=doi("doi_records_core"),
                                     EXTENDED_DOI=doi("doi_records_extended")))
    write(folder / "zenodo.json", json.dumps(zenodo("C", h, [code_zip.name]), indent=1, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- part 3: the manuscript
CELL_FILES = [  # (Editorial Manager item, file in cell_submission/, source)
    ("Cover letter", "Cover_letter.pdf", "build/letter/cover_letter.pdf"),
    ("Manuscript (text and legends, no figures)", "Manuscript.pdf", "build/submission-text/main-submission-text.pdf"),
    *[(f"Figure {i}", f"Figure_{i}.pdf", f"../figures/v4/main/fig{i}.pdf") for i in range(1, 8)],
    ("Graphical abstract", "Graphical_abstract.tif", "../figures/v4/ga/graphical_abstract.tif"),
    ("Supplemental information: Document S1 (Figures S1–S7, Table S1)", "Document_S1.pdf", "build/si/document_s1.pdf"),
    *[(f"Supplemental information: Table S{i} (Excel)", f"Table_S{i}.xlsx", f"build/supplement/Table_S{i}.xlsx") for i in (2, 3, 4)],
    ("Supplemental information: Data S1 (one .csv per figure)", "Data_S1.zip", "build/supplement/Data_S1.zip"),
    *[(f"Supplemental information: Video S{i}", f"Video_S{i}.mp4", f"../figures/v4/videos/Video_S{i}.mp4") for i in range(1, 5)],
]


def front_matter(led):
    """The plain text of the title, Summary, Highlights, In Brief, keywords and the availability statement."""
    t = {k: tex_text((PAPER / "front" / f"{k}.tex").read_text(encoding="utf-8"), led) for k in
         ("title", "summary", "highlights", "inbrief", "keywords")}
    strip = lambda s, head: re.sub(r"\s+", " ", s.replace(f"{head}.", "", 1)).strip()   # noqa: E731
    return dict(
        title=TITLE, summary=strip(t["summary"], "Summary"), inbrief=strip(t["inbrief"], "In brief"),
        keywords=strip(t["keywords"], "Keywords"),
        highlights=[re.sub(r"\s+", " ", x).strip() for x in t["highlights"].split("\n- ")[1:]],
        resource=unwrap(tex_text((PAPER / "back" / "resource.tex").read_text(encoding="utf-8"), led)))


def tex_check():
    r = subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / "manuscript" / "writing" / "check_tex.py")], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = r.stdout.splitlines()
    return dict(ok=r.returncode == 0, lines=lines, warnings=[x[6:] for x in lines if x.startswith("warn: ")],
                failures=[x[6:] for x in lines if x.startswith("FAIL: ")],
                count=next((x for x in lines if x.startswith("Cell word count")), ""))


def pdf_pages(p):
    out = run(["pdfinfo", str(p)])
    return int(re.search(r"Pages:\s+(\d+)", out).group(1))


def latex_source(h, out):
    """The LaTeX sources with their figures as one folder that compiles on its own, and its zip."""
    src = reset(out / "latex_source")
    tree = src / "cell-manuscript-latex"
    tracked = [p for p in git("ls-files", "manuscript/paper").splitlines() if p.endswith((".tex", ".sty", ".bst", ".bib"))]
    tracked += [p.relative_to(ROOT).as_posix() for p in (PAPER / "letter").glob("*.tex")]
    for p in sorted(set(tracked)):
        rel = Path(p).relative_to("manuscript/paper").as_posix()
        text = (ROOT / p).read_text(encoding="utf-8")
        if rel in ("legends/figures.tex", "legends/si_figures.tex"):
            text = text.replace("../figures/v4/main/", "figures/main/").replace("../figures/v4/supplement/", "figures/supplement/")
        if "../" in re.sub(r"(?<!\\)%.*", "", text):
            raise SystemExit(f"latex source: {rel} refers outside the folder")
        write(tree / rel, text)
    for i in range(1, 8):
        copy(FIG / "v4" / "main" / f"fig{i}.pdf", tree / "figures" / "main" / f"fig{i}.pdf")
        copy(FIG / "v4" / "supplement" / f"s{i}.pdf", tree / "figures" / "supplement" / f"s{i}.pdf")
    write(tree / "main_text.tex", "\\def\\cellmode{submission}\\def\\cellnofigures{1}\\input{main}\n")
    write(tree / "main_preprint.tex", "\\def\\cellmode{preprint}\\input{main}\n")
    for job, bbl in (("main", "submission/main-submission.bbl"), ("main_text", "submission-text/main-submission-text.bbl"),
                     ("main_preprint", "preprint/main-preprint.bbl"), ("si", "si/document_s1.bbl")):
        copy(PAPER / "build" / bbl, tree / f"{job}.bbl")
    write(tree / "README.txt", fill("latex_readme.md", DATE=h["date"], COMMIT=h["commit"][:12]))
    zip_tree(tree, src / "cell-manuscript-latex.zip", "cell-manuscript-latex")
    # compile it alone, in a copy, and compare the text with the main builds
    test = WORK / "latex_test"
    if test.exists():
        shutil.rmtree(test)
    shutil.copytree(tree, test)
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep) if p and os.path.isdir(p))
    env = {**os.environ, "PATH": path, "MIKTEX_AUTOINSTALL": "0", "FORCE_SOURCE_DATE": "1",
           "SOURCE_DATE_EPOCH": git("log", "-1", "--format=%ct")}
    checks = []
    for doc, ref in (("main_text.tex", "build/submission-text/main-submission-text.pdf"), ("main_preprint.tex", "build/preprint/main-preprint.pdf"),
                     ("si.tex", "build/si/document_s1.pdf"), ("letter/cover_letter.tex", "build/letter/cover_letter.pdf")):
        r = subprocess.run(["latexmk", "-pdfxe", "-interaction=nonstopmode", "-halt-on-error",
                            "-e", "$xelatex=q/xelatex --disable-installer %O %S/", "-e", "$bibtex=q/bibtex --disable-installer %O %S/", doc],
                           cwd=test, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        pdf = test / Path(doc).with_suffix(".pdf").name
        ok = r.returncode == 0 and pdf.exists()
        same = ok and pdf_text(pdf) == pdf_text(PAPER / ref)
        checks.append((doc, ok, same, pdf_pages(pdf) if ok else 0))
    return checks


def pdf_text(p):
    return run(["pdftotext", "-q", str(p), "-"])


def build_paper():
    h, led = head(), ledger()
    for s in ("manuscript/writing/build_table_s1.py", "manuscript/writing/build_tables_s2_s4.py", "manuscript/writing/build_data_s1.py"):
        py(s)
    out_build = py("manuscript/paper/build.py", "all")
    if "FAILED" in out_build:
        raise SystemExit(out_build)
    chk = tex_check()
    if chk["failures"]:
        raise SystemExit("check_tex failures:\n  " + "\n  ".join(chk["failures"]))
    out = reset(REL / "3_manuscript")
    cell = out / "cell_submission"
    table = []
    for item, name, src in CELL_FILES:
        p = copy((PAPER / src).resolve(), cell / name)
        n = pdf_pages(p) if p.suffix == ".pdf" else 0
        extra = f"{n} page{'s' if n > 1 else ''}" if n else ""
        table.append((item, name, size(p.stat().st_size), extra))
    copy(PAPER / "build" / "submission" / "main-submission.pdf", cell / "optional" / "Manuscript_with_figures.pdf")
    body = (PAPER / "letter" / "cover_letter.tex").read_text(encoding="utf-8").split("\\begin{document}")[1].split("\\end{document}")[0]
    letter = tex_text(body.replace("\\today", h["date"]).replace("\\\\\n", " ⏎ "), led)   # forced line breaks kept
    letter = unwrap(letter.replace("``", "“").replace("''", "”")).replace(" ⏎ ", "\n").replace("⏎", "")
    write(cell / "Cover_letter.txt", letter + "\n")
    fm = front_matter(led)
    words = lambda s: len(check_tex.words(s))   # noqa: E731
    form = [f"# Text for the *Cell* submission form", "", f"**Title.** {fm['title']}", "",
            "**Article type.** Article", "",
            CONF["author_form"], "",
            f"**Summary** ({words(fm['summary'])} words; at most 150).", "", fm["summary"], "",
            "**Highlights** (each at most 85 characters with spaces).", "",
            *[f"- {x} ({len(x)})" for x in fm["highlights"]], "",
            f"**In Brief** ({words(fm['inbrief'])} words; at most 50).", "", fm["inbrief"], "",
            f"**Keywords.** {fm['keywords']}", "",
            f"**Length.** {chk['count'] or 'see check_tex.py'}. Seven main figures; no tables in the main text.", "",
            "**Supplemental information.** Document S1 (Figures S1–S7 and Table S1), Tables S2–S4 (Excel), Data S1 (.csv, "
            "one table per figure), Videos S1–S4 (.mp4).", "",
            "**Resource availability, as in the manuscript.**", "", fm["resource"], "",
            "**Competing interests.** " + CONF["competing_interests"], "",
            "**Funding.** " + CONF["funding_statement"], "", "**AI declaration.** " + CONF["ai_statement"], "",
            "**Suggested and opposed reviewers.** [optional; the form asks for them]", "",
            f"**Preprint.** bioRxiv, {doi('doi_biorxiv', 'DOI once posted')}", ""]
    write(cell / "Submission_form.md", "\n".join(form))
    # bioRxiv
    bx = out / "biorxiv"
    pre = copy(PAPER / "build" / "preprint" / "main-preprint.pdf", bx / "Preprint.pdf")
    for item, name, src in CELL_FILES:
        if item.startswith("Supplemental"):
            copy(cell / name, bx / "supplementary" / name)
    write(bx / "bioRxiv_form.md", "\n".join([
        "# Text for the bioRxiv form", "", f"**Title.** {fm['title']}", "",
        CONF["author_form"], "",
        "**Abstract** (the Summary).", "", fm["summary"], "",
        "**Subject area.** Neuroscience", "", "**License.** " + CONF["preprint_license"], "",
        "**Funding.** " + CONF["funding_statement"], "", "**Competing interests.** " + CONF["competing_interests"], "", "**AI declaration.** " + CONF["ai_statement"], "",
        "**Intended journal.** Cell.", ""]))
    write(bx / "README.md", fill("biorxiv_readme.md", DATE=h["date"], COMMIT=h["commit"][:12], PAGES=pdf_pages(pre),
                                 SIZE=size(pre.stat().st_size)))
    latex = latex_source(h, out)
    tofill = [w for w in chk["warnings"] if "fill" in w.lower() or "blank" in w.lower() or "draft" in w.lower()]
    ppi = min_ppi([FIG / "v4" / "main" / f"fig{i}.pdf" for i in range(1, 8)])
    files_md = "| Item | File | Size | |\n|---|---|---:|---|\n" + "\n".join(f"| {a} | `{b}` | {c} | {d} |" for a, b, c, d in table)
    checks_md = "\n".join([f"- `check_tex.py`: {len(chk['failures'])} failures, {len(chk['warnings'])} warnings; {chk['count']}",
                           *[f"- LaTeX source alone, `{d}`: {'compiles' if ok else 'FAILS'}"
                             + (f", {n} page{'s' if n > 1 else ''}, text identical to the main build" if same else
                                (", text differs" if ok else ""))
                             for d, ok, same, n in latex]])
    write(cell / "README.md", fill("cell_readme.md", DATE=h["date"], COMMIT=h["commit"][:12], FILE_TABLE=files_md, MIN_PPI=ppi,
                                   TOFILL="\n".join(f"- {w}" for w in tofill) or "- none", CHECKS=checks_md))
    report = dict(tex=chk, latex=latex, cell=table, tofill=tofill, min_ppi=ppi)
    write(WORK / "paper.json", json.dumps(report, indent=1, ensure_ascii=False))
    print(f"paper: {len(table)} Cell files, preprint {pdf_pages(pre)} pages; check_tex {len(chk['failures'])} failures, "
          f"{len(chk['warnings'])} warnings; LaTeX source alone: {sum(1 for x in latex if x[1] and x[2])}/{len(latex)} identical")
    return report


def min_ppi(pdfs):
    """The lowest resolution at print size of the embedded images (rendered or photographic content). Images under 16 px a
    side are data drawn as pixel blocks (color-bar gradients, small matrices), sharp at any resolution, and are excepted."""
    low = None
    for p in pdfs:
        for line in run(["pdfimages", "-list", str(p)]).splitlines()[2:]:
            f = line.split()
            if len(f) > 13 and f[2] == "image" and min(int(f[3]), int(f[4])) >= 16:
                v = min(int(f[12]), int(f[13]))
                low = v if low is None else min(low, v)
    return low


# ---------------------------------------------------------------- part 4: figures and tables
def build_figures():
    from PIL import Image                      # the launcher's environment (flybrain's packages)
    out = reset(REL / "4_figures_and_tables")
    titles = __import__("build_data_s1").titles()
    rows = []
    tmp = WORK / "tiff"
    tmp.mkdir(parents=True, exist_ok=True)
    for kind, folder, prefix, n in (("main", "main", "fig", "{}"), ("supplemental", "supplement", "s", "S{}")):
        for i in range(1, 8):
            num = n.format(i)
            for ext in ("pdf", "svg", "jpg"):
                copy(FIG / "v4" / folder / f"{prefix}{i}.{ext}", out / kind / f"Figure_{num}.{ext}")
            pdf = out / kind / f"Figure_{num}.pdf"
            info = run(["pdfinfo", str(pdf)])
            w, hgt = (float(x) for x in re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).groups())
            files = "PDF, SVG, JPG"
            if kind == "main":
                run(["pdftoppm", "-r", "600", "-png", "-singlefile", str(pdf), str(tmp / f"fig{i}")])
                with Image.open(tmp / f"fig{i}.png") as im:
                    im.convert("RGB").save(out / kind / f"Figure_{num}.tif", compression="tiff_lzw", dpi=(600, 600))
                files += ", TIFF (600 ppi)"
            rows.append((f"Figure {num}", titles[num], f"{w / 72 * 25.4:.0f} × {hgt / 72 * 25.4:.0f} mm", files, min_ppi([pdf])))
    for ext in ("tif", "pdf", "jpg", "png"):
        copy(FIG / "v4" / "ga" / f"graphical_abstract.{ext}", out / "graphical_abstract" / f"Graphical_abstract.{ext}")
    for i in range(1, 5):
        copy(SUPP / f"Table_S{i}.xlsx", out / "tables" / f"Table_S{i}.xlsx")
    copy(SUPP / "Data_S1.zip", out / "source_data" / "Data_S1.zip")
    for p in sorted((SUPP / "Data_S1").iterdir()):
        copy(p, out / "source_data" / "Data_S1" / p.name)
    for i in range(1, 5):
        copy(FIG / "v4" / "videos" / f"Video_S{i}.mp4", out / "videos" / f"Video_S{i}.mp4")
        copy(FIG / "v4" / "videos" / f"Video_S{i}_poster.png", out / "videos" / f"Video_S{i}_poster.png")
    vids = [video(out / "videos" / f"Video_S{i}.mp4") for i in range(1, 5)]
    with Image.open(out / "graphical_abstract" / "Graphical_abstract.tif") as im:
        ga = dict(size=im.size, mode=im.mode, dpi=tuple(round(float(x)) for x in im.info.get("dpi", (0, 0))))
    idx = ["# Figures and tables of \"" + TITLE + "\"", "", "## Figures", "",
           "| Figure | Title | Print size | Files | Lowest image resolution (ppi) |", "|---|---|---|---|---:|",
           *[f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows], "",
           f"Graphical abstract: `graphical_abstract/`, TIFF {ga['size'][0]} × {ga['size'][1]} px at {ga['dpi'][0]} dpi ({ga['mode']}), "
           "with PDF, JPG and PNG copies.", "",
           "## Tables and data", "", "- `tables/Table_S1.xlsx`: Table S1, the credibility screen of fly pathologies (also in "
           "Document S1)", "- `tables/Table_S2.xlsx` to `Table_S4.xlsx`: every declared test, the run registry and the "
           "single-cell panels", "- `source_data/Data_S1.zip` (and unpacked in `source_data/Data_S1/`): the values behind every "
           "panel, one table per figure", "", "## Videos", "", "| Video | Duration | Frame | Codec | Size |", "|---|---:|---|---|---:|",
           *[f"| Video S{i} | {v['duration']:.1f} s | {v['width']} × {v['height']} | {v['codec']} ({v['pix_fmt']}) | {size(v['bytes'])} |"
             for i, v in enumerate(vids, 1)], ""]
    write(out / "INDEX.md", "\n".join(idx))
    report = dict(figures=rows, ga=ga, videos=vids)
    write(WORK / "figures.json", json.dumps(report, indent=1, ensure_ascii=False))
    print(f"figures: {len(rows)} figures, graphical abstract {ga}, {len(vids)} videos")
    return report


def video(p):
    j = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,width,height,pix_fmt:format=duration", "-of", "json", str(p)]))
    s = j["streams"][0]
    return dict(codec=s["codec_name"], width=s["width"], height=s["height"], pix_fmt=s["pix_fmt"],
                duration=float(j["format"]["duration"]), bytes=p.stat().st_size)


# ---------------------------------------------------------------- part 5: checks and the index
def build_check():
    h = head()
    load = lambda n: json.loads((WORK / n).read_text(encoding="utf-8")) if (WORK / n).exists() else None   # noqa: E731
    code, data, paper, figs = load("code.json"), load("data.json"), load("paper.json"), load("figures.json")
    lines, tofill = [], []
    if h["dirty"]:
        lines.append(f"- **Uncommitted changes** ({len(h['dirty'])}) are not in the code release; commit and rebuild `code`.")
    if code:
        cred = [f for f in code["findings"] if f[1] == "credential"]
        lines.append(f"- Code release: {code['files']} files from commit `{code['commit'][:12]}`; scan: {len(cred)} credentials, "
                     f"{len(code['findings']) - len(cred)} connection details" + (" (listed in `_work/code.json`)" if code["findings"] else ""))
    if data:
        for r, s in data["summary"].items():
            lines.append(f"- Zenodo record {r}: {s['files']:,} files in {s['archives']} archives, {size(s['bytes'])} (limit 50 GB); "
                         f"scan: {len(s['findings'])} findings")
        lines.append(f"- Left out of the deposit: {data['dropped']}; listed but missing on disk: {data['n_missing']}")
    if paper:
        t = paper["tex"]
        lines.append(f"- `check_tex.py`: {len(t['failures'])} failures, {len(t['warnings'])} warnings; {t['count']}")
        lines.append(f"- Main figures: embedded images at {paper['min_ppi']} ppi or more at print size (Cell asks for 300)")
        si = REL / "3_manuscript" / "cell_submission" / "Document_S1.pdf"
        if si.exists():
            lines.append(f"- Document S1: {size(si.stat().st_size)} (limit 50 MB)")
        lines += [f"- LaTeX source alone, `{d}`: " + ("compiles, text identical to the main build" if ok and same else
                  "compiles, text DIFFERS" if ok else "FAILS") for d, ok, same, _ in paper["latex"]]
        tofill = paper["tofill"]
        fm = front_matter(ledger())
        long = [x for x in fm["highlights"] if len(x) > 85]
        lines.append(f"- Front matter: Summary {len(check_tex.words(fm['summary']))} words (≤150), In Brief "
                     f"{len(check_tex.words(fm['inbrief']))} words (≤50), Highlights {len(fm['highlights'])}, longest "
                     f"{max(len(x) for x in fm['highlights'])} characters (≤85)" + (f"; TOO LONG: {long}" if long else ""))
        bad = [m.group(0) for m in DEV_LABELS.finditer(" ".join([fm["summary"], fm["inbrief"], *fm["highlights"]]))]
        if bad:
            lines.append(f"- Development labels in the front matter: {bad}")
    if figs:
        g = figs["ga"]
        lines.append(f"- Graphical abstract: {g['size'][0]} × {g['size'][1]} px, {g['dpi'][0]} dpi, {g['mode']} (Cell: 1,200 px square at 300 dpi)")
        lines += [f"- Video S{i}: {v['codec']}, {v['pix_fmt']}, {v['width']} × {v['height']}, {v['duration']:.1f} s, {size(v['bytes'])}"
                  for i, v in enumerate(figs["videos"], 1)]
    code_dir = f"{CONF['name']}-{CONF['version']}"
    zips = {r: sum(p.stat().st_size for p in (REL / "2_zenodo" / f).glob("*.zip")) if (REL / "2_zenodo" / f).exists() else 0
            for r, f in (("A", "A_records_core"), ("B", "B_records_extended"), ("C", "C_code"))}
    write(REL / "README.md", fill("release_index.md", DATE=h["date"], COMMIT=h["commit"][:12], CODE_DIR=code_dir, CODE_ZIP=f"{code_dir}.zip",
                                  CODE_FILES=code["files"] if code else "?", CORE_SIZE=size(zips["A"]), EXT_SIZE=size(zips["B"]),
                                  CODE_SIZE=size(zips["C"]), TOFILL="\n".join(f"- {w}" for w in tofill) or "- none",
                                  CHECKS="\n".join(lines)))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", choices=["code", "data", "paper", "figures", "check", []], default=[])
    ap.add_argument("--workers", type=int, default=3, help="processes writing the deposit's archives")
    a = ap.parse_args()
    parts = a.parts or ["code", "paper", "figures", "data", "check"]
    WORK.mkdir(parents=True, exist_ok=True)
    for part in ("code", "paper", "figures", "data", "check"):
        if part in parts:
            if part == "data":
                build_data(a.workers)
            else:
                globals()[f"build_{part}"]()


if __name__ == "__main__":
    main()
