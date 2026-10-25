r"""Builds the manuscript's PDFs from manuscript/paper with XeLaTeX (latexmk; BibTeX with cellnum.bst):

  submission       build/submission/main-submission.pdf            double spaced, line numbers, legends then figures at
                                                                    the end (one PDF of the whole manuscript)
  submission-text  build/submission-text/main-submission-text.pdf  the same without the figure pages: the manuscript file
                                                                    Cell takes, with each figure uploaded as its own file
  preprint         build/preprint/main-preprint.pdf                the same text, each figure with its legend after its
                                                                    Results section (bioRxiv)
  si               build/si/document_s1.pdf                        Document S1: Figures S1-S7 and Table S1 (the guide's form)
  letter           build/letter/cover_letter.pdf                   the cover letter (letter/cover_letter.tex)

  .\MIGRATION\python\python.exe -B -X utf8 manuscript\paper\build.py [submission|submission-text|preprint|si|letter|all] [--draft]

--draft leaves the figure pages out (build/<mode>-draft/), a light PDF for the review gates.

Every TeX program runs with --disable-installer: this MiKTeX installs missing packages on the fly, so a missing file must
fail the build instead of starting a download. SOURCE_DATE_EPOCH (the last commit's time) and FORCE_SOURCE_DATE make the PDFs
reproducible. The values are regenerated from the ledger first, and the log is scanned for undefined references or
citations, missing glyphs and overfull lines; the run fails on the first three.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# mode: (source, job, what the driver defines before it inputs the source)
MODES = {"submission": ("main", "main-submission", r"\def\cellmode{submission}"),
         "submission-text": ("main", "main-submission-text", r"\def\cellmode{submission}\def\cellnofigures{1}"),
         "preprint": ("main", "main-preprint", r"\def\cellmode{preprint}"),
         "si": ("si", "document_s1", r"\def\cellmode{si}"),
         "letter": ("letter/cover_letter", "cover_letter", "")}
OVERFULL_PT = 2.0


def epoch():
    out = subprocess.run(["git", "log", "-1", "--format=%ct"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return out or "0"


def scan(log: Path):
    text = log.read_text(encoding="utf-8", errors="replace")
    undefined = re.findall(r"LaTeX Warning: (?:Reference|Citation) `([^']*)'", text)
    undefined += re.findall(r"Package natbib Warning: Citation `([^']*)'", text)
    missing = re.findall(r"Missing character: There is no (.+?) in font", text)
    overfull = [float(x) for x in re.findall(r"Overfull \\hbox \((\d+(?:\.\d+)?)pt too wide\)", text)]
    errors = re.findall(r"^! (.*)$", text, flags=re.M)
    return dict(undefined=sorted(set(undefined)), missing=sorted(set(missing)), errors=errors,
                overfull=[x for x in overfull if x > OVERFULL_PT])


def build(mode, draft=False):
    src, job, prelude = MODES[mode]
    if draft:
        job += "-draft"
    out = HERE / "build" / (mode + ("-draft" if draft else ""))
    out.mkdir(parents=True, exist_ok=True)
    driver = out / f"{job}.tex"
    driver.write_text(("\\def\\celldraft{1}" if draft else "") + prelude + f"\\input{{{src}}}\n", encoding="utf-8", newline="\n")
    # MiKTeX's latexmk wrapper stops on a PATH entry that is not a folder (here a stray DLL path); drop such entries for the
    # build's own processes only
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep) if p and os.path.isdir(p))
    env = {**os.environ, "PATH": path, "SOURCE_DATE_EPOCH": epoch(), "FORCE_SOURCE_DATE": "1",
           "MIKTEX_AUTOINSTALL": "0",           # a second guard for programs without --disable-installer (xdvipdfmx)
           # BibTeX runs inside the output folder: point it at the style and the references here (the trailing separator
           # keeps the default search path)
           "BIBINPUTS": str(HERE) + os.pathsep, "BSTINPUTS": str(HERE) + os.pathsep}
    cmd = ["latexmk", "-pdfxe", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", f"-outdir={out}",
           "-e", "$xelatex=q/xelatex --disable-installer %O %S/",
           "-e", "$bibtex=q/bibtex --disable-installer %O %S/",
           str(driver.relative_to(HERE))]
    r = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = out / f"{job}.log"
    s = scan(log) if log.exists() else dict(undefined=[], missing=[], errors=["no log"], overfull=[])
    pdf = out / f"{job}.pdf"
    ok = r.returncode == 0 and pdf.exists() and not (s["undefined"] or s["missing"] or s["errors"])
    print(f"{mode}: {'ok' if ok else 'FAILED'} -> {pdf.relative_to(ROOT) if pdf.exists() else 'no PDF'}"
          + (f"; {len(s['overfull'])} overfull line(s) > {OVERFULL_PT:g} pt" if s["overfull"] else ""), flush=True)
    for key in ("errors", "undefined", "missing"):
        for x in s[key][:20]:
            print(f"  {key}: {x}")
    if r.returncode != 0 and not s["errors"]:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="all", choices=[*MODES, "all"])
    ap.add_argument("--draft", action="store_true", help="leave the figure pages out (a light PDF for review)")
    a = ap.parse_args()
    subprocess.run([sys.executable, "-B", "-X", "utf8", "-I", str(ROOT / "manuscript" / "writing" / "build_macros.py")], check=True)
    modes = list(MODES) if a.mode == "all" else [a.mode]
    results = [build(m, a.draft) for m in modes]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
