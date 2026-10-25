r"""Builds Table S1 of Document S1, the credibility screen of fly pathologies (Figure 7A), as manuscript/paper/generated/
table_s1.tex: one row per candidate of studies/brain-pathology-v1/credibility-table.csv (the screen's record, grades and
outcomes as scored) with the sources of studies/brain-pathology-v1/literature/candidates.csv (key DOIs). The case codes of
the study are development labels, so each row is written for the reader here (TEXT): the fly condition, the change measured
in flies and the model's operator, from the candidate record; the grades, passes, totals and outcomes are the table's own.
The output is checked against the figures' text rules (internal and development labels, American spelling). The same rows,
in plain text and with the legend of Document S1, go to manuscript/paper/build/supplement/Table_S1.xlsx (xlsx.py), the copy
of the table among the release's original tables.

  .\MIGRATION\python\python.exe -B -X utf8 -I manuscript\writing\build_table_s1.py
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "manuscript" / "figures"))
import check_tex  # noqa: E402
import xlsx  # noqa: E402
from lib.names import DEV_LABELS, INTERNAL, british  # noqa: E402

STUDY = ROOT / "studies" / "brain-pathology-v1"
OUT = ROOT / "manuscript" / "paper" / "generated" / "table_s1.tex"
XLSX = ROOT / "manuscript" / "paper" / "build" / "supplement" / "Table_S1.xlsx"
CA, K = r"Ca\textsuperscript{2+}", r"K\textsuperscript{+}"
# code: (fly condition, change measured in flies, the model's operator, outcome when not simply passed)
TEXT = {
    "C01": ("Aβ42, pan-neuronal", f"Kv4 (A-type {K}) channel degraded; less current needed for the first spike",
            "Threshold lowered in every cell", "Retained; the only case that reaches the visual tasks"),
    "C02": ("Aβ42 in mushroom-body neurons", "Kv4 current lost in Kenyon cells", "Threshold lowered in Kenyon cells", None),
    "C03": ("Aβ42 inside identified neurons", f"Lower release probability, fewer {CA}-channel clusters, less Bruchpilot",
            "Release scaled down", "Deferred (C3): no published magnitude to calibrate"),
    "C04": ("Aβ42, late (25 to 50 days)", "Kenyon cells lost first, then other neurons", "Kenyon cells removed", "Retained"),
    "C05": ("Tau in Kenyon cells, several isoforms", f"Less depolarization-evoked {CA} influx; learning impaired",
            "Threshold raised in Kenyon cells", "Merged with the tau case in Kenyon cells (the same measurement)"),
    "C06": ("Human tau, adult-onset, pan-neuronal", "Slower learning; protein-synthesis-dependent memory impaired, reversibly",
            "None available", "Excluded (C2, C3)"),
    "C07": ("Normal aging (3 vs 30 days)", "Larger active zones and more release at projection neuron to Kenyon cell synapses",
            "Release raised at these synapses", "Excluded (C5): Kenyon-cell responses doubled; in flies they stay flat or fall"),
    "C08": ("Normal aging (10 vs 30 days)", "Fewer contacts from DPM neurons onto α/β Kenyon cells",
            "None: DPM output carries no weight in the model", "Excluded (C3)"),
    "C09": ("α-Synuclein, parkin or PINK1", "Dopaminergic neurons and their synapses lost",
            "None: dopaminergic synapses carry no weight in the model", "Excluded (C3)"),
    "C10": (r"Fragile X (\textit{dfmr1} null), 7 to 9 days", "Less 5-HT2A receptor in the mushroom body",
            "None: no serotonin receptor signaling in the model", "Excluded (C3)"),
    "C11": ("Aβ42, early (0 to 2 days)", "More spontaneous transmission onto projection neurons",
            "Release onto projection neurons scaled up", None),
    "C12": ("Aβ42, 9 to 13 days", "Less spontaneous transmission onto projection neurons",
            "Release onto projection neurons scaled down", None),
    "C13": (r"Fragile X (\textit{dfmr1} null)", "Weaker GABAergic lateral inhibition in the antennal lobe; broader odor tuning",
            "Release of GABAergic antennal-lobe local neurons scaled down", "Retained"),
    "C14": ("Normal aging (1 to 5 vs 25 to 30 days)", r"\textit{KCNQ} (M-current) transcripts fall in the brain",
            "Threshold lowered in α/β Kenyon cells", "Excluded (C5): no change at the circuit level, and no matched assay"),
    "C15": ("Human tau in output neurons M4 and M6", f"Depolarization-evoked {CA} influx up about 2.7-fold",
            "Threshold lowered in these output neurons", "Excluded (C5): the fly's increase lies beyond the operator's range"),
    "C16": ("Human tau or Aβ42 in Kenyon cells", f"Depolarization-evoked {CA} influx down by about half",
            "Threshold raised in Kenyon cells", "Retained for the planned comparison of Aβ42 with tau"),
    "C17": ("Human tau isoforms, giant fiber system", "More response failures at high frequency; slower conduction",
            "Stronger short-term depression at the identified synapse", "Deferred (C3): its magnitude needs the full text"),
    "C18": ("Aβ42 inside a flight motor neuron", "Faster synaptic fatigue at high frequency",
            "Stronger short-term depression; release scaled down", "Deferred (C2, C3): weak mapping, no magnitude"),
    "C19": ("Sleep loss with Aβ-Arctic", "Large ventral lateral clock neurons fire more",
            "Threshold lowered in these clock neurons", "Excluded (C3): the change cannot propagate in the model"),
    "C20": ("α-Synuclein with sleep deprivation", "Persistent loss of short-term memory that requires dopamine receptors",
            "None available", "Excluded (C2, C3)"),
    "C21": ("Normal aging", "Mushroom-body fiber number declines after 3 to 4 weeks", "Kenyon cells removed",
            "Excluded (C1): a single early study, contradicted since"),
    "C22": ("Normal aging (2 vs 60 days)", "Dopaminergic neurons preserved", "No change", "Control: no change to reproduce"),
    "C23": ("Aβ42, tau or α-synuclein in photoreceptors", "Photoreceptors lost (Aβ42)", "Photoreceptors removed",
            "Not tested: confined to the retina, which the visual task input bypasses"),
    "C24": ("Human tau in dopaminergic neurons", "Vesicular dopamine release lost", "None: dopamine release carries no weight",
            "Excluded (C3)"),
}
ORDER = ["C13", "C04", "C01", "C16", "C02", "C11", "C12", "C07", "C15", "C14", "C03", "C17", "C18", "C05", "C22", "C23", "C21",
         "C06", "C20", "C08", "C09", "C10", "C19", "C24"]


def tex(s):
    return s.replace("%", r"\%").replace("&", r"\&")


def plain(s):
    out = re.sub(r"\\(?:textit|emph)\{([^{}]*)\}", r"\1", s.replace(CA, "Ca²⁺").replace(K, "K⁺"))
    if "\\" in out:
        raise SystemExit(f"Table S1 (Excel): TeX left in {out!r}")
    return out


def legend():
    """The table's title and legend as Document S1 prints them (si.tex, \\tableShead), in plain text."""
    si = (ROOT / "manuscript" / "paper" / "si.tex").read_text(encoding="utf-8")
    body, _ = check_tex.group(si, si.index(r"\newcommand{\tableShead}{") + len(r"\newcommand{\tableShead}"))
    text = check_tex.convert(body, check_tex.Doc(), "Supplemental figure legends", {}, False)
    title, rest = [re.sub(r"\s+", " ", t).strip() for t in text.strip().split("\n", 1)]
    return title, rest


def main():
    screen = {r["candidate"]: r for r in csv.DictReader(open(STUDY / "credibility-table.csv", encoding="utf-8"))}
    lit = {r["candidate_id"].split("_")[0]: r for r in csv.DictReader(open(STUDY / "literature" / "candidates.csv", encoding="utf-8"))}
    if set(screen) != set(TEXT) or set(ORDER) != set(TEXT) or len(ORDER) != len(TEXT):
        raise SystemExit("Table S1: the screen's candidates and the table's rows differ")
    rows, sheet = [], []
    for code in ORDER:
        s, (cond, change, operator, outcome) = screen[code], TEXT[code]
        passed = s["c1_c3_pass"] == "True"
        if outcome is None:
            if s["decision"] != "passed, not retained (rank)":
                raise SystemExit(f"Table S1: {code} needs an outcome ({s['decision']})")
            outcome = "Passed; not retained (rank)"
        if s["decision"] == "retained" and not outcome.startswith("Retained"):
            raise SystemExit(f"Table S1: {code} is retained in the screen")

        def yes(v):
            return {"True": "pass", "False": "fail"}.get(v, "--")
        cells = [s["c1"], s["c2"], s["c3"],
                 yes(s["c4_pass"]) if passed else "--",
                 s["c5_grade"] if passed else "--",
                 yes(s["c6_pass"]) if passed else "--",
                 s["total"] if passed else "--"]
        dois = r"\newline ".join(r"\url{" + d.strip() + "}" for d in lit[code]["key_dois"].split(";") if d.strip())
        rows.append(" & ".join([tex(cond), tex(change), tex(operator), *cells, tex(outcome), dois]) + r" \\")
        sheet.append([plain(cond), plain(change), plain(operator), *[int(c) if c.isdigit() else c.replace("--", "–") for c in cells],
                      plain(outcome), "; ".join(d.strip() for d in lit[code]["key_dois"].split(";") if d.strip())])
    head = (r"\textbf{Fly condition} & \textbf{Change measured in flies} & \textbf{Operator in the model} & \textbf{C1} & "
            r"\textbf{C2} & \textbf{C3} & \textbf{C4} & \textbf{C5} & \textbf{C6} & \textbf{Total} & \textbf{Outcome} & "
            r"\textbf{Sources (DOI)} \\")
    body = "\n".join(rows)
    text = body.replace(r"\newline", " ")
    for rx in (INTERNAL, DEV_LABELS):
        hits = [m.group(0) for m in rx.finditer(text.replace(r"\url", " ").replace("_", " "))]
        if hits:
            raise SystemExit(f"Table S1: internal or development labels {hits}")
    for word, us in british(text):
        raise SystemExit(f"Table S1: British spelling {word!r} (American: {us})")
    col = r">{\raggedright\arraybackslash}p"
    lines = [
        "% Generated by manuscript/writing/build_table_s1.py from the credibility screen's record; do not edit.",
        r"\begin{landscape}",
        r"\ifdefined\tableShead\tableShead\fi% the title and legend, defined in si.tex",
        r"{\footnotesize\setlength{\tabcolsep}{2.5pt}\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}\renewcommand{\arraystretch}{1.1}",
        r"\begin{longtable}{@{}" + f"{col}{{27mm}}{col}{{41mm}}{col}{{30mm}}" + "*{6}{>{\\centering\\arraybackslash}p{6.5mm}}"
        + f">{{\\centering\\arraybackslash}}p{{7.5mm}}{col}{{38mm}}{col}{{41mm}}" + "@{}}",
        r"\toprule", head, r"\midrule\endhead", r"\bottomrule\endlastfoot", body, r"\end{longtable}}", r"\end{landscape}", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"{len(rows)} rows -> {OUT.relative_to(ROOT)}")
    title, text = legend()
    for word, us in british(title + " " + text + " " + " ".join(str(c) for r in sheet for c in r)):
        raise SystemExit(f"Table S1 (Excel): British spelling {word!r} (American: {us})")
    header = ["Fly condition", "Change measured in flies", "Operator in the model", "C1", "C2", "C3", "C4", "C5", "C6", "Total",
              "Outcome", "Sources (DOI)"]
    XLSX.parent.mkdir(parents=True, exist_ok=True)
    xlsx.write(XLSX, [xlsx.Sheet("Legend", rows=[[title], [text], ["Dashes (–): not tested."]], widths=[150], wrap=True),
                      xlsx.Sheet("Screen", header=header, rows=sheet, widths=[30, 44, 32, 6, 6, 6, 6, 6, 6, 7, 44, 60])])
    if len(xlsx.read(XLSX)["Screen"]) != len(sheet) + 1:
        raise SystemExit("Table S1 (Excel): the sheet reads back with a different number of rows")
    print(f"{len(sheet)} rows -> {XLSX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
