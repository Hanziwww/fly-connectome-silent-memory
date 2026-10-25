r"""Builds manuscript/paper/refs.bib from manuscript/paper/citations.csv and the bibliographic records kept in
manuscript/paper/bib/: PubMed metadata (pubmed_records.json, fetched with the PubMed tool) and bioRxiv metadata
(biorxiv_records.json, from the bioRxiv tool). Nothing is fetched here. Each entry keeps its PMID or DOI, and its author list
is written in the Cell form (field authorlist, printed by cellnum.bst): at most 10 names as "Surname, I.J.", ", and" before
the last name, "et al." after the tenth. Journal names get the periods of the Cell form ("Nat. Neurosci."), page ranges are
written in full, missing eLife, PNAS, J. Neurosci. and Science article numbers come from the DOI, taxon names are italic, and
citations.csv may override a year or a title where PubMed's record differs from the published article. A record that PubMed
lists without a DOI is keyed by its PMID, written "pmid:N" in the doi column of citations.csv.

  .\MIGRATION\python\python.exe -B -X utf8 -I manuscript\writing\build_bib.py
  .\MIGRATION\python\python.exe -B -X utf8 -I manuscript\writing\build_bib.py --import FILE [FILE ...]

--import adds PubMed get_article_metadata results (JSON, as the tool returns or saves them) to bib/pubmed_records.json,
keeping only the fields a reference needs (no abstracts or affiliations).
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
BIB = PAPER / "bib"
PUBMED = BIB / "pubmed_records.json"
BIORXIV = BIB / "biorxiv_records.json"
TAXA = ["Drosophila melanogaster", "Caenorhabditis elegans", "Danio rerio", "Mus musculus", "Drosophila", "C. elegans"]
JOURNAL_FORM = {"Proc Natl Acad Sci U S A": "Proc. Natl. Acad. Sci. USA", "Elife": "eLife"}
MAX_NAMES = 10


def slim(a):
    """The fields a reference needs from one PubMed record."""
    ids = a.get("identifiers", {})
    return dict(pmid=ids.get("pmid"), pmc=ids.get("pmc"), doi=(a.get("doi") or ids.get("doi") or "").lower(), title=a.get("title", ""),
                journal=a.get("journal", {}).get("title", ""), iso=a.get("journal", {}).get("iso_abbreviation", ""),
                authors=[{k: au[k] for k in ("last_name", "fore_name", "initials", "collective_name") if au.get(k)}
                         for au in a.get("authors", []) if au.get("last_name") or au.get("collective_name")],
                year=a.get("publication_date", {}).get("year"), volume=a.get("citation", {}).get("volume"),
                pages=a.get("citation", {}).get("pages"))


def import_records(files):
    recs = json.loads(PUBMED.read_text(encoding="utf-8"))["records"] if PUBMED.exists() else {}
    for f in files:
        for a in json.loads(Path(f).read_text(encoding="utf-8"))["articles"]:
            r = slim(a)
            recs[r["doi"] or f"pmid:{r['pmid']}"] = r          # a record PubMed lists without a DOI is keyed by its PMID
    PUBMED.parent.mkdir(parents=True, exist_ok=True)
    PUBMED.write_text(json.dumps(dict(note="PubMed metadata from the PubMed tool (get_article_metadata), slimmed by build_bib.py",
                                      records=dict(sorted(recs.items()))), indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"{len(recs)} PubMed records in {PUBMED.relative_to(PAPER.parents[1])}")


def initials(au):
    fore = (au.get("fore_name") or "").replace(".", " ").strip()
    if not fore:
        return "".join(c + "." for c in au.get("initials", "") if c.isalpha())
    out = []
    for part in fore.split():
        if "-" in part:
            out.append(".-".join(p[0].upper() for p in part.split("-") if p) + ".")
        elif part.isupper() and len(part) <= 3:
            out.append("".join(c + "." for c in part))
        else:
            out.append(part[0].upper() + ".")
    return "".join(out)


def tex_escape(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def names(authors):
    out = []
    for au in authors:
        if au.get("collective_name"):
            out.append(tex_escape(au["collective_name"]))
        else:
            out.append(f"{tex_escape(au['last_name'])}, {initials(au)}")
    return out


def author_list(authors):
    n = names(authors)
    if len(n) > MAX_NAMES:
        return ", ".join(n[:MAX_NAMES]) + ", et~al."
    if len(n) == 1:
        return n[0]
    return ", ".join(n[:-1]) + ", and " + n[-1]


def bibtex_authors(authors):
    out = []
    for au in authors:
        if au.get("collective_name"):
            out.append("{" + tex_escape(au["collective_name"]) + "}")
        else:
            out.append(f"{tex_escape(au['last_name'])}, {tex_escape(au.get('fore_name') or au.get('initials', ''))}")
    return " and ".join(out)


def title(s):
    s = re.sub(r"<i>(.*?)</i>", r"\\textit{\1}", s.strip())
    s = re.sub(r"<sup>(.*?)</sup>", r"\\textsuperscript{\1}", s)
    s = re.sub(r"<sub>(.*?)</sub>", r"\\textsubscript{\1}", s)
    s = re.sub(r"<[^>]+>", "", s).rstrip(".")
    s = tex_escape(s)
    for cmd in ("textit", "textsuperscript", "textsubscript"):
        s = s.replace(r"\textbackslash{}" + cmd, "\\" + cmd)
    rx = r"(?<!\\textit\{)\b(" + "|".join(re.escape(t) for t in TAXA) + r")\b"
    return re.sub(rx, r"\\textit{\1}", s)


def journal(rec):
    iso = rec.get("iso") or rec.get("journal", "")
    if iso in JOURNAL_FORM:
        return JOURNAL_FORM[iso]
    full = set(re.findall(r"[a-z]+", re.sub(r"\s*:.*$", "", rec.get("journal", "")).lower()))
    return " ".join(w if (w.lower() in full or (w.isupper() and len(w) > 1) or not w.isalpha()) else w + "."  # "J" -> "J."
                    for w in iso.split())


def pages(rec):
    p = rec.get("pages") or ""
    doi = rec["doi"]
    if not p and "elife" in doi:
        return "e" + doi.rsplit(".", 1)[-1]
    if not p and doi.startswith("10.1073/pnas."):               # PNAS, J. Neurosci. and Science article numbers from the DOI
        return "e" + doi.split("pnas.", 1)[1]
    if not p and doi.startswith("10.1523/jneurosci."):
        return "e" + re.sub(r"\D", "", doi.split("jneurosci.", 1)[1])
    if not p and re.fullmatch(r"10\.1126/science\.a[a-z]{2}\d+", doi):
        return "e" + doi.split("science.", 1)[1]
    m = re.fullmatch(r"(\d+)-(\d+)(\..*)?", p)
    if m and len(m.group(2)) < len(m.group(1)):                 # 1543-6 -> 1543-1546
        p = m.group(1) + "-" + m.group(1)[: len(m.group(1)) - len(m.group(2))] + m.group(2) + (m.group(3) or "")
    return p


def entry(row, rec, kind):
    year = row.get("year_override") or rec.get("year") or ""
    t = title(row["title_override"]) if row.get("title_override") else title(rec.get("title", ""))
    fields = [("author", bibtex_authors(rec["authors"])), ("authorlist", author_list(rec["authors"])), ("title", t), ("year", year)]
    if kind == "article":
        fields += [("journal", journal(rec)), ("volume", rec.get("volume") or ""), ("pages", pages(rec))]
    else:
        fields += [("howpublished", f"Preprint at {rec.get('server', 'bioRxiv')}")]
    doi = "" if row["doi"].startswith("pmid:") else row["doi"]
    fields += [("doi", doi), ("pmid", rec.get("pmid") or ""), ("pmcid", rec.get("pmc") or "")]
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields if v)
    return f"@{kind}{{{row['key']},\n{body}\n}}\n"


def build():
    pub = json.loads(PUBMED.read_text(encoding="utf-8"))["records"] if PUBMED.exists() else {}
    bio = {r["doi"].lower(): r for r in json.loads(BIORXIV.read_text(encoding="utf-8"))["records"]} if BIORXIV.exists() else {}
    rows = list(csv.DictReader(open(PAPER / "citations.csv", encoding="utf-8")))
    keys = [r["key"] for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"duplicate citation keys: {sorted(k for k in keys if keys.count(k) > 1)}")
    out, missing = [], []
    for r in rows:
        doi = r["doi"].lower()
        if r["source"] == "pubmed" and doi in pub:
            out.append(entry(r, pub[doi], "article"))
        elif r["source"] == "biorxiv" and doi in bio:
            out.append(entry(r, bio[doi], "misc"))
        else:
            missing.append(f"{r['key']} ({r['doi']}, {r['source']})")
    if missing:
        raise SystemExit("no record for: " + "; ".join(missing))
    head = "% Generated by manuscript/writing/build_bib.py from citations.csv and bib/*.json; do not edit.\n\n"
    (PAPER / "refs.bib").write_text(head + "\n".join(out), encoding="utf-8", newline="\n")
    print(f"{len(out)} references -> manuscript/paper/refs.bib")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--import", dest="files", nargs="+")
    a = ap.parse_args()
    if a.files:
        import_records(a.files)
    build()


if __name__ == "__main__":
    sys.exit(main())
