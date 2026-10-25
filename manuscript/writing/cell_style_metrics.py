"""Style metrics of a corpus of Cell research articles (PubMed metadata and PMC full texts saved by the PubMed tools), for
manuscript/CELL_STYLE_ANALYSIS.md. The JSON it writes holds derived numbers only; the printout shows headings for checking.
Usage: python -I cell_style_metrics.py <out.json> <metadata.json> [<metadata.json> ...] -- <fulltext.json> [...]"""
import json
import re
import statistics as st
import sys

CORPUS = ["42691995", "42691997", "42691996", "40460825", "40967216", "38729112", "37607537", "37192620", "36563666", "38306983",
          "40695276", "41713415", "37774677", "42777707", "42361798"]
INTENSIFIERS = ["notably", "interestingly", "importantly", "remarkably", "strikingly", "surprisingly", "crucial", "critical", "key", "pivotal",
                "intricate", "delve", "underscore", "landscape", "novel", "robust", "comprehensive", "furthermore", "moreover", "taken together",
                "collectively", "thus", "together"]
HEDGES = ["may", "might", "could", "suggest", "suggests", "suggesting", "likely", "possibly", "potentially", "appear", "appears"]
LEVEL1 = {"introduction", "results", "discussion", "limitations of the study", "limitation of the study", "star★methods", "star methods",
          "resource availability", "method details", "quantification and statistical analysis"}


def sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text)
    return [p for p in parts if len(p.split()) >= 3]


def words(s):
    return len(s.split())


def summary_roles(abstract):
    ab = re.sub(r"^SUMMARY\s*", "", abstract.strip())
    ss = sentences(ab)
    roles = []
    for i, s in enumerate(ss):
        low = s.lower()
        if i == 0:
            r = "question" if s.endswith("?") else "context"
        elif low.startswith("however") or " remain" in low or "unknown" in low or "unclear" in low or "elusive" in low or "poorly understood" in low:
            r = "gap"
        elif low.startswith("here") or (low.startswith("we ") and i <= 3) or " we present" in low or "here, we" in low:
            r = "approach"
        elif i == len(ss) - 1:
            r = "implication"
        else:
            r = "finding"
        roles.append(r)
    return ss, roles


def sections(full):
    paras = [p.strip() for p in full.split("\n\n") if p.strip()]
    out, cur = [], None
    for p in paras:
        is_head = len(p) < 160 and not p.endswith(".") and len(p.split()) >= 2 and not p.lower().startswith("figure")
        if (p.isupper() and len(p) < 60) or p.lower() in LEVEL1:
            cur = dict(level=1, title=p.upper(), paras=[])
            out.append(cur)
        elif is_head and cur is not None and cur["title"] in ("RESULTS",) or (is_head and cur is not None and cur.get("parent") == "RESULTS"):
            cur = dict(level=2, title=p, paras=[], parent="RESULTS")
            out.append(cur)
        elif cur is not None:
            cur["paras"].append(p)
    return out


def analyse_full(a):
    full = a["full_text"]
    sec = sections(full)
    get = lambda name: [s for s in sec if s["title"].upper().startswith(name)]  # noqa: E731
    intro = get("INTRODUCTION")
    intro_p = intro[0]["paras"] if intro else []
    res_heads = [s for s in sec if s.get("parent") == "RESULTS" and len(s["paras"]) and sum(words(p) for p in s["paras"]) > 60]
    disc = get("DISCUSSION")
    disc_p = [p for p in (disc[0]["paras"] if disc else []) if words(p) > 25]
    lim = [s for s in sec if s["title"].lower().startswith("limitations")]
    lim_w = sum(words(p) for p in lim[0]["paras"]) if lim else 0
    if not lim and disc:                                      # "Limitations of the study" as a sub-heading inside the Discussion
        txt = "\n\n".join(disc[0]["paras"])
        m = re.search(r"Limitations of the study\s*\n\n(.+?)(\n\n[A-Z ]{6,}\n|\Z)", full, flags=re.S)
        lim_w = words(m.group(1)) if m else 0
    res_text = " ".join(p for s in res_heads for p in s["paras"])
    rs = sentences(res_text)
    lens = [words(s) for s in rs]
    low = res_text.lower()
    n_words = max(1, len(res_text.split()))
    callouts = len(re.findall(r"\(Figures? \d", res_text)) + len(re.findall(r"\(Figure S\d", res_text))
    heads = [s["title"] for s in res_heads]
    return dict(intro_paragraphs=len(intro_p), intro_words=sum(words(p) for p in intro_p),
                intro_last_opens=(sentences(intro_p[-1])[0][:60] if intro_p else ""),
                results_sections=len(heads), results_headings=heads, heading_words=[len(h.split()) for h in heads],
                results_words=n_words, results_paras_per_section=[len(s["paras"]) for s in res_heads],
                sentence_len_mean=round(st.mean(lens), 1) if lens else None, sentence_len_sd=round(st.pstdev(lens), 1) if lens else None,
                sentence_len_p10=sorted(lens)[len(lens) // 10] if lens else None, sentence_len_p90=sorted(lens)[9 * len(lens) // 10] if lens else None,
                we_per_1000=round(1000 * len(re.findall(r"\bwe\b", low)) / n_words, 1),
                passive_per_1000=round(1000 * len(re.findall(r"\b(?:was|were|is|are|been|be)\s+\w+ed\b", low)) / n_words, 1),
                hedges_per_1000=round(1000 * sum(len(re.findall(rf"\b{h}\b", low)) for h in HEDGES) / n_words, 1),
                intensifiers={k: len(re.findall(rf"\b{k}\b", low)) for k in INTENSIFIERS if re.search(rf"\b{k}\b", low)},
                figure_callouts_per_1000=round(1000 * callouts / n_words, 1),
                discussion_paragraphs=len(disc_p), discussion_words=sum(words(p) for p in disc_p),
                discussion_first=(sentences(disc_p[0])[0][:80] if disc_p else ""), limitations_words=lim_w)


def main():
    out = sys.argv[1]
    args = sys.argv[2:]
    cut = args.index("--")
    metas, fulls = args[:cut], args[cut + 1:]
    papers = {}
    for m in metas:
        for a in json.load(open(m, encoding="utf-8"))["articles"]:
            pmid = a["identifiers"]["pmid"]
            if pmid in CORPUS:
                ss, roles = summary_roles(a.get("abstract") or "")
                papers[pmid] = dict(pmid=pmid, doi=a["doi"], year=a["publication_date"].get("year"), title=a["title"],
                                    summary_words=sum(words(s) for s in ss), summary_sentences=len(ss),
                                    summary_sentence_words=[words(s) for s in ss], summary_roles=roles,
                                    summary_last_opens=" ".join(ss[-1].split()[:3]) if ss else "")
    for f in fulls:
        for a in json.load(open(f, encoding="utf-8"))["articles"]:
            pmid = a["identifiers"]["pmid"]
            if pmid in papers:
                papers[pmid]["full"] = analyse_full(a)
                papers[pmid]["pmcid"] = a["identifiers"]["pmcid"]
                papers[pmid]["in_brief_found"] = "In brief" in a["full_text"]
    missing = [p for p in CORPUS if p not in papers]
    stored = []
    for p in CORPUS:                                          # derived numbers only: no verbatim sentences or headings (copyright)
        if p in papers:
            x = {k: v for k, v in papers[p].items() if k != "summary_last_opens"}
            if "full" in x:
                x["full"] = {k: v for k, v in x["full"].items() if k not in ("intro_last_opens", "discussion_first", "results_headings")}
            stored.append(x)
    json.dump(dict(papers=stored, missing=missing, note="Derived numbers only; verbatim sentences and headings are not stored (copyright)."),
              open(out, "w", encoding="utf-8", newline="\n"), indent=1)
    sw = [p["summary_words"] for p in papers.values()]
    sn = [p["summary_sentences"] for p in papers.values()]
    print(f"papers {len(papers)} missing {missing}")
    print(f"summary words {min(sw)}-{max(sw)} median {st.median(sw)}; sentences {min(sn)}-{max(sn)} median {st.median(sn)}")
    for p in papers.values():
        f = p.get("full")
        print(p["pmid"], p["year"], p["title"][:60], "| roles", "-".join(r[0] for r in p["summary_roles"]), "| last:", p["summary_last_opens"])
        if f:
            print("   intro", f["intro_paragraphs"], "paras", f["intro_words"], "w | last opens:", f["intro_last_opens"])
            print("   results", f["results_sections"], "sections", f["results_words"], "w | paras/section", f["results_paras_per_section"],
                  "| sentence", f["sentence_len_mean"], "±", f["sentence_len_sd"], f"(p10 {f['sentence_len_p10']}, p90 {f['sentence_len_p90']})")
            print("   we/1000", f["we_per_1000"], "| passive/1000", f["passive_per_1000"], "| hedges/1000", f["hedges_per_1000"], "| callouts/1000",
                  f["figure_callouts_per_1000"])
            print("   intensifiers", f["intensifiers"])
            print("   discussion", f["discussion_paragraphs"], "paras", f["discussion_words"], "w | first:", f["discussion_first"], "| limitations", f["limitations_words"], "w")
            for h, n in zip(f["results_headings"], f["heading_words"]):
                print(f"      H({n}w): {h[:110]}")


if __name__ == "__main__":
    main()
