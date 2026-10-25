"""Checks a Cell manuscript draft against manuscript/CELL_VOICE_GUIDE.md: counts, dropped and rationed words, American
spelling, internal labels, sentence length, "Here, we" placement, and every number against the number ledger. Exit code 1
when a check fails; warnings are printed but do not fail.

  python -I manuscript/writing/check_text.py <draft.md> [--ledger manuscript/writing/number_ledger.csv]

The draft is Markdown: a level-1 title line, then level-2 parts named Summary, Highlights, In Brief, Introduction, Results,
Discussion, Limitations of the study and Figure legends (any subset), with level-3 headings inside Results.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from lib.names import DEV_LABELS, INTERNAL, british  # noqa: E402  (the figure QC's text rules)

_VERBS = ("delve", "underscore", "unlock", "unveil", "harness", "leverage")
DROPPED = ([v + s for v in _VERBS for s in ("", "s", "d", "ed", "es", "ing")]
           + ["delving", "underscoring", "leveraging", "pivotal", "crucial", "intricate", "intricacies", "tapestry", "landscape", "realm",
              "multifaceted", "paramount", "holistic", "synergy", "testament", "game-changing", "groundbreaking", "cutting-edge",
              "seamless", "shed light", "sheds light", "shedding light", "pave the way", "paves the way", "paving the way",
              "plays a key role", "plays a crucial role", "plays a pivotal role", "it is worth noting", "it is important to note",
              "a wide range of", "a myriad of", "in conclusion", "not only", "novel", "for the first time", "first to"])
RATIONED = {"notably": 2, "interestingly": 2, "importantly": 2, "moreover": 2, "furthermore": 2, "taken together": 2, "collectively": 2,
            "remarkably": 1, "strikingly": 1}
HEDGES = ["may", "might", "could", "suggest", "suggests", "suggesting", "likely", "possibly", "potentially", "appear", "appears"]
LIMITS = dict(summary_words=150, highlight_chars=85, highlights=(3, 4), in_brief_words=50, title_words=15, heading_words=12,
              sentence_words=45, hedges_per_1000=10.0)
LEDGER_PARTS = ("Summary", "Highlights", "In Brief", "Results")
# numbers that are references, not results: figure and panel call-outs, supplemental items, publication years in citations
REFERENCE = re.compile(r"(Figures?|Tables?|Fig\.)\s+S?\d+[A-Z]?(?:[–-]S?\d*[A-Z]?)?(?:,?\s*(?:and\s+)?S?\d+[A-Z]?)*|\bS\d+\b"
                       r"|(?:et al\.,?|,)\s*(?:19|20)\d{2}[a-z]?(?=[);])")
NUMBER = re.compile(r"(?<![\w.])[−-]?\d[\d,]*(?:\.\d+)?%?(?![\w])")


def parse(text: str) -> dict:
    out = dict(title="", parts={}, headings=[])
    part = None
    for line in text.splitlines():
        if line.startswith("# "):
            out["title"] = line[2:].strip()
        elif line.startswith("## "):
            part = line[3:].strip()
            out["parts"][part] = ""
        elif line.startswith("### "):
            if part == "Results":
                out["headings"].append(line[4:].strip())
        elif part is not None:
            out["parts"][part] += line + "\n"
    return out


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9α-ωΑ-Ω⁺′'’-]+", re.sub(r"[*_`#>]", " ", text))


def sentences(text: str) -> list[str]:
    """Sentences of a part; each bullet and each paragraph is its own unit, so unpunctuated bullets (Highlights) never merge."""
    chunks, cur = [], []
    for ln in text.splitlines():
        if re.match(r"^\s*[-*]\s+", ln) or not ln.strip():
            if cur:
                chunks.append(" ".join(cur))
                cur = []
            if ln.strip():
                chunks.append(re.sub(r"^\s*[-*]\s+", "", ln))
        else:
            cur.append(ln.strip())
    if cur:
        chunks.append(" ".join(cur))
    out = []
    for c in chunks:
        flat = re.sub(r"\s+", " ", c).strip()
        out += [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[*])", flat) if words(s)]
    return out


def acronyms(text: str) -> list[str]:
    """Tokens that read as acronyms or abbreviations: two or more capitals (CNS, SK) or letters mixed with digits (Kv4)."""
    found = []
    for tok in re.findall(r"[A-Za-zα-ωΑ-Ω0-9⁺]+", text):
        if sum(c.isupper() for c in tok) >= 2 or (re.search(r"\d", tok) and re.search(r"[A-Za-zα-ωΑ-Ω]", tok)):
            found.append(tok)
    return found


def bullets(text: str) -> list[str]:
    return [re.sub(r"^\s*[-*]\s+", "", ln).strip() for ln in text.splitlines() if re.match(r"^\s*[-*]\s+", ln)]


def count_phrase(text: str, phrase: str) -> int:
    return len(re.findall(r"\b" + re.escape(phrase) + r"\b", text, flags=re.IGNORECASE))


def ledger_values(path: Path) -> list[float]:
    with open(path, encoding="utf-8") as f:
        out = []
        for row in csv.DictReader(f):
            try:
                out.append(float(str(row["value"]).replace(",", "").replace("−", "-")))
            except ValueError:
                continue
        return out


def numbers(text: str) -> list[str]:
    return NUMBER.findall(REFERENCE.sub(" ", text))


def in_ledger(token: str, values: list[float]) -> bool:
    t = token.replace(",", "").replace("−", "-")
    pct = t.endswith("%")
    t = t.rstrip("%")
    decimals = len(t.split(".")[1]) if "." in t else 0
    x = float(t)
    for v in values:
        for cand in ((v, v * 100) if pct else (v,)):
            if round(cand, decimals) == x or (decimals == 0 and round(cand) == x):
                return True
    return False


def check(text: str, ledger: list[float] | None = None) -> tuple[list[str], list[str]]:
    d = parse(text)
    parts = d["parts"]
    fails, warns = [], []
    if d["title"] and len(words(d["title"])) > LIMITS["title_words"]:
        fails.append(f"title: {len(words(d['title']))} words (> {LIMITS['title_words']})")
    if "Summary" in parts:
        n = len(words(parts["Summary"]))
        if n > LIMITS["summary_words"]:
            fails.append(f"Summary: {n} words (> {LIMITS['summary_words']})")
        if acronyms(parts["Summary"]):
            fails.append(f"Summary: acronyms or abbreviations {sorted(set(acronyms(parts['Summary'])))}")
    if "Highlights" in parts:
        hl = bullets(parts["Highlights"])
        lo, hi = LIMITS["highlights"]
        if not lo <= len(hl) <= hi:
            fails.append(f"Highlights: {len(hl)} bullets (expected {lo}-{hi})")
        for h in hl:
            if len(h) > LIMITS["highlight_chars"]:
                fails.append(f"Highlight over {LIMITS['highlight_chars']} characters ({len(h)}): {h}")
            if acronyms(h):
                fails.append(f"Highlight with acronyms or abbreviations {acronyms(h)}: {h}")
    if "In Brief" in parts:
        n = len(words(parts["In Brief"]))
        if n > LIMITS["in_brief_words"]:
            fails.append(f"In Brief: {n} words (> {LIMITS['in_brief_words']})")
        if re.search(r"\b(we|our|us)\b", parts["In Brief"], flags=re.IGNORECASE):
            fails.append("In Brief: first person (write it in the third person)")
    for h in d["headings"]:
        if len(words(h)) > LIMITS["heading_words"]:
            fails.append(f"Results heading over {LIMITS['heading_words']} words: {h}")
    body = "\n".join([d["title"], *d["headings"], *parts.values()])
    for p in DROPPED:
        if count_phrase(body, p):
            fails.append(f"dropped word or phrase: {p!r} ({count_phrase(body, p)}x)")
    for p, cap in RATIONED.items():
        n = count_phrase(body, p)
        if n > cap:
            warns.append(f"rationed word {p!r}: {n} (cap {cap})")
    for word, us in british(body):
        fails.append(f"British spelling {word!r} (American: {us})")
    for name, rx in (("internal identifier", INTERNAL), ("development label", DEV_LABELS)):
        for m in rx.finditer(re.sub(r"\$[^$]*\$", "", body)):
            fails.append(f"{name} {m.group(0)!r}")
    for name, txt in parts.items():
        for s in sentences(txt):
            if len(words(s)) > LIMITS["sentence_words"]:
                fails.append(f"{name}: sentence of {len(words(s))} words: {s[:80]}…")
    if "Results" in parts:
        w = max(1, len(words(parts["Results"])))
        rate = 1000 * sum(count_phrase(parts["Results"], h) for h in HEDGES) / w
        if rate > LIMITS["hedges_per_1000"]:
            warns.append(f"Results: {rate:.1f} hedges per 1,000 words (Cell range 0.6-10)")
    for name in ("Summary", "Introduction"):
        if name in parts and count_phrase(parts[name], "Here, we") != 1:
            warns.append(f"{name}: 'Here, we' appears {count_phrase(parts[name], 'Here, we')} times (expected once)")
    if ledger is not None:
        for name in LEDGER_PARTS:
            if name in parts:
                for tok in numbers(parts[name]):
                    if not in_ledger(tok, ledger):
                        fails.append(f"{name}: number {tok} is not in the ledger")
    return fails, warns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("draft")
    ap.add_argument("--ledger")
    a = ap.parse_args()
    text = Path(a.draft).read_text(encoding="utf-8")
    fails, warns = check(text, ledger_values(Path(a.ledger)) if a.ledger else None)
    for w in warns:
        print("warn:", w)
    for f in fails:
        print("FAIL:", f)
    print(f"{len(fails)} failure(s), {len(warns)} warning(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
