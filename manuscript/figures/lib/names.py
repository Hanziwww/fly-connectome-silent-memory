"""Text rules shared by the figure QC (lib/qc.py) and the manuscript text check (manuscript/writing/check_text.py): internal
identifiers, development labels and American spelling. Pure regular expressions, no plotting imports."""
import re

INTERNAL = re.compile(r"(remote-dynamic|supp-|difficulty-|head_only|_|\bnan\b)")
# Development labels never shown to a reader under the Cell profile (v4; the user, 2026-10-22; NARRATIVE_V4_OUTLINE.md §12):
# model and scene codes, experiment / family / gate / task / open-question codes, map codes, pathology case and model codes,
# regimes by letter and the word "frozen". Published cell types (L1, T4, H1, C2, LC12, DNp01) do not match; nor do
# references to Figures, Tables, Videos or Data S1-S7.
DEV_LABELS = re.compile(r"(?<!Figure )(?<!Fig\. )(?<!Table )(?<!Video )(?<!Data )\b(?:M[0-2]|S[12])\b|\bX\d{1,2}\b|\bF1\d\b"
                        r"|\bG\d{1,2}[a-z]?\b"
                        r"|\bT-[A-Z]{2,}\d*\b|\bT\d{2}\b|\bD\d(?:-(?:T\d|H))?\b|\bP0\d\b|\bC\d{2}\b|\bO\d{1,2}\b"
                        r"|\b[Rr]egimes? [A-C]\b|\b[Ff]rozen\b")
# American spelling under the Cell profile (Cell copyedits to American English; the user's decision, 2026-10-23). Stems are
# matched at word starts, so "analysed" is caught but the plural noun "analyses" and words such as "exercise" are not; -ise
# verbs whose stems begin American words ("realistic", "specialist", "hypothesis", "emphasis") are listed by their full forms.
BRITISH = {"behaviour": "behavior", "colour": "color", "odour": "odor", "ageing": "aging", "analysed": "analyzed", "analysing": "analyzing",
           "analyse": "analyze", "centre": "center", "grey": "gray", "modelling": "modeling", "modelled": "modeled", "labelled": "labeled",
           "labelling": "labeling", "signalling": "signaling", "travelled": "traveled", "neighbour": "neighbor", "favour": "favor",
           "fibre": "fiber", "programme": "program", "organis": "organiz", "normalis": "normaliz", "generalis": "generaliz",
           "characteris": "characteriz", "optimis": "optimiz", "recognis": "recogniz", "visualis": "visualiz", "minimis": "minimiz",
           "maximis": "maximiz", "summaris": "summariz", "synchronis": "synchroniz", "stabilis": "stabiliz", "randomis": "randomiz",
           "categoris": "categoriz", "parameteris": "parameteriz", "haem": "hem", "oesophag": "esophag", "artefact": "artifact",
           "judgement": "judgment", "towards": "toward", "whilst": "while", "amongst": "among", "learnt": "learned",
           "polaris": "polariz", "depolaris": "depolariz", "hyperpolaris": "hyperpolariz", "repolaris": "repolariz",
           "localis": "localiz", "utilis": "utiliz", "initialis": "initializ", "standardis": "standardiz", "sensitis": "sensitiz",
           "desensitis": "desensitiz", "memoris": "memoriz", "mobilis": "mobiliz", "realise": "realize", "realising": "realizing",
           "realisation": "realization", "specialise": "specialize", "specialising": "specializing",
           "specialisation": "specialization", "emphasise": "emphasize", "emphasising": "emphasizing",
           "hypothesise": "hypothesize", "hypothesising": "hypothesizing", "practis": "practic", "metre": "meter",
           "millimetre": "millimeter", "micrometre": "micrometer", "nanometre": "nanometer", "centimetre": "centimeter",
           "litre": "liter", "millilitre": "milliliter", "microlitre": "microliter", "travelling": "traveling",
           "cancelled": "canceled", "cancelling": "canceling", "signalled": "signaled", "levelled": "leveled",
           "channelled": "channeled", "fuelled": "fueled", "catalogue": "catalog", "analogue": "analog", "licence": "license",
           "defence": "defense", "tumour": "tumor", "honour": "honor", "labour": "labor", "flavour": "flavor", "vigour": "vigor",
           "paediatr": "pediatr", "anaesthe": "anesthe", "oedema": "edema", "oestr": "estr", "foet": "fet", "leukaem": "leukem",
           "enrolment": "enrollment", "acknowledgement": "acknowledgment", "sulphur": "sulfur", "manoeuvr": "maneuver"}
WHOLE_WORDS = {"analyse", "towards", "whilst", "amongst", "learnt"}           # "analyses" (noun) is not a match
BRITISH_RE = re.compile(r"\b(" + "|".join(k + (r"\b" if k in WHOLE_WORDS else "") for k in sorted(BRITISH, key=len, reverse=True)) + r")",
                        re.IGNORECASE)


def british(text: str) -> list[tuple[str, str]]:
    """British spellings in a text, each with its American form."""
    return [(m.group(0), BRITISH[m.group(1).lower()]) for m in BRITISH_RE.finditer(text)]
