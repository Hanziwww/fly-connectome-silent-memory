"""Unit tests for the figure QC text rules (internal identifiers, published cell-type names, canvas bounds)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import figstyle as fs, qc  # noqa: E402


def issues_for(text, x=10.0, allow_names=()):
    cv = fs.Canvas(fs.FULL, 20.0)
    cv.text(x, 10.0, text, size=fs.PT_SMALL, va="center")
    try:
        return qc.check(cv, allow_names=allow_names)
    finally:
        cv.close()


class InternalIdentifiers(unittest.TestCase):
    def test_underscore_is_flagged(self):
        self.assertTrue(any("internal identifier" in i for i in issues_for("aIPg_m4")))

    def test_code_names_are_flagged(self):
        for s in ("remote-dynamic-M0", "head_only model", "value nan"):
            with self.subTest(s=s):
                self.assertTrue(any("internal identifier" in i for i in issues_for(s)))

    def test_allowed_cell_type_name_passes(self):
        self.assertEqual(issues_for("aIPg_m4", allow_names=("aIPg_m4",)), [])

    def test_allow_list_exempts_only_the_name(self):
        found = issues_for("aIPg_m4 head_only", allow_names=("aIPg_m4",))
        self.assertTrue(any("internal identifier" in i for i in found))

    def test_mathtext_is_exempt(self):
        self.assertEqual(issues_for(r"$x_{1}$ and 10$^{-3}$"), [])

    def test_plain_text_passes(self):
        self.assertEqual(issues_for("cue 1 minus cue 0"), [])


class DevelopmentLabels(unittest.TestCase):
    """The v4 (Cell) rule: reader-facing names only (manuscript/NARRATIVE_V4_OUTLINE.md §12)."""
    CODES = ("M0 readout-only", "S2 mirror", "X10 set 0", "F13.X1.lock", "G3a removal", "T-DY1", "T19 rescue", "D7-T0",
             "map D7", "P05", "C01 dose", "C13", "O14", "regime B", "Regime A", "frozen set")
    READER = ("LC12 (input)", "i2 MN (readout)", "DNp01 (giant fibre)", "T4", "L1", "H1 cell", "C2", "R7", "0A", "13A/19A",
              "Kv4", "see Figure S1", "Fig. S4", "Table S2", "Scene 1", "brain + readout trained", "2.36 M synapses")

    def test_codes_match(self):
        for s in self.CODES:
            with self.subTest(s=s):
                self.assertIsNotNone(qc.DEV_LABELS.search(s))

    def test_biology_and_reader_names_do_not_match(self):
        for s in self.READER:
            with self.subTest(s=s):
                self.assertIsNone(qc.DEV_LABELS.search(s))

    def test_only_the_cell_profile_applies_the_rule(self):
        self.assertFalse(any("development label" in i for i in issues_for("scene M0")))
        saved, fs.PROFILE = fs.PROFILE, "cell"
        try:
            self.assertTrue(any("development label" in i for i in issues_for("scene M0")))
        finally:
            fs.PROFILE = saved


class AmericanSpelling(unittest.TestCase):
    """The v4 (Cell) rule: American spelling in figure text (the user's decision, 2026-10-23)."""
    BRITISH = ("Odour memory by delay", "the ageing operator", "every analysed model", "colour: growth", "grey: unmapped",
               "Giant fibre", "push towards the target", "normalised rate", "generalisation range", "behaviour",
               "depolarised more by", "readout arbors, coloured as in A", "localised", "1 millimetre", "hyperpolarising")
    AMERICAN = ("Odor memory by delay", "the aging operator", "every analyzed model", "two analyses", "exercise", "noise",
                "gray: unmapped", "Giant fiber", "push toward the target", "normalized rate", "behavior", "precise",
                "depolarized more by", "a realistic model", "a specialist", "the hypothesis", "with emphasis", "parameter")

    def test_british_forms_are_found(self):
        for s in self.BRITISH:
            with self.subTest(s=s):
                self.assertTrue(qc.british(s))

    def test_american_forms_pass(self):
        for s in self.AMERICAN:
            with self.subTest(s=s):
                self.assertEqual(qc.british(s), [])

    def test_only_the_cell_profile_applies_the_rule(self):
        self.assertFalse(any("British spelling" in i for i in issues_for("odour")))
        saved, fs.PROFILE = fs.PROFILE, "cell"
        try:
            self.assertTrue(any("British spelling" in i for i in issues_for("odour")))
        finally:
            fs.PROFILE = saved


class LightTextOnFills(unittest.TestCase):
    """White text is allowed on a registered filled box (a chip, a bar segment) when it contrasts 4.5:1 with the fill."""

    def _issues(self, fill):
        cv = fs.Canvas(60.0, 20.0)
        cv.box(5.0, 5.0, 20.0, 6.0, fill)
        cv.fills.append(dict(rect=(5.0, 5.0, 20.0, 6.0), fill=fill))
        cv.text(6.0, 8.0, "label", color="#FFFFFF", va="center")
        out = qc.check(cv, width=60.0)
        cv.close()
        return out

    def test_dark_fill_passes(self):
        self.assertFalse(any("contrast" in i or "light text" in i for i in self._issues("#1F4E79")))

    def test_pale_fill_fails(self):
        self.assertTrue(any("contrast" in i for i in self._issues("#F2C14E")))


class Bounds(unittest.TestCase):
    def test_text_past_the_right_edge_is_flagged(self):
        self.assertTrue(any("outside the canvas" in i for i in issues_for("a long label at the edge", x=fs.FULL - 2.0)))


if __name__ == "__main__":
    unittest.main()
