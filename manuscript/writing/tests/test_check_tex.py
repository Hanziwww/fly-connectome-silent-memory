"""Unit tests for manuscript/writing/check_tex.py (the TeX conversion and its bookkeeping).
  python -I manuscript/writing/tests/test_check_tex.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_tex as cx  # noqa: E402

LEDGER = {"L01": {"display": "167,106"}, "A09": {"display": "−0.70"}}


def run(tex, part="Results", sentinel=False):
    doc = cx.Doc()
    return cx.convert(tex, doc, part, LEDGER, sentinel), doc


class Conversion(unittest.TestCase):
    def test_values_expand_and_are_recorded(self):
        text, doc = run(r"The model has \val{L01} neurons.")
        self.assertIn("167,106 neurons", text)
        self.assertEqual(doc.vals, [("Results", "L01")])

    def test_sentinel_hides_values_from_the_number_check(self):
        text, _ = run(r"The model has \val{L01} neurons.", sentinel=True)
        self.assertNotIn("167", text)

    def test_callouts_expand_ranges(self):
        text, doc = run(r"Spikes stop (\figr{2B}{2D}) and swaps flip (\figs{2F}{2G}; \video{S2}).")
        self.assertEqual([c[2] for c in doc.callouts], ["2B", "2C", "2D", "2F", "2G", "Video S2"])
        self.assertIn("Figures 2B–2D", text)

    def test_citations_and_placeholders_are_recorded(self):
        _, doc = run(r"Connectomes exist\cellcite{dorkenwald2024,berg2026}. \todraft{Gate B} \tofill{Affiliation}")
        self.assertEqual(doc.cites, {"dorkenwald2024", "berg2026"})
        self.assertEqual(doc.todraft, [("Results", "Gate B")])
        self.assertEqual(doc.tofill, [("Results", "Affiliation")])

    def test_results_headings_are_collected(self):
        _, doc = run(r"\subsection{Trace-carrying cells have a molecular signature: \gene{SK}}")
        self.assertEqual(doc.headings, ["Trace-carrying cells have a molecular signature: SK"])

    def test_uncounted_text_is_marked(self):
        text, _ = run(r"Counted. \begin{uncounted}Not counted here.\end{uncounted}", part="Supplemental information")
        self.assertEqual(len(cx.words(cx.re.sub(r"⟪U⟫.*?⟪/U⟫", " ", text, flags=cx.re.S))), 1)

    def test_comments_and_escapes(self):
        text, _ = run("Half of the cells (50\\%) % a comment\nfire.")
        self.assertIn("50%", text)
        self.assertNotIn("comment", text)

    def test_identifiers_are_printed_but_not_numbers(self):
        text, _ = run(r"Python \ident{3.12.13} ran.", part="STAR Methods")
        self.assertIn("3.12.13", text)
        text, _ = run(r"Python \ident{3.12.13} ran.", part="STAR Methods", sentinel=True)
        self.assertNotIn("3.12", text)

    def test_formulas_hide_symbols_but_keep_values(self):
        text, doc = run(r"With $w_{ij} = 2$ and \begin{equation*}v_i = 3\end{equation*} at $\tau = \val{L01}$ ms.", part="STAR Methods")
        self.assertNotIn("_", text)
        self.assertNotIn("3", text.replace("167,106", ""))
        self.assertEqual(doc.vals, [("STAR Methods", "L01")])

    def test_key_resources_table_is_set_aside(self):
        tex = r"Before. \begin{krt}Python \ident{3.12} & PSF\cite{berg2026} & \url{https://www.python.org}\\\end{krt} After."
        self.assertNotIn("Python", cx.KRT.sub(" ", tex))
        self.assertEqual(cx.KRT.findall(tex)[0].count("&"), 2)


class Manuscript(unittest.TestCase):
    def test_the_manuscript_passes(self):
        fails, _, _ = cx.check(strict=False)
        self.assertEqual(fails, [])


if __name__ == "__main__":
    unittest.main(verbosity=1)
