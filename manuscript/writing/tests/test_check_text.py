"""Unit tests for manuscript/writing/check_text.py.
  python -I manuscript/writing/tests/test_check_text.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_text as ct  # noqa: E402

GOOD = """# Memory locked to the brain state that wrote it

## Summary
Whole nervous systems now have wiring diagrams. How such a diagram could learn is open. Here, we trained a model.

## Highlights
- A trained fly nervous system holds a cue without spiking
- Memory is read back only from the state that wrote it
- Disease shifts the state and removes access, not the memory

## In Brief
Wang et al. train a model of the fly nervous system and find that memories are read only from the state that wrote them.

## Introduction
Connectomes exist. Here, we ask what they can learn.

## Results
### The trace is held without spiking
Access fell from 18 to 7 trained histories (Figure 7C; Smith et al., 2024).
"""


class Counts(unittest.TestCase):
    def test_a_compliant_draft_passes(self):
        fails, warns = ct.check(GOOD, ledger=[18.0, 7.0])
        self.assertEqual(fails, [])

    def test_long_highlight_fails(self):
        bad = GOOD.replace("- Disease shifts the state and removes access, not the memory",
                           "- Disease shifts the excitability state of every neuron and so removes access to earlier memories")
        self.assertTrue(any("over 85 characters" in f for f in ct.check(bad)[0]))

    def test_acronyms_in_highlights_fail(self):
        bad = GOOD.replace("A trained fly nervous system", "A trained fly CNS")
        self.assertTrue(any("acronyms" in f for f in ct.check(bad)[0]))
        bad = GOOD.replace("Disease shifts the state", "Kv4 loss shifts the state")
        self.assertTrue(any("acronyms" in f for f in ct.check(bad)[0]))

    def test_in_brief_must_be_third_person(self):
        bad = GOOD.replace("Wang et al. train", "We train")
        self.assertTrue(any("first person" in f for f in ct.check(bad)[0]))


class Words(unittest.TestCase):
    def test_dropped_words_fail(self):
        for w in ("This sheds light on", "a pivotal result", "a novel trace", "for the first time"):
            with self.subTest(w=w):
                bad = GOOD.replace("Connectomes exist.", f"Connectomes exist. {w} memory.")
                self.assertTrue(ct.check(bad)[0])

    def test_british_spelling_fails(self):
        bad = GOOD.replace("Connectomes exist.", "Connectomes guide behaviour and odour memory.")
        fails = ct.check(bad)[0]
        self.assertTrue(any("behaviour" in f for f in fails) and any("odour" in f for f in fails))

    def test_internal_labels_fail(self):
        bad = GOOD.replace("Connectomes exist.", "The M0 model and regime B learn.")
        self.assertTrue(any("development label" in f for f in ct.check(bad)[0]))

    def test_rationed_words_warn(self):
        bad = GOOD.replace("Connectomes exist.", "Notably, a. Notably, b. Notably, c.")
        self.assertTrue(any("rationed" in w for w in ct.check(bad)[1]))


class Ledger(unittest.TestCase):
    def test_numbers_must_be_in_the_ledger(self):
        self.assertTrue(any("number 7" in f for f in ct.check(GOOD, ledger=[18.0])[0]))

    def test_rounding_and_percent_match(self):
        self.assertTrue(ct.in_ledger("0.41", [0.4083]))
        self.assertTrue(ct.in_ledger("41%", [0.4083]))
        self.assertTrue(ct.in_ledger("116,326", [116326.0]))
        self.assertTrue(ct.in_ledger("−0.70", [-0.7012]))
        self.assertFalse(ct.in_ledger("0.42", [0.4083]))

    def test_references_are_not_numbers(self):
        self.assertEqual(ct.numbers("as shown (Figures 2A and 2B; Figure S5; Smith et al., 2024)"), [])


if __name__ == "__main__":
    unittest.main()
