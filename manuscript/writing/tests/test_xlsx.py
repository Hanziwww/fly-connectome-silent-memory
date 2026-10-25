"""Unit tests for manuscript/writing/xlsx.py (the dependency-free workbook writer of Tables S2-S4).
  python -I manuscript/writing/tests/test_xlsx.py"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xlsx  # noqa: E402


class Scalar:
    """Stands in for a numpy scalar (the writer calls .item()); the tests run without numpy."""
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class Workbook(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def sheets(self):
        return [xlsx.Sheet("Legend", rows=[["Table S9. A <test> & its legend"], [" leading space"]], widths=[80], wrap=True),
                xlsx.Sheet("Tests", header=["Test", "n", "P", "Primary"],
                           rows=[["swap vs baseline", 18, 0.000732421875, True], ["Kv4 loss, dose 0.5", Scalar(6), Scalar(0.25), False],
                                 ["missing value", None, float("nan"), None]], widths=[30, 8, 12, 8])]

    def test_round_trip(self):
        path = self.dir / "t.xlsx"
        xlsx.write(path, self.sheets())
        back = xlsx.read(path)
        self.assertEqual(list(back), ["Legend", "Tests"])
        self.assertEqual(back["Legend"][0], ["Table S9. A <test> & its legend"])
        self.assertEqual(back["Legend"][1], [" leading space"])
        self.assertEqual(back["Tests"][0], ["Test", "n", "P", "Primary"])
        self.assertEqual(back["Tests"][1], ["swap vs baseline", 18.0, 0.000732421875, "yes"])
        self.assertEqual(back["Tests"][2], ["Kv4 loss, dose 0.5", 6.0, 0.25, "no"])
        self.assertEqual(back["Tests"][3], ["missing value"])

    def test_rebuild_is_byte_identical(self):
        a, b = self.dir / "a.xlsx", self.dir / "b.xlsx"
        xlsx.write(a, self.sheets())
        xlsx.write(b, self.sheets())
        self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_columns_and_bad_names(self):
        self.assertEqual([xlsx.col(i) for i in (0, 25, 26, 701, 702)], ["A", "Z", "AA", "ZZ", "AAA"])
        with self.assertRaises(ValueError):
            xlsx.write(self.dir / "x.xlsx", [xlsx.Sheet("a/b", rows=[])])


if __name__ == "__main__":
    unittest.main(verbosity=1)
