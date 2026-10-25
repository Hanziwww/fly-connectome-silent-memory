"""Unit tests for the release builder's rules and the deposit checker (manuscript/package).
  python -I manuscript/package/tests/test_package.py"""
import csv
import hashlib
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_release as br  # noqa: E402


class CodeRelease(unittest.TestCase):
    def test_what_the_code_release_holds(self):
        for p in ("README.md", "sokoban/lib/brain_model.py", "studies/brain-pathology-v1/amendments/0001-x.json",
                  "manuscript/analysis/run_all.py", "manuscript/package/build_release.py", "tools/remote/ssh.py",
                  "MIGRATION/launch.py", "manuscript/FREEZE_MANIFEST.csv"):
            self.assertTrue(br.in_code(p), p)
        for p in ("PROJECT_MEMORY.md", "manuscript/PLAN.md", "studies/brain-pathology-v1/results/dy2_models.csv",
                  "manuscript/figures/design/board.py", "archive/INDEX.md", "runs/linear-joint/20261002-p05-01/report.md",
                  "tools/remote_study.py", "MIGRATION/restore_bundle.py", "manuscript/README.md.bak"):
            self.assertFalse(br.in_code(p), p)

    def test_scan(self):
        # the fixtures are joined at run time, so that the release's own scan does not match this file
        pw, ssh, at = "pass" + "word", "ssh" + " -p 2222", "root" + "@" + "connect" + ".example.com"
        found = []
        br.scan(f'{pw} = "x1y2z3w4"\n{ssh} {at}\nhan@hanziwww.com\nhttp://127.0.0.1:8774/', "f", found, limit=10)
        kinds = [k for _, k, _ in found]
        self.assertIn("credential", kinds)
        self.assertGreaterEqual(kinds.count("connection detail"), 2)
        self.assertFalse(any("hanziwww" in t or "127.0.0.1" in t for _, _, t in found))
        clean = []
        br.scan("password and keyboard-interactive authentication are refused; token budget 12", "g", clean)
        self.assertEqual(clean, [])

    def test_unwrap(self):
        self.assertEqual(br.unwrap("Lead\ncontact.\n \nSome text\nwrapped.\n\n- one\n item\n- two\n"),
                         "Lead contact.\n\nSome text wrapped.\n\n- one item\n\n- two")

    def test_zip_is_deterministic(self):
        d = Path(tempfile.mkdtemp())
        (d / "src" / "a").mkdir(parents=True)
        (d / "src" / "a" / "x.json").write_text('{"k": 1}', encoding="utf-8")
        (d / "src" / "b.npz").write_bytes(b"\x00\x01")
        one = br.zip_tree(d / "src", d / "one.zip", "top").read_bytes()
        two = br.zip_tree(d / "src", d / "two.zip", "top").read_bytes()
        self.assertEqual(one, two)
        with zipfile.ZipFile(d / "one.zip") as z:
            self.assertEqual(z.namelist(), ["top/a/x.json", "top/b.npz"])
            self.assertEqual(z.getinfo("top/b.npz").compress_type, zipfile.ZIP_STORED)
            self.assertEqual(z.getinfo("top/a/x.json").compress_type, zipfile.ZIP_DEFLATED)


class Deposit(unittest.TestCase):
    def test_private_list_stays_private(self):
        self.assertFalse(br.in_code("manuscript/package/private/exclusions.py"))
        saved = br.PRIVATE
        try:
            br.PRIVATE = None                      # without the workspace's private list nothing is left out or trimmed
            self.assertFalse(br.unreported("studies/x/runs/a/model.pt"))
            self.assertFalse(br.trims("studies/x/results/a.csv"))
            self.assertTrue(br.in_code("studies/x/code/a.py"))
        finally:
            br.PRIVATE = saved

    def test_records_placed(self):
        read = {"studies/brain-supplement-v2/runs/x1-M0-head_only-r0/verification.json",
                "studies/brain-sokoban-v1/mechanisms-v1/runs/record-x/a.npz"}
        self.assertEqual(br.place("studies/brain-supplement-v2/runs/x1-M0-head_only-r0/verification.json", read), ("A", "runs"))
        self.assertEqual(br.place("studies/brain-supplement-v2/runs/x2-S1-mirror-r0/verification.json", read), ("B", "verification"))
        self.assertEqual(br.place("studies/brain-supplement-v2/runs/x2-S1-mirror-r0/behavior-features/000001.npz", read),
                         ("B", "behavior-features"))
        self.assertEqual(br.place("studies/brain-sokoban-v1/mechanisms-v1/runs/record-x/a.npz", read), ("A", "recordings"))
        self.assertEqual(br.place("studies/brain-sokoban-v1/mechanisms-v1/runs/record-x/b.npz", read), ("B", "recordings"))
        self.assertEqual(br.place("studies/brain-pathology-v1/results/all_tests.csv", read), ("A", "results"))
        self.assertEqual(br.place("manuscript/figures/v4/main/fig1.pdf", read), ("A", "figures"))
        self.assertEqual(br.place("flybrain/cache/W_post_pre.npz", read), ("A", "assets"))
        self.assertTrue(br.FORBIDDEN.search("studies/x/runs/j/.cluster.json"))
        self.assertTrue(br.FORBIDDEN.search("local/notes.txt"))
        self.assertTrue(br.NOT_DATA.search("flybrain/.venv/Lib/site-packages/numpy/__init__.py"))
        self.assertFalse(br.NOT_DATA.search("flybrain/data/morphology/skeletons/10019"))

    def test_verify_deposit(self):
        d = Path(tempfile.mkdtemp())
        rec, into = d / "record", d / "into"
        rec.mkdir()
        data = {"runs/a/model.pt": b"weights", "runs/a/result.json": b'{"ok": true}'}
        with zipfile.ZipFile(rec / "core-runs.zip", "w") as z:
            for p, b in data.items():
                z.writestr(p, b)
        rows = [dict(archive="core-runs.zip", path=p, bytes=len(b), sha256=hashlib.sha256(b).hexdigest(), same_as="") for p, b in data.items()]
        rows.append(dict(archive="core-runs.zip", path="runs/b/model.pt", bytes=7, sha256=rows[0]["sha256"], same_as="runs/a/model.pt"))
        with open(rec / "MANIFEST.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["archive", "path", "bytes", "sha256", "same_as"], lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        script = HERE / "verify_deposit.py"
        ok = subprocess.run([sys.executable, "-I", str(script), str(rec), "--into", str(into)], capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        self.assertEqual((into / "runs/b/model.pt").read_bytes(), b"weights")
        rows[1]["sha256"] = "0" * 64
        with open(rec / "MANIFEST.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["archive", "path", "bytes", "sha256", "same_as"], lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        bad = subprocess.run([sys.executable, "-I", str(script), str(rec), "--check-only"], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 1)
        self.assertIn("differs", bad.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=1)
