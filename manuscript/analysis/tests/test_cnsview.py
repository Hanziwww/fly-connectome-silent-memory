"""Unit tests for the unfolded-CNS display transform."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msa import cnsview  # noqa: E402

Z, Y = cnsview.NECK_Z, cnsview.NECK_Y


class Unfold(unittest.TestCase):
    def test_brain_side_unchanged(self):
        p = np.array([[100.0, 200.0, 300.0], [700.0, 50.0, Z]])
        u, vnc = cnsview.unfold(p)
        self.assertFalse(vnc.any())
        np.testing.assert_array_equal(u, p)

    def test_vnc_hangs_below_in_ventral_view(self):
        u, vnc = cnsview.unfold([[400.0, Y + 30.0, Z + 100.0]])
        self.assertTrue(vnc[0])
        np.testing.assert_allclose(u[0], [400.0, Y + 100.0, Z - 30.0])   # posterior goes down; ventral comes towards the viewer

    def test_continuous_on_the_axis(self):
        a, _ = cnsview.unfold([[400.0, Y + 2.0, Z]])
        b, _ = cnsview.unfold([[400.0, Y + 2.0, Z + 1e-6]])
        self.assertLess(np.linalg.norm(a - b), 3.0)

    def test_rigid_on_each_side(self):
        rng = np.random.default_rng(0)
        p = rng.uniform([0, 300, Z + 1], [700, 550, 1000], size=(50, 3))
        u, vnc = cnsview.unfold(p)
        self.assertTrue(vnc.all())
        d0 = np.linalg.norm(p[:, None] - p[None], axis=2)
        d1 = np.linalg.norm(u[:, None] - u[None], axis=2)
        np.testing.assert_allclose(d0, d1, atol=1e-9)

    def test_vectors_follow_the_rotation(self):
        p = np.array([[400.0, Y, Z + 50.0]])
        u, vnc = cnsview.unfold(p)
        step, _ = cnsview.unfold(p + [0.0, 0.0, 1.0])
        v = cnsview.unfold_vectors([[0.0, 0.0, 1.0]], vnc)
        np.testing.assert_allclose(step - u, v)


if __name__ == '__main__':
    unittest.main()
