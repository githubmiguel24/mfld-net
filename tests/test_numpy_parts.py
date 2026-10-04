"""Tests for the pure-numpy modules (no PyTorch needed):  python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mfld.heatmaps import decode_expectation, render_gaussian_heatmaps
from mfld.metrics import (average_precision_recall, mean_absolute_difference, object_keypoint_similarity,
                          standard_deviation_of_difference)
from mfld.morphometry import DEFAULT_SPEC, KEYPOINT_NAMES, measure
from mfld.splits import make_splits
from mfld import synthetic


class HeatmapTests(unittest.TestCase):
    def test_shape_and_normalisation(self):
        hm = render_gaussian_heatmaps(np.random.rand(16, 2), 56, 1.5)
        self.assertEqual(hm.shape, (16, 56, 56))
        np.testing.assert_allclose(hm.sum((1, 2)), 1.0, rtol=1e-5)

    def test_decode_roundtrip(self):
        pts = np.random.default_rng(0).uniform(0.15, 0.85, (16, 2))
        dec = decode_expectation(render_gaussian_heatmaps(pts, 56, 1.5))
        np.testing.assert_allclose(dec, pts, atol=2e-3)

    def test_peak_location(self):
        hm = render_gaussian_heatmaps(np.array([[0.5 + 0.5 / 56, 0.25 + 0.5 / 56]]), 56, 1.5)[0]
        y, x = np.unravel_index(hm.argmax(), hm.shape)
        self.assertEqual((x, y), (28, 14))


class MetricTests(unittest.TestCase):
    def test_perfect_prediction(self):
        gt = np.random.default_rng(1).uniform(0, 200, (10, 16, 2))
        oks = object_keypoint_similarity(gt, gt, 100.0)
        np.testing.assert_allclose(oks, 1.0)
        res = average_precision_recall(oks, np.linspace(0, 1, 10))
        self.assertAlmostEqual(res["AP"], 1.0)
        self.assertAlmostEqual(res["AR"], 1.0)

    def test_oks_decreases_with_error(self):
        gt = np.zeros((1, 16, 2))
        near = object_keypoint_similarity(gt + 1.0, gt, 100.0)[0]
        far = object_keypoint_similarity(gt + 10.0, gt, 100.0)[0]
        self.assertTrue(1.0 > near > far > 0.0)

    def test_oks_formula(self):
        d, s, k = 3.0, 50.0, 0.05
        pred = np.zeros((16, 2))
        pred[:, 0] = d
        expected = np.exp(-d ** 2 / (2 * s ** 2 * k ** 2))
        self.assertAlmostEqual(float(object_keypoint_similarity(pred, np.zeros((16, 2)), s, k)), expected, places=9)

    def test_ap_hand_example(self):
        oks = np.array([0.9, 0.9, 0.1, 0.9])
        conf = np.array([0.9, 0.8, 0.7, 0.6])
        from mfld.metrics import _ap_ar_at_threshold
        ap, ar = _ap_ar_at_threshold(oks, conf, 0.5)
        self.assertAlmostEqual(ap, (26 + 25 + 0.75 * 25) / 101, places=2)
        self.assertAlmostEqual(ar, 0.75)

    def test_mad_sdd(self):
        x, y = [1, 2, 3], [2, 2, 5]
        self.assertAlmostEqual(mean_absolute_difference(x, y), 1.0)
        self.assertAlmostEqual(standard_deviation_of_difference(x, y), 1.0)
        self.assertAlmostEqual(standard_deviation_of_difference([1, 2, 3], [2, 3, 4]), 0.0)   # constant offset -> perfectly precise


class MorphometryTests(unittest.TestCase):
    def test_names_and_spec(self):
        self.assertEqual(len(KEYPOINT_NAMES), 16)
        self.assertEqual(set(DEFAULT_SPEC), {"total_length", "standard_length", "body_depth", "head_length"})

    def test_measure_simple(self):
        kp = np.zeros((16, 2))
        kp[9] = [100, 0]
        kp[3] = [20, 0]
        kp[10] = [120, -5]
        kp[11] = [120, 5]
        kp[6] = [30, -10]
        kp[15] = [30, 10]
        m = measure(kp, mm_per_pixel=2.0)
        self.assertAlmostEqual(m["standard_length"], 200.0)
        self.assertAlmostEqual(m["total_length"], 240.0)
        self.assertAlmostEqual(m["head_length"], 40.0)
        self.assertAlmostEqual(m["body_depth"], 40.0)


class SplitTests(unittest.TestCase):
    def test_paper_split(self):
        s = make_splits(1000, 0.4, 0.7, seed=0)
        self.assertEqual((len(s["train"]), len(s["val"]), len(s["test"])), (280, 120, 600))
        self.assertEqual(len(set(s["train"]) | set(s["val"]) | set(s["test"])), 1000)
        self.assertFalse(set(s["test"]) & (set(s["train"]) | set(s["val"])))


class SyntheticTests(unittest.TestCase):
    def test_sample(self):
        rng = np.random.default_rng(3)
        img, kp, scale = synthetic.generate_sample(rng, (512, 288))
        self.assertEqual(img.shape, (288, 512, 3))
        self.assertEqual(kp.shape, (16, 2))
        self.assertTrue((kp[:, 0] >= 0).all() and (kp[:, 0] < 512).all() and (kp[:, 1] >= 0).all() and (kp[:, 1] < 288).all())
        self.assertGreater(scale, 50)
        tl = measure(kp)["total_length"]
        self.assertTrue(0.6 * 512 < tl < 0.85 * 512)

    def test_dataset_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = synthetic.generate_dataset(d, n=3, size=(256, 144), seed=1)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(os.path.join(d, "images", "fish_00000.png")))


if __name__ == "__main__":
    unittest.main()
