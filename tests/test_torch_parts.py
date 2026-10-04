"""Tests that need PyTorch (skipped automatically when it is not installed)."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    HAVE_TORCH = True
except ImportError:                                         # pragma: no cover
    HAVE_TORCH = False

try:
    import albumentations  # noqa: F401
    HAVE_ALB = True
except ImportError:                                         # pragma: no cover
    HAVE_ALB = False


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class ModelTests(unittest.TestCase):
    def setUp(self):
        from mfld.config import ModelConfig
        from mfld.model import MFLDNet
        self.cfg = ModelConfig(num_keypoints=16)
        self.model = MFLDNet(self.cfg).eval()

    def test_output_shapes(self):
        out = self.model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(out.heatmaps.shape), (2, 16, 56, 56))
        self.assertEqual(tuple(out.coords.shape), (2, 16, 2))
        self.assertEqual(tuple(out.logits.shape), (2, 16, 56, 56))

    def test_heatmaps_are_distributions(self):
        out = self.model(torch.randn(2, 3, 224, 224))
        s = out.heatmaps.sum(dim=(2, 3))
        self.assertTrue(torch.allclose(s, torch.ones_like(s), atol=1e-4))
        self.assertTrue(((out.coords > 0) & (out.coords < 1)).all())

    def test_isometric_blocks(self):
        x = torch.randn(1, 256, 56, 56)
        for blk in self.model.blocks:
            self.assertEqual(blk(x).shape, x.shape)

    def test_architecture_constants(self):
        pe = self.model.patch_embed
        self.assertEqual((pe.in_channels, pe.out_channels, pe.kernel_size, pe.stride), (3, 256, (4, 4), (4, 4)))
        self.assertEqual(len(self.model.blocks), 8)
        dw = self.model.blocks[0].depthwise[0]
        self.assertEqual((dw.groups, dw.kernel_size), (256, (9, 9)))
        self.assertAlmostEqual(self.model.blocks[0].dropout.p, 0.2)

    def test_parameter_count(self):
        from mfld.model import count_parameters
        k, d = 9, 256
        expected = (3 * d * 16 + d) + 8 * ((d * k * k + d) + 2 * d + (d * d + d) + 2 * d) + (d * 16 + 16)
        self.assertEqual(count_parameters(self.model), expected)
        self.assertLess(count_parameters(self.model), 0.75e6)          # lightweight: < 0.75 M

    def test_soft_argmax_matches_numpy(self):
        from mfld.heatmaps import decode_expectation, render_gaussian_heatmaps
        from mfld.model import soft_argmax
        pts = np.random.default_rng(0).uniform(0.15, 0.85, (16, 2))
        hm = render_gaussian_heatmaps(pts, 56, 1.5)
        got = soft_argmax(torch.from_numpy(hm)[None])[0].numpy()
        np.testing.assert_allclose(got, decode_expectation(hm), atol=1e-5)
        np.testing.assert_allclose(got, pts, atol=2e-3)

    def test_backward(self):
        from mfld.heatmaps import render_gaussian_heatmaps
        from mfld.losses import MultiTaskLoss
        model = self.model.train()
        pts = np.random.default_rng(1).uniform(0.2, 0.8, (2, 16, 2)).astype(np.float32)
        hm = torch.from_numpy(np.stack([render_gaussian_heatmaps(p, 56, 1.5) for p in pts]))
        out = model(torch.randn(2, 3, 224, 224))
        loss = MultiTaskLoss()(out, hm, torch.from_numpy(pts))
        loss["total"].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class LossTests(unittest.TestCase):
    def test_jsd_properties(self):
        from mfld.losses import jensen_shannon_distance
        p = torch.softmax(torch.randn(2, 4, 8, 8).flatten(2), -1).reshape(2, 4, 8, 8)
        q = torch.softmax(torch.randn(2, 4, 8, 8).flatten(2), -1).reshape(2, 4, 8, 8)
        self.assertLess(jensen_shannon_distance(p, p).max().item(), 1e-3)
        d = jensen_shannon_distance(p, q)
        self.assertTrue((d > 0).all() and (d <= 1.0).all())               # sqrt of JSD in nats <= sqrt(ln 2)
        self.assertTrue(torch.allclose(d, jensen_shannon_distance(q, p), atol=1e-6))

    def test_euclidean(self):
        from mfld.losses import euclidean_distance
        d = euclidean_distance(torch.zeros(1, 2, 2), torch.tensor([[[3.0, 4.0], [0.0, 0.0]]]))
        self.assertAlmostEqual(d[0, 0].item(), 5.0, places=5)
        self.assertLess(d[0, 1].item(), 1e-5)

    def test_total_is_average(self):
        from mfld.losses import MultiTaskLoss
        from mfld.model import MFLDOutput
        hm = torch.softmax(torch.randn(1, 3, 6, 6).flatten(2), -1).reshape(1, 3, 6, 6)
        out = MFLDOutput(hm, torch.rand(1, 3, 2), hm)
        l = MultiTaskLoss()(out, torch.softmax(torch.randn(1, 3, 6, 6).flatten(2), -1).reshape(1, 3, 6, 6), torch.rand(1, 3, 2))
        self.assertAlmostEqual(l["total"].item(), 0.5 * (l["heatmap"].item() + l["coords"].item()), places=6)


@unittest.skipUnless(HAVE_TORCH and HAVE_ALB, "PyTorch/albumentations not installed")
class PipelineTests(unittest.TestCase):
    def test_dataset_and_one_epoch(self):
        from mfld.config import ModelConfig, TrainConfig
        from mfld.data import build_datasets
        from mfld.engine import collect_predictions, fit, keypoint_metrics
        from mfld.model import MFLDNet
        from mfld.synthetic import generate_dataset
        with tempfile.TemporaryDirectory() as d:
            ann = generate_dataset(d, n=20, size=(256, 144), seed=0)
            mcfg = ModelConfig(dim=32, depth=2, kernel_size=5, num_keypoints=16)             # tiny model keeps the test fast
            tcfg = TrainConfig(epochs=1, batch_size=4, labelled_frac=0.5)
            tr, va, te, _ = build_datasets(d, ann, mcfg, tcfg)
            item = tr[0]
            self.assertEqual(tuple(item["image"].shape), (3, 224, 224))
            self.assertEqual(tuple(item["heatmaps"].shape), (16, 56, 56))
            self.assertEqual(tuple(item["coords"].shape), (16, 2))
            model = MFLDNet(mcfg)
            hist = fit(model, tr, va, os.path.join(d, "run"), mcfg, tcfg, torch.device("cpu"), num_workers=0, log=lambda *_: None)
            self.assertEqual(len(hist), 1)
            self.assertTrue(os.path.exists(os.path.join(d, "run", "best.pt")))
            preds = collect_predictions(model, te, torch.device("cpu"))
            res = keypoint_metrics(preds, 224)
            self.assertIn("AP", res)


class TestPixelMapping(unittest.TestCase):
    def test_crop_coordinates_map_back_to_original_photo(self):
        from evaluate import to_original_pixels
        recs = [{"crop": [100, 50, 300, 250]}, {}]                       # 200x200 crop at (100, 50); whole image
        norm = np.array([[[0.5, 0.25]], [[0.5, 0.5]]], np.float32)        # (N=2, K=1, 2)
        size = np.array([[200, 200], [640, 480]], np.float32)             # crop / image size (w, h)
        out = to_original_pixels(norm, size, recs)
        np.testing.assert_allclose(out[0, 0], [200.0, 100.0])             # 0.5*200+100, 0.25*200+50
        np.testing.assert_allclose(out[1, 0], [320.0, 240.0])


if __name__ == "__main__":
    unittest.main()
