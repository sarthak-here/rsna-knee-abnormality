import unittest

import numpy as np
import torch

from models import prepare_clip, migrate_video_biases
from search import groups_from_reports, partition


class ContractTests(unittest.TestCase):
    def test_video_bias_conversion_is_exact(self):
        prefix = "videomae.encoder.layer.0.attention.attention."
        q, v = torch.randn(4), torch.randn(4)
        source = {prefix + "q_bias": q, prefix + "v_bias": v, "other": torch.ones(2)}
        expected = {prefix + "query.bias": q, prefix + "value.bias": v,
                    prefix + "key.bias": torch.zeros(4), "other": torch.ones(2)}
        converted = migrate_video_biases(source, expected)
        for key in expected:
            torch.testing.assert_close(converted[key], expected[key], rtol=0, atol=0)
        # Old layout must remain untouched under an older library.
        self.assertEqual(set(migrate_video_biases(source, source)), set(source))
        with self.assertRaises(ValueError):
            migrate_video_biases(source, {"wrong.architecture": q})

    def test_report_normalization(self):
        groups = groups_from_reports(["  NORMAL   Knee\n", "normal knee", "abnormal"])
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])

    def test_missing_report_fails(self):
        for value in (None, "", "   ", np.nan):
            with self.assertRaises(ValueError):
                groups_from_reports([value])

    def test_partition_excludes_gold_reports_and_is_repeatable(self):
        ids = np.asarray([f"id_{i}" for i in range(102)])
        groups = groups_from_reports([f"report {i // 2}" for i in range(102)])
        gold = np.zeros(102, bool); gold[0] = True
        first, second = partition(ids, groups, gold), partition(ids, groups, gold)
        used = np.concatenate(list(first.values()))
        self.assertEqual(set(used), set(range(2, 102)))
        self.assertEqual(len(set(used)), 100)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
            for other in first:
                if other != key:
                    self.assertFalse(set(groups[first[key]]) & set(groups[first[other]]))

    def test_clip_shape_and_replicated_grayscale(self):
        raw = torch.arange(12, dtype=torch.uint8)[:, None, None].expand(12, 32, 32)
        clip = prepare_clip(raw)
        self.assertEqual(tuple(clip.shape), (3, 16, 224, 224))
        self.assertTrue(torch.isfinite(clip).all())
        # Recover identical unnormalized grayscale from each normalized channel.
        restored = clip * torch.tensor([.229, .224, .225])[:, None, None, None]
        restored += torch.tensor([.485, .456, .406])[:, None, None, None]
        self.assertTrue(torch.allclose(restored[0], restored[1], atol=1e-6))
        self.assertEqual(torch.unique(restored[0, :, 0, 0]).numel(), 12)


if __name__ == "__main__":
    unittest.main()
