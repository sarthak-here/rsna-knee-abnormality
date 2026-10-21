import unittest
import torch
from audit_checkpoint import inventory


class CheckpointTests(unittest.TestCase):
    def test_finite_inventory_does_not_export_study_ids(self):
        result = inventory({"model": {"encoder.x": torch.ones(2)},
                            "train_ids": ["private-test-id"]})
        self.assertEqual(result["model_elements"], 2)
        self.assertEqual(result["metadata_types"]["train_ids"], "list")
        self.assertNotIn("private-test-id", str(result))
        self.assertFalse(result["metadata_is_independent_training_lineage_proof"])

    def test_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            inventory({"model": {"head.x": torch.tensor([float("nan")])}})

    def test_missing_or_nontensor_model_rejected(self):
        for payload in ({}, {"model": {}}, {"model": {"x": 4}}):
            with self.assertRaises(ValueError):
                inventory(payload)


if __name__ == "__main__":
    unittest.main()
