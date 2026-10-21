import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import prepare_splits
from build_notebook import training_source

# Exercise unchanged numerical/gate helpers against THIS generated trainer.
old_tests = Path(__file__).resolve().parents[1] / "r6-own-report-adaptation/test_training.py"
spec = importlib.util.spec_from_file_location("shared_runtime_tests", old_tests)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
TrainingContractTests = module.TrainingContractTests


class NewContractTests(unittest.TestCase):
    def test_no_gold_or_old_group_parse(self):
        here = Path(__file__).resolve().parent
        repo = here.parents[1]
        train = repo / ".audit/competition-metadata/train.csv"
        state = repo / ".audit/r6-strict-load/state_spec.json"
        if not train.exists():
            self.skipTest("pinned local metadata unavailable")
        excluded = set(json.loads((here / "excluded_previous_split.json").read_text())["report_groups"])
        with train.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        forbidden = {r["Report"] for r in rows if all(r[l].strip() for l in prepare_splits.LABELS)}
        original = prepare_splits.extract
        def checked(report):
            self.assertNotIn(report, forbidden)
            group = hashlib.sha256(" ".join(report.casefold().split()).encode()).hexdigest()
            self.assertNotIn(group, excluded)
            return original(report)
        with patch.object(prepare_splits, "extract", side_effect=checked):
            actual = prepare_splits.prepare(train, state)
        expected = json.loads((here / "splits.json").read_text())
        self.assertEqual(actual, expected)
        self.assertEqual([len(actual[p]) for p in ("fit", "tune", "confirmation")], [192, 64, 64])

    def test_generated_trainer_is_frozen(self):
        here = Path(__file__).resolve().parent
        self.assertEqual(training_source(), (here / "train.py").read_text(encoding="utf-8"))
        text = training_source()
        self.assertNotIn("candidate.enc.stages[3].requires_grad_(True)", text)
        self.assertIn("names = install(candidate)", text)
        self.assertIn("adapter_names = assert_frozen(reference, candidate)", text)
        self.assertIn('"trainable_parameters": 92160', text)


if __name__ == "__main__":
    unittest.main()
