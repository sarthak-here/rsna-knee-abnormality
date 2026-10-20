import csv
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import audit_signal as module


class SignalAuditTests(unittest.TestCase):
    def test_only_existing_fit_reports_are_parsed(self):
        repo = module.REPO
        metadata = repo/'.audit/competition-metadata/train.csv'
        spec = repo/'.audit/r6-strict-load/state_spec.json'
        split_path = repo/'experiments/r6-own-report-adaptation/splits.json'
        split = json.loads(split_path.read_text())
        fit_ids = {r['id'] for r in split['fit']}
        with metadata.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        fit_reports = {r['Report'] for r in rows if r['StudyInstanceUID'] in fit_ids}
        original = module.extract
        calls = []
        def guarded(report):
            self.assertIn(report, fit_reports)
            calls.append(report)
            return original(report)
        with patch.object(module, 'extract', side_effect=guarded):
            result = module.audit(metadata, spec, split_path)
        self.assertEqual(len(calls), 192)
        self.assertEqual(result['known_fit_cells'], 536)
        self.assertEqual(result['per_class']['Fracture']['state_counts'], {'none': 4, 'old_or_healed_fracture': 1})
        self.assertAlmostEqual(result['per_class']['Fracture']['coefficient_per_known_ce_cell'], 4.8)
        self.assertAlmostEqual(result['max_to_min_positive_cell_coefficient'], 24.8)
        self.assertEqual(result['remaining_guarded_source_fold0_groups_not_in_pilot'], 844)
        self.assertFalse(result['globally_fresh_or_independent_evaluation_proven'])
        self.assertEqual(result['gold_report_parser_calls'], 0)
        self.assertTrue(all(abs(x['epoch_sum_ce_coefficients']-24) < 1e-12
                            for x in result['per_class'].values() if x['direct_report_supervision']))


if __name__ == '__main__':
    unittest.main()
