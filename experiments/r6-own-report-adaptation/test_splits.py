import csv
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import prepare_splits


class SplitContractTests(unittest.TestCase):
    def test_real_fixed_split_no_gold_parse_or_report_overlap(self):
        repo = Path(__file__).resolve().parents[2]
        path = repo / '.audit/competition-metadata/train.csv'
        spec = repo / '.audit/r6-strict-load/state_spec.json'
        if not path.exists() or not spec.exists():
            self.skipTest('local pinned audit metadata unavailable')
        with path.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        gold_reports = {r['Report'] for r in rows if all(r[l].strip() for l in prepare_splits.LABELS)}
        original = prepare_splits.extract

        def no_gold(report):
            self.assertNotIn(report, gold_reports)
            return original(report)

        with patch.object(prepare_splits, 'extract', side_effect=no_gold):
            split = prepare_splits.prepare(path, spec)
        self.assertEqual(split, json.loads(Path(__file__).with_name('splits.json').read_text()))
        self.assertEqual([len(split[k]) for k in ('fit', 'tune', 'confirmation')], [192, 64, 64])
        for partition in ('fit', 'tune', 'confirmation'):
            for r in split[partition]:
                fold = int(hashlib.sha1(r['id'].encode()).hexdigest()[:8], 16) % 4
                self.assertEqual(fold == 0, partition != 'fit')


if __name__ == '__main__':
    unittest.main()
