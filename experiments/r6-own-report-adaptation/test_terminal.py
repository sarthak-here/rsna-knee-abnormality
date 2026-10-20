import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_terminal import prediction_report


class TerminalPredictionTests(unittest.TestCase):
    def fixture(self, path, **updates):
        records = [{'id': str(i), 'state_targets': [0]*12} for i in range(64)]
        payload = dict(ids=np.array([r['id'] for r in records]),
                       state_targets=np.zeros((64,12), dtype=np.int16),
                       control_states=np.full((64,12,2), .5),
                       candidate_states=np.full((64,12,2), .5),
                       control_probabilities=np.full((64,12), .5),
                       candidate_probabilities=np.full((64,12), .5))
        payload.update(updates)
        np.savez(path, **payload)
        return records

    def test_valid_and_corrupted_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'predictions.npz'
            records = self.fixture(path)
            valid = np.ones((12,2), dtype=bool)
            self.assertFalse(prediction_report(path, records, valid)['passed'])
            records = self.fixture(path, candidate_probabilities=np.full((64,12), np.nan))
            with self.assertRaises(AssertionError):
                prediction_report(path, records, valid)
            records = self.fixture(path, ids=np.array(['duplicate']*64))
            with self.assertRaises(AssertionError):
                prediction_report(path, records, valid)
            records = self.fixture(path, state_targets=np.ones((64,12), dtype=np.int16))
            with self.assertRaises(AssertionError):
                prediction_report(path, records, valid)


if __name__ == '__main__':
    unittest.main()
