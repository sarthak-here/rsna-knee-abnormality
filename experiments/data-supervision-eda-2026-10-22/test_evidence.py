import unittest
from evidence import ledger, norm


class EvidenceTests(unittest.TestCase):
    def test_history_never_mislabeled_current(self):
        result = ledger('Clinical history:\nACL tear.\nFindings:\nACL normal.')
        self.assertEqual([e['section'] for e in result['entries']], ['history', 'findings'])
        self.assertIsNone(result['training_targets'])

    def test_no_heading_retains_review_not_normal(self):
        entry = ledger('Possible fracture without displacement')['entries'][0]
        self.assertEqual(entry['section'], 'unsectioned_requires_review')
        self.assertTrue(entry['uncertainty_cue'])
        self.assertTrue(entry['negation_cue'])
        self.assertTrue(entry['requires_review'])

    def test_multiple_entities_are_retained_not_cross_labeled(self):
        result = ledger('Hallazgos:\nACL normal y MCL lesion.')
        self.assertEqual({m['finding'] for m in result['entries'][0]['mentions']}, {'ACL', 'MCL'})
        self.assertTrue(result['entries'][0]['multi_entity'])
        self.assertIsNone(result['training_targets'])

    def test_compartment_mention_is_not_positive_oa_label(self):
        result = ledger('Findings:\nNormal patellofemoral cartilage.')
        self.assertEqual(result['entries'][0]['mentions'][0]['finding'], 'PF OA')
        self.assertIsNone(result['training_targets'])
        self.assertFalse(result['clinically_validated'])

    def test_normalized_offsets(self):
        text = 'Conclusión:\nSin fractura; contusión ósea.'
        result = ledger(text)
        for entry in result['entries']:
            for mention in entry['mentions']:
                fragment = norm(text)[mention['start']:mention['end']]
                self.assertTrue(fragment)
                self.assertLessEqual(entry['start'], mention['start'])
                self.assertLessEqual(mention['end'], entry['end'])


if __name__ == '__main__':
    unittest.main()
