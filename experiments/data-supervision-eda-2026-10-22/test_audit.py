import unittest
from audit_data import LABELS, fold, summarize


class Guards(unittest.TestCase):
    def test_never_parse_gold_source_fold0_or_duplicates(self):
        uid = {k: next(str(n) for n in range(1000) if fold(str(n)) == k) for k in (0,1)}
        def row(id_, text, gold=False):
            return dict(StudyInstanceUID=id_, Report=text, **{l: '1' if gold else '' for l in LABELS})
        reports = [row('gold', 'gold report', True), row('duplicate', 'GOLD   REPORT'),
                   row(uid[0], 'source holdout'), row(uid[1], 'fit report')]
        parsed = []
        def parser(text):
            parsed.append(text)
            return {}
        result = summarize(reports, [], parser)
        self.assertEqual(parsed, ['fit report'])
        self.assertEqual(result['fitting_report_groups'], 1)
        self.assertEqual(result['parsed_gold_reports'], 0)
        self.assertEqual(result['per_finding']['ACL']['known_report_groups'], 0)
        self.assertEqual(result['per_finding']['ACL']['unknown_report_groups'], 1)


if __name__ == '__main__':
    unittest.main()
