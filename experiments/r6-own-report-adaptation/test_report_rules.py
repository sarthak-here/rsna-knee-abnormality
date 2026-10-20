import unittest
from report_rules import extract


class ReportRulesTests(unittest.TestCase):
    def test_positive_and_explicit_normal(self):
        self.assertEqual(extract("Findings: Complete ACL tear. Medial meniscus is intact."), {"ACL": "complete_tear", "Medial Meniscus": "normal"})

    def test_question_missing_and_uncertainty_are_not_negatives(self):
        self.assertEqual(extract("Clinical history: ACL tear?"), {})
        self.assertEqual(extract("Findings: Possible ACL tear. No joint effusion mentioned?"), {})
        self.assertEqual(extract("Findings: Normal appearance of visualized muscles."), {})

    def test_mixed_entities_postoperative_and_conflicts(self):
        self.assertEqual(extract("Findings: Medial meniscus tear and normal lateral meniscus."), {})
        self.assertEqual(extract("Findings: ACL graft is intact."), {})
        self.assertEqual(extract("Findings: ACL is intact. Complete ACL rupture."), {})

    def test_multilingual_literal_evidence(self):
        self.assertEqual(extract("Resultados: Rotura completa del LCA. Derrame moderado."), {"ACL": "complete_tear", "Effusion": "moderate"})
        self.assertEqual(extract("Befund: Vorderes Kreuzband intakt. Kein Gelenkerguss."), {"ACL": "normal", "Effusion": "none_or_trace"})

    def test_no_unwarranted_grade(self):
        self.assertEqual(extract("Findings: Partial ACL tear. Small to moderate effusion. Moderate chondral loss."), {})
        self.assertEqual(extract("Findings: No tear of the medial meniscus."), {})

    def test_negation_does_not_destroy_or_invent_fine_states(self):
        self.assertEqual(extract("Findings: Degeneration of medial meniscus without tear."), {})
        self.assertEqual(extract("Findings: MCL intact with periligamentous edema."), {})
        self.assertEqual(extract("Findings: No large joint effusion. No acute fracture."), {})
        self.assertEqual(extract("Findings: Small joint effusion without loose bodies."), {"Effusion": "small"})
        self.assertEqual(extract("Findings: Baker cyst is not ruptured."), {})


if __name__ == "__main__":
    unittest.main()
