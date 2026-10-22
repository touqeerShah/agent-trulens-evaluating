import unittest

from ollama_practice.evaluate import evaluate


class LocalEvaluationTest(unittest.TestCase):
    def test_evaluation_is_local_and_bounded(self):
        scores = evaluate(
            {"answer": "Atlas Health has the largest deal at 120000 EUR.", "evidence": ["Atlas Health"]},
            "Which customer has the largest deal?",
            "Atlas Health has the largest deal at 120000 EUR.",
        )
        self.assertTrue(0 <= scores["relevance"] <= 1)
        self.assertTrue(0 <= scores["groundedness"] <= 1)
        self.assertEqual(scores["has_evidence"], 1)
