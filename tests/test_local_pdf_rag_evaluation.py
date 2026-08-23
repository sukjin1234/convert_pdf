import unittest

from scripts.evaluate_local_pdf_rag import match_matches_source, summarize_retrieval


class LocalPdfRagEvaluationTests(unittest.TestCase):
    def test_source_match_accepts_declared_page_or_section_region(self):
        source = {"pages": [12], "section_any": [["Retry", "Policy"]]}

        self.assertTrue(match_matches_source({"page": 12, "section_path": "Other"}, source))
        self.assertTrue(match_matches_source({"page": 2, "section_path": "Retry Policy"}, source))
        self.assertFalse(match_matches_source({"page": 2, "section_path": "Overview"}, source))

    def test_retrieval_summary_separates_evidence_source_and_direct_answer(self):
        summary = summarize_retrieval(
            [
                {
                    "retrieval_pass": True,
                    "evidence_answer_recall": True,
                    "source_recall_at_k": True,
                    "reciprocal_rank": 1.0,
                    "has_direct_answer": True,
                    "direct_answer_correct": True,
                    "latency_seconds": 0.01,
                },
                {
                    "retrieval_pass": False,
                    "evidence_answer_recall": True,
                    "source_recall_at_k": False,
                    "reciprocal_rank": 0.0,
                    "has_direct_answer": False,
                    "direct_answer_correct": False,
                    "latency_seconds": 0.02,
                },
            ]
        )

        self.assertEqual(summary["retrieval_pass_rate"], 0.5)
        self.assertEqual(summary["evidence_answer_recall"], 1.0)
        self.assertEqual(summary["source_recall_at_k"], 0.5)
        self.assertEqual(summary["direct_answer_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
