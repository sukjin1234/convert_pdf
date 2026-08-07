import unittest
from unittest.mock import patch

from scripts.eval_rag_http import (
    TimedCaseResult,
    build_case_tasks,
    evaluate_case_http_timed,
    evaluate_latency_gates,
    percentile,
    summarize_latency,
)
from scripts.eval_rag_quality import CaseResult


def case_result(case_id: str, *, passed: bool = True) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        document_id="doc",
        question="question",
        answerable=True,
        passed=passed,
        query_type="date_lookup",
        expected_query_type="date_lookup",
        query_type_match=True,
        evidence_recall=passed,
        exact_value_match=passed,
        answerable_detection=passed,
        unsupported_guard=None,
        top_score=10,
        selected_count=1 if passed else 0,
        missing_terms=[],
        missing_values=[],
        notes=[],
    )


class RagHttpEvalTest(unittest.TestCase):
    def test_build_case_tasks_repeats_with_stable_indexes(self):
        tasks = build_case_tasks([{"id": "a"}, {"id": "b"}], repeat=2)

        self.assertEqual([index for index, _ in tasks], [1, 2, 3, 4])
        self.assertEqual([case["_iteration"] for _, case in tasks], [1, 1, 2, 2])

    def test_summarize_latency_reports_percentiles(self):
        results = [
            TimedCaseResult(1, case_result("a"), 0.1),
            TimedCaseResult(2, case_result("b"), 0.3),
            TimedCaseResult(3, case_result("c", passed=False), 1.0, error="HTTP 500"),
        ]

        summary = summarize_latency(results)

        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["successes"], 2)
        self.assertEqual(summary["avg_seconds"], 0.466667)
        self.assertEqual(summary["p50_seconds"], 0.3)
        self.assertEqual(summary["max_seconds"], 1.0)
        self.assertEqual(percentile([0.1, 0.3, 1.0], 95), 0.93)

    def test_evaluate_latency_gates_flags_avg_and_p95(self):
        failures = evaluate_latency_gates(
            {"avg_seconds": 1.2, "p95_seconds": 2.5},
            max_avg_latency=1.0,
            max_p95_latency=2.0,
        )

        self.assertEqual(
            failures,
            [
                "p95_seconds 2.500s > 2.000s",
                "avg_seconds 1.200s > 1.000s",
            ],
        )

    def test_evaluate_case_http_timed_captures_http_failure(self):
        with patch("scripts.eval_rag_http.evaluate_case_http", side_effect=RuntimeError("boom")):
            timed = evaluate_case_http_timed(
                1,
                {"id": "case-1", "document_id": "doc", "question": "question"},
                base_url="http://example.test",
                limit=8,
            )

        self.assertFalse(timed.result.passed)
        self.assertIn("RuntimeError: boom", timed.error)
        self.assertIn("RuntimeError: boom", timed.result.notes)


if __name__ == "__main__":
    unittest.main()
