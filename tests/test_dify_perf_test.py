import unittest
from unittest.mock import patch

from scripts.dify_perf_test import SendResult, drain_sse_response, percentile, summarize_results


class DifyPerfTestScriptTest(unittest.TestCase):
    def test_summarize_results_reports_latency_percentiles(self):
        results = [
            SendResult(1, 1, "q1", "first", True, 200, 1.0, "c1", "", time_to_first_token_seconds=0.4, total_tokens=100),
            SendResult(2, 1, "q2", "second", True, 200, 2.0, "c2", "", time_to_first_token_seconds=0.8, total_tokens=200),
            SendResult(3, 1, "q3", "third", False, 500, 10.0, "", "server error"),
        ]

        summary = summarize_results(results, elapsed_seconds=4.0)

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["success"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["success_rate"], 0.6667)
        self.assertEqual(summary["latency_avg_seconds"], 1.5)
        self.assertEqual(summary["latency_p50_seconds"], 1.5)
        self.assertEqual(summary["latency_max_seconds"], 2.0)
        self.assertEqual(summary["throughput_requests_per_second"], 0.75)
        self.assertEqual(summary["ttft_avg_seconds"], 0.6)
        self.assertEqual(summary["ttft_p50_seconds"], 0.6)
        self.assertEqual(summary["total_tokens"], 300)
        self.assertEqual(summary["tokens_avg"], 150.0)

    def test_percentile_handles_empty_and_single_value(self):
        self.assertEqual(percentile([], 95), 0)
        self.assertEqual(percentile([1.25], 95), 1.25)

    def test_drain_sse_response_extracts_stream_timing_and_usage(self):
        lines = [
            b'data: {"event":"workflow_started","workflow_run_id":"run-1"}\n',
            b'data: {"event":"message","conversation_id":"conv-1","answer":"hello"}\n',
            b'data: {"event":"message_end","metadata":{"usage":{"total_tokens":42}}}\n',
        ]
        with patch("scripts.dify_perf_test.time.perf_counter", side_effect=[10.75]):
            stats = drain_sse_response(lines, started_at=10.0)

        self.assertEqual(stats.conversation_id, "conv-1")
        self.assertEqual(stats.workflow_run_id, "run-1")
        self.assertEqual(stats.time_to_first_token_seconds, 0.75)
        self.assertEqual(stats.total_tokens, 42)


if __name__ == "__main__":
    unittest.main()
