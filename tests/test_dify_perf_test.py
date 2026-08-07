import unittest

from scripts.dify_perf_test import SendResult, percentile, summarize_results


class DifyPerfTestScriptTest(unittest.TestCase):
    def test_summarize_results_reports_latency_percentiles(self):
        results = [
            SendResult(1, 1, "q1", "first", True, 200, 1.0, "c1", ""),
            SendResult(2, 1, "q2", "second", True, 200, 2.0, "c2", ""),
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

    def test_percentile_handles_empty_and_single_value(self):
        self.assertEqual(percentile([], 95), 0)
        self.assertEqual(percentile([1.25], 95), 1.25)


if __name__ == "__main__":
    unittest.main()
