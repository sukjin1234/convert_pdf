import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.dify_app_api_diagnostics import (
    deployment_contract_status,
    flatten_stream_event,
    has_document_id_key,
    load_inputs,
    node_response_body_chars,
    node_timing_summary,
    parse_json_object,
    parse_sse_events,
    summarize_stream_events,
)


class DifyAppApiDiagnosticsTests(unittest.TestCase):
    def test_parse_sse_events_skips_ping_and_flattens_data_events(self):
        events = parse_sse_events(
            [
                "event: ping\n",
                "\n",
                'data: {"event":"workflow_started","data":{"inputs":{"sys.query":"hello"}}}\n',
                'data: {"event":"node_finished","data":{"title":"Query Plan","outputs":{"body":"{}"}}}\n',
            ]
        )

        self.assertEqual([event["event"] for event in events], ["workflow_started", "node_finished"])
        self.assertEqual(flatten_stream_event(events[0])["inputs"], {"sys.query": "hello"})

    def test_summarize_stream_events_extracts_rag_signals(self):
        events = [
            {
                "event": "workflow_started",
                "workflow_run_id": "run-1",
                "inputs": {"sys.query": "hello", "document_id": "doc-1", "sys.app_id": "app-1"},
            },
            {
                "event": "node_finished",
                "title": "Query Plan",
                "status": "succeeded",
                "outputs": {"body": '{"query":"hello","document_id":"doc-1"}'},
            },
            {
                "event": "node_finished",
                "title": "Structured Lookup",
                "status": "succeeded",
                "outputs": {
                    "body": (
                        '{"direct_answer":"Answer: 42",'
                        '"answer_items":[{"value":"42","document_id":"doc-1"}]}'
                    )
                },
            },
            {
                "event": "node_finished",
                "title": "Verify Answer",
                "status": "succeeded",
                "outputs": {"body": '{"valid":true,"issues":[]}'},
            },
            {"event": "message", "answer": "Answer"},
            {"event": "message", "answer": ": 42"},
            {
                "event": "message_end",
                "metadata": {"usage": {"total_tokens": 10}, "retriever_resources": [{"id": "r1"}]},
            },
        ]

        summary = summarize_stream_events(events)

        self.assertEqual(summary["workflow_run_id"], "run-1")
        self.assertTrue(summary["document_id_in_workflow_inputs"])
        self.assertEqual(summary["query_plan_document_id"], "doc-1")
        self.assertEqual(summary["structured_direct_answer"], "Answer: 42")
        self.assertTrue(summary["verify_valid"])
        self.assertEqual(summary["streamed_answer"], "Answer: 42")
        self.assertEqual(summary["retriever_resource_count"], 1)

    def test_document_id_key_detection_accepts_system_scoped_key(self):
        self.assertTrue(has_document_id_key({"sys.document_id": "doc-1"}))
        self.assertFalse(has_document_id_key({"document_id": ""}))

    def test_load_inputs_can_read_json_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "inputs.json"
            path.write_text('{"document_id":"doc-1"}', encoding="utf-8")

            self.assertEqual(load_inputs("{}", str(path)), {"document_id": "doc-1"})

    def test_parse_json_object_requires_object(self):
        with self.assertRaises(ValueError):
            parse_json_object("[]")

    def test_current_deployment_contract_is_ready(self):
        status = deployment_contract_status(
            {
                "knowledge_query": "원서접수",
                "knowledge_queries": ["원서접수", "수시1차"],
            },
            {
                "matches": [{"evidence": "수시1차: 2026. 9. 7."}],
                "diagnostics": {"prefilter_limit": 160, "hydrated_candidate_count": 12},
            },
        )

        self.assertTrue(status["ready"])
        self.assertEqual(status["missing"], [])

    def test_legacy_deployment_contract_reports_missing_features(self):
        status = deployment_contract_status(
            {"expanded_queries": ["긴 자연어 질문"]},
            {
                "matches": [{"chunk_text": "duplicated", "evidence": "value"}],
                "diagnostics": {"candidate_count": 1000},
            },
        )

        self.assertFalse(status["ready"])
        self.assertEqual(
            status["missing"],
            ["knowledge_query", "knowledge_queries", "lookup_prefilter", "compact_matches"],
        )

    def test_node_timing_and_response_size_are_extracted(self):
        events = [
            {
                "event": "node_finished",
                "title": "Structured Lookup",
                "node_type": "http-request",
                "status": "succeeded",
                "elapsed_time": 1.23456789,
                "outputs": {"body": "12345"},
            },
            {"event": "message", "answer": "done"},
        ]

        self.assertEqual(node_response_body_chars(events, "Structured Lookup"), 5)
        self.assertEqual(
            node_timing_summary(events),
            [
                {
                    "title": "Structured Lookup",
                    "node_type": "http-request",
                    "status": "succeeded",
                    "elapsed_seconds": 1.234568,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
