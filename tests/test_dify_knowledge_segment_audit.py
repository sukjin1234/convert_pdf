import unittest

from scripts.dify_knowledge_segment_audit import (
    audit_document_segments,
    clean_metadata_text,
    evaluate_gates,
    extract_page,
    extract_section,
    outline_depth_inversion,
    section_issue_reasons,
    segment_issue_reasons,
    summarize_audits,
)


class DifyKnowledgeSegmentAuditTest(unittest.TestCase):
    def test_extracts_section_and_page_from_segment_markers(self):
        content = "[page: 5]\n[section: Ⅱ. 전형일정]\n본문"

        self.assertEqual(extract_page(content), "5")
        self.assertEqual(extract_section(content), "Ⅱ. 전형일정")

    def test_extracts_fallback_source_metadata(self):
        content = "source_page: 12\nsource_section: 제1장 개요\n본문"

        self.assertEqual(extract_page(content), "12")
        self.assertEqual(extract_section(content), "제1장 개요")

    def test_section_issue_reasons_flags_invalid_toc_and_inversion(self):
        self.assertIn("invalid_section", section_issue_reasons("Untitled"))
        self.assertIn("table_of_contents_section", section_issue_reasons("목   차"))
        self.assertIn("outline_depth_inversion", section_issue_reasons("1. 세부 > Ⅱ. 상위"))
        self.assertTrue(outline_depth_inversion("1. 세부 > Ⅱ. 상위"))

    def test_segment_issue_reasons_flags_control_and_missing_metadata(self):
        reasons = segment_issue_reasons("짧음\x00", "", "")

        self.assertIn("control_character", reasons)
        self.assertIn("missing_page", reasons)
        self.assertIn("invalid_section", reasons)
        self.assertIn("very_short_content", reasons)

    def test_audit_document_segments_summarizes_reason_counts(self):
        audit = audit_document_segments(
            {"id": "doc-1", "name": "guide.pdf", "display_status": "available", "enabled": True},
            [
                {
                    "id": "seg-1",
                    "content": "[page: 5]\n[section: Ⅱ. 전형일정]\n" + "정상 본문 " * 20,
                },
                {
                    "id": "seg-2",
                    "content": "[section: 목 차]\n본문\x00",
                },
            ],
        )

        self.assertEqual(audit["segment_count"], 2)
        self.assertEqual(audit["issue_count"], 1)
        self.assertEqual(audit["reason_counts"]["control_character"], 1)
        self.assertEqual(audit["reason_counts"]["missing_page"], 1)
        summary = summarize_audits([audit])
        self.assertEqual(summary["documents_checked"], 1)
        self.assertEqual(summary["segments_checked"], 2)
        self.assertEqual(summary["issue_count"], 1)
        self.assertEqual(summary["issue_rate"], 0.5)

    def test_evaluate_gates_flags_issue_and_request_thresholds(self):
        summary = {
            "issue_count": 3,
            "issue_rate": 0.15,
            "failed_requests": [{"document_id": "doc-1", "error": "HTTP 500"}],
        }

        failures = evaluate_gates(
            summary,
            fail_on_issues=True,
            max_issue_rate=0.1,
            max_issues=2,
            max_failed_requests=0,
        )

        self.assertEqual(
            failures,
            [
                "audit contains 3 issue(s) and 1 failed request(s)",
                "issue_rate 15.00% > 10.00%",
                "issue_count 3 > 2",
                "failed_requests 1 > 0",
            ],
        )

    def test_evaluate_gates_allows_disabled_thresholds(self):
        summary = {
            "issue_count": 3,
            "issue_rate": 0.15,
            "failed_requests": [{"document_id": "doc-1", "error": "HTTP 500"}],
        }

        failures = evaluate_gates(
            summary,
            fail_on_issues=False,
            max_issue_rate=-1,
            max_issues=-1,
            max_failed_requests=-1,
        )

        self.assertEqual(failures, [])

    def test_clean_metadata_text_removes_control_characters_and_normalizes_separator(self):
        self.assertEqual(clean_metadata_text("A\x00>B"), "A > B")


if __name__ == "__main__":
    unittest.main()
