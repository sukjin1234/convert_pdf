import unittest

from app.pipeline_metrics import StageTimer
from scripts.evaluate_rag_system import (
    SegmentView,
    StructureAnchor,
    classify_node_stage,
    evaluate_answer,
    evaluate_citation,
    evaluate_structure,
    percentile,
    to_segment_view,
)


class RagSystemEvaluationTest(unittest.TestCase):
    def test_answer_schema_uses_alias_groups_without_domain_logic(self):
        correct, missing = evaluate_answer(
            "The retry window is 30 seconds and the owner is Platform.",
            {"all_of": [["30 seconds", "30s"], ["Platform", "SRE"]]},
        )

        self.assertTrue(correct)
        self.assertEqual(missing, [])

    def test_answer_schema_reports_missing_groups(self):
        correct, missing = evaluate_answer("The retry window is 30 seconds.", {"all_of": [["30 seconds"], ["Platform"]]})

        self.assertFalse(correct)
        self.assertEqual(missing, [["Platform"]])

    def test_numeric_alias_does_not_match_a_larger_number(self):
        correct, missing = evaluate_answer("This guide is for version 2027.", {"all_of": [["20"]]})

        self.assertFalse(correct)
        self.assertEqual(missing, [["20"]])

    def test_segment_view_extracts_metadata_before_whitespace_is_collapsed(self):
        segment = {
            "id": "segment-1",
            "content": (
                "[document_id: doc-1]\n"
                "[page: 12]\n"
                "[section: Operations > Retry policy]\n"
                "Retry attempts are limited to three."
            ),
        }

        view = to_segment_view(segment)

        self.assertEqual(view.page, 12)
        self.assertEqual(view.section, "Operations > Retry policy")

    def test_citation_checks_file_and_region_separately(self):
        correct, checks = evaluate_citation(
            "Result. Source: service-guide.pdf (p.12, Retry policy)",
            file_name="service-guide.pdf",
            source={"pages": [12], "section_any": [["Retry policy"]]},
        )

        self.assertTrue(correct)
        self.assertTrue(all(checks.values()))

    def test_structure_score_compares_source_anchors_pages_and_segment_boundaries(self):
        anchors = [
            StructureAnchor(2, "heading", "2. Retry policy"),
            StructureAnchor(2, "table", "Error Code / Resolution"),
        ]
        segments = [
            SegmentView("s1", 2, "2. Retry policy", "[page: 2] [section: 2. Retry policy] Retry policy details"),
            SegmentView("s2", 2, "2. Retry policy", "[page: 2] [section: 2. Retry policy] Error Code Resolution"),
        ]

        result = evaluate_structure(anchors, {2}, segments)

        self.assertEqual(result["boundary_accuracy"], 1.0)
        self.assertEqual(result["matched_anchor_count"], 2)

    def test_stage_classifier_and_percentile_are_generic(self):
        self.assertEqual(classify_node_stage("Knowledge Retrieval", "knowledge-retrieval"), "retrieval")
        self.assertEqual(classify_node_stage("Final Answer", "llm"), "generation")
        self.assertEqual(percentile([1, 2, 3, 100], 0.95), 100)

    def test_stage_timer_exposes_total_and_named_stages(self):
        timer = StageTimer()
        timer.mark("first")
        result = timer.snapshot(final_stage="second")

        self.assertIn("first", result)
        self.assertIn("second", result)
        self.assertIn("total", result)
        self.assertGreaterEqual(result["total"], 0)


if __name__ == "__main__":
    unittest.main()
