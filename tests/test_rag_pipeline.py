import unittest

from app.rag import build_rag_artifact, extract_keywords, lookup_matches, plan_query, verify_answer


class RagPipelineTest(unittest.TestCase):
    def test_builds_row_expanded_chunks_and_structured_records(self):
        markdown = """
--- Page 12 ---

# 전형별 모집인원

| 모집단위 | 전형 | 모집인원 |
| --- | --- | --- |
| 컴퓨터정보공학과 | 일반전형 | 45 |
| 기계공학과 | 일반전형 | 38 |
"""

        artifact = build_rag_artifact(markdown, "admission.pdf", document_id="doc_test")

        self.assertTrue(artifact.success)
        self.assertEqual(artifact.document_id, "doc_test")
        self.assertGreaterEqual(len(artifact.chunks), 1)
        self.assertEqual(len(artifact.records), 2)
        self.assertIn("[document_id: doc_test]", artifact.dify_markdown)
        self.assertIn("record_type: quantity", artifact.dify_markdown)
        self.assertIn("컴퓨터정보공학과", artifact.dify_markdown)
        self.assertIn("모집인원: 45", artifact.dify_markdown)

        record = artifact.records[0]
        self.assertEqual(record.record_type, "quantity")
        self.assertEqual(record.fields["모집단위"], "컴퓨터정보공학과")
        self.assertEqual(record.fields["모집인원"], "45")
        self.assertTrue(record.chunk_id)

    def test_plans_and_looks_up_number_question(self):
        artifact = build_rag_artifact(
            """
--- Page 12 ---

# 전형별 모집인원

| 모집단위 | 전형 | 모집인원 |
| --- | --- | --- |
| 컴퓨터정보공학과 | 일반전형 | 45 |
""",
            "admission.pdf",
            document_id="doc_test",
        )

        question = "컴퓨터정보공학과 모집인원 알려줘"
        plan = plan_query(question)
        result = lookup_matches(
            query=question,
            records=artifact.records,
            chunks=artifact.chunks,
            entities=plan["entities"],
            query_type=plan["query_type"],
            limit=3,
        )

        self.assertEqual(plan["query_type"], "number_lookup")
        self.assertGreaterEqual(len(result["matches"]), 1)
        self.assertIn("45", result["context"])

        direct_result = lookup_matches(query=question, records=artifact.records, chunks=artifact.chunks, limit=3)
        self.assertGreaterEqual(len(direct_result["matches"]), 1)
        self.assertIn("45", direct_result["context"])

        verified = verify_answer(question, "컴퓨터정보공학과 모집인원은 45명입니다.", [result["context"]])
        self.assertTrue(verified["valid"])

        rejected = verify_answer(question, "컴퓨터정보공학과 모집인원은 40명입니다.", [result["context"]])
        self.assertFalse(rejected["valid"])
        self.assertTrue(any(issue["type"] == "unsupported_number" for issue in rejected["issues"]))

    def test_generic_technical_query_is_not_admission_only(self):
        question = "Kafka Consumer Lag 모니터링 기준과 장애 대응 절차 알려줘"

        plan = plan_query(question)
        keywords = extract_keywords(question)

        self.assertEqual(plan["query_type"], "troubleshooting")
        self.assertIn("Kafka Consumer Lag", keywords)
        self.assertIn("절차", keywords)

    def test_technical_error_table_lookup(self):
        artifact = build_rag_artifact(
            """
--- Page 4 ---

# API 오류 코드

| Error Code | HTTP Status | Cause | Resolution |
| --- | --- | --- | --- |
| AUTH_401 | 401 | Missing token | Reissue API key |
| RATE_429 | 429 | Too many requests | Apply exponential backoff |
""",
            "api_manual.pdf",
            document_id="doc_api",
        )

        question = "AUTH_401 오류 해결 방법 알려줘"
        plan = plan_query(question)
        result = lookup_matches(
            query=question,
            records=artifact.records,
            chunks=artifact.chunks,
            entities=plan["entities"],
            query_type=plan["query_type"],
            limit=3,
        )

        self.assertEqual(plan["query_type"], "troubleshooting")
        self.assertEqual(artifact.records[0].record_type, "troubleshooting")
        self.assertEqual(artifact.records[0].fields["Error Code"], "AUTH_401")
        self.assertGreaterEqual(len(result["matches"]), 1)
        self.assertIn("AUTH_401", result["context"])
        self.assertIn("Error Code: AUTH_401", result["context"])
        self.assertIn("Resolution: Reissue API key", result["context"])
        self.assertIn("Reissue API key", result["context"])

    def test_sla_metric_table_lookup(self):
        artifact = build_rag_artifact(
            """
--- Page 2 ---

# Service Level Targets

| Service | Availability | Response Time | Error Budget |
| --- | --- | --- | --- |
| Billing API | 99.9% | 200ms | 0.1% |
| Search API | 99.5% | 350ms | 0.5% |
""",
            "sla.pdf",
            document_id="doc_sla",
        )

        question = "Billing API Availability 얼마야?"
        plan = plan_query(question)
        result = lookup_matches(
            query=question,
            records=artifact.records,
            chunks=artifact.chunks,
            entities=plan["entities"],
            query_type=plan["query_type"],
            limit=3,
        )

        self.assertEqual(plan["query_type"], "number_lookup")
        self.assertEqual(artifact.records[0].record_type, "metric")
        self.assertGreaterEqual(len(result["matches"]), 1)
        self.assertIn("99.9%", result["context"])

    def test_contact_owner_table_lookup(self):
        artifact = build_rag_artifact(
            """
--- Page 7 ---

# 운영 담당자

| System | Owner Team | Contact Email |
| --- | --- | --- |
| Payment Gateway | FinOps | finops@example.com |
| Search Cluster | Platform | platform@example.com |
""",
            "ops_runbook.pdf",
            document_id="doc_ops",
        )

        question = "Payment Gateway 담당 팀 이메일 알려줘"
        plan = plan_query(question)
        result = lookup_matches(
            query=question,
            records=artifact.records,
            chunks=artifact.chunks,
            entities=plan["entities"],
            query_type=plan["query_type"],
            limit=3,
        )

        self.assertEqual(plan["query_type"], "contact_lookup")
        self.assertEqual(artifact.records[0].record_type, "contact")
        self.assertGreaterEqual(len(result["matches"]), 1)
        self.assertIn("finops@example.com", result["context"])

    def test_plan_exposes_sub_queries_and_answer_style(self):
        plan = plan_query("Billing API와 Search API 차이 비교해줘")

        self.assertEqual(plan["query_type"], "comparison")
        self.assertEqual(plan["answer_style"], "compare")
        self.assertIn("Billing API", plan["sub_queries"])
        self.assertIn("Search API 차이 비교해줘", plan["sub_queries"])
        self.assertNotIn("API와", plan["keywords"])

    def test_list_style_question_uses_table_lookup(self):
        plan = plan_query("전공심화 개설 학과 알려줘")

        self.assertEqual(plan["query_type"], "table_lookup")
        self.assertEqual(plan["answer_style"], "table_or_bullets")
        self.assertIn("전공심화", plan["keywords"])
        self.assertTrue(plan["retrieval_hints"]["prefer_structured_lookup"])

    def test_lookup_packs_supporting_context_for_record_matches(self):
        artifact = build_rag_artifact(
            """
--- Page 3 ---

# 장애 대응

API 키 교체 후에는 감사 로그에서 발급자를 확인해야 한다.

| Error Code | Cause | Resolution |
| --- | --- | --- |
| AUTH_401 | Missing token | Reissue API key |
""",
            "runbook.pdf",
            document_id="doc_runbook",
        )

        result = lookup_matches(
            query="AUTH_401 해결 후 확인할 것",
            records=artifact.records,
            chunks=artifact.chunks,
            limit=2,
        )

        self.assertGreaterEqual(len(result["matches"]), 1)
        self.assertIn("supporting_context", result["context"])
        self.assertIn("감사 로그", result["context"])
        self.assertGreater(result["diagnostics"]["average_coverage"], 0)

    def test_verify_rejects_unsupported_identifier(self):
        result = verify_answer(
            "AUTH_401 해결 방법 알려줘",
            "AUTH_999는 API 키를 재발급하면 됩니다.",
            ["Error Code: AUTH_401\nResolution: Reissue API key"],
        )

        self.assertFalse(result["valid"])
        self.assertIn("AUTH_999", result["answer_identifiers"])
        self.assertTrue(any(issue["type"] == "unsupported_identifier" for issue in result["issues"]))

    def test_verify_rejects_abstention_when_structured_lookup_has_answer(self):
        result = verify_answer(
            "전공심화 개설 학과 알려줘",
            "제공된 문서에는 충분한 근거가 없습니다.",
            [
                "[Structured Lookup - authoritative exact evidence]\n"
                "[match_type: structured_record]\n"
                "[section: 전공심화과정 개설 학과]\n"
                "answer_candidate: 컴퓨터정보공학과, 기계공학과\n"
                "evidence:\n"
                "과정: 전공심화\n"
                "개설 학과: 컴퓨터정보공학과, 기계공학과"
            ],
        )

        self.assertFalse(result["valid"])
        self.assertTrue(any(issue["type"] == "unnecessary_abstention" for issue in result["issues"]))

    def test_lookup_rejects_missing_specific_attribute(self):
        artifact = build_rag_artifact(
            """
--- Page 1 ---

# Product and Pricing Manual

| Plan | Monthly Price | Included Users | Support SLA |
| --- | --- | --- | --- |
| Basic | $29 | 5 users | 2 business days |
| Enterprise | Custom | Unlimited | 2 hours |
""",
            "product_manual.pdf",
            document_id="doc_product",
        )

        result = lookup_matches(
            query="Enterprise 저장공간 한도는 얼마야?",
            records=artifact.records,
            chunks=artifact.chunks,
            limit=5,
        )

        self.assertEqual(result["matches"], [])
        self.assertEqual(result["context"], "")
        self.assertFalse(result["diagnostics"]["answerability"]["answerable"])
        self.assertIn("storage", result["diagnostics"]["answerability"]["missing_strict_concepts"])


if __name__ == "__main__":
    unittest.main()
