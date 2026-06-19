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

    def test_table_list_lookup_builds_complete_direct_answer(self):
        artifact = build_rag_artifact(
            """
--- Page 4 ---

# 전공심화 개설 학과

| 과정 | 개설 학과 | 수업 방식 |
| --- | --- | --- |
| 전공심화 | 컴퓨터정보공학과 | 야간 |
| 전공심화 | 기계공학과 | 야간 |
| 전공심화 | 간호학과 | 주말 |
| 일반학위 | 산업디자인학과 | 주간 |
""",
            "admission.pdf",
            document_id="doc_major_intensive",
        )

        result = lookup_matches(
            query="전공심화 개설 학과 알려줘",
            records=artifact.records,
            chunks=artifact.chunks,
            limit=8,
        )

        self.assertEqual(result["answer_field"], "개설 학과")
        self.assertIn("컴퓨터정보공학과", result["direct_answer"])
        self.assertIn("기계공학과", result["direct_answer"])
        self.assertIn("간호학과", result["direct_answer"])
        self.assertNotIn("산업디자인학과", result["direct_answer"])
        self.assertIn("[Direct Answer - complete structured result]", result["context"])
        self.assertIn("complete_values:", result["context"])

        incomplete = verify_answer(
            "전공심화 개설 학과 알려줘",
            "전공심화 개설 학과는 컴퓨터정보공학과와 기계공학과입니다.",
            [result["context"]],
        )
        self.assertFalse(incomplete["valid"])
        self.assertTrue(any(issue["type"] == "missing_direct_answer_value" for issue in incomplete["issues"]))

    def test_marker_legend_rows_build_complete_direct_answer(self):
        artifact = build_rag_artifact(
            """
--- Page 26 ---

# 모집인원

| 계열 | 모집단위 | 수업연한 | 모집정원 |
| --- | --- | --- | --- |
| 공학 | 로봇자동화공학과 ♣ | 3 | 80 |
| 공학 | 반도체기계정비학과 | 2 | 60 |
| 공학 | 자동차공학과 ♣ | 2 | 90 |
| 공학 | 컴퓨터정보공학과 ♣ | 3 | 93 |
| 예체능 | 산업디자인학과 | 2 | 75 |

※ ♣표시 : 4년제 학사학위(전공심화)과정 개설 학과
""",
            "admission.pdf",
            document_id="doc_marker_admission",
        )

        result = lookup_matches(
            query="전공심화 개설 학과 알려줘",
            records=artifact.records,
            chunks=artifact.chunks,
            limit=10,
        )

        self.assertEqual(result["answer_field"], "모집단위")
        self.assertIn("로봇자동화공학과", result["direct_answer"])
        self.assertIn("자동차공학과", result["direct_answer"])
        self.assertIn("컴퓨터정보공학과", result["direct_answer"])
        self.assertNotIn("반도체기계정비학과", result["direct_answer"])
        self.assertNotIn("산업디자인학과", result["direct_answer"])
        self.assertIn("표시 의미", result["context"])
        self.assertIn("complete_values:", result["context"])

    def test_marker_legend_logic_is_domain_neutral(self):
        artifact = build_rag_artifact(
            """
--- Page 2 ---

# API Catalog

| API | Version | Owner |
| --- | --- | --- |
| Legacy Export API † | v1.4 | Data Platform |
| Bulk Export API | v2 | Data Platform |
| Classic Dashboard † | v2.1 | Analytics |

† indicates deprecated APIs
""",
            "api_catalog.pdf",
            document_id="doc_marker_api",
        )

        result = lookup_matches(
            query="deprecated APIs 목록 알려줘",
            records=artifact.records,
            chunks=artifact.chunks,
            limit=10,
        )

        self.assertEqual(result["answer_field"], "API")
        self.assertIn("Legacy Export API", result["direct_answer"])
        self.assertIn("Classic Dashboard", result["direct_answer"])
        self.assertNotIn("Bulk Export API", result["direct_answer"])

    def test_plain_text_marker_rows_are_structured_without_markdown_table(self):
        artifact = build_rag_artifact(
            """
--- Page 26 ---

# 모집인원

계열 모집단위 수업연한 모집정원
공학
자유전공학부 2 30
로봇자동화공학과 ♣ 3 80
반도체기계정비학과 2 60
자동차공학과 ♣ 2 90
전기공학과 ♣ 2 90
전자공학과 ♣ 2 90
정보통신공학과 ♣ 2 120
컴퓨터정보공학과 ♣ 3 93
컴퓨터시스템공학과 ♣ 3 93
디지털마케팅공학과 2 60
건설환경공학과 ♣ 2 90
공간정보빅데이터학과 ♣ 3 80
건축학과 ♣ 3 92
실내건축학과 ♣ 3 92
예체능
산업디자인학과 2 75
인문사회
항공운항과 ♣ 2 160
항공경영학과 ♣ 2 92
관광경영학과 ♣ 2 87
경영비서학과 ♣ 2 87
호텔경영학과 2 87

※ ♣표시 : 4년제 학사학위(전공심화)과정 개설 학과
""",
            "admission_plain_text.pdf",
            document_id="doc_plain_marker_admission",
        )

        result = lookup_matches(
            query="전공심화 개설 학과 알려줘",
            records=artifact.records,
            chunks=artifact.chunks,
            limit=30,
        )

        expected = [
            "로봇자동화공학과",
            "자동차공학과",
            "전기공학과",
            "전자공학과",
            "정보통신공학과",
            "컴퓨터정보공학과",
            "컴퓨터시스템공학과",
            "건설환경공학과",
            "공간정보빅데이터학과",
            "건축학과",
            "실내건축학과",
            "항공운항과",
            "항공경영학과",
            "관광경영학과",
            "경영비서학과",
        ]
        self.assertEqual(result["answer_field"], "항목")
        for department in expected:
            self.assertIn(department, result["direct_answer"])
        self.assertNotIn("자유전공학부", result["direct_answer"])
        self.assertNotIn("반도체기계정비학과", result["direct_answer"])
        self.assertNotIn("산업디자인학과", result["direct_answer"])
        self.assertNotIn("호텔경영학과", result["direct_answer"])

    def test_multilevel_header_table_supports_exact_column_lookup(self):
        artifact = build_rag_artifact(
            """
--- Page 26 ---

# 모집인원

| 계열 | 모집단위 | 수업 연한 | 모집 정원 | 수시1차<br>특기자 |  |  | 수시2차<br>특기자 |  |  | 정시 일반 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  | 일반고 | 특성화고 | (어학) | 일반고 | 특성화고 | (어학) | (수능) |
| 공학 | 항공기계공학과 | 3 | 120 | 50 | 21 | 13 | 18 | 8 | 4 | 6 |
| 공학 | 정보통신공학과 | 2 | 120 | 67 | 17 |  | 24 | 6 |  | 6 |
| 공학 | 컴퓨터정보공학과 | 3 | 93 | 53 | 13 |  | 19 | 4 |  | 4 |
| 공학 | 화학생명공학과 | 2 | 120 | 67 | 17 |  | 24 | 6 |  | 6 |
""",
            "admission_multilevel.pdf",
            document_id="doc_multilevel_admission",
        )

        special = lookup_matches(
            query="컴퓨터정보공학과 수시1차 특성화고 모집인원은?",
            records=artifact.records,
            chunks=artifact.chunks,
            limit=8,
        )
        self.assertEqual(special["answer_field"], "수시1차 특기자 특성화고")
        self.assertIn("13", special["direct_answer"])
        self.assertNotIn("53", special["direct_answer"])

        inverse = lookup_matches(
            query="모집정원이 120명인 학과를 알려줘",
            records=artifact.records,
            chunks=artifact.chunks,
            limit=8,
        )
        self.assertEqual(inverse["answer_field"], "모집단위")
        self.assertIn("항공기계공학과", inverse["direct_answer"])
        self.assertIn("정보통신공학과", inverse["direct_answer"])
        self.assertIn("화학생명공학과", inverse["direct_answer"])
        self.assertNotIn("컴퓨터정보공학과", inverse["direct_answer"])

    def test_verify_rejects_inverse_lookup_answer_that_drops_query_value(self):
        result = verify_answer(
            "모집정원이 120명인 학과를 알려줘",
            "로봇자동화공학과의 모집정원은 80명입니다.",
            [
                "[Structured Lookup - authoritative exact evidence]\n"
                "모집단위: 항공기계공학과\n모집 정원: 120\n"
                "모집단위: 로봇자동화공학과\n모집 정원: 80"
            ],
        )

        self.assertFalse(result["valid"])
        self.assertTrue(any(issue["type"] == "missing_query_constraint_value" for issue in result["issues"]))

    def test_verify_rejects_answer_for_different_named_entity(self):
        result = verify_answer(
            "정보통신공학과 모집정원은 몇 명이야?",
            "컴퓨터정보공학과의 모집정원은 93명입니다.",
            [
                "[Structured Lookup - authoritative exact evidence]\n"
                "모집단위: 정보통신공학과\n모집 정원: 120\n"
                "모집단위: 컴퓨터정보공학과\n모집 정원: 93"
            ],
        )

        self.assertFalse(result["valid"])
        self.assertTrue(any(issue["type"] == "missing_question_entity" for issue in result["issues"]))

    def test_inverse_value_lookup_is_domain_neutral(self):
        artifact = build_rag_artifact(
            """
--- Page 1 ---

# Mixed Operations Manual

## Error Catalog

| Error Code | HTTP Status | Cause | Resolution |
| --- | --- | --- | --- |
| AUTH_401 | 401 | Missing token | Reissue API key |
| PAY_503 | 503 | Payment gateway unavailable | Fail over to backup gateway |

## Data Retention

| Data Type | Retention Period | Storage Location |
| --- | --- | --- |
| Audit Logs | 3 years | Secure Archive |
| Session Records | 180 days | Analytics Store |

## Plans

| Plan | Monthly Price | Included Users | Support SLA |
| --- | --- | --- | --- |
| Basic | $29 | 5 users | 2 business days |
| Enterprise | Custom | Unlimited | 2 hours |
""",
            "mixed_manual.pdf",
            document_id="doc_mixed_manual",
        )

        error = lookup_matches(
            query="HTTP Status가 503인 Error Code 알려줘",
            records=artifact.records,
            chunks=artifact.chunks,
            limit=8,
        )
        self.assertEqual(error["answer_field"], "Error Code")
        self.assertIn("PAY_503", error["direct_answer"])
        self.assertNotIn("AUTH_401", error["direct_answer"])

        retention = lookup_matches(
            query="Retention Period가 3 years인 Data Type 알려줘",
            records=artifact.records,
            chunks=artifact.chunks,
            limit=8,
        )
        self.assertEqual(retention["answer_field"], "Data Type")
        self.assertIn("Audit Logs", retention["direct_answer"])
        self.assertNotIn("Session Records", retention["direct_answer"])

        price = lookup_matches(
            query="Monthly Price가 $29인 Plan 알려줘",
            records=artifact.records,
            chunks=artifact.chunks,
            limit=8,
        )
        self.assertEqual(price["answer_field"], "Plan")
        self.assertIn("Basic", price["direct_answer"])
        self.assertNotIn("Enterprise", price["direct_answer"])

        sla = lookup_matches(
            query="Support SLA가 2 hours인 Plan 알려줘",
            records=artifact.records,
            chunks=artifact.chunks,
            limit=8,
        )
        self.assertEqual(sla["answer_field"], "Plan")
        self.assertIn("Enterprise", sla["direct_answer"])
        self.assertNotIn("Basic", sla["direct_answer"])

    def test_boolean_matrix_lookup_builds_supported_value_list(self):
        artifact = build_rag_artifact(
            """
--- Page 3 ---

# 기능 비교

| Feature | Basic | Pro | Enterprise |
| --- | --- | --- | --- |
| Audit Log | No | Yes | Yes |
| SSO | No | Yes | Yes |
| Dedicated CSM | No | No | Yes |
""",
            "product_manual.pdf",
            document_id="doc_feature_matrix",
        )

        result = lookup_matches(
            query="SSO는 어떤 요금제에서 지원돼?",
            records=artifact.records,
            chunks=artifact.chunks,
            limit=8,
        )

        self.assertEqual(result["answer_field"], "supported_fields")
        self.assertIn("Pro", result["direct_answer"])
        self.assertIn("Enterprise", result["direct_answer"])
        self.assertNotIn("Basic", result["direct_answer"])

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
