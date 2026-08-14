import asyncio
import json
import os
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import app.main as main_module
from fastapi import UploadFile
from fastapi.testclient import TestClient

from app.eval_logs import EvalLogStore
from app.main import (
    LookupRequest,
    app,
    ChatflowDebugRequest,
    ConvertResponse,
    MarkdownIngestRequest,
    chatflow_debug,
    convert,
    convert_batch,
    convert_rag,
    rag_ingest_markdown,
)
from app.rag import RagStore, build_rag_artifact


class ApiContractTest(unittest.IsolatedAsyncioTestCase):
    def test_convert_response_has_only_success_and_markdown(self):
        response = ConvertResponse(success=True, markdown="# Done")
        payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()

        self.assertEqual(set(payload), {"success", "markdown"})

    async def test_convert_uses_uploaded_pdf_parameter(self):
        content = b"%PDF-1.4\n%%EOF"
        pdf = UploadFile(file=BytesIO(content), filename="policy.pdf")

        with patch("app.main.DocumentConverter") as converter_class:
            converter_class.return_value.convert_bytes.return_value = "# Converted"

            response = await convert(pdf)

        self.assertTrue(response.success)
        self.assertEqual(response.markdown, "# Converted")
        converter_class.return_value.convert_bytes.assert_called_once_with(content, "policy.pdf")

    async def test_convert_uses_generic_file_parameter(self):
        content = "# Manual\n\nAllowed domains: all".encode("utf-8")
        upload = UploadFile(file=BytesIO(content), filename="manual.md")

        with patch("app.main.DocumentConverter") as converter_class:
            converter_class.return_value.convert_bytes.return_value = "# Manual"

            response = await convert(file=upload)

        self.assertTrue(response.success)
        self.assertEqual(response.markdown, "# Manual")
        converter_class.return_value.convert_bytes.assert_called_once_with(content, "manual.md")

    async def test_convert_requests_are_serialized(self):
        first = UploadFile(file=BytesIO(b"# First"), filename="first.md")
        second = UploadFile(file=BytesIO(b"# Second"), filename="second.md")
        active = 0
        max_active = 0
        order: list[str] = []
        counter_lock = threading.Lock()

        def fake_convert(_content: bytes, name: str) -> str:
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with counter_lock:
                active -= 1
                order.append(name)
            return f"# {name}"

        with patch("app.main.DocumentConverter") as converter_class:
            converter_class.return_value.convert_bytes.side_effect = fake_convert
            first_task = asyncio.create_task(convert(file=first))
            await asyncio.sleep(0.01)
            second_task = asyncio.create_task(convert(file=second))
            responses = await asyncio.gather(first_task, second_task)

        self.assertTrue(all(response.success for response in responses))
        self.assertEqual(max_active, 1)
        self.assertEqual(order, ["first.md", "second.md"])

    async def test_convert_batch_processes_files_in_order(self):
        first = UploadFile(file=BytesIO(b"# First"), filename="first.md")
        second = UploadFile(file=BytesIO(b"# Second"), filename="second.md")
        order: list[str] = []

        def fake_convert(_content: bytes, name: str) -> str:
            order.append(name)
            return f"# {name}"

        with patch("app.main.DocumentConverter") as converter_class:
            converter_class.return_value.convert_bytes.side_effect = fake_convert
            response = await convert_batch(files=[first, second])

        self.assertTrue(response.success)
        self.assertEqual([item.file_name for item in response.files], ["first.md", "second.md"])
        self.assertEqual(order, ["first.md", "second.md"])

    async def test_rag_ingest_markdown_builds_artifact(self):
        response = await rag_ingest_markdown(
            MarkdownIngestRequest(
                document_id="doc_markdown_test",
                file_name="manual.md",
                markdown="""
--- Page 1 ---

# Manual

| Item | Value |
| --- | --- |
| API Limit | 1000 req/s |
""",
            )
        )

        self.assertTrue(response.success)
        self.assertEqual(response.document_id, "doc_markdown_test")
        self.assertGreaterEqual(len(response.chunks), 1)
        self.assertGreaterEqual(len(response.records), 1)
        self.assertIn("API Limit", response.dify_markdown)

    async def test_convert_rag_accepts_markdown_file_upload(self):
        content = b"# Manual\n\n| Item | Value |\n| --- | --- |\n| SLA | 99.9% |\n"
        upload = UploadFile(file=BytesIO(content), filename="manual.md")

        with TemporaryDirectory() as temp_dir:
            store = RagStore(Path(temp_dir))
            with patch("app.main.STORE", store):
                response = await convert_rag(file=upload, document_id="doc_manual")

        self.assertTrue(response.success)
        self.assertEqual(response.document_id, "doc_manual")
        self.assertEqual(response.file_name, "manual.md")
        self.assertIn("SLA", response.dify_markdown)
        self.assertGreaterEqual(len(response.records), 1)

    def test_rag_store_falls_back_when_configured_store_is_not_writable(self):
        artifact = build_rag_artifact(
            "--- Page 1 ---\n\n# Manual\n\n| Item | Value |\n| --- | --- |\n| SLA | 99.9% |",
            "manual.md",
            document_id="doc_store_fallback",
        )

        with TemporaryDirectory() as temp_dir:
            blocked_dir = Path(temp_dir) / "blocked"
            runtime_dir = Path(temp_dir) / "runtime"

            def fake_assert_writable(path: Path):
                if path == blocked_dir:
                    raise PermissionError("denied")

            with (
                patch.dict(os.environ, {"RAG_STORE_DIR": str(blocked_dir), "DIFY_RUNTIME_DIR": str(runtime_dir)}),
                patch("app.rag._assert_writable_dir", side_effect=fake_assert_writable),
            ):
                store = RagStore()
                store.upsert(artifact)

            stored_path = runtime_dir / "rag-store" / "doc_store_fallback.json"
            self.assertEqual(store.store_dir, runtime_dir / "rag-store")
            self.assertTrue(stored_path.exists())
            document_ids = {item["document_id"] for item in store.list_documents()}
            self.assertIn("doc_store_fallback", document_ids)

    def test_rag_store_writes_artifact_atomically(self):
        artifact = build_rag_artifact(
            "--- Page 1 ---\n\n# Manual\n\n| Item | Value |\n| --- | --- |\n| SLA | 99.9% |",
            "manual.md",
            document_id="doc_atomic_store",
        )

        with TemporaryDirectory() as temp_dir:
            store_dir = Path(temp_dir)
            store = RagStore(store_dir)
            store.upsert(artifact)

            stored_path = store_dir / "doc_atomic_store.json"
            self.assertTrue(stored_path.exists())
            self.assertFalse(list(store_dir.glob("*.tmp")))
            self.assertNotIn("\n", stored_path.read_text(encoding="utf-8"))

            reloaded = RagStore(store_dir)
            document_ids = {item["document_id"] for item in reloaded.list_documents()}
            self.assertIn("doc_atomic_store", document_ids)

    def test_rag_store_refreshes_artifacts_written_by_another_worker(self):
        first = build_rag_artifact(
            "--- Page 1 ---\n\n# Manual\n\n| Item | Value |\n| --- | --- |\n| SLA | 99.9% |",
            "manual.md",
            document_id="doc_shared_store",
        )
        second = build_rag_artifact(
            "--- Page 1 ---\n\n# Manual\n\n| Item | Value |\n| --- | --- |\n| SLA | 99.95% |",
            "manual.md",
            document_id="doc_shared_store",
        )

        with TemporaryDirectory() as temp_dir:
            store_dir = Path(temp_dir)
            reader = RagStore(store_dir, refresh_interval_seconds=0)
            writer = RagStore(store_dir, refresh_interval_seconds=0)

            self.assertEqual(reader.records("doc_shared_store"), [])
            initial_token = reader.cache_token("doc_shared_store")
            writer.upsert(first)
            self.assertIn("99.9", reader.records("doc_shared_store")[0].answer_text)
            first_token = reader.cache_token("doc_shared_store")

            writer.upsert(second)
            self.assertIn("99.95", reader.records("doc_shared_store")[0].answer_text)
            second_token = reader.cache_token("doc_shared_store")

            self.assertNotEqual(initial_token, first_token)
            self.assertNotEqual(first_token, second_token)

    def test_rag_store_migrates_legacy_control_characters_on_load(self):
        artifact = build_rag_artifact(
            "--- Page 1 ---\n\n# Manual\n\n| Item | Value |\n| --- | --- |\n| SLA | 99.9% |",
            "manual.md",
            document_id="doc_legacy_control",
        )

        with TemporaryDirectory() as temp_dir:
            store_dir = Path(temp_dir)
            payload = artifact.to_dict()
            payload["markdown"] = "# Manual\x00\n\n| Item | Value |"
            payload["dify_markdown"] = payload["dify_markdown"].replace("SLA", "S\x00LA")
            payload["dify_markdown"] = payload["dify_markdown"].replace(
                "[section: Manual]",
                "[section: Guide\x00 > 2. 정원외 전형 모집인원 > Ⅱ. 전형일정]",
            )
            payload["dify_markdown"] = payload["dify_markdown"].replace(
                "source_section: Manual",
                "source_section: Guide\x00 > 2. 정원외 전형 모집인원 > Ⅱ. 전형일정",
            )
            payload["chunks"][0]["section_path"] = "Guide\x00 > 2. 정원외 전형 모집인원 > Ⅱ. 전형일정"
            payload["chunks"][0]["text"] = payload["chunks"][0]["text"].replace("SLA", "S\x00LA")
            payload["records"][0]["section_path"] = "Guide\x00 > 2. 정원외 전형 모집인원 > Ⅱ. 전형일정"
            payload["records"][0]["fields"] = {"I\x00tem": "S\x00LA", "Value": "99.9\x00%"}
            payload["records"][0]["answer_text"] = "S\x00LA의 Value은/는 99.9\x00%입니다."
            payload["document_map"][0]["section_path"] = "Guide\x00 > 2. 정원외 전형 모집인원 > Ⅱ. 전형일정"

            stored_path = store_dir / "doc_legacy_control.json"
            stored_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            reloaded = RagStore(store_dir)
            records = reloaded.records("doc_legacy_control")
            chunks = reloaded.chunks("doc_legacy_control")

            self.assertTrue(records)
            self.assertTrue(chunks)
            joined = "\n".join(
                [
                    chunks[0].section_path,
                    chunks[0].text,
                    records[0].answer_text,
                    *records[0].fields.keys(),
                    *records[0].fields.values(),
                ]
            )
            self.assertNotIn("\x00", joined)
            self.assertEqual(chunks[0].section_path, "Guide > Ⅱ. 전형일정")
            self.assertEqual(records[0].section_path, "Guide > Ⅱ. 전형일정")
            self.assertEqual(records[0].fields["Item"], "SLA")
            migrated_text = stored_path.read_text(encoding="utf-8")
            self.assertNotIn("\\u0000", migrated_text)
            self.assertIn("[section: Guide > Ⅱ. 전형일정]", migrated_text)
            self.assertIn("source_section: Guide > Ⅱ. 전형일정", migrated_text)
            self.assertNotIn("source_section: Guide > 2. 정원외 전형 모집인원", migrated_text)

    def test_lookup_cache_reuses_results_until_store_changes(self):
        first = build_rag_artifact(
            "--- Page 1 ---\n\n# Manual\n\n| Item | Value |\n| --- | --- |\n| SLA | 99.9% |",
            "manual.md",
            document_id="doc_cached",
        )
        second = build_rag_artifact(
            "--- Page 1 ---\n\n# Manual\n\n| Item | Value |\n| --- | --- |\n| SLA | 99.95% |",
            "manual.md",
            document_id="doc_cached",
        )

        with TemporaryDirectory() as temp_dir:
            store = RagStore(Path(temp_dir))
            store.upsert(first)
            request = LookupRequest(
                query="SLA 값 알려줘",
                document_id="doc_cached",
                entities=["SLA", "Value"],
                query_type="number_lookup",
                limit=8,
            )
            main_module._lookup_from_store_cached.cache_clear()
            with (
                patch.object(main_module, "STORE", store),
                patch.object(main_module, "lookup_matches", wraps=main_module.lookup_matches) as lookup_spy,
            ):
                first_result = main_module._lookup_from_store(request, 8)
                second_result = main_module._lookup_from_store(request, 8)
                self.assertEqual(lookup_spy.call_count, 1)
                self.assertEqual(first_result["direct_answer"], second_result["direct_answer"])

                store.upsert(second)
                third_result = main_module._lookup_from_store(request, 8)
                self.assertEqual(lookup_spy.call_count, 2)
                self.assertIn("99.95", third_result["direct_answer"])

    async def test_chatflow_debug_uses_structured_lookup_without_knowledge_retrieval(self):
        await rag_ingest_markdown(
            MarkdownIngestRequest(
                document_id="doc_chatflow_debug",
                file_name="admission.md",
                markdown="""
--- Page 4 ---

# 전공심화 개설 학과

| 과정 | 개설 학과 | 수업 방식 |
| --- | --- | --- |
| 전공심화 | 컴퓨터정보공학과 | 야간 |
| 전공심화 | 기계공학과 | 야간 |
| 전공심화 | 간호학과 | 주말 |
""",
            )
        )

        response = await chatflow_debug(
            ChatflowDebugRequest(
                query="전공심화 개설 학과 알려줘",
                document_id="doc_chatflow_debug",
                knowledge_result=[],
                answer="제공된 문서에는 충분한 근거가 없습니다.",
            )
        )
        payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()

        self.assertEqual(payload["query_plan"]["query_type"], "table_lookup")
        self.assertTrue(payload["node_status"]["structured_lookup_has_context"])
        self.assertIn("Structured Lookup - authoritative exact evidence", payload["merge_evidence"]["evidence_context"])
        self.assertEqual(payload["answer_contract"]["status"], "direct_answer")
        self.assertEqual(payload["answer_contract"]["required_values"], ["컴퓨터정보공학과", "기계공학과", "간호학과"])
        self.assertIn("Answer Contract - obey before writing", payload["merge_evidence"]["evidence_context"])
        self.assertIn("컴퓨터정보공학과", payload["draft_answer"])
        self.assertIn("기계공학과", payload["draft_answer"])
        self.assertIn("간호학과", payload["draft_answer"])
        self.assertTrue(payload["draft_verification"]["valid"])
        self.assertFalse(payload["supplied_answer_verification"]["valid"])

    def test_chatflow_merge_evidence_endpoint_handles_nested_knowledge_result(self):
        client = TestClient(app)
        payload = {
            "lookup_body": {
                "query": "전공심화 개설 학과 알려줘",
                "query_type": "table_lookup",
                "answer_style": "table_or_bullets",
                "direct_answer": "개설 학과: 컴퓨터정보공학과, 기계공학과",
                "answer_field": "개설 학과",
                "answer_items": [{"value": "컴퓨터정보공학과"}, {"value": "기계공학과"}],
                "context": "[Direct Answer - complete structured result]\ncomplete_values:\n- 컴퓨터정보공학과\n- 기계공학과",
                "diagnostics": {"answerability": {"answerable": True}},
            },
            "knowledge_result": {
                "data": {
                    "records": [
                        {
                            "segment": {
                                "content": "전공심화 과정은 야간 수업으로 운영됩니다.",
                                "metadata": {"file_name": "admission.md", "page": 4},
                            },
                            "score": 0.91,
                        }
                    ]
                }
            },
        }

        response = client.post("/chatflow/merge-evidence", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("Answer Contract - obey before writing", body["evidence_context"])
        self.assertIn("Structured Lookup - authoritative exact evidence", body["evidence_context"])
        self.assertIn("[knowledge 1] source=admission.md page=4 score=0.91", body["knowledge_context"])
        self.assertIn("참조 문서: admission.md (p.4)", body["source_summary"])
        self.assertIn("- 컴퓨터정보공학과", body["answer_contract_text"])

    def test_chatflow_merge_evidence_endpoint_accepts_single_item_array_body(self):
        client = TestClient(app)
        payload = [
            {
                "lookup_body": {
                    "query": "기숙사 비용과 신청기간 알려줘",
                    "query_type": "date_lookup",
                    "direct_answer": "",
                    "answer_items": [],
                    "context": "",
                    "diagnostics": {"answerability": {"answerable": False}},
                },
                "knowledge_result": [],
            }
        ]

        response = client.post("/chatflow/merge-evidence", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["structured_context"], "")
        self.assertEqual(body["knowledge_context"], "")

    def test_eval_log_api_writes_jsonl_and_detail_file(self):
        with TemporaryDirectory() as temp_dir:
            store = EvalLogStore(Path(temp_dir))
            client = TestClient(app)
            payload = {
                "run_id": "test-run-1",
                "query": "전공심화 개설 학과 알려줘",
                "document_id": "sample_2026",
                "query_plan": {"query_type": "table_lookup"},
                "structured_lookup_body": {"direct_answer": "모집단위: 컴퓨터정보공학과"},
                "knowledge_retrieval": [],
                "final_answer": "컴퓨터정보공학과입니다.",
                "verify_result": {"valid": True},
            }

            with patch("app.main.EVAL_LOG_STORE", store):
                response = client.post("/eval/log", json=payload)
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertTrue(body["success"])
                self.assertEqual(body["run_id"], "test-run-1")
                self.assertTrue(Path(body["log_path"]).exists())
                self.assertTrue(Path(body["jsonl_path"]).exists())

                list_response = client.get("/eval/logs", params={"limit": 10})
                self.assertEqual(list_response.status_code, 200)
                logs = list_response.json()["logs"]
                self.assertEqual(len(logs), 1)
                self.assertEqual(logs[0]["run_id"], "test-run-1")
                self.assertIn("전공심화", logs[0]["query"])

                detail_response = client.get("/eval/logs/test-run-1")
                self.assertEqual(detail_response.status_code, 200)
                detail = detail_response.json()
                self.assertTrue(detail["success"])
                self.assertEqual(detail["log"]["query_plan"]["query_type"], "table_lookup")


if __name__ == "__main__":
    unittest.main()
