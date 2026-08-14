import unittest
from unittest.mock import patch

from scripts.dify_knowledge_pdf_perf_test import KnowledgeConfig
from scripts.dify_knowledge_reingest import (
    build_update_payload,
    extract_artifact_document_id,
    filename_from_content_disposition,
    download_document_pdf,
    poll_document_status,
    rebuild_dify_markdown_from_segments,
    select_documents,
    split_recovered_parent_bodies,
    update_started_despite_response_error,
    validate_artifact,
)


class DifyKnowledgeReingestTest(unittest.TestCase):
    def test_select_documents_defaults_to_enabled_documents(self):
        selected = select_documents(
            [
                {"id": "doc-2", "name": "b.pdf", "enabled": False},
                {"id": "doc-1", "name": "a.pdf", "enabled": True},
            ],
            document_ids=[],
            include_disabled=False,
        )

        self.assertEqual([document["id"] for document in selected], ["doc-1"])

    def test_select_documents_rejects_unknown_explicit_id(self):
        with self.assertRaisesRegex(RuntimeError, "doc-missing"):
            select_documents(
                [{"id": "doc-1", "name": "a.pdf", "enabled": True}],
                document_ids=["doc-missing"],
                include_disabled=False,
            )

    def test_extract_artifact_document_id_from_existing_segment(self):
        document_id = extract_artifact_document_id(
            [
                {"content": "plain continuation"},
                {"content": "[document_id: stable-artifact-id]\n[page: 37]"},
            ]
        )

        self.assertEqual(document_id, "stable-artifact-id")

    def test_content_disposition_supports_utf8_and_plain_names(self):
        self.assertEqual(
            filename_from_content_disposition("attachment; filename*=UTF-8''%EC%9E%85%EC%8B%9C.pdf"),
            "입시.pdf",
        )
        self.assertEqual(filename_from_content_disposition('attachment; filename="guide.pdf"'), "guide.pdf")

    def test_text_download_is_treated_as_missing_source_pdf(self):
        class Response:
            headers = {"Content-Disposition": 'attachment; filename="guide.md"'}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return b"[document_id: existing-text-source]"

        config = KnowledgeConfig("https://example.test/v1", "secret", "dataset")
        with patch("urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(RuntimeError, "text rather than a PDF"):
                download_document_pdf(config, "doc-1", "guide.pdf", timeout=1)

    def test_update_payload_preserves_hierarchical_chunk_boundaries(self):
        payload = build_update_payload(
            KnowledgeConfig(
                base_url="https://example.test/v1",
                api_key="secret",
                dataset_id="dataset",
                doc_form="hierarchical_model",
            ),
            "입시.pdf",
            "retrieval markdown",
            "Korean",
        )

        self.assertEqual(payload["name"], "입시.pdf")
        self.assertEqual(payload["doc_form"], "hierarchical_model")
        self.assertEqual(payload["process_rule"]["mode"], "hierarchical")
        segmentation = payload["process_rule"]["rules"]["segmentation"]
        self.assertEqual(segmentation["separator"], "\n\n---\n\n")
        self.assertEqual(segmentation["max_tokens"], 1500)

    def test_artifact_validation_requires_chunks_and_context_metadata(self):
        valid = "[document_id: doc-1]\n[context: file=a.pdf | page=1 | section=A | paragraph=p1]"
        validate_artifact(valid, 1, "doc-1")

        with self.assertRaisesRegex(RuntimeError, "missing retrieval metadata"):
            validate_artifact("[document_id: doc-1]", 1, "doc-1")

    def test_segment_fallback_resections_numbered_headings_and_continuations(self):
        common = (
            "[document_id: stable]\n[chunk_id: old]\n[file_name: guide.pdf]\n[page: 37]\n"
            "[section: IX. 안내사항 > 2. 입학학기 휴학]\n[record_types: money]\n[keywords: 장학금]\n\n"
        )
        markdown, chunk_count = rebuild_dify_markdown_from_segments(
            [
                {
                    "position": 1,
                    "content": common
                    + "#### 2. 입학학기 휴학\n\n휴학 안내\n\n3. 장학제도\n\n"
                    + "  source_section: IX. 안내사항 > 2. 입학학기 휴학\n  answer_hint: 수업료 전액",
                },
                {
                    "position": 2,
                    "content": common
                    + "- structured_record:\n  source_section: IX. 안내사항 > 2. 입학학기 휴학\n"
                    + "  answer_hint: 교내 장학금",
                },
                {
                    "position": 3,
                    "content": "장학금 유의사항\n\n4. 입학포기 및 등록금 반환\n\n반환 안내",
                },
                {
                    "position": 4,
                    "content": common
                    + "- structured_record:\n  source_section: IX. 안내사항 > 2. 입학학기 휴학\n"
                    + "  answer_hint: 등록금 전액 반환",
                },
            ],
            document_id="stable",
            fallback_file_name="guide.pdf",
        )

        self.assertGreaterEqual(chunk_count, 4)
        self.assertIn("[section: IX. 안내사항 > 3. 장학제도]", markdown)
        self.assertIn("source_section: IX. 안내사항 > 3. 장학제도", markdown)
        self.assertIn("[section: IX. 안내사항 > 4. 입학포기 및 등록금 반환]", markdown)
        self.assertIn("source_section: IX. 안내사항 > 4. 입학포기 및 등록금 반환", markdown)
        refund_context = next(
            line for line in markdown.splitlines() if "section=IX. 안내사항 > 4. 입학포기" in line
        )
        self.assertIn("page=37", refund_context)

    def test_recovered_parent_blocks_are_split_before_dify_token_limit(self):
        body = "\n\n".join(["structured record " + ("가" * 700)] * 4)

        blocks = split_recovered_parent_bodies(body, weight_limit=1000)

        self.assertEqual(len(blocks), 4)
        self.assertTrue(all(len(block) <= 780 for block in blocks))

    @patch("scripts.dify_knowledge_reingest.get_knowledge_document")
    def test_response_error_is_accepted_only_when_indexing_started(self, get_document):
        get_document.return_value = {"indexing_status": "indexing", "updated_at": 20}
        config = KnowledgeConfig("https://example.test/v1", "secret", "dataset")

        accepted = update_started_despite_response_error(
            config,
            "doc-1",
            previous_updated_at=10,
            error=RuntimeError('HTTP 400: {"message":"Document is not available"}'),
            timeout=1,
        )

        self.assertTrue(accepted)

    @patch("scripts.dify_knowledge_reingest.get_knowledge_document")
    def test_document_status_polling_reports_terminal_completion(self, get_document):
        get_document.side_effect = [
            {"indexing_status": "indexing", "updated_at": 20},
            {"indexing_status": "completed", "updated_at": 21},
        ]
        config = KnowledgeConfig("https://example.test/v1", "secret", "dataset")

        result = poll_document_status(
            config,
            "doc-1",
            timeout=1,
            index_timeout=1,
            poll_interval=0,
        )

        self.assertEqual(result["statuses"], ["completed"])
        self.assertFalse(result["failed"])


if __name__ == "__main__":
    unittest.main()
