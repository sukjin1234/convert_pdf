import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.rag import build_rag_artifact
from scripts.dify_knowledge_artifact_sync import (
    KnowledgeConfig,
    apply_duplicate_policy,
    artifact_sync_item,
    artifact_upload_file_name,
    create_document_from_artifact,
    existing_document_names,
    existing_documents_by_name,
    load_artifacts,
    replace_existing_documents,
    summarize_items,
)


class DifyKnowledgeArtifactSyncTest(unittest.TestCase):
    def test_artifact_upload_file_name_uses_safe_document_id(self):
        artifact = build_rag_artifact("# Manual\n\n본문", "manual.pdf", document_id="doc 2027/입시")

        self.assertEqual(artifact_upload_file_name(artifact), "doc_2027.rag.md")

    def test_load_artifacts_filters_and_cleans_dify_markdown_metadata(self):
        artifact = build_rag_artifact("# Manual\n\n| Item | Value |\n| --- | --- |\n| SLA | 99.9% |", "manual.pdf", document_id="doc_manual")
        payload = artifact.to_dict()
        payload["dify_markdown"] = payload["dify_markdown"].replace(
            "[section: Manual]",
            "[section: Guide > 2. 정원외 전형 모집인원 > Ⅱ. 전형일정]",
        )

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "doc_manual.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            artifacts = load_artifacts(Path(temp_dir), ["doc_manual"])

        self.assertEqual(len(artifacts), 1)
        self.assertIn("[section: Guide > Ⅱ. 전형일정]", artifacts[0].dify_markdown)
        self.assertNotIn("[section: Guide > 2. 정원외 전형 모집인원", artifacts[0].dify_markdown)

    def test_artifact_sync_item_marks_ready_and_summarizes_counts(self):
        artifact = build_rag_artifact("# Manual\n\n본문", "manual.pdf", document_id="doc_manual")

        item = artifact_sync_item(artifact)
        summary = summarize_items([item])

        self.assertEqual(item.status, "ready")
        self.assertEqual(item.upload_file_name, "doc_manual.rag.md")
        self.assertGreater(item.markdown_chars, 0)
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["ready"], 1)
        self.assertEqual(summary["markdown_chars"], item.markdown_chars)

    def test_existing_document_names_supports_nested_dify_file_metadata(self):
        names = existing_document_names(
            {
                "id": "doc-1",
                "name": "display name",
                "data_source_info": json.dumps({"upload_file_name": "doc_manual.rag.md"}),
                "data_source_detail_dict": {"upload_file": {"name": "manual.pdf"}},
            }
        )

        self.assertIn("display name", names)
        self.assertIn("doc_manual.rag.md", names)
        self.assertIn("manual.pdf", names)

    def test_apply_duplicate_policy_marks_dry_run_skip_and_replace(self):
        item = artifact_sync_item(build_rag_artifact("# Manual\n\n본문", "manual.pdf", document_id="doc_manual"))
        existing = {"doc_manual.rag.md": ["knowledge-doc-1"]}

        apply_duplicate_policy([item], existing, duplicate_action="skip", upload=False)

        self.assertEqual(item.status, "would_skip_existing")
        self.assertEqual(item.existing_document_ids, ["knowledge-doc-1"])
        summary = summarize_items([item])
        self.assertEqual(summary["skipped"], 1)

        replace_item = artifact_sync_item(build_rag_artifact("# Manual\n\n본문", "manual.pdf", document_id="doc_manual"))
        apply_duplicate_policy([replace_item], existing, duplicate_action="replace", upload=False)

        self.assertEqual(replace_item.status, "would_replace_existing")
        self.assertEqual(summarize_items([replace_item])["replace_pending"], 1)

    def test_apply_duplicate_policy_keeps_upload_when_requested(self):
        item = artifact_sync_item(build_rag_artifact("# Manual\n\n본문", "manual.pdf", document_id="doc_manual"))

        apply_duplicate_policy([item], {"doc_manual.rag.md": ["knowledge-doc-1"]}, duplicate_action="upload", upload=True)

        self.assertEqual(item.status, "ready")
        self.assertEqual(item.existing_document_ids, ["knowledge-doc-1"])

    def test_existing_documents_by_name_groups_ids(self):
        grouped = existing_documents_by_name(
            [
                {"id": "doc-1", "name": "doc_manual.rag.md"},
                {"id": "doc-2", "document_name": "doc_manual.rag.md"},
            ]
        )

        self.assertEqual(grouped["doc_manual.rag.md"], ["doc-1", "doc-2"])

    def test_replace_existing_documents_reports_delete_errors(self):
        item = artifact_sync_item(build_rag_artifact("# Manual\n\n본문", "manual.pdf", document_id="doc_manual"))
        item.existing_document_ids = ["doc-1", "doc-2"]
        config = KnowledgeConfig(base_url="https://dify.example/v1", api_key="test-key", dataset_id="dataset-1")

        with patch("scripts.dify_knowledge_artifact_sync.delete_document", side_effect=["", "HTTP 500"]):
            replace_existing_documents(config, item, timeout=30)

        self.assertEqual(item.status, "error")
        self.assertIn("doc-2: HTTP 500", item.error)

    def test_create_document_from_artifact_posts_markdown_file(self):
        config = KnowledgeConfig(
            base_url="https://dify.example/v1",
            api_key="test-key",
            dataset_id="dataset-1",
            doc_form="hierarchical_model",
        )

        with patch("scripts.dify_knowledge_artifact_sync.request_json", return_value={"batch": "batch-1"}) as request_json:
            payload = create_document_from_artifact(
                config,
                "doc_manual.rag.md",
                "# Manual\n\n본문",
                doc_language="Korean",
                timeout=30,
            )

        self.assertEqual(payload["batch"], "batch-1")
        request_json.assert_called_once()
        _, url, api_key = request_json.call_args.args[:3]
        self.assertEqual(url, "https://dify.example/v1/datasets/dataset-1/document/create-by-file")
        self.assertEqual(api_key, "test-key")
        body = request_json.call_args.kwargs["body"]
        self.assertIn(b'name="file"; filename="doc_manual.rag.md"', body)
        self.assertIn(b"Content-Type: text/markdown; charset=utf-8", body)
        self.assertIn(b'"mode": "hierarchical"', body)


if __name__ == "__main__":
    unittest.main()
