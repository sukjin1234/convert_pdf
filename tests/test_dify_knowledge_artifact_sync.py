import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.rag import build_rag_artifact
from scripts.dify_knowledge_artifact_sync import (
    KnowledgeConfig,
    artifact_sync_item,
    artifact_upload_file_name,
    create_document_from_artifact,
    load_artifacts,
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
