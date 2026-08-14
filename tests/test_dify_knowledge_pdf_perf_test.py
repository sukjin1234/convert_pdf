import unittest

from scripts.dify_knowledge_pdf_perf_test import (
    KnowledgeConfig,
    build_process_rule,
    encode_multipart,
    extract_created_document_id,
    extract_indexed_document_id,
    retrieval_items,
    sample_pdf_bytes,
    segment_items_from_payload,
)


class DifyKnowledgePdfPerfScriptTest(unittest.TestCase):
    def test_sample_pdf_bytes_are_valid_pdf(self):
        content = sample_pdf_bytes()

        self.assertTrue(content.startswith(b"%PDF-"))
        self.assertTrue(content.rstrip().endswith(b"%%EOF"))
        import fitz

        doc = fitz.open(stream=content, filetype="pdf")
        try:
            self.assertIn("Codex Knowledge PDF performance smoke test", doc[0].get_text())
        finally:
            doc.close()

    def test_encode_multipart_includes_json_data_and_pdf_file(self):
        body, content_type = encode_multipart(
            {"data": '{"indexing_technique":"high_quality"}'},
            {"file": ("sample.pdf", b"%PDF-1.4\n%%EOF", "application/pdf")},
        )

        self.assertIn("multipart/form-data; boundary=", content_type)
        self.assertIn(b'name="data"', body)
        self.assertIn(b'"indexing_technique":"high_quality"', body)
        self.assertIn(b'name="file"; filename="sample.pdf"', body)
        self.assertIn(b"Content-Type: application/pdf", body)

    def test_extract_document_ids_from_create_and_index_payloads(self):
        self.assertEqual(extract_created_document_id({"document": {"id": "doc-1"}}), "doc-1")
        self.assertEqual(extract_created_document_id({"document_id": "doc-2"}), "doc-2")
        self.assertEqual(extract_indexed_document_id({"entries": [{"id": "doc-3"}]}), "doc-3")

    def test_retrieval_items_supports_segment_payloads(self):
        items = retrieval_items(
            {
                "data": [
                    {
                        "score": 0.91,
                        "segment": {
                            "id": "seg-1",
                            "content": "Codex Knowledge PDF performance smoke test result",
                        },
                    }
                ]
            }
        )

        self.assertEqual(items[0]["score"], 0.91)
        self.assertEqual(items[0]["segment_id"], "seg-1")
        self.assertIn("Codex Knowledge", items[0]["content_preview"])

    def test_segment_items_from_payload_compacts_uploaded_document_segments(self):
        items = segment_items_from_payload(
            {
                "data": [
                    {
                        "id": "seg-1",
                        "word_count": 8,
                        "content": "Codex Knowledge PDF performance smoke test segment",
                    }
                ]
            }
        )

        self.assertEqual(items[0]["segment_id"], "seg-1")
        self.assertEqual(items[0]["word_count"], 8)
        self.assertIn("smoke test", items[0]["content_preview"])

    def test_hierarchical_dataset_uses_parent_child_process_rule(self):
        rule = build_process_rule(
            KnowledgeConfig(
                base_url="https://example.test/v1",
                api_key="test-key",
                dataset_id="dataset",
                doc_form="hierarchical_model",
            )
        )

        self.assertEqual(rule["mode"], "hierarchical")
        self.assertEqual(rule["rules"]["parent_mode"], "paragraph")
        self.assertEqual(rule["rules"]["segmentation"]["separator"], "\n\n---\n\n")
        self.assertEqual(rule["rules"]["segmentation"]["max_tokens"], 1500)
        self.assertEqual(rule["rules"]["subchunk_segmentation"], {"separator": "\n\n", "max_tokens": 500})


if __name__ == "__main__":
    unittest.main()
