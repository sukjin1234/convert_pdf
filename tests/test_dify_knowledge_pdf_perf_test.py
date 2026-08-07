import unittest

from scripts.dify_knowledge_pdf_perf_test import (
    encode_multipart,
    extract_created_document_id,
    extract_indexed_document_id,
    retrieval_items,
    sample_pdf_bytes,
)


class DifyKnowledgePdfPerfScriptTest(unittest.TestCase):
    def test_sample_pdf_bytes_are_valid_pdf(self):
        content = sample_pdf_bytes()

        self.assertTrue(content.startswith(b"%PDF-"))
        self.assertIn(b"Codex Knowledge PDF performance smoke test", content)
        self.assertTrue(content.rstrip().endswith(b"%%EOF"))

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


if __name__ == "__main__":
    unittest.main()
