import unittest

from app.main import (
    build_graph_json,
    enrich_dify_markdown,
    sanitize_relation_type,
    sha256_text,
)


class RagGraphTest(unittest.TestCase):
    def test_sanitizes_relation_type_to_allowed_values(self):
        self.assertEqual(sanitize_relation_type("generates"), "GENERATES")
        self.assertEqual(sanitize_relation_type("GENERATES; MATCH (n)"), "RELATED_TO")

    def test_builds_stable_chunk_metadata_for_dify_and_graph(self):
        markdown = """--- Page 3 ---

# System Architecture

## PDF Processing

OpenDataLoader converts PDF to Markdown.
"""

        graph_json, chunks = build_graph_json("doc_001", "sample.pdf", markdown)
        dify_markdown = enrich_dify_markdown(chunks)
        chunk = graph_json["sections"][0]["chunks"][0]

        self.assertEqual(chunk["chunk_id"], "doc_001_sec_001_child_001")
        self.assertEqual(chunk["parent_id"], "doc_001_sec_001_parent_001")
        self.assertEqual(chunk["section_path"], "System Architecture > PDF Processing")
        self.assertEqual(chunk["page_start"], 3)
        self.assertEqual(chunk["page_end"], 3)
        self.assertEqual(chunk["text_hash"], sha256_text(chunk["text"]))
        self.assertIn("[chunk_id: doc_001_sec_001_child_001]", dify_markdown)
        self.assertIn("[parent_id: doc_001_sec_001_parent_001]", dify_markdown)
        self.assertIn("[section_path: System Architecture > PDF Processing]", dify_markdown)
        self.assertIn("[page: 3]", dify_markdown)
        self.assertIn("[text_hash: ", dify_markdown)
        self.assertIn(
            {"source": "OpenDataLoader", "type": "GENERATES", "target": "Markdown"},
            chunk["relations"],
        )


if __name__ == "__main__":
    unittest.main()
