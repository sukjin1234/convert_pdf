import json
import unittest
from pathlib import Path

from app.rag import build_rag_artifact
from scripts.eval_rag_quality import evaluate_case, summarize


class RagQualityEvalTest(unittest.TestCase):
    def test_seed_quality_eval_passes_baseline(self):
        payload = json.loads(Path("evaluation/rag_quality_cases.json").read_text(encoding="utf-8"))
        artifacts = {
            document["document_id"]: build_rag_artifact(
                document["markdown"],
                document["file_name"],
                document_id=document["document_id"],
            )
            for document in payload["documents"]
        }

        results = [evaluate_case(case, artifacts, limit=8) for case in payload["cases"]]
        summary = summarize(results)

        self.assertGreaterEqual(summary["pass_rate"], 0.90)
        self.assertGreaterEqual(summary["evidence_recall"], 0.95)
        self.assertGreaterEqual(summary["exact_value_match"], 0.95)
        self.assertGreaterEqual(summary["unsupported_guard"], 0.95)


if __name__ == "__main__":
    unittest.main()
