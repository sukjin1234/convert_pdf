import os
import unittest
from pathlib import Path

from app.config import get_settings
from app.converter import PdfConverter
from app.rag import build_rag_artifact


class SamplePdfContractTest(unittest.TestCase):
    @unittest.skipUnless(os.getenv("RUN_SAMPLE_PDF_TEST") == "1", "Set RUN_SAMPLE_PDF_TEST=1 to run the sample PDF integration test.")
    def test_sample_pdf_can_build_rag_artifact(self):
        root = Path(__file__).resolve().parents[1]
        candidates = sorted(root.glob("*모집요강.pdf"))
        self.assertTrue(candidates, "Sample admission PDF was not found.")

        pdf_path = candidates[0]
        markdown = PdfConverter(get_settings()).convert_pdf_bytes(pdf_path.read_bytes(), pdf_path.name)
        artifact = build_rag_artifact(markdown, pdf_path.name)

        self.assertTrue(artifact.success)
        self.assertGreater(len(artifact.dify_markdown), 1000)
        self.assertGreater(len(artifact.chunks), 1)


if __name__ == "__main__":
    unittest.main()
