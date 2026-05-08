from io import BytesIO
import unittest
from unittest.mock import patch

from fastapi import UploadFile

from app.main import ConvertResponse, convert


class ApiContractTest(unittest.IsolatedAsyncioTestCase):
    def test_convert_response_has_only_success_and_markdown(self):
        response = ConvertResponse(success=True, markdown="# Done")
        payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()

        self.assertEqual(set(payload), {"success", "markdown"})

    async def test_convert_uses_uploaded_pdf_parameter(self):
        content = b"%PDF-1.4\n%%EOF"
        pdf = UploadFile(file=BytesIO(content), filename="policy.pdf")

        with patch("app.main.PdfConverter") as converter_class:
            converter_class.return_value.convert_pdf_bytes.return_value = "# Converted"

            response = await convert(pdf)

        self.assertTrue(response.success)
        self.assertEqual(response.markdown, "# Converted")
        converter_class.return_value.convert_pdf_bytes.assert_called_once_with(content, "policy.pdf", use_ocr=False)

    async def test_convert_passes_selected_ocr_option(self):
        content = b"%PDF-1.4\n%%EOF"
        pdf = UploadFile(file=BytesIO(content), filename="scan.pdf")

        with patch("app.main.PdfConverter") as converter_class:
            converter_class.return_value.convert_pdf_bytes.return_value = "# Converted"

            response = await convert(pdf, ocr=True)

        self.assertTrue(response.success)
        converter_class.return_value.convert_pdf_bytes.assert_called_once_with(content, "scan.pdf", use_ocr=True)


if __name__ == "__main__":
    unittest.main()
