import asyncio
from io import BytesIO
import threading
import unittest
from unittest.mock import patch

from fastapi import UploadFile

from app.main import ConvertResponse, _resolve_use_ocr, convert


class _Request:
    def __init__(self, query_params=None, headers=None):
        self.query_params = query_params or {}
        self.headers = headers or {}


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

            response = await convert(_Request(), pdf)

        self.assertTrue(response.success)
        self.assertEqual(response.markdown, "# Converted")
        converter_class.return_value.convert_pdf_bytes.assert_called_once_with(content, "policy.pdf", use_ocr=True)

    async def test_convert_passes_use_ocr_flag(self):
        content = b"%PDF-1.4\n%%EOF"
        pdf = UploadFile(file=BytesIO(content), filename="scan.pdf")

        with patch("app.main.PdfConverter") as converter_class:
            converter_class.return_value.convert_pdf_bytes.return_value = "# Converted"

            response = await convert(_Request(), pdf, use_ocr="false")

        self.assertTrue(response.success)
        converter_class.return_value.convert_pdf_bytes.assert_called_once_with(content, "scan.pdf", use_ocr=False)

    async def test_convert_reads_use_ocr_from_query_string(self):
        content = b"%PDF-1.4\n%%EOF"
        pdf = UploadFile(file=BytesIO(content), filename="scan.pdf")

        with patch("app.main.PdfConverter") as converter_class:
            converter_class.return_value.convert_pdf_bytes.return_value = "# Converted"

            response = await convert(_Request(query_params={"use_ocr": "0"}), pdf)

        self.assertTrue(response.success)
        converter_class.return_value.convert_pdf_bytes.assert_called_once_with(content, "scan.pdf", use_ocr=False)

    def test_resolve_use_ocr_reads_header_value(self):
        self.assertFalse(_resolve_use_ocr(None, _Request(headers={"x-use-ocr": "false"})))
        self.assertTrue(_resolve_use_ocr(None, _Request(headers={"x-use-ocr": "true"})))

    async def test_convert_joins_duplicate_inflight_conversion(self):
        content = b"%PDF-1.4\n%%EOF"
        first_pdf = UploadFile(file=BytesIO(content), filename="scan.pdf")
        second_pdf = UploadFile(file=BytesIO(content), filename="scan.pdf")
        started = threading.Event()
        release = threading.Event()

        def fake_convert(*_args, **_kwargs):
            started.set()
            release.wait(timeout=2)
            return "# Converted"

        with patch("app.main.PdfConverter") as converter_class:
            converter_class.return_value.convert_pdf_bytes.side_effect = fake_convert

            first_task = asyncio.create_task(convert(_Request(), first_pdf, use_ocr="true"))
            await asyncio.get_running_loop().run_in_executor(None, started.wait, 2)
            second_task = asyncio.create_task(convert(_Request(), second_pdf, use_ocr="true"))
            await asyncio.sleep(0.05)
            release.set()

            first_response, second_response = await asyncio.gather(first_task, second_task)
            await asyncio.sleep(0)

        self.assertTrue(first_response.success)
        self.assertTrue(second_response.success)
        self.assertEqual(first_response.markdown, "# Converted")
        self.assertEqual(second_response.markdown, "# Converted")
        converter_class.return_value.convert_pdf_bytes.assert_called_once_with(content, "scan.pdf", use_ocr=True)


if __name__ == "__main__":
    unittest.main()
