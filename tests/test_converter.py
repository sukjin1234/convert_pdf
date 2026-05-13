import json
import unittest
from contextlib import contextmanager
from http.client import BadStatusLine
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from app.config import Settings
from app.converter import (
    ConversionError,
    PdfConverter,
    _convert_pdf_file,
    _managed_ocr_server,
    _read_generated_markdown,
    _read_rendered_markdown,
    _wait_for_http_server,
    build_ocr_server_command,
    build_opendataloader_native_command,
    build_opendataloader_command,
    sanitize_filename,
)


@contextmanager
def _test_workspace():
    root = Path(__file__).resolve().parents[1] / ".tmp-tests"
    path = root / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    yield str(path)


class _RunningProcess:
    def poll(self):
        return None


class _ReadyResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _ExitedOcrServerProcess:
    pid = 12345
    returncode = 2

    def __init__(self, _command, **kwargs):
        kwargs["stderr"].write("address already in use\n")

    def poll(self):
        return self.returncode


class ConverterTest(unittest.TestCase):
    def test_command_forces_hybrid_with_no_fallback(self):
        settings = Settings()
        command = build_opendataloader_command(Path("input.pdf"), Path("out"), settings)

        self.assertIn("--hybrid", command)
        self.assertEqual(command[command.index("--hybrid") + 1], "docling-fast")
        self.assertIn("--hybrid-mode", command)
        self.assertEqual(command[command.index("--hybrid-mode") + 1], "auto")
        self.assertIn("--hybrid-timeout", command)
        self.assertEqual(command[command.index("--hybrid-timeout") + 1], "300000")
        self.assertIn("--table-method", command)
        self.assertEqual(command[command.index("--table-method") + 1], "cluster")
        self.assertNotIn("--hybrid-fallback", command)

    def test_command_can_target_dedicated_ocr_hybrid_url(self):
        settings = Settings(ocr_hybrid_url="http://127.0.0.1:5999")
        command = build_opendataloader_command(
            Path("input.pdf"),
            Path("out"),
            settings,
            hybrid_url=settings.ocr_hybrid_url,
        )

        self.assertEqual(command[command.index("--hybrid-url") + 1], "http://127.0.0.1:5999")

    def test_ocr_server_command_uses_force_ocr(self):
        settings = Settings(ocr_server_cli="hybrid-server", ocr_server_port=5999, ocr_lang="ko,en")
        command = build_ocr_server_command(settings)

        self.assertEqual(command[:3], ["hybrid-server", "--port", "5999"])
        self.assertIn("--force-ocr", command)
        self.assertIn("--ocr-lang", command)
        self.assertEqual(command[command.index("--ocr-lang") + 1], "ko,en")

    def test_wait_for_http_server_treats_bad_status_line_as_transient(self):
        calls = []

        def fake_urlopen(url, timeout):
            calls.append((url, timeout))
            if len(calls) == 1:
                raise BadStatusLine("handshake failed, invalid handshake message")
            return _ReadyResponse()

        with patch("app.converter.urlopen", side_effect=fake_urlopen):
            _wait_for_http_server("http://127.0.0.1:5003", _RunningProcess(), 1)

        self.assertGreaterEqual(len(calls), 2)

    def test_managed_ocr_server_reports_startup_stderr(self):
        settings = Settings(ocr_server_cli="hybrid-server")

        with patch("app.converter.subprocess.Popen", _ExitedOcrServerProcess):
            with self.assertRaisesRegex(ConversionError, "address already in use"):
                with _managed_ocr_server(settings):
                    pass

    def test_native_command_preserves_page_separators_without_hybrid(self):
        settings = Settings(opendataloader_jar="odl.jar")
        command = build_opendataloader_native_command(Path("input.pdf"), Path("out"), settings)

        self.assertEqual(command[:3], ["java", "-jar", "odl.jar"])
        self.assertNotIn("--hybrid", command)
        self.assertIn("--markdown-page-separator", command)
        self.assertEqual(command[command.index("--markdown-page-separator") + 1], "\n\n--- Page %page-number% ---\n\n")
        self.assertIn("--table-method", command)
        self.assertEqual(command[command.index("--table-method") + 1], "cluster")

    def test_rejects_non_pdf_bytes(self):
        converter = PdfConverter(Settings())

        with self.assertRaises(ConversionError):
            converter._validate_pdf_bytes(b"not a pdf")

    def test_rejects_text_corrupted_pdf_bytes(self):
        converter = PdfConverter(Settings())
        corrupted = b"%PDF-1.4\n" + (b"\xef\xbf\xbd" * 200) + b"\n%%EOF"

        with self.assertRaisesRegex(ConversionError, "text-decoded or corrupted"):
            converter._validate_pdf_bytes(corrupted)

    def test_sanitizes_filename(self):
        self.assertEqual(sanitize_filename("../bad:name?.pdf"), "bad_name_.pdf")

    def test_retries_qpdf_repaired_pdf_after_original_failure(self):
        with _test_workspace() as tmp:
            settings = Settings(
                tmp_root=Path(tmp),
                qpdf_repair_pdf_on_failure=True,
                repair_pdf_on_failure=False,
                rasterize_pdf_on_failure=False,
            )
            converter = PdfConverter(settings)

            with patch("app.converter._convert_pdf_file") as convert, patch("app.converter._repair_pdf_with_pikepdf") as repair:
                convert.side_effect = [ConversionError("broken xref"), "markdown"]

                result = converter.convert_pdf_bytes(b"%PDF-1.4\n%%EOF", "broken.pdf")

        self.assertEqual(result, "markdown")
        self.assertEqual(convert.call_count, 2)
        repair.assert_called_once()

    def test_retries_repaired_pdf_after_original_failure(self):
        with _test_workspace() as tmp:
            settings = Settings(
                tmp_root=Path(tmp),
                qpdf_repair_pdf_on_failure=False,
                repair_pdf_on_failure=True,
                rasterize_pdf_on_failure=False,
            )
            converter = PdfConverter(settings)

            with patch("app.converter._convert_pdf_file") as convert, patch("app.converter._repair_pdf") as repair:
                convert.side_effect = [ConversionError("broken xref"), "markdown"]

                result = converter.convert_pdf_bytes(b"%PDF-1.4\n%%EOF", "broken.pdf")

        self.assertEqual(result, "markdown")
        self.assertEqual(convert.call_count, 2)
        repair.assert_called_once()

    def test_rasterizes_after_repair_conversion_failure(self):
        with _test_workspace() as tmp:
            settings = Settings(
                tmp_root=Path(tmp),
                qpdf_repair_pdf_on_failure=False,
                repair_pdf_on_failure=True,
                rasterize_pdf_on_failure=True,
            )
            converter = PdfConverter(settings)

            with (
                patch("app.converter._convert_pdf_file") as convert,
                patch("app.converter._repair_pdf") as repair,
                patch("app.converter._rasterize_pdf") as rasterize,
            ):
                convert.side_effect = [
                    ConversionError("broken xref"),
                    ConversionError("empty markdown"),
                    "ocr markdown",
                ]

                result = converter.convert_pdf_bytes(b"%PDF-1.4\n%%EOF", "broken.pdf")

        self.assertEqual(result, "ocr markdown")
        self.assertEqual(convert.call_count, 3)
        repair.assert_called_once()
        rasterize.assert_called_once()

    def test_use_ocr_inserts_ocr_markdown_at_image_position(self):
        with _test_workspace() as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / "input.pdf"
            input_path.write_bytes(b"%PDF-1.4\n%%EOF")
            settings = Settings(
                tmp_root=tmp_dir,
                native_text_layer_first=False,
                ocr_hybrid_url="http://127.0.0.1:5999",
            )
            events = []

            @contextmanager
            def fake_ocr_server(_settings):
                events.append("start")
                try:
                    yield
                finally:
                    events.append("stop")

            def fake_run_command(command, _timeout):
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                if output_dir.name == "output":
                    doc = {
                        "kids": [
                            {"type": "paragraph", "page number": 1, "content": "Before image"},
                            {
                                "type": "picture",
                                "page number": 1,
                                "bounding box": [0, 0, 100, 100],
                                "description": "Admissions chart.",
                            },
                            {"type": "paragraph", "page number": 1, "content": "After image"},
                            {"type": "paragraph", "page number": 2, "content": "Text-only page"},
                        ]
                    }
                    (output_dir / "doc.json").write_text(json.dumps(doc), encoding="utf-8")
                    (output_dir / "doc.md").write_text("fallback", encoding="utf-8")
                else:
                    self.assertEqual(command[command.index("--hybrid-url") + 1], "http://127.0.0.1:5999")
                    (output_dir / "ocr.md").write_text("--- Page 1 ---\n\nOCR text from image", encoding="utf-8")

            with (
                patch("app.converter._managed_ocr_server", fake_ocr_server),
                patch("app.converter._extract_pdf_region_to_pdf") as extract_region,
                patch("app.converter._run_command", side_effect=fake_run_command),
            ):
                markdown = _convert_pdf_file(input_path, tmp_dir / "output", settings, use_ocr=True)

        self.assertEqual(events, ["start", "stop"])
        extract_region.assert_called_once()
        self.assertLess(markdown.index("Before image"), markdown.index("OCR text from image"))
        self.assertLess(markdown.index("OCR text from image"), markdown.index("After image"))
        self.assertIn("Text-only page", markdown)

    def test_use_ocr_does_not_start_server_without_image_targets(self):
        with _test_workspace() as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / "input.pdf"
            input_path.write_bytes(b"%PDF-1.4\n%%EOF")
            settings = Settings(tmp_root=tmp_dir, native_text_layer_first=False)

            def fake_run_command(command, _timeout):
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                doc = {"kids": [{"type": "paragraph", "page number": 1, "content": "Only text"}]}
                (output_dir / "doc.json").write_text(json.dumps(doc), encoding="utf-8")
                (output_dir / "doc.md").write_text("Only text", encoding="utf-8")

            with (
                patch("app.converter._managed_ocr_server") as server,
                patch("app.converter._run_command", side_effect=fake_run_command),
            ):
                markdown = _convert_pdf_file(input_path, tmp_dir / "output", settings, use_ocr=True)

        self.assertEqual(markdown, "Only text")
        server.assert_not_called()

    def test_rejects_page_separator_only_markdown(self):
        with _test_workspace() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text("\n\n--- Page 1 ---\n\n--- Page 2 ---\n\n", encoding="utf-8")

            with self.assertRaises(ConversionError):
                _read_rendered_markdown(output_dir)

    def test_rejects_image_link_only_markdown(self):
        with _test_workspace() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text(
                "\n\n--- Page 1 ---\n\n![image 1](<images/page1.png>)\n\n",
                encoding="utf-8",
            )

            with self.assertRaises(ConversionError):
                _read_rendered_markdown(output_dir)

    def test_reads_native_markdown_with_page_order_and_tables(self):
        markdown = "\n\n--- Page 26 ---\n\n| Category | Unit |\n| --- | --- |\n| Engineering | Free Major |\n"
        with _test_workspace() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text(markdown, encoding="utf-8")

            result = _read_generated_markdown(output_dir)

        self.assertIn("--- Page 26 ---", result)
        self.assertIn("| Category | Unit |", result)

    def test_rejects_empty_table_only_markdown(self):
        with _test_workspace() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text(
                "\n\n--- Page 1 ---\n\n| | |\n| --- | --- |\n| | |\n",
                encoding="utf-8",
            )

            with self.assertRaises(ConversionError):
                _read_generated_markdown(output_dir)

    def test_rejects_sparse_page_content_markdown(self):
        markdown = "\n".join(
            [
                "--- Page 1 ---",
                "Only one useful page",
                *[f"--- Page {page} ---" for page in range(2, 21)],
            ]
        )
        with _test_workspace() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text(markdown, encoding="utf-8")

            with self.assertRaises(ConversionError):
                _read_generated_markdown(output_dir)


if __name__ == "__main__":
    unittest.main()
