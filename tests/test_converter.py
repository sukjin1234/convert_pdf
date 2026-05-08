import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.config import Settings
from app.converter import (
    ConversionError,
    PdfConverter,
    _read_generated_markdown,
    _read_rendered_markdown,
    build_opendataloader_native_command,
    build_opendataloader_command,
    build_hybrid_server_command,
    sanitize_filename,
)


class ConverterTest(unittest.TestCase):
    def test_hybrid_server_command_disables_ocr_when_not_selected(self):
        settings = Settings(hybrid_server_cli="opendataloader-pdf-hybrid")
        command = build_hybrid_server_command(settings, 5101, use_ocr=False)

        self.assertEqual(command[:5], ["opendataloader-pdf-hybrid", "--host", "127.0.0.1", "--port", "5101"])
        self.assertIn("--no-ocr", command)
        self.assertNotIn("--force-ocr", command)
        self.assertNotIn("--enrich-picture-description", command)

    def test_hybrid_server_command_enables_ocr_options_when_selected(self):
        settings = Settings(
            hybrid_server_cli="opendataloader-pdf-hybrid",
            hybrid_server_ocr_engine="easyocr",
            hybrid_server_ocr_lang="ko,en",
            hybrid_server_enrich_picture_description=True,
        )
        command = build_hybrid_server_command(settings, 5102, use_ocr=True)

        self.assertIn("--ocr-engine", command)
        self.assertEqual(command[command.index("--ocr-engine") + 1], "easyocr")
        self.assertIn("--ocr-lang", command)
        self.assertEqual(command[command.index("--ocr-lang") + 1], "ko,en")
        self.assertIn("--enrich-picture-description", command)
        self.assertNotIn("--no-ocr", command)

    def test_uses_primary_hybrid_server_first_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                tmp_root=Path(tmp),
                hybrid_url="http://localhost:5002",
                native_text_layer_first=False,
                spawn_hybrid_per_conversion=True,
                prepare_dify_parent_child_chunks=False,
            )
            converter = PdfConverter(settings)

            with (
                patch("app.converter._hybrid_health_ok", return_value=True),
                patch("app.converter._start_temporary_hybrid_server") as start_server,
                patch("app.converter._run_command") as run_command,
                patch("app.converter._read_rendered_markdown", return_value="markdown"),
            ):
                result = converter.convert_pdf_bytes(b"%PDF-1.4\n%%EOF", "policy.pdf")

        self.assertEqual(result, "markdown")
        start_server.assert_not_called()
        command = run_command.call_args.args[0]
        self.assertEqual(command[command.index("--hybrid-url") + 1], "http://localhost:5002")

    def test_uses_temporary_hybrid_server_when_primary_is_busy_or_unavailable(self):
        class FakeHybridServer:
            url = "http://127.0.0.1:5103"

            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                tmp_root=Path(tmp),
                native_text_layer_first=False,
                spawn_hybrid_per_conversion=True,
                prepare_dify_parent_child_chunks=False,
            )
            converter = PdfConverter(settings)
            fake_server = FakeHybridServer()

            with (
                patch("app.converter._try_acquire_primary_hybrid", return_value=None),
                patch("app.converter._start_temporary_hybrid_server", return_value=fake_server) as start_server,
                patch("app.converter._run_command") as run_command,
                patch("app.converter._read_rendered_markdown", return_value="markdown"),
            ):
                result = converter.convert_pdf_bytes(b"%PDF-1.4\n%%EOF", "policy.pdf")

        self.assertEqual(result, "markdown")
        start_server.assert_called_once()
        command = run_command.call_args.args[0]
        self.assertEqual(command[command.index("--hybrid-url") + 1], fake_server.url)
        self.assertTrue(fake_server.closed)

    def test_uses_shared_hybrid_server_when_per_conversion_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                tmp_root=Path(tmp),
                hybrid_url="http://localhost:5002",
                native_text_layer_first=False,
                spawn_hybrid_per_conversion=False,
                prepare_dify_parent_child_chunks=False,
            )
            converter = PdfConverter(settings)

            with (
                patch("app.converter._start_temporary_hybrid_server") as start_server,
                patch("app.converter._run_command") as run_command,
                patch("app.converter._read_rendered_markdown", return_value="markdown"),
            ):
                result = converter.convert_pdf_bytes(b"%PDF-1.4\n%%EOF", "policy.pdf")

        self.assertEqual(result, "markdown")
        start_server.assert_not_called()
        command = run_command.call_args.args[0]
        self.assertEqual(command[command.index("--hybrid-url") + 1], "http://localhost:5002")

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

    def test_rejects_hybrid_server_binary_as_converter_cli(self):
        settings = Settings(opendataloader_cli="opendataloader-pdf-hybrid")

        with self.assertRaisesRegex(ValueError, "ODL_CLI must point to opendataloader-pdf"):
            settings.validate()

    def test_retries_qpdf_repaired_pdf_after_original_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                tmp_root=Path(tmp),
                qpdf_repair_pdf_on_failure=True,
                repair_pdf_on_failure=False,
                rasterize_pdf_on_failure=False,
                prepare_dify_parent_child_chunks=False,
            )
            converter = PdfConverter(settings)

            with patch("app.converter._convert_pdf_file") as convert, patch("app.converter._repair_pdf_with_pikepdf") as repair:
                convert.side_effect = [ConversionError("broken xref"), "markdown"]

                result = converter.convert_pdf_bytes(b"%PDF-1.4\n%%EOF", "broken.pdf", use_ocr=True)

        self.assertEqual(result, "markdown")
        self.assertEqual(convert.call_count, 2)
        repair.assert_called_once()

    def test_retries_repaired_pdf_after_original_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                tmp_root=Path(tmp),
                qpdf_repair_pdf_on_failure=False,
                repair_pdf_on_failure=True,
                rasterize_pdf_on_failure=False,
                prepare_dify_parent_child_chunks=False,
            )
            converter = PdfConverter(settings)

            with patch("app.converter._convert_pdf_file") as convert, patch("app.converter._repair_pdf") as repair:
                convert.side_effect = [ConversionError("broken xref"), "markdown"]

                result = converter.convert_pdf_bytes(b"%PDF-1.4\n%%EOF", "broken.pdf", use_ocr=True)

        self.assertEqual(result, "markdown")
        self.assertEqual(convert.call_count, 2)
        repair.assert_called_once()

    def test_rasterizes_after_repair_conversion_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                tmp_root=Path(tmp),
                qpdf_repair_pdf_on_failure=False,
                repair_pdf_on_failure=True,
                rasterize_pdf_on_failure=True,
                prepare_dify_parent_child_chunks=False,
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

                result = converter.convert_pdf_bytes(b"%PDF-1.4\n%%EOF", "broken.pdf", use_ocr=True)

        self.assertEqual(result, "ocr markdown")
        self.assertEqual(convert.call_count, 3)
        repair.assert_called_once()
        rasterize.assert_called_once()

    def test_rasterizes_directly_when_original_needs_ocr_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                tmp_root=Path(tmp),
                qpdf_repair_pdf_on_failure=True,
                repair_pdf_on_failure=True,
                rasterize_pdf_on_failure=True,
                prepare_dify_parent_child_chunks=False,
            )
            converter = PdfConverter(settings)

            with (
                patch("app.converter._convert_pdf_file") as convert,
                patch("app.converter._rasterize_pdf") as rasterize,
                patch("app.converter._repair_pdf") as repair,
                patch("app.converter._repair_pdf_with_pikepdf") as qpdf_repair,
            ):
                convert.side_effect = [
                    ConversionError("OpenDataLoader produced no usable text or image description; OCR fallback required."),
                    "ocr markdown",
                ]

                result = converter.convert_pdf_bytes(b"%PDF-1.4\n%%EOF", "scan.pdf", use_ocr=True)

        self.assertEqual(result, "ocr markdown")
        self.assertEqual(convert.call_count, 2)
        rasterize.assert_called_once()
        repair.assert_not_called()
        qpdf_repair.assert_not_called()

    def test_does_not_run_ocr_fallback_unless_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                tmp_root=Path(tmp),
                qpdf_repair_pdf_on_failure=True,
                repair_pdf_on_failure=True,
                rasterize_pdf_on_failure=True,
                prepare_dify_parent_child_chunks=False,
            )
            converter = PdfConverter(settings)

            with (
                patch("app.converter._convert_pdf_file") as convert,
                patch("app.converter._rasterize_pdf") as rasterize,
                patch("app.converter._repair_pdf") as repair,
                patch("app.converter._repair_pdf_with_pikepdf") as qpdf_repair,
            ):
                convert.side_effect = ConversionError(
                    "OpenDataLoader produced no usable text or image description; OCR fallback required."
                )

                with self.assertRaisesRegex(ConversionError, "OCR was not selected"):
                    converter.convert_pdf_bytes(b"%PDF-1.4\n%%EOF", "scan.pdf")

        convert.assert_called_once()
        rasterize.assert_not_called()
        repair.assert_not_called()
        qpdf_repair.assert_not_called()

    def test_attempts_ocr_fallback_only_once_when_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                tmp_root=Path(tmp),
                qpdf_repair_pdf_on_failure=True,
                repair_pdf_on_failure=True,
                rasterize_pdf_on_failure=True,
                prepare_dify_parent_child_chunks=False,
            )
            converter = PdfConverter(settings)

            with (
                patch("app.converter._convert_pdf_file") as convert,
                patch("app.converter._rasterize_pdf") as rasterize,
                patch("app.converter._repair_pdf") as repair,
                patch("app.converter._repair_pdf_with_pikepdf") as qpdf_repair,
            ):
                convert.side_effect = [
                    ConversionError("OpenDataLoader produced no usable text or image description; OCR fallback required."),
                    ConversionError("OpenDataLoader produced visual pages without OCR text or image description."),
                ]

                with self.assertRaisesRegex(ConversionError, "after OCR fallback attempt"):
                    converter.convert_pdf_bytes(b"%PDF-1.4\n%%EOF", "scan.pdf", use_ocr=True)

        self.assertEqual(convert.call_count, 2)
        rasterize.assert_called_once()
        repair.assert_not_called()
        qpdf_repair.assert_not_called()

    def test_rejects_page_separator_only_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text("\n\n--- Page 1 ---\n\n--- Page 2 ---\n\n", encoding="utf-8")

            with self.assertRaises(ConversionError):
                _read_rendered_markdown(output_dir)

    def test_rejects_image_link_only_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text(
                "\n\n--- Page 1 ---\n\n![image 1](images/page1.png)\n\n",
                encoding="utf-8",
            )

            with self.assertRaises(ConversionError):
                _read_rendered_markdown(output_dir)

    def test_rejects_image_only_placeholder_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text(
                "\n\n--- Page 1 ---\n\n> Image-only page. No embedded text layer was available.\n\n",
                encoding="utf-8",
            )

            with self.assertRaises(ConversionError):
                _read_rendered_markdown(output_dir)

    def test_allows_image_only_placeholder_when_ocr_is_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text(
                "\n\n--- Page 1 ---\n\n> Image-only page. No embedded text layer was available.\n\n",
                encoding="utf-8",
            )

            result = _read_rendered_markdown(output_dir, allow_unresolved_visual_pages=True)

        self.assertIn("Image-only page", result)

    def test_prefers_json_image_description_over_markdown_image_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text(
                "\n\n--- Page 1 ---\n\n# Product Overview\n\n![image 1](images/page1.png)\n\n",
                encoding="utf-8",
            )
            (output_dir / "out.json").write_text(
                json.dumps(
                    {
                        "kids": [
                            {"type": "heading", "page number": 1, "heading level": 1, "content": "Product Overview"},
                            {
                                "type": "picture",
                                "page number": 1,
                                "description": "The image says the application period is March 4 to March 8.",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = _read_rendered_markdown(output_dir)

        self.assertIn("--- Page 1 ---", result)
        self.assertIn("# Product Overview", result)
        self.assertIn("**Image summary:** The image says the application period is March 4 to March 8.", result)

    def test_rejects_visual_page_with_heading_but_no_ocr_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text("\n\n--- Page 1 ---\n\n# Product Overview\n\n", encoding="utf-8")
            (output_dir / "out.json").write_text(
                json.dumps(
                    {
                        "kids": [
                            {"type": "heading", "page number": 1, "heading level": 1, "content": "Product Overview"},
                            {"type": "picture", "page number": 1},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ConversionError):
                _read_rendered_markdown(output_dir)

    def test_allows_visual_page_without_ocr_text_when_ocr_is_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text("\n\n--- Page 1 ---\n\n# Product Overview\n\n", encoding="utf-8")
            (output_dir / "out.json").write_text(
                json.dumps(
                    {
                        "kids": [
                            {"type": "heading", "page number": 1, "heading level": 1, "content": "Product Overview"},
                            {"type": "picture", "page number": 1},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = _read_rendered_markdown(output_dir, allow_unresolved_visual_pages=True)

        self.assertIn("# Product Overview", result)
        self.assertIn("이미지/도식 중심 페이지", result)

    def test_reads_native_markdown_with_page_order_and_tables(self):
        markdown = "\n\n--- Page 26 ---\n\n| Category | Unit |\n| --- | --- |\n| Engineering | Free Major |\n"
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text(markdown, encoding="utf-8")

            result = _read_generated_markdown(output_dir)

        self.assertIn("--- Page 26 ---", result)
        self.assertIn("| Category | Unit |", result)

    def test_rejects_empty_table_only_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
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
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text(markdown, encoding="utf-8")

            with self.assertRaises(ConversionError):
                _read_generated_markdown(output_dir)


if __name__ == "__main__":
    unittest.main()
