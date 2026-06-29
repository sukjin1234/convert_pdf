import unittest
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from app.config import Settings
from app.converter import (
    ConversionError,
    DocumentConverter,
    PdfConverter,
    _read_generated_markdown,
    _read_rendered_markdown,
    build_opendataloader_native_command,
    build_opendataloader_command,
    sanitize_filename,
)


class ConverterTest(unittest.TestCase):
    def test_document_converter_reads_markdown_bytes(self):
        converter = DocumentConverter(Settings())

        result = converter.convert_bytes(b"# Policy\n\nAllowed domains: all", "policy.md")

        self.assertIn("# Policy", result)
        self.assertIn("Allowed domains", result)

    def test_document_converter_reads_cp949_text_bytes(self):
        converter = DocumentConverter(Settings())
        content = "담당자: 입학처\n연락처: 02-1234-5678".encode("cp949")

        result = converter.convert_bytes(content, "contact.txt")

        self.assertIn("담당자: 입학처", result)
        self.assertIn("02-1234-5678", result)

    def test_document_converter_reads_csv_as_markdown_table(self):
        converter = DocumentConverter(Settings())

        result = converter.convert_bytes(
            "항목,값\n요금,10000원\n기간,30일\n".encode("utf-8"),
            "pricing.csv",
        )

        self.assertIn("| 항목 | 값 |", result)
        self.assertIn("| 요금 | 10000원 |", result)

    def test_document_converter_reads_docx_paragraphs_and_tables(self):
        converter = DocumentConverter(Settings())
        content = _minimal_docx_bytes()

        result = converter.convert_bytes(content, "manual.docx")

        self.assertIn("# 운영 정책", result)
        self.assertIn("장애 대응은 30분 이내에 시작합니다.", result)
        self.assertIn("| 항목 | 값 |", result)
        self.assertIn("| SLA | 99.9% |", result)

    def test_document_converter_ocr_extracts_docx_embedded_images(self):
        converter = DocumentConverter(Settings())
        content = _minimal_docx_bytes(images={"word/media/image1.png": b"fake-png"})

        with patch("app.converter._ocr_embedded_images_with_opendataloader", return_value="이미지 안의 공지 텍스트") as ocr:
            result = converter.convert_bytes(content, "manual.docx")

        ocr.assert_called_once()
        self.assertIn("## Embedded Image OCR", result)
        self.assertIn("이미지 안의 공지 텍스트", result)

    def test_document_converter_rejects_unsupported_suffix(self):
        converter = DocumentConverter(Settings())

        with self.assertRaisesRegex(ConversionError, "Unsupported document type"):
            converter.convert_bytes(b"hello", "legacy.doc")

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

    def test_pdf_with_images_ocr_only_embedded_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                tmp_root=Path(tmp),
                qpdf_repair_pdf_on_failure=False,
                repair_pdf_on_failure=False,
                rasterize_pdf_on_failure=False,
            )
            converter = PdfConverter(settings)

            with (
                patch("app.converter._convert_pdf_file", return_value="text markdown") as convert,
                patch("app.converter._read_pdf_embedded_images", return_value=[("page-001-image-001.png", b"image")]),
                patch("app.converter._ocr_embedded_images_with_opendataloader", return_value="이미지 OCR 텍스트") as ocr,
            ):
                result = converter.convert_pdf_bytes(b"%PDF-1.4\n%%EOF", "image.pdf")

        self.assertIn("text markdown", result)
        self.assertIn("## Embedded Image OCR", result)
        self.assertIn("이미지 OCR 텍스트", result)
        self.assertEqual(convert.call_count, 1)
        self.assertEqual(convert.call_args.kwargs, {})
        ocr.assert_called_once()

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

    def test_falls_back_when_default_tmp_root_is_not_writable(self):
        original_mkdtemp = tempfile.mkdtemp
        with tempfile.TemporaryDirectory() as tmp:
            fallback_root = Path(tmp)
            blocked_root = fallback_root / "blocked-root"
            seen_input_paths = []
            settings = Settings(
                tmp_root=blocked_root,
                qpdf_repair_pdf_on_failure=False,
                repair_pdf_on_failure=False,
                rasterize_pdf_on_failure=False,
            )
            converter = PdfConverter(settings)

            def fake_mkdtemp(prefix: str, dir: str | Path):
                if Path(dir) == blocked_root:
                    raise PermissionError("denied")
                return original_mkdtemp(prefix=prefix, dir=dir)

            def fake_convert(input_path, _output_dir, _settings, **_kwargs):
                seen_input_paths.append(Path(input_path))
                return "markdown"

            with (
                patch("app.converter.tempfile.gettempdir", return_value=str(fallback_root)),
                patch("app.converter.tempfile.mkdtemp", side_effect=fake_mkdtemp),
                patch("app.converter._convert_pdf_file", side_effect=fake_convert),
            ):
                result = converter.convert_pdf_bytes(b"%PDF-1.4\n%%EOF", "fallback.pdf")

        self.assertEqual(result, "markdown")
        self.assertEqual(len(seen_input_paths), 1)
        self.assertEqual(seen_input_paths[0].parents[1], fallback_root)

    def test_retries_qpdf_repaired_pdf_after_original_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
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
        with tempfile.TemporaryDirectory() as tmp:
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
        with tempfile.TemporaryDirectory() as tmp:
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
                "\n\n--- Page 1 ---\n\n![image 1](<images/page1.png>)\n\n",
                encoding="utf-8",
            )

            with self.assertRaises(ConversionError):
                _read_rendered_markdown(output_dir)

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


def _minimal_docx_bytes(images: dict[str, bytes] | None = None) -> bytes:
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{namespace}">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
      <w:r><w:t>운영 정책</w:t></w:r>
    </w:p>
    <w:p><w:r><w:t>장애 대응은 30분 이내에 시작합니다.</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>항목</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>값</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>SLA</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>99.9%</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
        for name, content in (images or {}).items():
            archive.writestr(name, content)
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
