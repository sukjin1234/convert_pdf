import json
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
    EmbeddedImage,
    PdfConverter,
    _ocr_embedded_images_with_opendataloader,
    _read_generated_markdown,
    _read_rendered_markdown,
    _run_command,
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

    def test_resource_shortage_retries_opendataloader_command_until_success(self):
        settings = Settings(
            resource_retry_interval_seconds=1,
            resource_retry_max_interval_seconds=1,
            resource_retry_max_attempts=3,
        )

        with (
            patch(
                "app.converter.subprocess.Popen",
                side_effect=[
                    _FakeProcess(returncode=1, stderr="CUDA out of memory"),
                    _FakeProcess(returncode=0),
                ],
            ) as popen,
            patch("app.converter.time.sleep") as sleep,
        ):
            _run_command(["opendataloader-pdf", "input.pdf"], settings)

        self.assertEqual(popen.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_resource_shortage_retry_respects_max_attempts(self):
        settings = Settings(
            resource_retry_interval_seconds=1,
            resource_retry_max_interval_seconds=1,
            resource_retry_max_attempts=2,
        )

        with (
            patch(
                "app.converter.subprocess.Popen",
                side_effect=[
                    _FakeProcess(returncode=1, stderr="service unavailable"),
                    _FakeProcess(returncode=1, stderr="service unavailable"),
                ],
            ) as popen,
            patch("app.converter.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(ConversionError, "busy or unavailable"):
                _run_command(["opendataloader-pdf", "input.pdf"], settings)

        self.assertEqual(popen.call_count, 2)
        sleep.assert_called_once_with(1)

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

    def test_embedded_image_ocr_filters_and_batches_images(self):
        settings = Settings(
            embedded_image_ocr_max_images=3,
            embedded_image_ocr_batch_size=2,
            embedded_image_ocr_min_pixels=100,
            embedded_image_ocr_max_image_bytes=4,
        )
        images = [
            EmbeddedImage("too-small.png", b"ok", width=5, height=5),
            EmbeddedImage("page-1.png", b"ok", width=50, height=50),
            EmbeddedImage("too-large.png", b"12345", width=50, height=50),
            EmbeddedImage("page-2.png", b"ok", width=50, height=50),
            EmbeddedImage("page-3.png", b"ok", width=50, height=50),
            EmbeddedImage("over-limit.png", b"ok", width=50, height=50),
        ]
        batches: list[list[str]] = []

        def fake_images_to_pdf(batch, output_path, _settings):
            batches.append([image.name for image in batch])
            output_path.write_bytes(b"%PDF-1.4\n%%EOF")

        def fake_convert(input_path, _output_dir, _settings, **kwargs):
            self.assertEqual(kwargs, {"hybrid_mode": "full", "image_output": "off"})
            return f"text from {Path(input_path).stem}"

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("app.converter._images_to_pdf", side_effect=fake_images_to_pdf),
                patch("app.converter._convert_pdf_file", side_effect=fake_convert),
            ):
                result = _ocr_embedded_images_with_opendataloader(images, Path(tmp), settings, source_name="images.pdf")

        self.assertEqual(batches, [["page-1.png", "page-2.png"], ["page-3.png"]])
        self.assertIn("text from embedded-images-001", result)
        self.assertIn("text from embedded-images-002", result)

    def test_embedded_image_ocr_treats_no_markdown_as_no_text(self):
        settings = Settings()
        images = [EmbeddedImage("blank.png", b"ok", width=200, height=200)]

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("app.converter._images_to_pdf"),
                patch("app.converter._convert_pdf_file", side_effect=ConversionError("OpenDataLoader produced no Markdown.")),
            ):
                result = _ocr_embedded_images_with_opendataloader(images, Path(tmp), settings, source_name="blank.pdf")

        self.assertEqual(result, "")

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

    def test_rendered_markdown_uses_json_layout_pairs_for_card_grid(self):
        doc = {
            "kids": [
                {"type": "heading", "page number": 1, "bounding box": [70, 700, 210, 735], "font size": 26, "content": "국토교통부"},
                {"type": "heading", "page number": 1, "bounding box": [305, 700, 500, 735], "font size": 26, "content": "산업통상자원부"},
                {"type": "heading", "page number": 1, "bounding box": [560, 700, 690, 735], "font size": 26, "content": "교육부"},
                {"type": "paragraph", "page number": 1, "bounding box": [50, 610, 230, 650], "font size": 15, "content": "공간정보 특성화 전문대학"},
                {"type": "paragraph", "page number": 1, "bounding box": [320, 590, 500, 655], "font size": 15, "content": "창의융합형 공학인재 양성지원사업"},
                {"type": "paragraph", "page number": 1, "bounding box": [570, 590, 700, 655], "font size": 15, "content": "고교 장애학생 대학체험지원"},
            ]
        }
        bad_markdown = (
            "--- Page 1 ---\n\n"
            "국토교통부\n\n산업통상자원부\n\n교육부\n\n"
            "공간정보 특성화 전문대학\n\n창의융합형 공학인재 양성지원사업\n\n고교 장애학생 대학체험지원\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "out.md").write_text(bad_markdown, encoding="utf-8")
            (output_dir / "out.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

            result = _read_rendered_markdown(output_dir)

        self.assertIn("| 국토교통부 | 공간정보 특성화 전문대학 |", result)
        self.assertIn("| 산업통상자원부 | 창의융합형 공학인재 양성지원사업 |", result)
        self.assertLess(result.index("공간정보 특성화 전문대학"), result.index("산업통상자원부"))

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


class _FakeProcess:
    pid = 12345

    def __init__(self, *, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    def communicate(self, timeout: int | None = None):
        return self._stdout, self._stderr


if __name__ == "__main__":
    unittest.main()
