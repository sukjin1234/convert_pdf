from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree

from .config import Settings, get_settings
from .markdown import render_document_pages_to_markdown, render_document_to_markdown

logger = logging.getLogger(__name__)


class ConversionError(RuntimeError):
    pass


SUPPORTED_DOCUMENT_SUFFIXES = {
    ".pdf",
    ".txt",
    ".log",
    ".md",
    ".markdown",
    ".docx",
    ".csv",
    ".tsv",
}
TEXT_DOCUMENT_SUFFIXES = {".txt", ".log", ".md", ".markdown"}
CSV_DOCUMENT_SUFFIXES = {".csv", ".tsv"}
IMAGE_DOCUMENT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_DOCUMENT_LABEL = "PDF, TXT, Markdown, DOCX, CSV, TSV"
WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


@dataclass(frozen=True)
class DocumentConverter:
    settings: Settings

    def __init__(self, settings: Settings | None = None):
        object.__setattr__(self, "settings", settings or get_settings())

    def convert_bytes(self, file_bytes: bytes, filename: str = "document") -> str:
        if not file_bytes:
            raise ConversionError("Empty document input.")

        safe_name = sanitize_filename(filename or "document")
        suffix = Path(safe_name).suffix.lower()
        if suffix == ".pdf" or (not suffix and file_bytes.lstrip().startswith(b"%PDF-")):
            return PdfConverter(self.settings).convert_pdf_bytes(file_bytes, safe_name)
        if suffix in TEXT_DOCUMENT_SUFFIXES:
            return _read_text_document(file_bytes)
        if suffix in CSV_DOCUMENT_SUFFIXES:
            delimiter = "\t" if suffix == ".tsv" else ","
            return _read_csv_document(file_bytes, delimiter=delimiter)
        if suffix == ".docx":
            return _read_docx_document(file_bytes, self.settings, safe_name)

        if file_bytes.lstrip().startswith(b"%PDF-"):
            return PdfConverter(self.settings).convert_pdf_bytes(file_bytes, safe_name)
        raise ConversionError(
            f"Unsupported document type '{suffix or '(none)'}'. "
            f"Supported types: {SUPPORTED_DOCUMENT_LABEL}."
        )


@dataclass(frozen=True)
class PdfConverter:
    settings: Settings

    def __init__(self, settings: Settings | None = None):
        object.__setattr__(self, "settings", settings or get_settings())

    def convert_pdf_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> str:
        self._validate_pdf_bytes(pdf_bytes)
        safe_name = sanitize_filename(filename)
        if not safe_name.lower().endswith(".pdf"):
            safe_name = f"{Path(safe_name).stem or 'document'}.pdf"

        tmp_dir = _create_conversion_tmp_dir(self.settings)
        try:
            input_path = tmp_dir / safe_name
            input_path.write_bytes(pdf_bytes)

            errors = []
            repaired_path: Path | None = None
            qpdf_repaired_path: Path | None = None

            try:
                markdown = _convert_pdf_file(input_path, tmp_dir / "output-original", self.settings)
                return _append_pdf_image_ocr(
                    markdown,
                    input_path,
                    tmp_dir,
                    self.settings,
                    source_name=safe_name,
                )
            except ConversionError as exc:
                errors.append(f"original: {exc}")
                logger.info("Original PDF conversion failed; trying fallbacks. error=%s", exc)

            if self.settings.qpdf_repair_pdf_on_failure:
                qpdf_repaired_path = tmp_dir / "qpdf-repaired.pdf"
                try:
                    _repair_pdf_with_pikepdf(input_path, qpdf_repaired_path)
                    markdown = _convert_pdf_file(
                        qpdf_repaired_path,
                        tmp_dir / "output-qpdf-repaired",
                        self.settings,
                    )
                    return _append_pdf_image_ocr(
                        markdown,
                        qpdf_repaired_path,
                        tmp_dir,
                        self.settings,
                        source_name=safe_name,
                    )
                except ConversionError as exc:
                    errors.append(f"qpdf-repaired: {exc}")
                    logger.info("qpdf/pikepdf repaired PDF conversion failed. error=%s", exc)

            if self.settings.repair_pdf_on_failure:
                repaired_path = tmp_dir / "repaired.pdf"
                repair_source = qpdf_repaired_path if qpdf_repaired_path and qpdf_repaired_path.exists() else input_path
                try:
                    _repair_pdf(repair_source, repaired_path)
                    markdown = _convert_pdf_file(repaired_path, tmp_dir / "output-repaired", self.settings)
                    return _append_pdf_image_ocr(
                        markdown,
                        repaired_path,
                        tmp_dir,
                        self.settings,
                        source_name=safe_name,
                    )
                except ConversionError as exc:
                    errors.append(f"repaired: {exc}")
                    logger.info("Repaired PDF conversion failed. error=%s", exc)

                if repaired_path.exists():
                    try:
                        return _convert_pdf_file(
                            repaired_path,
                            tmp_dir / "output-repaired-ocr",
                            self.settings,
                            hybrid_mode="full",
                            image_output="off",
                        )
                    except ConversionError as exc:
                        errors.append(f"repaired-ocr: {exc}")
                        logger.info("Repaired PDF OCR conversion failed. error=%s", exc)

            if self.settings.rasterize_pdf_on_failure:
                rasterized_path = tmp_dir / "rasterized.pdf"
                raster_source = _first_existing_path(repaired_path, qpdf_repaired_path, input_path)
                try:
                    _rasterize_pdf(raster_source, rasterized_path, self.settings.rasterize_dpi)
                    return _convert_pdf_file(
                        rasterized_path,
                        tmp_dir / "output-rasterized",
                        self.settings,
                        hybrid_mode="full",
                        image_output="off",
                    )
                except ConversionError as exc:
                    errors.append(f"rasterized: {exc}")
                    logger.info("Rasterized PDF conversion failed. error=%s", exc)

            raise ConversionError("PDF conversion failed after fallback attempts: " + "; ".join(errors))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _validate_pdf_bytes(self, pdf_bytes: bytes) -> None:
        if not pdf_bytes:
            raise ConversionError("Empty PDF input.")
        if len(pdf_bytes) > self.settings.max_pdf_bytes:
            raise ConversionError("PDF input exceeds ODL_MAX_PDF_BYTES.")
        if self.settings.require_pdf_signature and not pdf_bytes.lstrip().startswith(b"%PDF-"):
            raise ConversionError("Input is not a valid PDF stream.")
        if _looks_text_corrupted_pdf(pdf_bytes):
            raise ConversionError(
                "PDF binary streams appear text-decoded or corrupted. "
                "Upload the original binary PDF again."
            )


def sanitize_filename(filename: str) -> str:
    name = Path(filename or "document.pdf").name
    name = re.sub(r"[^A-Za-z0-9_ .-]+", "_", name).strip(" .")
    return name or "document.pdf"


def _read_text_document(file_bytes: bytes) -> str:
    markdown = _normalize_markdown(_decode_text_bytes(file_bytes))
    if not _has_meaningful_markdown(markdown):
        raise ConversionError("Text document produced no Markdown.")
    return markdown


def _decode_text_bytes(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("latin-1", errors="replace")


def _read_csv_document(file_bytes: bytes, *, delimiter: str) -> str:
    text = _decode_text_bytes(file_bytes)
    rows = [
        [_normalize_table_cell(cell) for cell in row]
        for row in csv.reader(text.splitlines(), delimiter=delimiter)
        if any(cell.strip() for cell in row)
    ]
    markdown = _markdown_table(rows)
    if not _has_meaningful_markdown(markdown):
        raise ConversionError("CSV document produced no Markdown.")
    return markdown


def _read_docx_document(file_bytes: bytes, settings: Settings, filename: str) -> str:
    images: list[tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
            parts = _read_docx_main_parts(archive)
            images = _read_docx_media_images(archive)
    except zipfile.BadZipFile as exc:
        raise ConversionError("Input is not a valid DOCX file.") from exc
    except KeyError as exc:
        raise ConversionError("DOCX file is missing word/document.xml.") from exc
    except ElementTree.ParseError as exc:
        raise ConversionError("DOCX XML is not readable.") from exc

    if images:
        tmp_dir = _create_conversion_tmp_dir(settings)
        try:
            try:
                image_ocr = _ocr_embedded_images_with_opendataloader(
                    images,
                    tmp_dir,
                    settings,
                    source_name=filename,
                )
            except ConversionError as exc:
                logger.warning("DOCX embedded image OCR failed for %s: %s", filename, exc)
                image_ocr = ""
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if image_ocr:
            parts.append("## Embedded Image OCR\n\n" + image_ocr)

    markdown = _normalize_markdown("\n\n".join(part for part in parts if part.strip()))
    if not _has_meaningful_markdown(markdown):
        raise ConversionError("DOCX document produced no Markdown.")
    return markdown


def _read_docx_main_parts(archive: zipfile.ZipFile) -> list[str]:
    parts = _read_docx_xml_member(archive, "word/document.xml")
    auxiliary_members = [
        member
        for member in archive.namelist()
        if re.match(r"word/(header|footer|footnotes|endnotes|comments)\d*\.xml$", member)
    ]
    for member in sorted(auxiliary_members):
        blocks = _read_docx_xml_member(archive, member)
        if blocks:
            parts.append(f"## {_docx_member_label(member)}")
            parts.extend(blocks)
    return parts


def _read_docx_media_images(archive: zipfile.ZipFile) -> list[tuple[str, bytes]]:
    images: list[tuple[str, bytes]] = []
    for member_name in sorted(archive.namelist()):
        suffix = Path(member_name).suffix.lower()
        if not member_name.startswith("word/media/") or suffix not in IMAGE_DOCUMENT_SUFFIXES:
            continue
        try:
            image_bytes = archive.read(member_name)
        except KeyError:
            continue
        if image_bytes:
            images.append((member_name, image_bytes))
    return images


def _ocr_embedded_images_with_opendataloader(
    images: list[tuple[str, bytes]],
    tmp_dir: Path,
    settings: Settings,
    *,
    source_name: str,
) -> str:
    if not images:
        return ""

    image_pdf_path = tmp_dir / "embedded-images.pdf"
    _images_to_pdf(images, image_pdf_path)
    try:
        markdown = _convert_pdf_file(
            image_pdf_path,
            tmp_dir / "output-embedded-images-ocr",
            settings,
            hybrid_mode="full",
            image_output="off",
        )
    except ConversionError as exc:
        raise ConversionError(f"Embedded image OCR failed for {source_name}: {exc}") from exc
    return _normalize_markdown(markdown)


def _append_pdf_image_ocr(
    markdown: str,
    pdf_path: Path,
    tmp_dir: Path,
    settings: Settings,
    *,
    source_name: str,
) -> str:
    try:
        images = _read_pdf_embedded_images(pdf_path)
    except ConversionError as exc:
        logger.warning("PDF image extraction failed for %s: %s", source_name, exc)
        return markdown

    if not images:
        return markdown

    try:
        image_ocr = _ocr_embedded_images_with_opendataloader(
            images,
            tmp_dir,
            settings,
            source_name=source_name,
        )
    except ConversionError as exc:
        logger.warning("PDF embedded image OCR failed for %s: %s", source_name, exc)
        return markdown

    if not _has_meaningful_markdown(image_ocr):
        return markdown
    return _normalize_markdown(markdown + "\n\n## Embedded Image OCR\n\n" + image_ocr)


def _read_pdf_embedded_images(pdf_path: Path) -> list[tuple[str, bytes]]:
    try:
        fitz = _load_pymupdf()
        doc = fitz.open(str(pdf_path))
    except ConversionError as exc:
        raise ConversionError("PyMuPDF is required for PDF image extraction.") from exc
    except Exception as exc:
        raise ConversionError("Could not open PDF for image extraction.") from exc

    images: list[tuple[str, bytes]] = []
    seen_xrefs: set[int] = set()
    try:
        for page_number, page in enumerate(doc, start=1):
            for image_number, image_info in enumerate(page.get_images(full=True), start=1):
                xref = int(image_info[0])
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    extracted = doc.extract_image(xref)
                except Exception:
                    logger.info("Could not extract PDF image. pdf=%s page=%s xref=%s", pdf_path, page_number, xref)
                    continue
                image_bytes = extracted.get("image") or b""
                if not image_bytes:
                    continue
                ext = str(extracted.get("ext") or "png").lower()
                image_name = f"{pdf_path.stem}-page{page_number:03d}-image{image_number:03d}.{ext}"
                images.append((image_name, image_bytes))
        return images
    except Exception as exc:
        raise ConversionError("Could not inspect PDF images.") from exc
    finally:
        doc.close()


def _images_to_pdf(images: list[tuple[str, bytes]], output_path: Path) -> None:
    fitz = _load_pymupdf()
    target = fitz.open()
    try:
        for image_name, image_bytes in images:
            image_doc = None
            image_pdf = None
            try:
                image_doc = fitz.open(stream=image_bytes, filetype=_pymupdf_image_type(image_name))
                image_pdf_bytes = image_doc.convert_to_pdf()
                image_pdf = fitz.open(stream=image_pdf_bytes, filetype="pdf")
                target.insert_pdf(image_pdf)
            except Exception as exc:
                logger.info("Could not prepare embedded image for OCR: %s error=%s", image_name, exc)
                continue
            finally:
                if image_pdf is not None:
                    image_pdf.close()
                if image_doc is not None:
                    image_doc.close()

        if target.page_count <= 0:
            raise ConversionError("Document contains no OCR-readable images.")
        target.save(str(output_path), garbage=4, deflate=True)
    finally:
        target.close()


def _pymupdf_image_type(image_name: str) -> str:
    suffix = Path(image_name).suffix.lower().lstrip(".")
    return "jpeg" if suffix == "jpg" else suffix


def _read_docx_xml_member(archive: zipfile.ZipFile, member_name: str) -> list[str]:
    root = ElementTree.fromstring(archive.read(member_name))
    container = _first_child(root, "body") or root
    return _docx_blocks(container)


def _docx_blocks(container: ElementTree.Element) -> list[str]:
    blocks: list[str] = []
    for child in list(container):
        local_name = _xml_local_name(child.tag)
        if local_name == "p":
            paragraph = _docx_paragraph_markdown(child)
            if paragraph:
                blocks.append(paragraph)
        elif local_name == "tbl":
            table = _docx_table_markdown(child)
            if table:
                blocks.append(table)
    return blocks


def _docx_paragraph_markdown(paragraph: ElementTree.Element) -> str:
    text = _normalize_inline_text(_docx_text(paragraph))
    if not text:
        return ""

    style = _docx_paragraph_style(paragraph).lower()
    heading_match = re.search(r"heading\s*([1-6])|heading([1-6])", style)
    if heading_match:
        level = int(heading_match.group(1) or heading_match.group(2))
        return f"{'#' * level} {text}"
    if _docx_is_numbered_paragraph(paragraph):
        return f"- {text}"
    return text


def _docx_paragraph_style(paragraph: ElementTree.Element) -> str:
    ppr = _first_child(paragraph, "pPr")
    style = _first_child(ppr, "pStyle") if ppr is not None else None
    if style is None:
        return ""
    return (
        style.attrib.get(f"{{{WORD_NAMESPACE}}}val")
        or style.attrib.get("val")
        or ""
    )


def _docx_is_numbered_paragraph(paragraph: ElementTree.Element) -> bool:
    ppr = _first_child(paragraph, "pPr")
    return ppr is not None and _first_child(ppr, "numPr") is not None


def _docx_table_markdown(table: ElementTree.Element) -> str:
    rows: list[list[str]] = []
    for row in _children(table, "tr"):
        values = [_normalize_table_cell(_docx_text(cell)) for cell in _children(row, "tc")]
        if any(values):
            rows.append(values)
    return _markdown_table(rows)


def _markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized_rows = [
        [_escape_markdown_table_cell(cell) for cell in row + [""] * (width - len(row))]
        for row in rows
    ]
    header = normalized_rows[0]
    body = normalized_rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _docx_text(element: ElementTree.Element) -> str:
    parts: list[str] = []
    for node in element.iter():
        local_name = _xml_local_name(node.tag)
        if local_name == "t" and node.text:
            parts.append(node.text)
        elif local_name == "tab":
            parts.append("\t")
        elif local_name in {"br", "cr"}:
            parts.append("\n")
    return "".join(parts)


def _normalize_inline_text(value: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _normalize_table_cell(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _escape_markdown_table_cell(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace("|", "\\|")


def _docx_member_label(member_name: str) -> str:
    stem = Path(member_name).stem
    if stem.startswith("header"):
        return "DOCX Header"
    if stem.startswith("footer"):
        return "DOCX Footer"
    if stem == "footnotes":
        return "DOCX Footnotes"
    if stem == "endnotes":
        return "DOCX Endnotes"
    if stem == "comments":
        return "DOCX Comments"
    return f"DOCX {stem}"


def _first_child(element: ElementTree.Element | None, local_name: str) -> ElementTree.Element | None:
    if element is None:
        return None
    for child in list(element):
        if _xml_local_name(child.tag) == local_name:
            return child
    return None


def _children(element: ElementTree.Element, local_name: str) -> list[ElementTree.Element]:
    return [child for child in list(element) if _xml_local_name(child.tag) == local_name]


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _create_conversion_tmp_dir(settings: Settings) -> Path:
    candidates = [settings.tmp_root]
    if not os.getenv("ODL_TMP_ROOT"):
        system_tmp = Path(tempfile.gettempdir())
        if system_tmp != settings.tmp_root:
            candidates.append(system_tmp)

    errors = []
    for root in candidates:
        try:
            root.mkdir(parents=True, exist_ok=True)
            return Path(tempfile.mkdtemp(prefix="convert-", dir=root))
        except OSError as exc:
            errors.append(f"{root}: {exc}")
            logger.warning("Conversion temp root is not writable. root=%s error=%s", root, exc)

    raise ConversionError(
        "Could not create a writable conversion temp directory. "
        "Set ODL_TMP_ROOT to a directory writable by the FastAPI process. "
        + "; ".join(errors)
    )


def _convert_pdf_file(
    input_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    hybrid_mode: str | None = None,
    image_output: str = "external",
) -> str:
    native_error = None
    if settings.native_text_layer_first and hybrid_mode is None:
        try:
            return _convert_pdf_file_native(input_path, output_dir.with_name(f"{output_dir.name}-native"), settings)
        except ConversionError as exc:
            native_error = exc
            logger.info("Native PDF conversion failed; trying hybrid. error=%s", exc)

    try:
        return _convert_pdf_file_hybrid(
            input_path,
            output_dir,
            settings,
            hybrid_mode=hybrid_mode,
            image_output=image_output,
        )
    except ConversionError as exc:
        if native_error is not None:
            raise ConversionError(f"{exc}; native: {native_error}") from exc
        raise


def _convert_pdf_file_native(input_path: Path, output_dir: Path, settings: Settings) -> str:
    output_dir.mkdir()
    command = build_opendataloader_native_command(input_path, output_dir, settings)
    _run_command(command, settings.conversion_timeout_seconds)
    return _read_generated_markdown(output_dir)


def _convert_pdf_file_hybrid(
    input_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    hybrid_mode: str | None = None,
    image_output: str = "external",
) -> str:
    output_dir.mkdir()
    command = build_opendataloader_command(
        input_path,
        output_dir,
        settings,
        hybrid_mode=hybrid_mode,
        image_output=image_output,
    )
    _run_command(command, settings.conversion_timeout_seconds)
    return _read_rendered_markdown(output_dir)


def build_opendataloader_command(
    input_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    hybrid_mode: str | None = None,
    image_output: str = "external",
) -> list[str]:
    command = [
        settings.opendataloader_cli,
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--format",
        "json,markdown",
        "--hybrid",
        settings.hybrid_backend,
        "--hybrid-mode",
        hybrid_mode or settings.hybrid_mode,
        "--hybrid-url",
        settings.hybrid_url,
        "--hybrid-timeout",
        str(settings.hybrid_timeout_ms),
        "--table-method",
        settings.table_method,
        "--reading-order",
        settings.reading_order,
        "--image-output",
        image_output,
        "--markdown-page-separator",
        "\n\n--- Page %page-number% ---\n\n",
        "--quiet",
    ]
    if settings.use_struct_tree:
        command.append("--use-struct-tree")
    return command


def build_opendataloader_native_command(input_path: Path, output_dir: Path, settings: Settings) -> list[str]:
    if settings.opendataloader_jar:
        command = [
            "java",
            "-jar",
            settings.opendataloader_jar,
            str(input_path),
        ]
    else:
        command = [
            settings.opendataloader_cli,
            str(input_path),
        ]

    command.extend(
        [
            "--output-dir",
            str(output_dir),
            "--format",
            "json,markdown",
            "--table-method",
            settings.table_method,
            "--reading-order",
            settings.reading_order,
            "--image-output",
            "off",
            "--markdown-page-separator",
            "\n\n--- Page %page-number% ---\n\n",
            "--quiet",
        ]
    )
    if settings.use_struct_tree:
        command.append("--use-struct-tree")
    return command


def _run_command(command: Sequence[str], timeout_seconds: int) -> None:
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    process = subprocess.Popen(command, **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _kill_process_tree(process.pid)
        stdout, stderr = process.communicate()
        logger.warning("OpenDataLoader timed out. stdout=%s stderr=%s", stdout[-2000:], stderr[-2000:])
        raise ConversionError("PDF conversion timed out.") from exc

    if process.returncode != 0:
        logger.warning("OpenDataLoader failed. stdout=%s stderr=%s", stdout[-2000:], stderr[-2000:])
        raise ConversionError("OpenDataLoader conversion failed.")


def _kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        if shutil.which("taskkill"):
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        return

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _read_rendered_markdown(output_dir: Path) -> str:
    markdown_path = _first_file(output_dir, (".md", ".markdown"))
    if markdown_path:
        markdown = _normalize_markdown(markdown_path.read_text(encoding="utf-8"))
        if _has_page_separators(markdown) and _has_meaningful_markdown(markdown):
            return markdown

    json_path = _first_file(output_dir, (".json",))

    fallback = markdown_path.read_text(encoding="utf-8") if markdown_path else ""
    doc = None
    if json_path:
        with json_path.open(encoding="utf-8") as f:
            doc = json.load(f)

    markdown = render_document_to_markdown(doc, fallback)
    if not _has_meaningful_markdown(markdown):
        raise ConversionError("OpenDataLoader produced no Markdown.")
    return markdown


def _read_generated_markdown(output_dir: Path) -> str:
    json_path = _first_file(output_dir, (".json",))
    markdown_path = _first_file(output_dir, (".md", ".markdown"))
    if not markdown_path:
        raise ConversionError("OpenDataLoader produced no Markdown.")

    fallback = markdown_path.read_text(encoding="utf-8")
    if json_path:
        with json_path.open(encoding="utf-8") as f:
            doc = json.load(f)
        markdown = render_document_pages_to_markdown(doc, fallback)
    else:
        markdown = _normalize_markdown(fallback)

    if not _has_meaningful_markdown(markdown):
        raise ConversionError("OpenDataLoader produced no Markdown.")
    if _has_page_separators(markdown) and not _has_enough_page_content(markdown):
        raise ConversionError("OpenDataLoader produced too few content pages.")
    return markdown


def _repair_pdf(input_path: Path, output_path: Path) -> None:
    fitz = _load_pymupdf()
    doc = None
    try:
        doc = fitz.open(str(input_path))
        if doc.page_count <= 0:
            raise ConversionError("PDF has no pages.")
        doc.save(str(output_path), garbage=4, deflate=True)
    except Exception as exc:
        raise ConversionError("Could not repair PDF with PyMuPDF.") from exc
    finally:
        if doc is not None:
            doc.close()


def _repair_pdf_with_pikepdf(input_path: Path, output_path: Path) -> None:
    try:
        import pikepdf
    except ImportError as exc:
        raise ConversionError("pikepdf is required for qpdf-style PDF repair.") from exc

    try:
        with pikepdf.open(input_path) as pdf:
            if len(pdf.pages) <= 0:
                raise ConversionError("PDF has no pages.")
            pdf.save(output_path)
    except Exception as exc:
        raise ConversionError("Could not repair PDF with pikepdf/qpdf.") from exc


def _rasterize_pdf(input_path: Path, output_path: Path, dpi: int) -> None:
    fitz = _load_pymupdf()
    source = None
    target = None
    try:
        source = fitz.open(str(input_path))
        if source.page_count <= 0:
            raise ConversionError("PDF has no pages.")

        target = fitz.open()
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        for page in source:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_bytes = pixmap.tobytes("png")
            output_page = target.new_page(width=page.rect.width, height=page.rect.height)
            output_page.insert_image(output_page.rect, stream=image_bytes)

        target.save(str(output_path), garbage=4, deflate=True)
    except Exception as exc:
        raise ConversionError("Could not rasterize PDF with PyMuPDF.") from exc
    finally:
        if target is not None:
            target.close()
        if source is not None:
            source.close()


def _load_pymupdf():
    try:
        import fitz
    except ImportError as exc:
        raise ConversionError("PyMuPDF is required for PDF fallback conversion.") from exc
    return fitz


def _has_meaningful_markdown(markdown: str) -> bool:
    stripped = markdown.strip()
    if not stripped:
        return False
    stripped = re.sub(r"(?m)^\s*--- Page \d+ ---\s*$", "", stripped)
    stripped = re.sub(r"(?m)^\s*!\[[^\]]*\]\(<[^>]+>\)\s*$", "", stripped)
    stripped = re.sub(r"(?m)^\s*\|(?:\s*\|)+\s*$", "", stripped)
    stripped = re.sub(r"(?m)^\s*\|(?:\s*:?-+:?\s*\|)+\s*$", "", stripped)
    return bool(stripped.strip())


def _has_page_separators(markdown: str) -> bool:
    return bool(re.search(r"(?m)^\s*--- Page \d+ ---\s*$", markdown))


def _has_enough_page_content(markdown: str) -> bool:
    pages = re.split(r"(?m)^\s*--- Page \d+ ---\s*$", markdown)
    if len(pages) <= 2:
        return True

    page_count = len(pages) - 1
    content_pages = sum(1 for page in pages[1:] if _has_meaningful_markdown(page))
    if page_count <= 3:
        return content_pages > 0

    minimum_content_pages = max(2, page_count // 5)
    return content_pages >= minimum_content_pages


def _normalize_markdown(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.split("\n")]
    value = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def _looks_text_corrupted_pdf(pdf_bytes: bytes) -> bool:
    replacement_count = pdf_bytes.count(b"\xef\xbf\xbd")
    if replacement_count < 100:
        return False
    replacement_ratio = (replacement_count * 3) / max(len(pdf_bytes), 1)
    return replacement_ratio > 0.01


def _first_existing_path(*paths: Path | None) -> Path:
    for path in paths:
        if path and path.exists():
            return path
    raise ConversionError("No fallback PDF path exists.")


def _first_file(directory: Path, suffixes: tuple[str, ...]) -> Path | None:
    candidates = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    return sorted(candidates)[0] if candidates else None
