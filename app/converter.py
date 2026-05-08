from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence

from .chunking import ChunkingOptions, prepare_parent_child_markdown
from .config import Settings, get_settings
from .markdown import render_document_pages_to_markdown

logger = logging.getLogger(__name__)
_LOCAL_CONVERSION_LOCK = threading.Lock()


class ConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PdfConverter:
    settings: Settings

    def __init__(self, settings: Settings | None = None):
        object.__setattr__(self, "settings", settings or get_settings())

    def convert_pdf_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf", *, use_ocr: bool = False) -> str:
        self._validate_pdf_bytes(pdf_bytes)
        safe_name = sanitize_filename(filename)
        if not safe_name.lower().endswith(".pdf"):
            safe_name = f"{Path(safe_name).stem or 'document'}.pdf"

        self.settings.tmp_root.mkdir(parents=True, exist_ok=True)
        with _conversion_lock(self.settings):
            markdown = self._convert_pdf_bytes_locked(pdf_bytes, safe_name, use_ocr=use_ocr)
        return _prepare_for_dify_chunks(markdown, self.settings)

    def _convert_pdf_bytes_locked(self, pdf_bytes: bytes, safe_name: str, *, use_ocr: bool) -> str:
        with tempfile.TemporaryDirectory(prefix="convert-", dir=self.settings.tmp_root) as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / safe_name
            input_path.write_bytes(pdf_bytes)

            errors = []
            repaired_path: Path | None = None
            qpdf_repaired_path: Path | None = None
            ocr_attempted = False

            def try_ocr_fallback(source_path: Path, label: str) -> str:
                nonlocal ocr_attempted
                if not use_ocr:
                    raise ConversionError("OCR is disabled. Submit ocr=true to convert image-only or scanned PDFs.")
                if not self.settings.rasterize_pdf_on_failure:
                    raise ConversionError("OCR fallback is disabled by ODL_RASTERIZE_PDF_ON_FAILURE.")
                if ocr_attempted:
                    raise ConversionError("OCR fallback was already attempted.")

                ocr_attempted = True
                rasterized_ocr_path = tmp_dir / f"{label}-ocr.pdf"
                _rasterize_pdf(source_path, rasterized_ocr_path, self.settings.rasterize_dpi)
                return _convert_pdf_file(
                    rasterized_ocr_path,
                    tmp_dir / f"output-{label}-ocr",
                    self.settings,
                    hybrid_mode="full",
                    image_output="external",
                )

            def handle_ocr_required(exc: ConversionError, source_path: Path, label: str) -> str:
                if not use_ocr:
                    raise ConversionError("PDF appears to require OCR, but OCR was not selected.") from exc
                try:
                    return try_ocr_fallback(source_path, label)
                except ConversionError as ocr_exc:
                    errors.append(f"{label}-ocr: {ocr_exc}")
                    logger.info("OCR fallback failed. label=%s error=%s", label, ocr_exc)
                    raise ConversionError("PDF conversion failed after OCR fallback attempt: " + "; ".join(errors)) from ocr_exc

            try:
                return _convert_pdf_file(input_path, tmp_dir / "output-original", self.settings)
            except ConversionError as exc:
                errors.append(f"original: {exc}")
                logger.info("Original PDF conversion failed; trying fallbacks. error=%s", exc)
                if _requires_ocr_fallback(exc):
                    return handle_ocr_required(exc, input_path, "original")

            if self.settings.qpdf_repair_pdf_on_failure:
                qpdf_repaired_path = tmp_dir / "qpdf-repaired.pdf"
                try:
                    _repair_pdf_with_pikepdf(input_path, qpdf_repaired_path)
                    return _convert_pdf_file(
                        qpdf_repaired_path,
                        tmp_dir / "output-qpdf-repaired",
                        self.settings,
                    )
                except ConversionError as exc:
                    errors.append(f"qpdf-repaired: {exc}")
                    logger.info("qpdf/pikepdf repaired PDF conversion failed. error=%s", exc)
                    if _requires_ocr_fallback(exc):
                        return handle_ocr_required(exc, qpdf_repaired_path, "qpdf-repaired")

            if self.settings.repair_pdf_on_failure:
                repaired_path = tmp_dir / "repaired.pdf"
                repair_source = qpdf_repaired_path if qpdf_repaired_path and qpdf_repaired_path.exists() else input_path
                try:
                    _repair_pdf(repair_source, repaired_path)
                    return _convert_pdf_file(repaired_path, tmp_dir / "output-repaired", self.settings)
                except ConversionError as exc:
                    errors.append(f"repaired: {exc}")
                    logger.info("Repaired PDF conversion failed. error=%s", exc)
                    if _requires_ocr_fallback(exc):
                        return handle_ocr_required(exc, repaired_path, "repaired")

            if use_ocr and self.settings.rasterize_pdf_on_failure and not ocr_attempted:
                raster_source = _first_existing_path(repaired_path, qpdf_repaired_path, input_path)
                try:
                    return try_ocr_fallback(raster_source, "rasterized")
                except ConversionError as exc:
                    errors.append(f"rasterized-ocr: {exc}")
                    logger.info("Final OCR fallback failed. error=%s", exc)

            raise ConversionError("PDF conversion failed after fallback attempts: " + "; ".join(errors))

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


def _prepare_for_dify_chunks(markdown: str, settings: Settings) -> str:
    if not settings.prepare_dify_parent_child_chunks:
        return markdown
    return prepare_parent_child_markdown(
        markdown,
        ChunkingOptions(
            parent_delimiter=settings.dify_parent_delimiter,
            child_delimiter=settings.dify_child_delimiter,
            parent_max_chars=settings.dify_parent_max_chars,
            child_target_chars=settings.dify_child_target_chars,
            child_overlap_chars=settings.dify_child_overlap_chars,
        ),
    )


def _requires_ocr_fallback(exc: ConversionError) -> bool:
    message = str(exc).lower()
    return "ocr fallback required" in message or "visual pages without ocr text" in message


@contextmanager
def _conversion_lock(settings: Settings):
    lock_path = settings.tmp_root / "opendataloader-conversion.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCAL_CONVERSION_LOCK:
        with lock_path.open("a+b") as lock_file:
            _lock_file(lock_file)
            try:
                yield
            finally:
                _unlock_file(lock_file)


def _lock_file(lock_file: BinaryIO) -> None:
    lock_file.seek(0)
    if not lock_file.read(1):
        lock_file.write(b"\0")
        lock_file.flush()
    lock_file.seek(0)

    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)


def _unlock_file(lock_file: BinaryIO) -> None:
    lock_file.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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
    json_path = _first_file(output_dir, (".json",))
    fallback = _normalize_markdown(markdown_path.read_text(encoding="utf-8")) if markdown_path else ""
    json_requires_ocr_fallback = False

    if json_path:
        with json_path.open(encoding="utf-8") as f:
            doc = json.load(f)
        markdown = render_document_pages_to_markdown(doc, fallback)
        if _has_meaningful_markdown(markdown):
            if not _has_unresolved_visual_pages(markdown):
                return markdown
            json_requires_ocr_fallback = True

    if json_requires_ocr_fallback:
        raise ConversionError("OpenDataLoader produced visual pages without OCR text or image description.")

    if (
        fallback
        and _has_meaningful_markdown(fallback)
        and not _has_unresolved_visual_pages(fallback)
    ):
        return fallback

    raise ConversionError("OpenDataLoader produced no usable text or image description; OCR fallback required.")


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
    if _has_unresolved_visual_pages(markdown):
        raise ConversionError("OpenDataLoader produced visual pages without OCR text or image description.")
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
    stripped = re.sub(r"(?m)^\s*!\[[^\]]*\]\((?:<[^>]+>|[^)]+)\)\s*$", "", stripped)
    stripped = re.sub(r"(?m)^>\s*Image-only page\. No embedded text layer was available\.\s*$", "", stripped)
    stripped = re.sub(
        r"(?m)^>\s*\uc774\ubbf8\uc9c0/\ub3c4\uc2dd \uc911\uc2ec \ud398\uc774\uc9c0\ub85c, PDF \ud14d\uc2a4\ud2b8 \ub808\uc774\uc5b4\uc5d0\uc11c \ud655\uc778\ub418\ub294 \ubb38\uad6c\ub9cc \ud3ec\ud568\ud588\uc2b5\ub2c8\ub2e4\.\s*$",
        "",
        stripped,
    )
    stripped = re.sub(r"(?m)^\s*\|(?:\s*\|)+\s*$", "", stripped)
    stripped = re.sub(r"(?m)^\s*\|(?:\s*:?-+:?\s*\|)+\s*$", "", stripped)
    return bool(stripped.strip())


def _has_unresolved_visual_pages(markdown: str) -> bool:
    pages = re.split(r"(?m)^\s*--- Page \d+ ---\s*$", markdown)
    pages = pages[1:] if len(pages) > 1 else pages
    for page in pages:
        if not _has_visual_placeholder(page):
            continue
        content = _strip_visual_noise(page)
        content = re.sub(r"(?m)^#{1,6}\s+.*$", "", content)
        if not _content_fingerprint(content):
            return True
    return False


def _has_visual_placeholder(markdown: str) -> bool:
    return bool(
        re.search(r"(?m)^>\s*Image-only page\. No embedded text layer was available\.\s*$", markdown)
        or re.search(
            r"(?m)^>\s*\uc774\ubbf8\uc9c0/\ub3c4\uc2dd \uc911\uc2ec \ud398\uc774\uc9c0\ub85c, PDF \ud14d\uc2a4\ud2b8 \ub808\uc774\uc5b4\uc5d0\uc11c \ud655\uc778\ub418\ub294 \ubb38\uad6c\ub9cc \ud3ec\ud568\ud588\uc2b5\ub2c8\ub2e4\.\s*$",
            markdown,
        )
    )


def _strip_visual_noise(markdown: str) -> str:
    stripped = re.sub(r"(?m)^\s*!\[[^\]]*\]\((?:<[^>]+>|[^)]+)\)\s*$", "", markdown)
    stripped = re.sub(r"(?m)^>\s*Image-only page\. No embedded text layer was available\.\s*$", "", stripped)
    stripped = re.sub(
        r"(?m)^>\s*\uc774\ubbf8\uc9c0/\ub3c4\uc2dd \uc911\uc2ec \ud398\uc774\uc9c0\ub85c, PDF \ud14d\uc2a4\ud2b8 \ub808\uc774\uc5b4\uc5d0\uc11c \ud655\uc778\ub418\ub294 \ubb38\uad6c\ub9cc \ud3ec\ud568\ud588\uc2b5\ub2c8\ub2e4\.\s*$",
        "",
        stripped,
    )
    return stripped


def _content_fingerprint(value: str) -> str:
    return "".join(re.findall(r"[0-9A-Za-z\uac00-\ud7a3]+", value.lower()))


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
