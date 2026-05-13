from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from uuid import uuid4

from .config import Settings, get_settings
from .markdown import render_document_pages_to_markdown, render_document_to_markdown

logger = logging.getLogger(__name__)

_ocr_server_lock = threading.RLock()
_ocr_server_states: dict[str, OcrServerState] = {}


class ConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class OcrTarget:
    element: dict[str, Any]
    page_number: int
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class OcrServerSpec:
    index: int
    port: int
    hybrid_url: str
    cuda_visible_devices: str | None = None


@dataclass
class OcrServerState:
    process: subprocess.Popen
    stdout_file: Any
    stderr_file: Any
    users: int


@dataclass(frozen=True)
class PdfConverter:
    settings: Settings

    def __init__(self, settings: Settings | None = None):
        object.__setattr__(self, "settings", settings or get_settings())

    def convert_pdf_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf", *, use_ocr: bool = True) -> str:
        self._validate_pdf_bytes(pdf_bytes)
        safe_name = sanitize_filename(filename)
        if not safe_name.lower().endswith(".pdf"):
            safe_name = f"{Path(safe_name).stem or 'document'}.pdf"

        self.settings.tmp_root.mkdir(parents=True, exist_ok=True)
        with _temporary_directory("convert-", self.settings.tmp_root) as tmp_dir:
            input_path = tmp_dir / safe_name
            input_path.write_bytes(pdf_bytes)

            errors = []
            repaired_path: Path | None = None
            qpdf_repaired_path: Path | None = None

            try:
                return _convert_pdf_file(input_path, tmp_dir / "output-original", self.settings, use_ocr=use_ocr)
            except ConversionError as exc:
                errors.append(f"original: {exc}")
                logger.info("Original PDF conversion failed; trying fallbacks. error=%s", exc)

            if self.settings.qpdf_repair_pdf_on_failure:
                qpdf_repaired_path = tmp_dir / "qpdf-repaired.pdf"
                try:
                    _repair_pdf_with_pikepdf(input_path, qpdf_repaired_path)
                    return _convert_pdf_file(
                        qpdf_repaired_path,
                        tmp_dir / "output-qpdf-repaired",
                        self.settings,
                        use_ocr=use_ocr,
                    )
                except ConversionError as exc:
                    errors.append(f"qpdf-repaired: {exc}")
                    logger.info("qpdf/pikepdf repaired PDF conversion failed. error=%s", exc)

            if self.settings.repair_pdf_on_failure:
                repaired_path = tmp_dir / "repaired.pdf"
                repair_source = qpdf_repaired_path if qpdf_repaired_path and qpdf_repaired_path.exists() else input_path
                try:
                    _repair_pdf(repair_source, repaired_path)
                    return _convert_pdf_file(
                        repaired_path,
                        tmp_dir / "output-repaired",
                        self.settings,
                        use_ocr=use_ocr,
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
                            use_ocr=use_ocr,
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
                        use_ocr=use_ocr,
                    )
                except ConversionError as exc:
                    errors.append(f"rasterized: {exc}")
                    logger.info("Rasterized PDF conversion failed. error=%s", exc)

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


@contextmanager
def _temporary_directory(prefix: str, directory: Path) -> Iterator[Path]:
    path = directory / f"{prefix}{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _convert_pdf_file(
    input_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    hybrid_mode: str | None = None,
    image_output: str = "external",
    use_ocr: bool = False,
) -> str:
    native_error = None
    if settings.native_text_layer_first and hybrid_mode is None:
        try:
            return _convert_pdf_file_native(
                input_path,
                output_dir.with_name(f"{output_dir.name}-native"),
                settings,
                use_ocr=use_ocr,
            )
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
            use_ocr=use_ocr,
        )
    except ConversionError as exc:
        if native_error is not None:
            raise ConversionError(f"{exc}; native: {native_error}") from exc
        raise


def _convert_pdf_file_native(input_path: Path, output_dir: Path, settings: Settings, *, use_ocr: bool = False) -> str:
    output_dir.mkdir()
    command = build_opendataloader_native_command(input_path, output_dir, settings)
    _run_command(command, settings.conversion_timeout_seconds)
    if use_ocr:
        return _read_generated_markdown_with_ocr(input_path, output_dir, settings)
    return _read_generated_markdown(output_dir)


def _convert_pdf_file_hybrid(
    input_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    hybrid_mode: str | None = None,
    image_output: str = "external",
    use_ocr: bool = False,
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
    if use_ocr:
        return _read_rendered_markdown_with_ocr(input_path, output_dir, settings)
    return _read_rendered_markdown(output_dir)


def build_opendataloader_command(
    input_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    hybrid_mode: str | None = None,
    image_output: str = "external",
    hybrid_url: str | None = None,
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
        hybrid_url or settings.hybrid_url,
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


def build_ocr_server_command(settings: Settings, port: int | None = None) -> list[str]:
    command = [
        settings.ocr_server_cli,
        "--port",
        str(port or settings.ocr_server_port),
        "--force-ocr",
        "--ocr-lang",
        settings.ocr_lang,
    ]
    if settings.ocr_enrich_picture_description:
        command.append("--enrich-picture-description")
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


@contextmanager
def _managed_ocr_server(settings: Settings) -> Iterator[OcrServerSpec | None]:
    with _managed_ocr_servers(settings) as specs:
        yield specs[0] if specs else None


@contextmanager
def _managed_ocr_servers(settings: Settings) -> Iterator[list[OcrServerSpec]]:
    specs = _ocr_server_specs(settings)
    managed_specs: list[OcrServerSpec] = []
    try:
        for spec in specs:
            if _acquire_ocr_server(settings, spec):
                managed_specs.append(spec)
        yield specs
    finally:
        for spec in reversed(managed_specs):
            _release_ocr_server(settings, spec)


def _acquire_ocr_server(settings: Settings, spec: OcrServerSpec) -> bool:
    with _ocr_server_lock:
        state = _ocr_server_states.get(spec.hybrid_url)
        if state is not None and state.process.poll() is None:
            _wait_for_http_server(spec.hybrid_url, state.process, settings.ocr_server_start_timeout_seconds)
            state.users += 1
            logger.info("Using managed OCR server. url=%s users=%s", spec.hybrid_url, state.users)
            return True

        if _http_server_ready(spec.hybrid_url):
            logger.info("Using existing OCR server. url=%s", spec.hybrid_url)
            return False

        stdout_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        process = None
        try:
            command = build_ocr_server_command(settings, port=spec.port)
            env = os.environ.copy()
            if spec.cuda_visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = spec.cuda_visible_devices

            kwargs = {
                "stdout": stdout_file,
                "stderr": stderr_file,
                "text": True,
                "env": env,
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True

            logger.info(
                "Starting OCR server. url=%s cuda_visible_devices=%s command=%s",
                spec.hybrid_url,
                spec.cuda_visible_devices or "inherit",
                command,
            )
            process = subprocess.Popen(command, **kwargs)
            _wait_for_http_server(spec.hybrid_url, process, settings.ocr_server_start_timeout_seconds)
            _ocr_server_states[spec.hybrid_url] = OcrServerState(
                process=process,
                stdout_file=stdout_file,
                stderr_file=stderr_file,
                users=1,
            )
            return True
        except ConversionError as exc:
            if process is not None and process.poll() is not None:
                stdout = _read_temp_output(stdout_file)
                stderr = _read_temp_output(stderr_file)
                logger.warning(
                    "OCR server exited before readiness. returncode=%s stdout=%s stderr=%s",
                    process.returncode,
                    _tail_output(stdout),
                    _tail_output(stderr),
                )
                raise ConversionError(f"{exc}{_process_output_detail(stdout, stderr)}") from exc
            raise
        finally:
            state = _ocr_server_states.get(spec.hybrid_url)
            if state is None or state.process is not process:
                if process is not None:
                    _terminate_process_tree(process, settings.ocr_server_shutdown_timeout_seconds)
                    logger.info("Cleaned up OCR server process. url=%s", spec.hybrid_url)
                stdout_file.close()
                stderr_file.close()


def _release_ocr_server(settings: Settings, spec: OcrServerSpec) -> None:
    with _ocr_server_lock:
        state = _ocr_server_states.get(spec.hybrid_url)
        if state is None:
            return

        if state.users > 0:
            state.users -= 1
        if state.users > 0:
            logger.info("OCR server still in use. url=%s users=%s", spec.hybrid_url, state.users)
            return

        _ocr_server_states.pop(spec.hybrid_url, None)

        _terminate_process_tree(state.process, settings.ocr_server_shutdown_timeout_seconds)
        logger.info("Stopped OCR server. url=%s", spec.hybrid_url)
        state.stdout_file.close()
        state.stderr_file.close()


def _ocr_server_specs(settings: Settings) -> list[OcrServerSpec]:
    explicit_ports = list(settings.ocr_server_ports)
    explicit_urls = list(settings.ocr_hybrid_urls)
    devices = [_normalize_cuda_visible_devices(device) for device in settings.ocr_server_devices]
    worker_count = settings.ocr_server_workers or max(len(explicit_ports), len(explicit_urls), len(devices), 1)

    if explicit_ports and len(explicit_ports) != worker_count:
        raise ConversionError("ODL_OCR_SERVER_PORTS count must match OCR worker count.")
    if explicit_urls and len(explicit_urls) != worker_count:
        raise ConversionError("ODL_OCR_HYBRID_URLS count must match OCR worker count.")
    if devices and len(devices) != worker_count:
        raise ConversionError("ODL_OCR_SERVER_DEVICES count must match OCR worker count.")

    ports = explicit_ports or [settings.ocr_server_port + index for index in range(worker_count)]
    if explicit_urls:
        urls = explicit_urls
    elif worker_count == 1 and not explicit_ports and not devices and not settings.ocr_server_workers:
        urls = [settings.ocr_hybrid_url]
    else:
        urls = [f"http://{settings.ocr_server_host}:{port}" for port in ports]

    return [
        OcrServerSpec(
            index=index,
            port=port,
            hybrid_url=url,
            cuda_visible_devices=devices[index - 1] if devices else None,
        )
        for index, (port, url) in enumerate(zip(ports, urls), start=1)
    ]


def _normalize_cuda_visible_devices(device: str) -> str:
    value = device.strip()
    lowered = value.lower()
    if lowered.startswith("cuda:"):
        return value.split(":", 1)[1]
    if lowered.startswith("cuda") and value[4:].isdigit():
        return value[4:]
    return value


def _read_temp_output(output_file: Any) -> str:
    try:
        output_file.flush()
        output_file.seek(0)
        return output_file.read()
    except OSError:
        return ""


def _tail_output(value: str, limit: int = 2000) -> str:
    return (value or "")[-limit:]


def _process_output_detail(stdout: str, stderr: str) -> str:
    parts = []
    if stdout:
        parts.append(f"stdout={_tail_output(stdout)!r}")
    if stderr:
        parts.append(f"stderr={_tail_output(stderr)!r}")
    return " " + " ".join(parts) if parts else ""


def _wait_for_http_server(url: str, process: subprocess.Popen, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ConversionError("OCR server exited before it became ready.")

        ready, last_error = _probe_http_server(url)
        if ready:
            return

        time.sleep(0.25)

    detail = f" Last error: {last_error}" if last_error else ""
    raise ConversionError(f"OCR server did not become ready at {url} within {timeout_seconds} seconds.{detail}")


def _http_server_ready(url: str) -> bool:
    ready, _last_error = _probe_http_server(url)
    return ready


def _probe_http_server(url: str) -> tuple[bool, Exception | None]:
    last_error: Exception | None = None
    urls = [f"{url.rstrip('/')}/health", url.rstrip("/")]
    for candidate in urls:
        try:
            with urlopen(candidate, timeout=1) as response:
                if response.status < 500:
                    return True, None
        except HTTPError as exc:
            if exc.code < 500:
                return True, None
            last_error = exc
        except (OSError, URLError, HTTPException) as exc:
            last_error = exc
    return False, last_error


def _terminate_process_tree(process: subprocess.Popen, timeout_seconds: int) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        if shutil.which("taskkill"):
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.terminate()
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            return

    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process.pid)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("OCR server process did not exit after forced shutdown. pid=%s", process.pid)


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


def _read_rendered_markdown_with_ocr(input_path: Path, output_dir: Path, settings: Settings) -> str:
    json_path = _first_file(output_dir, (".json",))
    if not json_path:
        return _read_rendered_markdown(output_dir)

    markdown_path = _first_file(output_dir, (".md", ".markdown"))
    fallback = markdown_path.read_text(encoding="utf-8") if markdown_path else ""
    with json_path.open(encoding="utf-8") as f:
        doc = json.load(f)

    if _apply_ocr_to_document_images(input_path, doc, output_dir, settings) <= 0:
        return _read_rendered_markdown(output_dir)

    markdown = render_document_pages_to_markdown(doc, fallback)
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


def _read_generated_markdown_with_ocr(input_path: Path, output_dir: Path, settings: Settings) -> str:
    json_path = _first_file(output_dir, (".json",))
    if not json_path:
        return _read_generated_markdown(output_dir)

    markdown_path = _first_file(output_dir, (".md", ".markdown"))
    fallback = markdown_path.read_text(encoding="utf-8") if markdown_path else ""
    with json_path.open(encoding="utf-8") as f:
        doc = json.load(f)

    if _apply_ocr_to_document_images(input_path, doc, output_dir, settings) <= 0:
        return _read_generated_markdown(output_dir)

    markdown = render_document_pages_to_markdown(doc, fallback)
    if not _has_meaningful_markdown(markdown):
        raise ConversionError("OpenDataLoader produced no Markdown.")
    if _has_page_separators(markdown) and not _has_enough_page_content(markdown):
        raise ConversionError("OpenDataLoader produced too few content pages.")
    return markdown


def _apply_ocr_to_document_images(input_path: Path, doc: dict[str, Any], output_dir: Path, settings: Settings) -> int:
    targets = _collect_ocr_targets(doc, settings)
    logger.info("Collected OCR targets. count=%s", len(targets))
    if not targets:
        return 0

    ocr_root = output_dir / "ocr"
    ocr_root.mkdir(exist_ok=True)

    with _managed_ocr_servers(settings) as servers:
        logger.info("Applying OCR targets. count=%s servers=%s", len(targets), len(servers))
        results = _convert_ocr_targets_parallel(input_path, ocr_root, targets, settings, servers)

    applied = 0
    for target, ocr_markdown in results:
        if ocr_markdown:
            target.element["_ocr_markdown"] = ocr_markdown
            applied += 1
    return applied


def _convert_ocr_targets_parallel(
    input_path: Path,
    ocr_root: Path,
    targets: list[OcrTarget],
    settings: Settings,
    servers: list[OcrServerSpec],
) -> list[tuple[OcrTarget, str]]:
    work_queue: queue.Queue[tuple[int, OcrTarget]] = queue.Queue()
    for index, target in enumerate(targets, start=1):
        work_queue.put((index, target))

    def worker(server: OcrServerSpec) -> list[tuple[OcrTarget, str]]:
        converted: list[tuple[OcrTarget, str]] = []
        while True:
            try:
                index, target = work_queue.get_nowait()
            except queue.Empty:
                return converted

            region_pdf = ocr_root / f"target-{index}.pdf"
            region_output = ocr_root / f"output-{index}"
            try:
                _extract_pdf_region_to_pdf(input_path, region_pdf, target)
                ocr_markdown = _convert_ocr_region(region_pdf, region_output, settings, server.hybrid_url)
            except ConversionError as exc:
                logger.info(
                    "Skipping OCR target after conversion failure. page=%s target=%s server=%s error=%s",
                    target.page_number,
                    index,
                    server.hybrid_url,
                    exc,
                )
            else:
                if ocr_markdown:
                    converted.append((target, ocr_markdown))
            finally:
                work_queue.task_done()

    results: list[tuple[OcrTarget, str]] = []
    with ThreadPoolExecutor(max_workers=len(servers)) as executor:
        for server_results in executor.map(worker, servers):
            results.extend(server_results)
    return results


def _collect_ocr_targets(doc: dict[str, Any], settings: Settings) -> list[OcrTarget]:
    targets: list[OcrTarget] = []
    skipped = 0

    def visit(element: dict[str, Any], current_page: int) -> None:
        nonlocal skipped
        page_number = _safe_int_value(element.get("page number"), current_page)
        if _document_element_type(element) in {"image", "picture", "figure"} and page_number > 0:
            bbox = _extract_bbox(element)
            if settings.ocr_filter_decorative_images and _is_decorative_ocr_image(element, bbox):
                skipped += 1
            else:
                targets.append(OcrTarget(element=element, page_number=page_number, bbox=bbox))

        for child in _document_children(element):
            visit(child, page_number)

    visit(doc, 0)
    if skipped:
        logger.info("Skipped decorative OCR targets. count=%s", skipped)
    return targets


def _is_decorative_ocr_image(element: dict[str, Any], bbox: tuple[float, float, float, float] | None) -> bool:
    hint = " ".join(
        str(element.get(key) or "")
        for key in ("description", "content", "source", "path", "name", "alt")
    ).lower()
    if any(token in hint for token in ("logo", "icon", "seal", "emblem", "watermark", "decorative")):
        return True

    if bbox is None:
        return False

    x0, y0, x1, y1 = bbox
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    area = width * height
    if width < 60 and height < 60:
        return True
    if area < 3_600:
        return True
    if area < 8_000 and (y0 < 80 or y1 < 80 or y0 > 720 or y1 > 720):
        return True
    return False


def _extract_pdf_region_to_pdf(input_path: Path, output_path: Path, target: OcrTarget) -> None:
    fitz = _load_pymupdf()
    source = None
    target_doc = None
    try:
        source = fitz.open(str(input_path))
        if target.page_number < 1 or target.page_number > source.page_count:
            raise ConversionError(f"OCR target page {target.page_number} is outside the PDF page range.")

        page_index = target.page_number - 1
        page = source[page_index]
        clip = _clip_rect_from_bbox(fitz, page.rect, target.bbox)
        page_rect = clip or page.rect

        target_doc = fitz.open()
        target_page = target_doc.new_page(width=page_rect.width, height=page_rect.height)
        if clip is None:
            target_page.show_pdf_page(target_page.rect, source, page_index)
        else:
            target_page.show_pdf_page(target_page.rect, source, page_index, clip=clip)
        target_doc.save(str(output_path), garbage=4, deflate=True)
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError("Could not extract OCR target region from PDF.") from exc
    finally:
        if target_doc is not None:
            target_doc.close()
        if source is not None:
            source.close()


def _clip_rect_from_bbox(fitz: Any, page_rect: Any, bbox: tuple[float, float, float, float] | None) -> Any | None:
    if bbox is None:
        return None

    direct = fitz.Rect(*bbox).normalize()
    direct = direct & page_rect
    if not direct.is_empty and direct.width > 1 and direct.height > 1:
        return direct

    x0, y0, x1, y1 = bbox
    flipped = fitz.Rect(x0, page_rect.height - y1, x1, page_rect.height - y0).normalize()
    flipped = flipped & page_rect
    if not flipped.is_empty and flipped.width > 1 and flipped.height > 1:
        return flipped
    return None


def _convert_ocr_region(input_path: Path, output_dir: Path, settings: Settings, hybrid_url: str) -> str:
    output_dir.mkdir()
    command = build_opendataloader_command(
        input_path,
        output_dir,
        settings,
        hybrid_mode="full",
        image_output="off",
        hybrid_url=hybrid_url,
    )
    _run_command(command, settings.conversion_timeout_seconds)
    try:
        markdown = _read_rendered_markdown(output_dir)
    except ConversionError:
        return ""
    return _strip_page_separators(markdown)


def _strip_page_separators(markdown: str) -> str:
    markdown = re.sub(r"(?m)^\s*--- Page \d+ ---\s*$", "", markdown)
    return _normalize_markdown(markdown)


def _document_element_type(element: dict[str, Any]) -> str:
    return str(element.get("type", "")).strip().lower()


def _document_children(element: dict[str, Any]) -> list[dict[str, Any]]:
    children: list[dict[str, Any]] = []
    for key in ("kids", "children", "list items", "items", "rows", "cells"):
        values = element.get(key) or []
        if isinstance(values, list):
            children.extend(value for value in values if isinstance(value, dict))
    return children


def _extract_bbox(element: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = element.get("bounding box")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _safe_int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
