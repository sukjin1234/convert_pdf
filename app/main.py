from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from pydantic import BaseModel

from .config import get_settings
from .converter import PdfConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dify OpenDataLoader PDF Converter")
_inflight_lock = asyncio.Lock()
_inflight_conversions: dict[str, asyncio.Future[str]] = {}
_USE_OCR_KEYS = ("use_ocr", "useOcr", "useOCR", "ocr", "enable_ocr", "enableOcr")


class ConvertResponse(BaseModel):
    success: bool
    markdown: str


@app.post("/convert", response_model=ConvertResponse)
async def convert(
    request: Request,
    pdf: UploadFile | None = File(None),
    use_ocr: Annotated[str | None, Form()] = None,
) -> ConvertResponse:
    try:
        if pdf is None:
            raise ValueError("PDF file parameter is required.")
        content = await pdf.read()
        resolved_use_ocr = await _resolve_use_ocr(use_ocr, request)
        markdown = await _convert_pdf_once(content, pdf.filename or "document.pdf", use_ocr=resolved_use_ocr)
        return ConvertResponse(success=True, markdown=markdown)
    except Exception:
        logger.exception("PDF conversion failed")
        return ConvertResponse(success=False, markdown="")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def _convert_pdf_once(content: bytes, filename: str, *, use_ocr: bool) -> str:
    key = _conversion_key(content, filename, use_ocr)
    loop = asyncio.get_running_loop()

    async with _inflight_lock:
        future = _inflight_conversions.get(key)
        if future is None:
            converter = PdfConverter(get_settings())
            call = functools.partial(converter.convert_pdf_bytes, content, filename, use_ocr=use_ocr)
            future = loop.run_in_executor(None, call)
            _inflight_conversions[key] = future
            future.add_done_callback(
                lambda completed, conversion_key=key: loop.call_soon_threadsafe(
                    _schedule_inflight_cleanup,
                    conversion_key,
                    completed,
                )
            )
            logger.info("Started PDF conversion. key=%s use_ocr=%s", key[:12], use_ocr)
        else:
            logger.info("Joining in-flight PDF conversion. key=%s use_ocr=%s", key[:12], use_ocr)

    return await asyncio.shield(future)


def _schedule_inflight_cleanup(key: str, future: asyncio.Future[str]) -> None:
    asyncio.create_task(_cleanup_inflight_conversion(key, future))


async def _cleanup_inflight_conversion(key: str, future: asyncio.Future[str]) -> None:
    async with _inflight_lock:
        if _inflight_conversions.get(key) is future:
            _inflight_conversions.pop(key, None)


def _conversion_key(content: bytes, filename: str, use_ocr: bool) -> str:
    digest = hashlib.sha256(content).hexdigest()
    return f"{digest}:{filename}:{int(use_ocr)}"


async def _resolve_use_ocr(form_value: Any, request: Request, default: bool = True) -> bool:
    candidates: list[tuple[str, Any]] = [("form-param:use_ocr", form_value)]

    form_values = await _request_form_values(request)
    for key in _USE_OCR_KEYS:
        candidates.append((f"form:{key}", form_values.get(key)))
        candidates.append((f"query:{key}", request.query_params.get(key)))

    for key in ("x-use-ocr", "use-ocr", "x-useocr", "x-ocr"):
        candidates.append((f"header:{key}", request.headers.get(key)))

    for source, candidate in candidates:
        parsed = _parse_bool(candidate)
        if parsed is not None:
            logger.info("Resolved use_ocr=%s source=%s raw=%r", parsed, source, candidate)
            return parsed
    logger.info("Resolved use_ocr=%s source=default", default)
    return default


async def _request_form_values(request: Request) -> dict[str, Any]:
    form_method = getattr(request, "form", None)
    if form_method is None:
        return {}
    try:
        form = await form_method()
    except Exception:
        return {}
    return {str(key): value for key, value in form.multi_items() if not hasattr(value, "filename")}


def _parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in {0, 1}:
            return bool(value)
        raise ValueError(f"Invalid boolean value for use_ocr: {value!r}")

    text = str(value).strip().strip("\"'").strip().lower()
    if not text:
        return None
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for use_ocr: {value!r}")
