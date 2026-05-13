from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from .config import get_settings
from .converter import PdfConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dify OpenDataLoader PDF Converter")
_inflight_lock = asyncio.Lock()
_inflight_conversions: dict[str, asyncio.Future[str]] = {}


class ConvertResponse(BaseModel):
    success: bool
    markdown: str


@app.post("/convert", response_model=ConvertResponse)
async def convert(pdf: UploadFile | None = File(None), use_ocr: Annotated[bool, Form()] = True) -> ConvertResponse:
    try:
        if pdf is None:
            raise ValueError("PDF file parameter is required.")
        content = await pdf.read()
        markdown = await _convert_pdf_once(content, pdf.filename or "document.pdf", use_ocr=use_ocr)
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
