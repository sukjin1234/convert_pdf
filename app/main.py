from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from .config import get_settings
from .converter import PdfConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dify OpenDataLoader PDF Converter")


class ConvertResponse(BaseModel):
    success: bool
    markdown: str


@app.post("/convert", response_model=ConvertResponse)
async def convert(
    pdf: UploadFile | None = File(None),
    ocr: bool = Form(False),
) -> ConvertResponse:
    try:
        if pdf is None:
            raise ValueError("PDF file parameter is required.")
        content = await pdf.read()
        converter = PdfConverter(get_settings())
        use_ocr = ocr is True
        markdown = await asyncio.to_thread(
            converter.convert_pdf_bytes,
            content,
            pdf.filename or "document.pdf",
            use_ocr=use_ocr,
        )
        return ConvertResponse(success=True, markdown=markdown)
    except Exception:
        logger.exception("PDF conversion failed")
        return ConvertResponse(success=False, markdown="")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
