from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field

from .config import get_settings
from .converter import PdfConverter
from .rag import STORE, build_rag_artifact, lookup_matches, plan_query, verify_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dify OpenDataLoader RAG API")


class ConvertResponse(BaseModel):
    success: bool
    markdown: str


class ConvertRagResponse(BaseModel):
    success: bool
    document_id: str = ""
    file_name: str = ""
    markdown: str = ""
    dify_markdown: str = ""
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)
    document_map: list[dict[str, Any]] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class QueryPlanRequest(BaseModel):
    query: str
    document_id: str | None = None


class QueryPlanResponse(BaseModel):
    query: str
    document_id: str | None = None
    query_type: str
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    expanded_queries: list[str] = Field(default_factory=list)
    retrieval_hints: dict[str, Any] = Field(default_factory=dict)


class LookupRequest(BaseModel):
    query: str
    document_id: str | None = None
    entities: list[str] = Field(default_factory=list)
    query_type: str | None = None
    limit: int = 8


class LookupResponse(BaseModel):
    query: str
    query_type: str
    entities: list[str] = Field(default_factory=list)
    matches: list[dict[str, Any]] = Field(default_factory=list)
    context: str = ""


class VerifyRequest(BaseModel):
    question: str
    answer: str
    evidence: list[Any] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    valid: bool
    confidence: float
    query_type: str
    issues: list[dict[str, Any]] = Field(default_factory=list)
    answer_numbers: list[str] = Field(default_factory=list)
    evidence_numbers: list[str] = Field(default_factory=list)
    answer_dates: list[str] = Field(default_factory=list)
    evidence_dates: list[str] = Field(default_factory=list)


@app.post("/convert", response_model=ConvertResponse)
async def convert(pdf: UploadFile | None = File(None)) -> ConvertResponse:
    try:
        if pdf is None:
            raise ValueError("PDF file parameter is required.")
        content = await pdf.read()
        converter = PdfConverter(get_settings())
        markdown = await asyncio.to_thread(
            converter.convert_pdf_bytes,
            content,
            pdf.filename or "document.pdf",
        )
        return ConvertResponse(success=True, markdown=markdown)
    except Exception:
        logger.exception("PDF conversion failed")
        return ConvertResponse(success=False, markdown="")


@app.post("/convert/rag", response_model=ConvertRagResponse)
async def convert_rag(
    pdf: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
    document_id: str | None = Form(None),
    file_name: str | None = Form(None),
) -> ConvertRagResponse:
    try:
        upload = pdf or file
        if upload is None:
            raise ValueError("PDF file parameter is required. Use form-data field 'pdf' or 'file'.")

        content = await upload.read()
        source_file_name = file_name or upload.filename or "document.pdf"
        converter = PdfConverter(get_settings())
        markdown = await asyncio.to_thread(
            converter.convert_pdf_bytes,
            content,
            source_file_name,
        )
        artifact = build_rag_artifact(markdown, source_file_name, document_id=document_id)
        STORE.upsert(artifact)
        payload = artifact.to_dict()
        return ConvertRagResponse(**payload)
    except Exception as exc:
        logger.exception("PDF RAG conversion failed")
        return ConvertRagResponse(success=False, error=str(exc))


@app.post("/query/plan", response_model=QueryPlanResponse)
async def query_plan(request: QueryPlanRequest) -> QueryPlanResponse:
    return QueryPlanResponse(**plan_query(request.query, request.document_id))


@app.post("/lookup", response_model=LookupResponse)
async def lookup(request: LookupRequest) -> LookupResponse:
    limit = max(1, min(request.limit, 30))
    result = lookup_matches(
        query=request.query,
        records=STORE.records(request.document_id),
        chunks=STORE.chunks(request.document_id),
        entities=request.entities,
        query_type=request.query_type,
        limit=limit,
    )
    return LookupResponse(**result)


@app.post("/answer/verify", response_model=VerifyResponse)
async def answer_verify(request: VerifyRequest) -> VerifyResponse:
    return VerifyResponse(**verify_answer(request.question, request.answer, request.evidence))


@app.get("/rag/documents")
async def rag_documents() -> dict[str, Any]:
    return {"documents": STORE.list_documents()}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
