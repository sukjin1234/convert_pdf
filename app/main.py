from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field

from .config import get_settings
from .converter import PdfConverter
from .rag import (
    STORE,
    build_grounded_answer_draft,
    build_rag_artifact,
    lookup_matches,
    merge_evidence_context,
    plan_query,
    verify_answer,
)

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


class MarkdownIngestRequest(BaseModel):
    markdown: str
    document_id: str | None = None
    file_name: str = "document.md"


class QueryPlanResponse(BaseModel):
    query: str
    document_id: str | None = None
    query_type: str
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    expanded_queries: list[str] = Field(default_factory=list)
    answer_style: str = "grounded_explanation"
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
    sub_queries: list[str] = Field(default_factory=list)
    answer_style: str = "grounded_explanation"
    matches: list[dict[str, Any]] = Field(default_factory=list)
    context: str = ""
    evidence_items: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class VerifyRequest(BaseModel):
    question: str
    answer: str
    evidence: list[Any] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    valid: bool
    confidence: float
    query_type: str
    issues: list[dict[str, Any]] = Field(default_factory=list)
    evidence_coverage: float = 0
    answer_numbers: list[str] = Field(default_factory=list)
    evidence_numbers: list[str] = Field(default_factory=list)
    answer_dates: list[str] = Field(default_factory=list)
    evidence_dates: list[str] = Field(default_factory=list)
    answer_identifiers: list[str] = Field(default_factory=list)
    evidence_identifiers: list[str] = Field(default_factory=list)


class ChatflowDebugRequest(BaseModel):
    query: str
    document_id: str | None = None
    limit: int = 8
    knowledge_result: Any = None
    answer: str | None = None


class ChatflowDebugResponse(BaseModel):
    query_plan: dict[str, Any]
    structured_lookup: dict[str, Any]
    merge_evidence: dict[str, str]
    draft_answer: str
    draft_verification: dict[str, Any]
    supplied_answer_verification: dict[str, Any] | None = None
    node_status: dict[str, Any] = Field(default_factory=dict)


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


@app.post("/rag/ingest-markdown", response_model=ConvertRagResponse)
async def rag_ingest_markdown(request: MarkdownIngestRequest) -> ConvertRagResponse:
    try:
        artifact = build_rag_artifact(
            request.markdown,
            request.file_name or "document.md",
            document_id=request.document_id,
        )
        STORE.upsert(artifact)
        return ConvertRagResponse(**artifact.to_dict())
    except Exception as exc:
        logger.exception("Markdown RAG ingestion failed")
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


@app.post("/chatflow/debug", response_model=ChatflowDebugResponse)
async def chatflow_debug(request: ChatflowDebugRequest) -> ChatflowDebugResponse:
    limit = max(1, min(request.limit, 30))
    plan = plan_query(request.query, request.document_id)
    lookup_result = lookup_matches(
        query=request.query,
        records=STORE.records(request.document_id),
        chunks=STORE.chunks(request.document_id),
        entities=plan["entities"],
        query_type=plan["query_type"],
        limit=limit,
    )
    merged = merge_evidence_context(lookup_result, request.knowledge_result)
    draft_answer = build_grounded_answer_draft(request.query, lookup_result)
    draft_verification = verify_answer(request.query, draft_answer, [merged["evidence_context"]])
    supplied_verification = None
    if request.answer is not None:
        supplied_verification = verify_answer(request.query, request.answer, [merged["evidence_context"]])

    return ChatflowDebugResponse(
        query_plan=plan,
        structured_lookup=lookup_result,
        merge_evidence=merged,
        draft_answer=draft_answer,
        draft_verification=draft_verification,
        supplied_answer_verification=supplied_verification,
        node_status={
            "structured_lookup_has_context": bool(lookup_result.get("context")),
            "knowledge_retrieval_supplied": bool(request.knowledge_result),
            "evidence_context_has_structured_lookup": "[Structured Lookup" in merged["evidence_context"],
            "draft_answer_valid": bool(draft_verification.get("valid")),
        },
    )


@app.get("/rag/documents")
async def rag_documents() -> dict[str, Any]:
    return {"documents": STORE.list_documents()}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
