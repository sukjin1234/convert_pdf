from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .converter import DocumentConverter
from .eval_logs import EVAL_LOG_STORE
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
CONVERSION_LOCK = asyncio.Lock()
CONVERSION_LOCK_POLL_SECONDS = 2


class ConvertResponse(BaseModel):
    success: bool
    markdown: str


class ConvertBatchItem(BaseModel):
    success: bool
    file_name: str = ""
    markdown: str = ""
    error: str = ""


class ConvertBatchResponse(BaseModel):
    success: bool
    files: list[ConvertBatchItem] = Field(default_factory=list)
    error: str = ""


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


class ConvertRagBatchResponse(BaseModel):
    success: bool
    files: list[ConvertRagResponse] = Field(default_factory=list)
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
    direct_answer: str = ""
    answer_items: list[dict[str, Any]] = Field(default_factory=list)
    answer_field: str = ""
    filter_terms: list[str] = Field(default_factory=list)
    answer_contract: dict[str, Any] = Field(default_factory=dict)
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


class MergeEvidenceRequest(BaseModel):
    lookup_body: Any = None
    knowledge_result: Any = None
    knowledge_context: Any = None
    knowledge_body: Any = None


class MergeEvidenceResponse(BaseModel):
    structured_context: str = ""
    knowledge_context: str = ""
    evidence_context: str = ""
    evidence_priority: str = ""
    source_summary: str = ""
    answer_contract: str = ""
    answer_contract_text: str = ""
    lookup_diagnostics: str = ""
    lookup_answerability: str = ""


class ChatflowDebugResponse(BaseModel):
    query_plan: dict[str, Any]
    structured_lookup: dict[str, Any]
    merge_evidence: dict[str, str]
    answer_contract: dict[str, Any] = Field(default_factory=dict)
    draft_answer: str
    draft_verification: dict[str, Any]
    supplied_answer_verification: dict[str, Any] | None = None
    node_status: dict[str, Any] = Field(default_factory=dict)


def _select_upload(*uploads: Any) -> UploadFile | None:
    for upload in uploads:
        if upload is not None and hasattr(upload, "read") and hasattr(upload, "filename"):
            return upload
    return None


def _upload_list(value: Any) -> list[UploadFile]:
    if value is None:
        return []
    if isinstance(value, list):
        return [upload for upload in value if upload is not None and hasattr(upload, "read") and hasattr(upload, "filename")]
    if hasattr(value, "read") and hasattr(value, "filename"):
        return [value]
    return []


def _select_uploads(*uploads: Any) -> list[UploadFile]:
    selected: list[UploadFile] = []
    for upload in uploads:
        selected.extend(_upload_list(upload))
    return selected


def _optional_form_text(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _optional_form_text_at(value: Any, index: int) -> str | None:
    if isinstance(value, list):
        if index >= len(value):
            return None
        return _optional_form_text(value[index])
    return _optional_form_text(value)


async def _convert_upload_to_markdown(upload: UploadFile, source_file_name: str | None = None) -> ConvertBatchItem:
    content = await upload.read()
    safe_file_name = source_file_name or upload.filename or "document"
    async with CONVERSION_LOCK:
        logger.info("Starting queued document conversion. file=%s bytes=%s", safe_file_name, len(content))
        markdown = await asyncio.to_thread(_convert_bytes_with_process_lock, content, safe_file_name)
        logger.info("Finished queued document conversion. file=%s chars=%s", safe_file_name, len(markdown))
    return ConvertBatchItem(success=True, file_name=safe_file_name, markdown=markdown)


async def _convert_upload_to_rag(
    upload: UploadFile,
    *,
    document_id: str | None = None,
    source_file_name: str | None = None,
) -> ConvertRagResponse:
    content = await upload.read()
    safe_file_name = source_file_name or upload.filename or "document"
    async with CONVERSION_LOCK:
        logger.info("Starting queued RAG conversion. file=%s bytes=%s document_id=%s", safe_file_name, len(content), document_id or "")
        markdown = await asyncio.to_thread(_convert_bytes_with_process_lock, content, safe_file_name)
        artifact = build_rag_artifact(markdown, safe_file_name, document_id=document_id)
        STORE.upsert(artifact)
        logger.info(
            "Finished queued RAG conversion. file=%s document_id=%s chunks=%s records=%s",
            safe_file_name,
            artifact.document_id,
            artifact.stats.get("chunk_count"),
            artifact.stats.get("record_count"),
        )
    return ConvertRagResponse(**artifact.to_dict())


def _convert_bytes_with_process_lock(content: bytes, source_file_name: str) -> str:
    settings = get_settings()
    with _conversion_process_lock(settings):
        converter = DocumentConverter(settings)
        return converter.convert_bytes(content, source_file_name)


@contextmanager
def _conversion_process_lock(settings: Settings):
    lock_path = _conversion_lock_path(settings)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        _acquire_file_lock(lock_file)
        try:
            yield
        finally:
            _release_file_lock(lock_file)


def _conversion_lock_path(settings: Settings) -> Path:
    return settings.tmp_root.parent / "conversion.lock"


def _acquire_file_lock(lock_file: Any) -> None:
    while True:
        try:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                lock_file.write("1")
                lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError:
            time.sleep(CONVERSION_LOCK_POLL_SECONDS)


def _release_file_lock(lock_file: Any) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except OSError:
        logger.warning("Could not release conversion process lock cleanly.", exc_info=True)


@app.post("/convert", response_model=ConvertResponse)
async def convert(
    pdf: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
) -> ConvertResponse:
    try:
        upload = _select_upload(pdf, file)
        if upload is None:
            raise ValueError("File parameter is required. Use form-data field 'pdf' or 'file'.")

        converted = await _convert_upload_to_markdown(upload)
        return ConvertResponse(success=True, markdown=converted.markdown)
    except Exception:
        logger.exception("Document conversion failed")
        return ConvertResponse(success=False, markdown="")


@app.post("/convert/batch", response_model=ConvertBatchResponse)
async def convert_batch(
    pdfs: list[UploadFile] | None = File(None),
    files: list[UploadFile] | None = File(None),
    file_names: list[str] | None = Form(None),
) -> ConvertBatchResponse:
    uploads = _select_uploads(pdfs, files)
    if not uploads:
        return ConvertBatchResponse(success=False, error="File parameter is required. Use form-data field 'files' or 'pdfs'.")

    results: list[ConvertBatchItem] = []
    for index, upload in enumerate(uploads):
        try:
            results.append(await _convert_upload_to_markdown(upload, _optional_form_text_at(file_names, index)))
        except Exception as exc:
            logger.exception("Batch document conversion failed. file=%s", upload.filename)
            results.append(ConvertBatchItem(success=False, file_name=upload.filename or "document", error=str(exc)))
    return ConvertBatchResponse(success=all(item.success for item in results), files=results)


@app.post("/convert/rag", response_model=ConvertRagResponse)
async def convert_rag(
    pdf: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
    document_id: str | None = Form(None),
    file_name: str | None = Form(None),
) -> ConvertRagResponse:
    try:
        upload = _select_upload(pdf, file)
        if upload is None:
            raise ValueError("File parameter is required. Use form-data field 'pdf' or 'file'.")

        return await _convert_upload_to_rag(
            upload,
            document_id=_optional_form_text(document_id),
            source_file_name=_optional_form_text(file_name),
        )
    except Exception as exc:
        logger.exception("Document RAG conversion failed")
        return ConvertRagResponse(success=False, error=str(exc))


@app.post("/convert/rag/batch", response_model=ConvertRagBatchResponse)
async def convert_rag_batch(
    pdfs: list[UploadFile] | None = File(None),
    files: list[UploadFile] | None = File(None),
    document_ids: list[str] | None = Form(None),
    file_names: list[str] | None = Form(None),
) -> ConvertRagBatchResponse:
    uploads = _select_uploads(pdfs, files)
    if not uploads:
        return ConvertRagBatchResponse(success=False, error="File parameter is required. Use form-data field 'files' or 'pdfs'.")

    results: list[ConvertRagResponse] = []
    for index, upload in enumerate(uploads):
        try:
            results.append(
                await _convert_upload_to_rag(
                    upload,
                    document_id=_optional_form_text_at(document_ids, index),
                    source_file_name=_optional_form_text_at(file_names, index),
                )
            )
        except Exception as exc:
            logger.exception("Batch document RAG conversion failed. file=%s", upload.filename)
            results.append(ConvertRagResponse(success=False, file_name=upload.filename or "document", error=str(exc)))
    return ConvertRagBatchResponse(success=all(item.success for item in results), files=results)


@app.post("/rag/ingest-markdown", response_model=ConvertRagResponse)
async def rag_ingest_markdown(request: MarkdownIngestRequest) -> ConvertRagResponse:
    try:
        artifact = await asyncio.to_thread(_build_and_store_markdown_artifact, request)
        return ConvertRagResponse(**artifact.to_dict())
    except Exception as exc:
        logger.exception("Markdown RAG ingestion failed")
        return ConvertRagResponse(success=False, error=str(exc))


def _build_and_store_markdown_artifact(request: MarkdownIngestRequest):
    artifact = build_rag_artifact(
        request.markdown,
        request.file_name or "document.md",
        document_id=request.document_id,
    )
    STORE.upsert(artifact)
    return artifact


@app.post("/query/plan", response_model=QueryPlanResponse)
async def query_plan(request: QueryPlanRequest) -> QueryPlanResponse:
    return QueryPlanResponse(**plan_query(request.query, request.document_id))


@app.post("/lookup", response_model=LookupResponse)
async def lookup(request: LookupRequest) -> LookupResponse:
    limit = max(1, min(request.limit, 30))
    result = await asyncio.to_thread(_lookup_from_store, request, limit)
    return LookupResponse(**result)


def _lookup_from_store(request: LookupRequest, limit: int) -> dict[str, Any]:
    return lookup_matches(
        query=request.query,
        records=STORE.records(request.document_id),
        chunks=STORE.chunks(request.document_id),
        entities=request.entities,
        query_type=request.query_type,
        limit=limit,
    )


@app.post("/answer/verify", response_model=VerifyResponse)
async def answer_verify(request: VerifyRequest) -> VerifyResponse:
    result = await asyncio.to_thread(verify_answer, request.question, request.answer, request.evidence)
    return VerifyResponse(**result)


@app.post("/chatflow/merge-evidence", response_model=MergeEvidenceResponse)
async def chatflow_merge_evidence(request_body: Any = Body(None)) -> MergeEvidenceResponse:
    request = normalize_merge_evidence_request(request_body)
    knowledge_source = request.knowledge_result
    if knowledge_source is None:
        knowledge_source = request.knowledge_context
    if knowledge_source is None:
        knowledge_source = request.knowledge_body
    result = await asyncio.to_thread(merge_evidence_context, request.lookup_body, knowledge_source)
    return MergeEvidenceResponse(**result)


def normalize_merge_evidence_request(request_body: Any) -> MergeEvidenceRequest:
    if isinstance(request_body, list):
        request_body = request_body[0] if request_body else {}
    if isinstance(request_body, MergeEvidenceRequest):
        return request_body
    if not isinstance(request_body, dict):
        request_body = {}
    if hasattr(MergeEvidenceRequest, "model_validate"):
        return MergeEvidenceRequest.model_validate(request_body)
    return MergeEvidenceRequest.parse_obj(request_body)


@app.post("/chatflow/debug", response_model=ChatflowDebugResponse)
async def chatflow_debug(request: ChatflowDebugRequest) -> ChatflowDebugResponse:
    limit = max(1, min(request.limit, 30))
    payload = await asyncio.to_thread(_build_chatflow_debug_payload, request, limit)
    return ChatflowDebugResponse(**payload)


def _build_chatflow_debug_payload(request: ChatflowDebugRequest, limit: int) -> dict[str, Any]:
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

    return {
        "query_plan": plan,
        "structured_lookup": lookup_result,
        "merge_evidence": merged,
        "answer_contract": lookup_result.get("answer_contract") or {},
        "draft_answer": draft_answer,
        "draft_verification": draft_verification,
        "supplied_answer_verification": supplied_verification,
        "node_status": {
            "structured_lookup_has_context": bool(lookup_result.get("context")),
            "knowledge_retrieval_supplied": bool(request.knowledge_result),
            "evidence_context_has_structured_lookup": "[Structured Lookup" in merged["evidence_context"],
            "draft_answer_valid": bool(draft_verification.get("valid")),
        },
    }


@app.post("/eval/log")
async def eval_log(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    try:
        record = dict(payload)
        record["_http"] = {
            "client": request.client.host if request.client else "",
            "user_agent": request.headers.get("user-agent", ""),
        }
        return EVAL_LOG_STORE.append(record)
    except Exception as exc:
        logger.exception("Evaluation log write failed")
        return {"success": False, "error": str(exc)}


@app.get("/eval/logs")
async def eval_logs(limit: int = 50, date: str | None = None) -> dict[str, Any]:
    return {"success": True, "logs": EVAL_LOG_STORE.list(limit=limit, date=date)}


@app.get("/eval/logs/{run_id}")
async def eval_log_detail(run_id: str) -> dict[str, Any]:
    record = EVAL_LOG_STORE.get(run_id)
    if record is None:
        return {"success": False, "error": "log not found", "run_id": run_id}
    return {"success": True, "log": record}


@app.get("/rag/documents")
async def rag_documents() -> dict[str, Any]:
    return {"documents": STORE.list_documents()}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
