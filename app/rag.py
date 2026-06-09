from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


PAGE_RE = re.compile(r"(?m)^\s*---\s*Page\s+(\d+)\s*---\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

DATE_RE = re.compile(
    r"(?:(?:19|20)\d{2}\s*[.\-/년]\s*)?"
    r"(?:0?[1-9]|1[0-2])\s*(?:[.\-/월]\s*)"
    r"(?:0?[1-9]|[12]\d|3[01])\s*(?:일)?(?!\s*[%A-Za-z])"
    r"|(?:19|20)\d{2}\s*년"
    r"|(?:상반기|하반기|1학기|2학기|분기|quarter|Q[1-4])",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*"
    r"(?:명|개|건|회|원|만원|억원|달러|USD|KRW|%|점|학점|초|분|시간|일|개월|년|ms|s|KB|MB|GB|TB|req/s|rpm)?"
)

STRUCTURED_LOOKUP_TYPES = {
    "number_lookup",
    "date_lookup",
    "table_lookup",
    "condition_lookup",
    "procedure",
    "troubleshooting",
    "contact_lookup",
    "dependency_lookup",
}

CONTEXT_CHAR_BUDGET = 12_000
MATCH_EVIDENCE_CHAR_LIMIT = 1_800
SUPPORTING_CONTEXT_CHAR_LIMIT = 1_400

NUMBER_RECORD_TYPES = {"quantity", "quota", "money", "metric", "table_row", "key_value"}
TABLE_RECORD_TYPES = {
    "quantity",
    "quota",
    "date",
    "money",
    "metric",
    "requirement",
    "procedure",
    "troubleshooting",
    "contact",
    "dependency",
    "status",
    "identifier",
    "version",
    "policy",
    "table_row",
    "key_value",
}

VALUE_FIELD_RE = re.compile(
    r"수량|개수|건수|횟수|인원|정원|모집인원|한도|제한|임계값|기준값|금액|비용|가격|예산|매출|원가|단가|"
    r"점수|등급|비율|율|퍼센트|기간|날짜|일자|시간|기한|마감|상태|버전|코드|식별자|연락처|이메일|전화|"
    r"count|quantity|capacity|limit|threshold|amount|cost|price|budget|revenue|score|grade|rate|ratio|"
    r"percent|date|time|deadline|duration|status|version|code|id|identifier|email|phone|contact",
    re.IGNORECASE,
)

IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9]{1,}(?:[_-][A-Z0-9]+)+(?![A-Za-z0-9_])"
    r"|(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9._%+-])"
    r"|(?<![A-Za-z0-9_])(?:[A-Z]{2,}-)?\d{3,}[A-Z0-9_-]*(?![A-Za-z0-9_])"
)

ABSTENTION_RE = re.compile(
    r"충분한\s*근거|근거가\s*(?:부족|충분하지)|정보가\s*(?:없|부족)|"
    r"확인할\s*수\s*없|알\s*수\s*없|찾을\s*수\s*없|"
    r"문서(?:에는|에서)?\s*(?:확인|찾|포함).*없|"
    r"제공된\s*문서.*(?:없|부족|포함되어\s*있지)|"
    r"not\s+enough\s+evidence|insufficient\s+evidence|does\s+not\s+contain",
    re.IGNORECASE,
)

ANSWER_CANDIDATE_RE = re.compile(r"(?m)^answer_candidate:\s*(.+?)\s*$")

CONCEPT_TERMS = {
    "storage": ["저장공간", "저장 공간", "스토리지", "용량", "storage", "space", "quota"],
    "compensation": ["연봉", "급여", "임금", "보상", "인상률", "salary", "compensation", "raise"],
    "backup": ["백업", "backup", "restore", "복구"],
    "database": ["데이터베이스", "database", "db"],
    "location": ["장소", "위치", "location", "place", "venue"],
    "preparation": ["준비물", "지참", "bring", "required item"],
    "approval": ["승인", "approval", "approve", "manager approval"],
    "support": ["지원", "제공", "yes", "supported", "available", "included", "support"],
    "plan": ["요금제", "plan", "basic", "pro", "enterprise"],
    "price": ["가격", "비용", "요금", "금액", "price", "cost", "$", "monthly price"],
    "users": ["유저", "사용자", "users", "included users"],
    "retention": ["보존", "retention", "retain"],
    "deprecation": ["폐기", "폐기일", "deprecation", "deprecate"],
    "replacement": ["대체", "replacement", "replace"],
}

STRICT_ABSENCE_CONCEPTS = {"storage", "compensation", "backup", "database"}

STOPWORDS = {
    "그리고",
    "또는",
    "에서",
    "으로",
    "에게",
    "있는",
    "없는",
    "하는",
    "되는",
    "대한",
    "관련",
    "알려줘",
    "무엇",
    "어떻게",
    "please",
    "with",
    "from",
    "this",
    "that",
    "into",
    "and",
    "or",
    "the",
    "for",
    "are",
    "is",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
}

IMPORTANT_TERMS = [
    "SLA",
    "SLO",
    "SLI",
    "KPI",
    "OKR",
    "ROI",
    "MOU",
    "NDA",
    "SOP",
    "ETL",
    "ELT",
    "API",
    "HTTP",
    "HTTPS",
    "REST",
    "GraphQL",
    "gRPC",
    "SQL",
    "NoSQL",
    "JSON",
    "CSV",
    "XML",
    "YAML",
    "PDF",
    "Markdown",
    "DOCX",
    "XLSX",
    "RAG",
    "LLM",
    "OCR",
    "Embedding",
    "Vector DB",
    "Dify",
    "Neo4j",
    "FastAPI",
    "Docker",
    "Kubernetes",
    "Kafka",
    "Redis",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Elasticsearch",
    "OpenDataLoader",
    "Knowledge Retrieval",
    "Vector Search",
    "Hybrid Search",
    "Rerank",
    "장애",
    "오류",
    "에러",
    "장애 대응",
    "트러블슈팅",
    "원인",
    "해결 방법",
    "담당자",
    "연락처",
    "정책",
    "규정",
    "계약",
    "조항",
    "요건",
    "조건",
    "예외",
    "절차",
    "일정",
    "기한",
    "금액",
    "비용",
    "수량",
    "한도",
    "임계값",
    "버전",
    "상태",
    "의존성",
    "리스크",
    "지표",
    "모집인원",
    "모집단위",
    "원서접수",
    "전형",
    "전형일정",
    "지원자격",
    "제출서류",
    "면접",
    "수시",
    "정시",
    "학생부",
    "목록",
    "리스트",
    "명단",
    "개설",
    "학과",
    "부서",
    "종류",
    "카테고리",
]

QUERY_TYPE_TERMS = {
    "number_lookup": [
        "몇",
        "얼마",
        "수",
        "수량",
        "개수",
        "건수",
        "횟수",
        "인원",
        "정원",
        "모집인원",
        "선발",
        "한도",
        "제한",
        "임계값",
        "기준값",
        "점수",
        "금액",
        "비용",
        "가격",
        "예산",
        "매출",
        "가용성",
        "응답시간",
        "처리시간",
        "지연시간",
        "비율",
        "율",
        "percent",
        "percentage",
        "count",
        "quantity",
        "number",
        "amount",
        "capacity",
        "limit",
        "threshold",
        "score",
        "rate",
        "ratio",
        "budget",
        "revenue",
        "quota",
        "availability",
        "response time",
        "latency",
        "throughput",
    ],
    "date_lookup": [
        "언제",
        "날짜",
        "일정",
        "기간",
        "마감",
        "접수",
        "발표",
        "시작",
        "종료",
        "기한",
        "마감일",
        "소요",
        "주기",
        "보존",
        "폐기",
        "폐기일",
        "개정일",
        "date",
        "deadline",
        "schedule",
        "period",
        "start",
        "end",
        "duration",
        "cycle",
        "retention",
        "rotation",
        "deprecation",
        "expiry",
    ],
    "table_lookup": [
        "표",
        "목록",
        "리스트",
        "명단",
        "행",
        "열",
        "항목",
        "종류",
        "분류",
        "카테고리",
        "어떤",
        "어느",
        "무슨",
        "개설",
        "학과",
        "부서",
        "팀",
        "제품",
        "서비스",
        "전형별",
        "항목별",
        "비교표",
        "table",
        "list",
        "catalog",
        "category",
        "type",
        "kind",
        "available",
        "offered",
        "department",
        "team",
        "service",
        "product",
        "row",
        "column",
        "matrix",
    ],
    "procedure": [
        "방법",
        "절차",
        "순서",
        "단계",
        "어떻게",
        "처리",
        "진행",
        "운영",
        "수행",
        "신청",
        "설정",
        "구성",
        "process",
        "procedure",
        "step",
        "workflow",
        "runbook",
        "operate",
        "configure",
        "how",
    ],
    "condition_lookup": [
        "조건",
        "자격",
        "요건",
        "대상",
        "예외",
        "가능",
        "불가",
        "지원",
        "제공",
        "필수",
        "선택",
        "제외",
        "금지",
        "허용",
        "의무",
        "권한",
        "정책",
        "규정",
        "계약",
        "조항",
        "약관",
        "준수",
        "승인",
        "감사",
        "requirement",
        "condition",
        "eligible",
        "support",
        "supported",
        "available",
        "included",
        "exception",
        "mandatory",
        "optional",
        "exclude",
        "allowed",
        "forbidden",
        "permission",
        "policy",
        "rule",
        "contract",
        "clause",
        "terms",
        "compliance",
        "approval",
        "audit",
    ],
    "troubleshooting": [
        "오류",
        "에러",
        "장애",
        "실패",
        "문제",
        "원인",
        "해결",
        "대응",
        "복구",
        "로그",
        "경고",
        "error",
        "failure",
        "issue",
        "incident",
        "root cause",
        "troubleshoot",
        "resolution",
        "recover",
        "log",
        "warning",
    ],
    "contact_lookup": [
        "담당",
        "담당자",
        "부서",
        "팀",
        "연락처",
        "전화",
        "이메일",
        "주소",
        "owner",
        "assignee",
        "department",
        "team",
        "contact",
        "phone",
        "email",
        "address",
    ],
    "dependency_lookup": [
        "의존",
        "영향",
        "연동",
        "관계",
        "참조",
        "상위",
        "하위",
        "선행",
        "후행",
        "dependency",
        "depend",
        "impact",
        "integration",
        "relation",
        "upstream",
        "downstream",
        "parent",
        "child",
    ],
    "definition": [
        "정의",
        "의미",
        "개념",
        "무엇",
        "뭐야",
        "설명",
        "용어",
        "definition",
        "meaning",
        "concept",
        "what is",
        "describe",
        "term",
    ],
    "comparison": [
        "차이",
        "비교",
        "다른",
        "대비",
        "장단점",
        "versus",
        "vs",
        "compare",
        "difference",
        "pros",
        "cons",
    ],
    "summary": [
        "요약",
        "정리",
        "핵심",
        "summary",
        "summarize",
        "overview",
    ],
}


@dataclass
class RagChunk:
    chunk_id: str
    document_id: str
    file_name: str
    page_start: int | None
    page_end: int | None
    section_path: str
    title: str
    text: str
    metadata_text: str
    keywords: list[str] = field(default_factory=list)
    record_types: list[str] = field(default_factory=list)
    text_hash: str = ""


@dataclass
class StructuredRecord:
    record_id: str
    record_type: str
    document_id: str
    file_name: str
    chunk_id: str
    page: int | None
    section_path: str
    fields: dict[str, str]
    source_text: str
    answer_text: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class DocumentMapItem:
    section_path: str
    page_start: int | None
    page_end: int | None
    keywords: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    record_count: int = 0


@dataclass
class RagArtifact:
    success: bool
    document_id: str
    file_name: str
    markdown: str
    dify_markdown: str
    chunks: list[RagChunk] = field(default_factory=list)
    records: list[StructuredRecord] = field(default_factory=list)
    document_map: list[DocumentMapItem] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RagArtifact":
        chunks = [RagChunk(**item) for item in payload.get("chunks", [])]
        records = [StructuredRecord(**item) for item in payload.get("records", [])]
        document_map = [DocumentMapItem(**item) for item in payload.get("document_map", [])]
        return cls(
            success=bool(payload.get("success", False)),
            document_id=str(payload.get("document_id", "")),
            file_name=str(payload.get("file_name", "")),
            markdown=str(payload.get("markdown", "")),
            dify_markdown=str(payload.get("dify_markdown", "")),
            chunks=chunks,
            records=records,
            document_map=document_map,
            stats=dict(payload.get("stats", {})),
        )


class RagStore:
    def __init__(self, store_dir: Path | None = None):
        root = store_dir or Path(os.getenv("RAG_STORE_DIR", str(Path(tempfile.gettempdir()) / "dify-rag-store")))
        self.store_dir = root
        self._documents: dict[str, RagArtifact] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def upsert(self, artifact: RagArtifact) -> None:
        with self._lock:
            self._ensure_loaded_locked()
            self._documents[artifact.document_id] = artifact
            self.store_dir.mkdir(parents=True, exist_ok=True)
            path = self.store_dir / f"{_safe_id(artifact.document_id)}.json"
            path.write_text(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_loaded_locked()
            return [
                {
                    "document_id": artifact.document_id,
                    "file_name": artifact.file_name,
                    "chunk_count": len(artifact.chunks),
                    "record_count": len(artifact.records),
                    "stats": artifact.stats,
                }
                for artifact in sorted(self._documents.values(), key=lambda item: item.file_name)
            ]

    def records(self, document_id: str | None = None) -> list[StructuredRecord]:
        with self._lock:
            self._ensure_loaded_locked()
            artifacts = self._selected_artifacts(document_id)
            return [record for artifact in artifacts for record in artifact.records]

    def chunks(self, document_id: str | None = None) -> list[RagChunk]:
        with self._lock:
            self._ensure_loaded_locked()
            artifacts = self._selected_artifacts(document_id)
            return [chunk for artifact in artifacts for chunk in artifact.chunks]

    def _selected_artifacts(self, document_id: str | None) -> list[RagArtifact]:
        if document_id:
            artifact = self._documents.get(document_id)
            return [artifact] if artifact else []
        return list(self._documents.values())

    def _ensure_loaded_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.store_dir.exists():
            return
        for path in sorted(self.store_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                artifact = RagArtifact.from_dict(payload)
            except Exception:
                continue
            if artifact.document_id:
                self._documents[artifact.document_id] = artifact


STORE = RagStore()


def build_rag_artifact(
    markdown: str,
    file_name: str,
    document_id: str | None = None,
    *,
    chunk_char_limit: int = 3500,
) -> RagArtifact:
    normalized = normalize_markdown(markdown)
    doc_id = document_id or make_document_id(file_name, normalized)
    safe_file_name = file_name or "document.pdf"

    chunks: list[RagChunk] = []
    records: list[StructuredRecord] = []

    section_blocks = list(iter_section_blocks(normalized))
    chunk_index = 0
    for block in section_blocks:
        expanded_text, block_records = expand_tables_to_records(
            block["text"],
            document_id=doc_id,
            file_name=safe_file_name,
            page=block["page"],
            section_path=block["section_path"],
        )
        texts = split_chunk_text(expanded_text, chunk_char_limit)
        if not texts:
            continue

        first_chunk_id = ""
        for text in texts:
            chunk_index += 1
            chunk_id = make_chunk_id(doc_id, block["page"], chunk_index)
            if not first_chunk_id:
                first_chunk_id = chunk_id
            chunk_keywords = extract_keywords(
                "\n".join([block["section_path"], text, " ".join(record.answer_text for record in block_records)]),
                limit=18,
            )
            record_types = sorted({record.record_type for record in block_records})
            metadata_text = build_metadata_header(
                document_id=doc_id,
                file_name=safe_file_name,
                chunk_id=chunk_id,
                page_start=block["page"],
                page_end=block["page"],
                section_path=block["section_path"],
                keywords=chunk_keywords,
                record_types=record_types,
            )
            chunk = RagChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                file_name=safe_file_name,
                page_start=block["page"],
                page_end=block["page"],
                section_path=block["section_path"],
                title=block["title"],
                text=text,
                metadata_text=metadata_text,
                keywords=chunk_keywords,
                record_types=record_types,
                text_hash=short_hash(text),
            )
            chunks.append(chunk)

        for record in block_records:
            record.chunk_id = first_chunk_id
            records.append(record)

    dify_markdown = "\n\n---\n\n".join(f"{chunk.metadata_text}\n\n{chunk.text}" for chunk in chunks).strip()
    document_map = build_document_map(chunks, records)
    stats = {
        "chunk_count": len(chunks),
        "record_count": len(records),
        "page_count": len({chunk.page_start for chunk in chunks if chunk.page_start is not None}),
        "section_count": len(document_map),
        "dify_markdown_chars": len(dify_markdown),
    }
    return RagArtifact(
        success=True,
        document_id=doc_id,
        file_name=safe_file_name,
        markdown=normalized,
        dify_markdown=dify_markdown,
        chunks=chunks,
        records=records,
        document_map=document_map,
        stats=stats,
    )


def make_document_id(file_name: str, markdown: str) -> str:
    seed = f"{file_name}\n{markdown[:12000]}"
    return f"doc_{short_hash(seed, length=16)}"


def make_chunk_id(document_id: str, page: int | None, index: int) -> str:
    page_part = f"p{page:03d}" if page is not None else "p000"
    return f"{_safe_id(document_id)}_{page_part}_c{index:04d}"


def build_metadata_header(
    *,
    document_id: str,
    file_name: str,
    chunk_id: str,
    page_start: int | None,
    page_end: int | None,
    section_path: str,
    keywords: list[str],
    record_types: list[str],
) -> str:
    page_value = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
    return "\n".join(
        [
            f"[document_id: {document_id}]",
            f"[chunk_id: {chunk_id}]",
            f"[file_name: {file_name}]",
            f"[page: {page_value}]",
            f"[section: {section_path or 'Untitled'}]",
            f"[record_types: {', '.join(record_types) if record_types else 'text'}]",
            f"[keywords: {', '.join(keywords)}]",
        ]
    )


def normalize_markdown(value: str) -> str:
    value = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.split("\n")]
    value = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def iter_section_blocks(markdown: str) -> Iterable[dict[str, Any]]:
    section_stack: list[tuple[int, str]] = []
    for page, page_text in split_pages(markdown):
        current_lines: list[str] = []
        current_section = " > ".join(title for _, title in section_stack)
        current_title = section_stack[-1][1] if section_stack else ""

        def flush() -> dict[str, Any] | None:
            text = normalize_markdown("\n".join(current_lines))
            if not text:
                return None
            return {
                "page": page,
                "section_path": current_section,
                "title": current_title,
                "text": text,
            }

        for line in page_text.split("\n"):
            match = HEADING_RE.match(line.strip())
            if match:
                block = flush()
                if block:
                    yield block
                level = len(match.group(1))
                title = clean_cell(match.group(2))
                section_stack = [(item_level, item_title) for item_level, item_title in section_stack if item_level < level]
                section_stack.append((level, title))
                current_section = " > ".join(item_title for _, item_title in section_stack)
                current_title = title
                current_lines = [line]
                continue
            current_lines.append(line)

        block = flush()
        if block:
            yield block


def split_pages(markdown: str) -> list[tuple[int | None, str]]:
    matches = list(PAGE_RE.finditer(markdown))
    if not matches:
        return [(None, markdown)]

    pages: list[tuple[int | None, str]] = []
    preface = markdown[: matches[0].start()].strip()
    if preface:
        pages.append((None, preface))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        pages.append((int(match.group(1)), markdown[start:end].strip()))
    return pages


def split_chunk_text(text: str, limit: int) -> list[str]:
    text = normalize_markdown(text)
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > limit:
            for sentence_chunk in split_long_text(paragraph, limit):
                if current:
                    chunks.append(normalize_markdown("\n\n".join(current)))
                    current = []
                    current_length = 0
                chunks.append(sentence_chunk)
            continue
        if current and current_length + len(paragraph) + 2 > limit:
            chunks.append(normalize_markdown("\n\n".join(current)))
            current = []
            current_length = 0
        current.append(paragraph)
        current_length += len(paragraph) + 2

    if current:
        chunks.append(normalize_markdown("\n\n".join(current)))
    return chunks


def split_long_text(text: str, limit: int) -> list[str]:
    units = re.split(r"(?<=[.!?。！？])\s+|\n", text)
    chunks: list[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > limit:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(unit[index : index + limit].strip() for index in range(0, len(unit), limit))
            continue
        if current and len(current) + len(unit) + 1 > limit:
            chunks.append(current.strip())
            current = unit
        else:
            current = f"{current} {unit}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def expand_tables_to_records(
    text: str,
    *,
    document_id: str,
    file_name: str,
    page: int | None,
    section_path: str,
) -> tuple[str, list[StructuredRecord]]:
    lines = text.split("\n")
    output: list[str] = []
    records: list[StructuredRecord] = []
    table_index = 0
    index = 0

    while index < len(lines):
        if not looks_like_table_row(lines[index]):
            output.append(lines[index])
            index += 1
            continue

        table_lines = []
        while index < len(lines) and looks_like_table_row(lines[index]):
            table_lines.append(lines[index])
            index += 1

        parsed = parse_markdown_table(table_lines)
        output.extend(table_lines)
        if not parsed:
            continue

        table_index += 1
        headers, rows = parsed
        row_texts = []
        for row_index, row in enumerate(rows, start=1):
            fields = {
                header: clean_cell(row[column_index]) if column_index < len(row) else ""
                for column_index, header in enumerate(headers)
            }
            fields = {key: value for key, value in fields.items() if key and value}
            if not fields:
                continue
            answer_text = row_to_sentence(fields)
            source_text = " | ".join(fields.values())
            record_type = classify_record(fields, source_text)
            keywords = extract_keywords(" ".join([section_path, source_text, " ".join(fields), answer_text]), limit=12)
            record = StructuredRecord(
                record_id=f"{_safe_id(document_id)}_p{page or 0:03d}_t{table_index:02d}_r{row_index:03d}",
                record_type=record_type,
                document_id=document_id,
                file_name=file_name,
                chunk_id="",
                page=page,
                section_path=section_path,
                fields=fields,
                source_text=source_text,
                answer_text=answer_text,
                keywords=keywords,
            )
            records.append(record)
            row_texts.append(build_record_text(record))

        if row_texts:
            output.append("")
            output.append("표 행 설명 및 구조화 레코드:")
            output.extend(row_texts)

    return normalize_markdown("\n".join(output)), records


def looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def parse_markdown_table(lines: list[str]) -> tuple[list[str], list[list[str]]] | None:
    matrix = [split_table_row(line) for line in lines]
    matrix = [row for row in matrix if any(cell.strip() for cell in row)]
    if len(matrix) < 2:
        return None

    if is_separator_row(matrix[1]):
        headers = matrix[0]
        rows = matrix[2:]
    else:
        width = max(len(row) for row in matrix)
        headers = [f"Column {index}" for index in range(1, width + 1)]
        rows = matrix

    width = max([len(headers), *(len(row) for row in rows)] or [0])
    headers = [clean_cell(header) or f"Column {index}" for index, header in enumerate(headers + [""] * width, start=1)][:width]
    rows = [(row + [""] * width)[:width] for row in rows]
    rows = [row for row in rows if any(clean_cell(cell) for cell in row)]
    if not rows:
        return None
    return headers, rows


def split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = re.split(r"(?<!\\)\|", stripped)
    return [clean_cell(cell.replace(r"\|", "|")) for cell in cells]


def is_separator_row(row: list[str]) -> bool:
    return bool(row) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in row if cell.strip())


def clean_cell(value: Any) -> str:
    text = str(value or "")
    text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    text = re.sub(r"[*`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def row_to_sentence(fields: dict[str, str]) -> str:
    items = [(key, value) for key, value in fields.items() if value]
    if not items:
        return ""

    subject_key, subject_value = choose_subject_field(items)
    attributes = [(key, value) for key, value in items if key != subject_key]
    if not attributes:
        return f"{subject_key}은/는 {subject_value}입니다."
    if len(attributes) == 1:
        key, value = attributes[0]
        return f"{subject_value}의 {key}은/는 {value}입니다."
    detail = ", ".join(f"{key}: {value}" for key, value in attributes)
    return f"{subject_value} 항목은 {detail}입니다."


def choose_subject_field(items: list[tuple[str, str]]) -> tuple[str, str]:
    for key, value in items:
        if not looks_like_numeric_or_date(value) and not VALUE_FIELD_RE.search(key):
            return key, value
    return items[0]


def looks_like_numeric_or_date(value: str) -> bool:
    return bool(NUMBER_RE.search(value) or DATE_RE.search(value))


def classify_record(fields: dict[str, str], source_text: str) -> str:
    haystack = " ".join([source_text, *fields.keys(), *fields.values()]).lower()
    if DATE_RE.search(source_text) or re.search(r"날짜|일정|기간|마감|접수|시작|종료|기한|소요|date|deadline|schedule|period|start|end|duration", haystack):
        return "date"
    if re.search(r"점수|등급|비율|율|퍼센트|%|성공률|가용성|처리율|score|grade|rate|ratio|percentage|availability|throughput|latency", haystack):
        return "metric"
    if re.search(r"금액|비용|수수료|가격|예산|매출|원가|단가|[\d,]+\s*(?:원|만원|억원|달러)|amount|cost|price|fee|budget|revenue", haystack):
        return "money"
    if re.search(r"모집인원|정원|인원|선발|수량|개수|건수|횟수|한도|제한|임계값|기준값|count|quantity|quota|headcount|capacity|limit|threshold", haystack):
        return "quantity"
    if re.search(r"오류|에러|장애|실패|문제|원인|해결|대응|복구|로그|경고|error|failure|issue|incident|cause|resolution|troubleshoot|recover|log|warning", haystack):
        return "troubleshooting"
    if re.search(r"담당|담당자|부서|팀|연락처|전화|이메일|주소|owner|assignee|department|team|contact|phone|email|address", haystack):
        return "contact"
    if re.search(r"의존|영향|연동|관계|참조|상위|하위|선행|후행|dependency|depend|impact|integration|relation|upstream|downstream|parent|child", haystack):
        return "dependency"
    if re.search(r"상태|진행상태|결과|단계상태|status|state|result", haystack):
        return "status"
    if re.search(r"버전|릴리스|개정|revision|version|release|build", haystack):
        return "version"
    if re.search(r"정책|규정|계약|조항|약관|준수|승인|감사|보존|policy|rule|contract|clause|terms|compliance|approval|audit|retention", haystack):
        return "policy"
    if re.search(r"코드|식별자|번호|id|identifier|code|key", haystack):
        return "identifier"
    if re.search(r"조건|자격|요건|대상|예외|필수|선택|제외|금지|허용|의무|권한|requirement|condition|eligible|exception|mandatory|optional|forbidden|allowed|permission", haystack):
        return "requirement"
    if re.search(r"단계|절차|순서|방법|처리|진행|운영|수행|신청|설정|구성|step|procedure|process|workflow|runbook|operate|configure", haystack):
        return "procedure"
    if len(fields) <= 2:
        return "key_value"
    return "table_row"


def build_record_text(record: StructuredRecord) -> str:
    fields = "\n".join(f"  {key}: {value}" for key, value in record.fields.items())
    return "\n".join(
        [
            f"- record_id: {record.record_id}",
            f"  record_type: {record.record_type}",
            f"  answer_hint: {record.answer_text}",
            fields,
        ]
    )


def build_document_map(chunks: list[RagChunk], records: list[StructuredRecord]) -> list[DocumentMapItem]:
    grouped: dict[str, DocumentMapItem] = {}
    for chunk in chunks:
        key = chunk.section_path or "Untitled"
        item = grouped.setdefault(
            key,
            DocumentMapItem(
                section_path=key,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                keywords=[],
                chunk_ids=[],
                record_count=0,
            ),
        )
        pages = [page for page in [item.page_start, item.page_end, chunk.page_start, chunk.page_end] if page is not None]
        if pages:
            item.page_start = min(pages)
            item.page_end = max(pages)
        item.chunk_ids.append(chunk.chunk_id)
        item.keywords = unique_keep_order([*item.keywords, *chunk.keywords])[:20]

    for record in records:
        key = record.section_path or "Untitled"
        item = grouped.setdefault(
            key,
            DocumentMapItem(
                section_path=key,
                page_start=record.page,
                page_end=record.page,
                keywords=record.keywords[:20],
                chunk_ids=[],
                record_count=0,
            ),
        )
        item.record_count += 1
    return list(grouped.values())


def plan_query(query: str, document_id: str | None = None) -> dict[str, Any]:
    normalized_query = clean_cell(query)
    keywords = extract_keywords(normalized_query, limit=16)
    query_type = classify_query_type(normalized_query)
    sub_queries = split_sub_queries(normalized_query, keywords)
    expanded_queries = build_query_variants(normalized_query, keywords, query_type)
    expanded_queries = unique_keep_order([*expanded_queries, *sub_queries])
    answer_style = classify_answer_style(normalized_query, query_type)
    return {
        "query": normalized_query,
        "document_id": document_id,
        "query_type": query_type,
        "entities": keywords,
        "keywords": keywords,
        "sub_queries": sub_queries,
        "expanded_queries": expanded_queries,
        "answer_style": answer_style,
        "retrieval_hints": {
            "prefer_structured_lookup": query_type in STRUCTURED_LOOKUP_TYPES,
            "prefer_exact_terms": keywords[:8],
            "answer_must_include_evidence": True,
            "answer_style": answer_style,
            "min_evidence_count": 2 if query_type in {"comparison", "summary"} else 1,
        },
    }


def classify_query_type(query: str) -> str:
    lower = query.lower()
    scores = {
        query_type: sum(1 for term in terms if term.lower() in lower)
        for query_type, terms in QUERY_TYPE_TERMS.items()
    }
    if NUMBER_RE.search(query) and scores["date_lookup"] == 0:
        scores["number_lookup"] += 1
    if DATE_RE.search(query):
        scores["date_lookup"] += 2
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    return best_type if best_score > 0 else "semantic"


def split_sub_queries(query: str, keywords: list[str]) -> list[str]:
    parts = [
        clean_cell(part)
        for part in re.split(r"\s+(?:그리고|또한|및|\band\b|\bor\b)\s+|(?<=[A-Za-z0-9가-힣])(?:와|과)\s+|[,;/]\s*", query)
        if clean_cell(part)
    ]
    useful_parts = [part for part in parts if len(part) >= 4]
    if 1 < len(useful_parts) <= 5:
        return unique_keep_order([query, *useful_parts, *keywords[:5]])
    return unique_keep_order([query, *keywords[:5]])


def classify_answer_style(query: str, query_type: str) -> str:
    lower = query.lower()
    if query_type == "comparison":
        return "compare"
    if query_type == "summary":
        return "summarize"
    if query_type == "procedure":
        return "steps"
    if re.search(r"표|목록|리스트|명단|항목|종류|카테고리|어떤|어느|무슨|개설|list|table|catalog|category|available|offered", lower):
        return "table_or_bullets"
    if query_type in {"number_lookup", "date_lookup", "contact_lookup"}:
        return "direct"
    return "grounded_explanation"


def build_query_variants(query: str, keywords: list[str], query_type: str) -> list[str]:
    variants = [query]
    if keywords:
        variants.append(" ".join(unique_keep_order([query, *keywords])))
    if query_type == "number_lookup":
        variants.append(" ".join(unique_keep_order([*keywords, "숫자", "수량", "개수", "한도", "임계값", "amount", "count", "quantity", "limit", "threshold"])))
    elif query_type == "date_lookup":
        variants.append(" ".join(unique_keep_order([*keywords, "날짜", "일정", "기간", "기한", "시작", "종료", "date", "deadline", "schedule", "duration"])))
    elif query_type == "table_lookup":
        variants.append(" ".join(unique_keep_order([*keywords, "표", "목록", "리스트", "명단", "행", "열", "항목", "종류", "table", "list", "catalog", "row", "column", "record"])))
    elif query_type == "procedure":
        variants.append(" ".join(unique_keep_order([*keywords, "절차", "단계", "방법", "워크플로우", "procedure", "step", "workflow", "runbook"])))
    elif query_type == "condition_lookup":
        variants.append(" ".join(unique_keep_order([*keywords, "조건", "요건", "예외", "권한", "requirement", "condition", "exception", "permission"])))
    elif query_type == "troubleshooting":
        variants.append(" ".join(unique_keep_order([*keywords, "오류", "장애", "원인", "해결", "복구", "error", "incident", "cause", "resolution"])))
    elif query_type == "contact_lookup":
        variants.append(" ".join(unique_keep_order([*keywords, "담당자", "부서", "연락처", "이메일", "owner", "team", "contact", "email"])))
    elif query_type == "dependency_lookup":
        variants.append(" ".join(unique_keep_order([*keywords, "의존성", "영향", "연동", "관계", "dependency", "impact", "integration"])))
    elif query_type == "definition":
        variants.append(" ".join(unique_keep_order([*keywords, "정의", "의미", "개념", "definition", "meaning", "concept"])))
    elif query_type == "comparison":
        variants.append(" ".join(unique_keep_order([*keywords, "비교", "차이", "대비", "compare", "difference", "versus"])))
    return [variant for variant in unique_keep_order(variants) if variant.strip()]


def lookup_matches(
    *,
    query: str,
    records: list[StructuredRecord],
    chunks: list[RagChunk],
    entities: list[str] | None = None,
    query_type: str | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    plan = plan_query(query)
    active_entities = unique_keep_order([*(entities or []), *plan["entities"]])
    active_query_type = query_type or plan["query_type"]
    terms = normalize_terms([query, *active_entities])
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunk_order = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}

    record_matches = []
    for record in records:
        score = score_record(record, terms, active_query_type)
        if score <= 0:
            continue
        chunk = chunk_by_id.get(record.chunk_id)
        evidence = format_record_evidence(record)
        supporting_context = format_supporting_context(chunk, chunks, chunk_order) if chunk else ""
        coverage_text = " ".join([record.section_path, evidence, supporting_context])
        record_matches.append(
            {
                "match_type": "structured_record",
                "score": score,
                "answer": choose_answer_candidate(record, active_query_type, terms),
                "record_id": record.record_id,
                "record_type": record.record_type,
                "document_id": record.document_id,
                "file_name": record.file_name,
                "page": record.page,
                "section_path": record.section_path,
                "chunk_id": record.chunk_id,
                "fields": record.fields,
                "evidence": evidence,
                "supporting_context": supporting_context,
                "answer_hint": record.answer_text,
                "chunk_text": chunk.text if chunk else "",
                "coverage": term_coverage(coverage_text, terms),
                "matched_terms": matched_terms(coverage_text, terms),
                "reason": explain_match(record.record_type, active_query_type, coverage_text, terms),
            }
        )

    chunk_matches = []
    for chunk in chunks:
        score = score_chunk(chunk, terms, active_query_type)
        if score <= 0:
            continue
        chunk_matches.append(
            {
                "match_type": "chunk",
                "score": score,
                "answer": "",
                "record_id": "",
                "record_type": ",".join(chunk.record_types) or "text",
                "document_id": chunk.document_id,
                "file_name": chunk.file_name,
                "page": chunk.page_start,
                "section_path": chunk.section_path,
                "chunk_id": chunk.chunk_id,
                "fields": {},
                "evidence": trim_text(chunk.text, MATCH_EVIDENCE_CHAR_LIMIT),
                "supporting_context": "",
                "answer_hint": "",
                "chunk_text": chunk.text,
                "coverage": term_coverage(chunk.text, terms),
                "matched_terms": matched_terms(chunk.text, terms),
                "reason": explain_match(",".join(chunk.record_types) or "text", active_query_type, chunk.text, terms),
            }
        )

    ranked = sorted(
        [*record_matches, *chunk_matches],
        key=lambda item: (-item["score"], -item["coverage"], item["match_type"]),
    )
    combined = select_evidence_matches(ranked, limit)
    answerability = assess_answerability(query, active_query_type, active_entities, combined)
    if not answerability["answerable"]:
        combined = []
    return {
        "query": query,
        "query_type": active_query_type,
        "entities": active_entities,
        "sub_queries": plan.get("sub_queries", []),
        "answer_style": plan.get("answer_style", "grounded_explanation"),
        "matches": combined,
        "context": format_lookup_context(combined),
        "evidence_items": build_evidence_items(combined),
        "diagnostics": {
            "candidate_count": len(ranked),
            "selected_count": len(combined),
            "top_score": ranked[0]["score"] if ranked else 0,
            "average_coverage": round(sum(match["coverage"] for match in combined) / len(combined), 3) if combined else 0,
            "answerability": answerability,
        },
    }


def format_record_evidence(record: StructuredRecord) -> str:
    field_lines = [f"{key}: {value}" for key, value in record.fields.items() if value]
    return "\n".join(
        [
            f"record_type: {record.record_type}",
            f"answer_hint: {record.answer_text}",
            *field_lines,
        ]
    )


def format_supporting_context(chunk: RagChunk, chunks: list[RagChunk], chunk_order: dict[str, int]) -> str:
    index = chunk_order.get(chunk.chunk_id)
    if index is None:
        return trim_text(chunk.text, SUPPORTING_CONTEXT_CHAR_LIMIT)

    neighbors = []
    for neighbor_index in range(max(0, index - 1), min(len(chunks), index + 2)):
        neighbor = chunks[neighbor_index]
        if neighbor.document_id != chunk.document_id:
            continue
        if neighbor_index != index and neighbor.section_path != chunk.section_path and neighbor.page_start != chunk.page_start:
            continue
        label = "current" if neighbor_index == index else "neighbor"
        neighbors.append(f"[{label} chunk {neighbor.chunk_id} page={neighbor.page_start}]\n{neighbor.text}")
    return trim_text("\n\n".join(neighbors), SUPPORTING_CONTEXT_CHAR_LIMIT)


def matched_terms(text: str, terms: list[str]) -> list[str]:
    haystack = normalize_for_match(text)
    matches = []
    for term in terms:
        term_norm = normalize_for_match(term)
        if not term_norm:
            continue
        if term_norm in haystack:
            matches.append(term)
            continue
        tokens = [token for token in term_norm.split() if len(token) >= 2]
        if tokens and all(token in haystack for token in tokens):
            matches.append(term)
    return unique_keep_order(matches)[:16]


def term_coverage(text: str, terms: list[str]) -> float:
    important_terms = [term for term in unique_keep_order(terms) if len(normalize_for_match(term)) >= 2]
    if not important_terms:
        return 0.0
    return len(matched_terms(text, important_terms)) / len(important_terms)


def explain_match(record_type: str, query_type: str, text: str, terms: list[str]) -> str:
    reasons = []
    matches = matched_terms(text, terms)
    if matches:
        reasons.append("matched_terms=" + ", ".join(matches[:8]))
    if query_type in STRUCTURED_LOOKUP_TYPES:
        reasons.append(f"query_type={query_type}")
    if record_type:
        reasons.append(f"record_type={record_type}")
    return "; ".join(reasons)


def select_evidence_matches(ranked: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_chunk_ids: set[str] = set()
    selected_record_ids: set[str] = set()

    for match in ranked:
        if len(selected) >= limit:
            break
        record_id = match.get("record_id") or ""
        chunk_id = match.get("chunk_id") or ""

        if record_id and record_id in selected_record_ids:
            continue
        if match.get("match_type") == "chunk" and chunk_id in selected_chunk_ids:
            continue

        selected.append(match)
        if record_id:
            selected_record_ids.add(record_id)
        if chunk_id:
            selected_chunk_ids.add(chunk_id)

    return selected


def build_evidence_items(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "match_type": match.get("match_type", ""),
            "score": match.get("score", 0),
            "coverage": match.get("coverage", 0),
            "file_name": match.get("file_name", ""),
            "page": match.get("page"),
            "section_path": match.get("section_path", ""),
            "chunk_id": match.get("chunk_id", ""),
            "answer": match.get("answer") or match.get("answer_hint") or "",
            "matched_terms": match.get("matched_terms", []),
        }
        for match in matches
    ]


def assess_answerability(
    query: str,
    query_type: str,
    entities: list[str],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    if not matches:
        return {
            "answerable": False,
            "confidence": 0.0,
            "query_concepts": sorted(detect_concepts(query)),
            "evidence_concepts": [],
            "missing_strict_concepts": [],
            "reason": "no evidence selected",
        }

    evidence_text = "\n".join(
        " ".join(
            [
                str(match.get("answer") or match.get("answer_hint") or ""),
                str(match.get("evidence") or ""),
                str(match.get("supporting_context") or ""),
                " ".join(str(term) for term in match.get("matched_terms", [])),
            ]
        )
        for match in matches
    )
    query_concepts = detect_concepts(" ".join([query, *entities]))
    evidence_concepts = detect_concepts(evidence_text)
    missing_strict_concepts = sorted((query_concepts & STRICT_ABSENCE_CONCEPTS) - evidence_concepts)
    average_coverage = sum(float(match.get("coverage") or 0) for match in matches) / len(matches)
    top_score = max(float(match.get("score") or 0) for match in matches)

    if missing_strict_concepts:
        return {
            "answerable": False,
            "confidence": 0.2,
            "query_concepts": sorted(query_concepts),
            "evidence_concepts": sorted(evidence_concepts),
            "missing_strict_concepts": missing_strict_concepts,
            "reason": "query asks for a specific concept that is absent from evidence",
        }

    weak_signal = average_coverage < 0.12 and top_score < 6.0 and query_type in STRUCTURED_LOOKUP_TYPES
    if weak_signal:
        return {
            "answerable": False,
            "confidence": 0.35,
            "query_concepts": sorted(query_concepts),
            "evidence_concepts": sorted(evidence_concepts),
            "missing_strict_concepts": [],
            "reason": "evidence has weak query coverage",
        }

    return {
        "answerable": True,
        "confidence": round(min(0.95, 0.55 + min(average_coverage, 1.0) * 0.35 + min(top_score / 40.0, 0.25)), 3),
        "query_concepts": sorted(query_concepts),
        "evidence_concepts": sorted(evidence_concepts),
        "missing_strict_concepts": [],
        "reason": "evidence passed answerability checks",
    }


def detect_concepts(text: str) -> set[str]:
    normalized = normalize_for_match(text)
    concepts = set()
    for concept, terms in CONCEPT_TERMS.items():
        for term in terms:
            term_norm = normalize_for_match(term)
            if term_norm and term_norm in normalized:
                concepts.add(concept)
                break
    return concepts


def score_record(record: StructuredRecord, terms: list[str], query_type: str) -> float:
    haystack = normalize_for_match(
        " ".join(
            [
                record.section_path,
                record.record_type,
                record.source_text,
                record.answer_text,
                " ".join(record.fields.keys()),
                " ".join(record.fields.values()),
                " ".join(record.keywords),
            ]
        )
    )
    score = 0.0
    matched_terms = 0
    for term in terms:
        term_norm = normalize_for_match(term)
        if not term_norm:
            continue
        if term_norm in haystack:
            matched_terms += 1
            score += 5.0 if " " in term_norm else 3.0
        else:
            for token in term_norm.split():
                if len(token) >= 2 and token in haystack:
                    score += 0.8
    if matched_terms >= 2:
        score += 4.0
    coverage = term_coverage(haystack, terms)
    has_term_signal = matched_terms > 0 or coverage > 0
    score += coverage * 6.0
    if query_type == "number_lookup" and has_term_signal and record.record_type in NUMBER_RECORD_TYPES:
        if any(NUMBER_RE.search(value) for value in record.fields.values()):
            score += 5.0
    if query_type == "date_lookup" and has_term_signal and (record.record_type == "date" or DATE_RE.search(record.source_text)):
        score += 5.0
    if query_type == "table_lookup" and has_term_signal and record.record_type in TABLE_RECORD_TYPES:
        score += 3.0
    query_record_bonus = {
        "condition_lookup": {"requirement"},
        "procedure": {"procedure"},
        "troubleshooting": {"troubleshooting"},
        "contact_lookup": {"contact"},
        "dependency_lookup": {"dependency"},
        "definition": {"key_value", "table_row", "policy"},
        "comparison": TABLE_RECORD_TYPES,
    }
    if has_term_signal and record.record_type in query_record_bonus.get(query_type, set()):
        score += 5.0
    return score


def score_chunk(chunk: RagChunk, terms: list[str], query_type: str) -> float:
    haystack = normalize_for_match(" ".join([chunk.section_path, chunk.text, " ".join(chunk.keywords), " ".join(chunk.record_types)]))
    score = 0.0
    exact_term_matches = 0
    for term in terms:
        term_norm = normalize_for_match(term)
        if not term_norm:
            continue
        if term_norm in haystack:
            exact_term_matches += 1
            score += 2.0
    coverage = term_coverage(haystack, terms)
    has_term_signal = exact_term_matches > 0 or coverage > 0
    score += coverage * 3.0
    if query_type == "number_lookup" and has_term_signal and NUMBER_RE.search(chunk.text):
        score += 1.0
    if query_type == "date_lookup" and has_term_signal and DATE_RE.search(chunk.text):
        score += 1.0
    if has_term_signal and query_type == "troubleshooting" and re.search(r"오류|에러|장애|실패|원인|해결|error|failure|incident|resolution", chunk.text, re.IGNORECASE):
        score += 1.0
    if has_term_signal and query_type == "condition_lookup" and re.search(r"조건|요건|예외|권한|requirement|condition|exception|permission|지원|제공|support|available|included", chunk.text, re.IGNORECASE):
        score += 1.0
    if has_term_signal and query_type == "procedure" and re.search(r"절차|단계|방법|workflow|procedure|step|runbook", chunk.text, re.IGNORECASE):
        score += 1.0
    return score


def choose_answer_candidate(record: StructuredRecord, query_type: str, terms: list[str]) -> str:
    if query_type in {"number_lookup", "date_lookup"}:
        candidate_fields = []
        for key, value in record.fields.items():
            key_norm = normalize_for_match(key)
            value_has_target = NUMBER_RE.search(value) if query_type == "number_lookup" else DATE_RE.search(value)
            key_matches_query = any(normalize_for_match(term) in key_norm or key_norm in normalize_for_match(term) for term in terms if term)
            generic_key_match = bool(VALUE_FIELD_RE.search(key))
            if value_has_target and (key_matches_query or generic_key_match):
                candidate_fields.append((key, value))
        if len(candidate_fields) == 1:
            key, value = candidate_fields[0]
            return f"{key}: {value}"
        if candidate_fields:
            return ", ".join(f"{key}: {value}" for key, value in candidate_fields)
    if query_type == "contact_lookup":
        contact_fields = [
            (key, value)
            for key, value in record.fields.items()
            if re.search(r"담당|부서|팀|연락처|전화|이메일|주소|owner|team|contact|phone|email|address", key, re.IGNORECASE)
        ]
        if contact_fields:
            return ", ".join(f"{key}: {value}" for key, value in contact_fields)
    return record.answer_text


def format_lookup_context(matches: list[dict[str, Any]]) -> str:
    blocks = []
    used_chars = 0
    for match in matches:
        evidence = trim_text(str(match.get("evidence", "")), MATCH_EVIDENCE_CHAR_LIMIT)
        supporting_context = trim_text(str(match.get("supporting_context", "")), SUPPORTING_CONTEXT_CHAR_LIMIT)
        block = "\n".join(
            [
                f"[match_type: {match['match_type']}]",
                f"[score: {match['score']:.2f}]",
                f"[coverage: {match.get('coverage', 0):.2f}]",
                f"[file_name: {match['file_name']}]",
                f"[page: {match['page']}]",
                f"[section: {match['section_path']}]",
                f"[chunk_id: {match['chunk_id']}]",
                f"[matched_terms: {', '.join(match.get('matched_terms', []))}]",
                f"[reason: {match.get('reason', '')}]",
                f"answer_candidate: {match['answer'] or match['answer_hint']}",
                f"evidence:\n{evidence}",
                f"supporting_context:\n{supporting_context}" if supporting_context else "",
            ]
        ).strip()
        if used_chars and used_chars + len(block) > CONTEXT_CHAR_BUDGET:
            remaining = CONTEXT_CHAR_BUDGET - used_chars
            if remaining <= 400:
                break
            block = trim_text(block, remaining)
        blocks.append(block)
        used_chars += len(block) + 2
        if used_chars >= CONTEXT_CHAR_BUDGET:
            break
    return "\n\n".join(blocks)


def trim_text(text: str, limit: int) -> str:
    text = normalize_markdown(text)
    if len(text) <= limit:
        return text
    head_limit = max(0, limit - 80)
    return text[:head_limit].rstrip() + "\n...[trimmed]"


def verify_answer(question: str, answer: str, evidence: list[Any]) -> dict[str, Any]:
    evidence_text = evidence_to_text(evidence)
    issues = []
    if not answer.strip():
        issues.append({"type": "empty_answer", "message": "Answer is empty."})
    if not evidence_text.strip():
        issues.append({"type": "missing_evidence", "message": "No evidence was provided."})

    query_type = classify_query_type(question)
    answer_numbers = extract_number_values(answer)
    evidence_numbers = extract_number_values(evidence_text)
    answer_dates = extract_date_values(answer)
    evidence_dates = extract_date_values(evidence_text)
    answer_identifiers = extract_identifier_values(answer)
    evidence_identifiers = extract_identifier_values(evidence_text)

    if query_type == "number_lookup" and not answer_numbers:
        issues.append({"type": "missing_number", "message": "The question asks for a numeric answer, but the answer has no number."})
    for number in answer_numbers:
        if number not in evidence_numbers:
            issues.append(
                {
                    "type": "unsupported_number",
                    "message": f"The answer contains number '{number}' that is not present in evidence.",
                }
            )

    if query_type == "date_lookup" and not answer_dates:
        issues.append({"type": "missing_date", "message": "The question asks for a date or period, but the answer has no date."})
    for date in answer_dates:
        if date not in evidence_dates:
            issues.append(
                {
                    "type": "unsupported_date",
                    "message": f"The answer contains date '{date}' that is not present in evidence.",
                }
            )

    for identifier in answer_identifiers:
        if identifier not in evidence_identifiers:
            issues.append(
                {
                    "type": "unsupported_identifier",
                    "message": f"The answer contains identifier '{identifier}' that is not present in evidence.",
                }
            )

    question_terms = extract_keywords(question, limit=8)
    question_coverage = term_coverage(evidence_text, question_terms)
    missing_terms = [
        term
        for term in question_terms
        if len(term) >= 3 and normalize_for_match(term) not in normalize_for_match(evidence_text)
    ][:5]
    if missing_terms and evidence_text:
        issues.append(
            {
                "type": "weak_entity_coverage",
                "message": "Some question terms were not found in evidence.",
                "terms": missing_terms,
            }
        )

    if is_abstention_answer(answer) and evidence_has_answer_candidate(question, evidence_text, query_type):
        issues.append(
            {
                "type": "unnecessary_abstention",
                "message": "The answer says evidence is insufficient, but structured evidence contains an answer candidate.",
            }
        )

    severe_issue_types = {
        "empty_answer",
        "missing_evidence",
        "missing_number",
        "missing_date",
        "unsupported_number",
        "unsupported_date",
        "unsupported_identifier",
        "unnecessary_abstention",
    }
    valid = not any(issue["type"] in severe_issue_types for issue in issues)
    if valid and not issues:
        confidence = 0.92
    elif valid:
        confidence = max(0.55, round(0.72 + min(question_coverage, 1.0) * 0.12, 2))
    else:
        confidence = 0.35
    return {
        "valid": valid,
        "confidence": confidence,
        "query_type": query_type,
        "issues": issues,
        "evidence_coverage": round(question_coverage, 3),
        "answer_numbers": answer_numbers,
        "evidence_numbers": evidence_numbers[:20],
        "answer_dates": answer_dates,
        "evidence_dates": evidence_dates[:20],
        "answer_identifiers": answer_identifiers,
        "evidence_identifiers": evidence_identifiers[:20],
    }


def is_abstention_answer(answer: str) -> bool:
    return bool(ABSTENTION_RE.search(answer or ""))


def evidence_has_answer_candidate(question: str, evidence_text: str, query_type: str) -> bool:
    if not evidence_text.strip():
        return False

    for match in ANSWER_CANDIDATE_RE.finditer(evidence_text):
        candidate = clean_cell(match.group(1))
        if candidate and candidate not in {"-", "N/A", "None", "null"}:
            return True

    if "[Structured Lookup" not in evidence_text:
        return False

    terms = extract_keywords(question, limit=8)
    coverage = term_coverage(evidence_text, terms)
    return query_type in STRUCTURED_LOOKUP_TYPES and coverage >= 0.1


def evidence_to_text(evidence: list[Any]) -> str:
    parts = []
    for item in evidence:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            parts.append(json.dumps(item, ensure_ascii=False))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def extract_number_values(text: str) -> list[str]:
    values = []
    for match in NUMBER_RE.finditer(text):
        raw = match.group(0).strip()
        number = re.search(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?", raw)
        if number:
            values.append(number.group(0).replace(",", ""))
    return unique_keep_order(values)


def extract_date_values(text: str) -> list[str]:
    return unique_keep_order(clean_cell(match.group(0)) for match in DATE_RE.finditer(text))


def extract_identifier_values(text: str) -> list[str]:
    return unique_keep_order(clean_cell(match.group(0)) for match in IDENTIFIER_RE.finditer(text))


def extract_keywords(text: str, limit: int = 20) -> list[str]:
    text = clean_cell(text)
    candidates: list[str] = []

    for term in IMPORTANT_TERMS:
        if term.lower() in text.lower():
            candidates.append(term)

    suffix_pattern = re.compile(
        r"[가-힣A-Za-z0-9._/-]{2,50}(?:"
        r"학과|학부|전공|대학|계열|기관|부서|센터|팀|조직|회사|고객|사용자|"
        r"제품|상품|서비스|시스템|플랫폼|애플리케이션|앱|모듈|컴포넌트|엔드포인트|"
        r"모델|데이터베이스|DB|API|테이블|컬럼|필드|스키마|인덱스|큐|토픽|"
        r"워크플로우|파이프라인|프로세스|잡|배치|정책|규정|계약|조항|약관|"
        r"가이드|매뉴얼|기준|절차|요건|조건|자격|권한|역할|예외|일정|기간|기한|"
        r"수량|개수|건수|인원|정원|한도|임계값|금액|비용|가격|예산|매출|점수|등급|비율|"
        r"오류|에러|장애|원인|해결|대응|복구|담당자|연락처|주소|이메일|전화|"
        r"지표|목표|성과|상태|버전|릴리스|의존성|리스크|항목|내용|정의|개념)"
    )
    candidates.extend(match.group(0) for match in suffix_pattern.finditer(text))

    english_pattern = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_/\.][A-Za-z0-9]+)*(?:\s+[A-Za-z][A-Za-z0-9]*(?:[-_/\.][A-Za-z0-9]+)*){0,3}\b")
    for match in english_pattern.finditer(text):
        value = match.group(0).strip()
        if len(value) >= 3 and (re.search(r"[A-Z]", value) or "-" in value or "_" in value or "/" in value):
            candidates.append(value)

    token_pattern = re.compile(r"[가-힣A-Za-z0-9._/-]{2,40}")
    token_counts = Counter(match.group(0) for match in token_pattern.finditer(text))
    for token, _ in token_counts.most_common(40):
        if token.lower() in STOPWORDS:
            continue
        if len(token) < 2:
            continue
        candidates.append(token)

    return unique_keep_order(candidates)[:limit]


def normalize_terms(terms: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for term in terms:
        term = clean_cell(term)
        if not term:
            continue
        expanded.append(term)
        expanded.extend(extract_keywords(term, limit=8))
    return unique_keep_order(expanded)


def normalize_for_match(value: str) -> str:
    value = clean_cell(value).lower()
    value = re.sub(r"[^0-9a-z가-힣._/-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def unique_keep_order(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        item = strip_trailing_particle(clean_cell(value))
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def strip_trailing_particle(value: str) -> str:
    if not re.search(r"[A-Za-z0-9]", value):
        return value
    stripped = re.sub(r"(?:은|는|이|가|을|를|와|과|의|에|에서|으로|로)$", "", value)
    return stripped or value


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value or "doc").strip("_")
    return safe or "doc"
