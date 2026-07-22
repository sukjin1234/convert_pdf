from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import APP_RUNTIME_DIR_NAME


logger = logging.getLogger(__name__)


PAGE_RE = re.compile(r"(?m)^\s*---\s*Page\s+(\d+)\s*---\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

DATE_RE = re.compile(
    r"(?:(?:19|20)\d{2}\s*[.\-/년]\s*)?"
    r"(?:1[0-2]|0?[1-9])\s*(?:[.\-/월]\s*)"
    r"(?:3[01]|[12]\d|0?[1-9])\s*(?:일)?(?!\s*[%A-Za-z])"
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
MARKER_SYMBOL_RE = re.compile(r"[♣★☆◆◇■□▲△●○◎◈◉◯✓✔✕×†‡※]")
MARKER_LEGEND_PATTERNS = [
    re.compile(
        r"(?:^|\s)(?P<symbol>[♣★☆◆◇■□▲△●○◎◈◉◯✓✔✕×†‡※])\s*표시\s*[:：=\-]?\s*(?P<meaning>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\s)(?P<symbol>[♣★☆◆◇■□▲△●○◎◈◉◯✓✔✕×†‡※])\s*(?:indicates|means|denotes)\s*(?P<meaning>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\s)(?P<symbol>[♣★☆◆◇■□▲△●○◎◈◉◯✓✔✕×†‡※])\s*[:：=]\s*(?P<meaning>.+)$",
        re.IGNORECASE,
    ),
]

POSITIVE_CELL_RE = re.compile(r"^(?:yes|true|supported|available|included|enabled|y|o|가능|지원|제공|포함|허용|필수|사용\s*가능)$", re.IGNORECASE)
NEGATIVE_CELL_RE = re.compile(r"^(?:no|false|none|unsupported|unavailable|disabled|n|x|불가|미지원|없음|제외|금지)$", re.IGNORECASE)
GENERIC_LIST_TERMS = {
    "어떤",
    "어느",
    "무슨",
    "목록",
    "리스트",
    "명단",
    "항목",
    "종류",
    "카테고리",
    "알려줘",
    "필요해",
    "지원돼",
    "지원",
    "제공",
    "가능",
    "what",
    "which",
    "list",
    "type",
    "kind",
    "category",
    "available",
    "supported",
}
MARKER_METADATA_FIELDS = {"표시", "표시 의미", "표시 대상 필드"}

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

ATTRIBUTE_CONCEPT_TERMS = {
    "price": [
        "가격",
        "비용",
        "금액",
        "전형료",
        "등록금",
        "수업료",
        "price",
        "cost",
        "fee",
        "amount",
        "monthly price",
        "$",
        "usd",
        "krw",
    ],
    "capacity": ["수용인원", "수용 인원", "정원", "모집정원", "모집 정원", "capacity", "quota", "limit"],
    "schedule": [
        "일정",
        "기간",
        "날짜",
        "일자",
        "일시",
        "마감",
        "기한",
        "접수",
        "등록",
        "발표",
        "예약",
        "schedule",
        "date",
        "period",
        "deadline",
        "time",
    ],
    "location": ["장소", "위치", "location", "place", "venue"],
    "contact": ["연락처", "전화", "이메일", "문의", "contact", "phone", "email"],
    "document": ["서류", "문서", "document", "file"],
    "eligibility": ["자격", "요건", "조건", "requirement", "eligibility", "condition"],
    "score": ["점수", "총점", "가산점", "등급", "score", "grade", "point"],
    "status": ["상태", "지원", "가능", "제공", "포함", "status", "support", "available", "included"],
    "version": ["버전", "폐기", "대체", "version", "deprecation", "replacement"],
}

STRICT_ATTRIBUTE_CONCEPTS = {"price", "schedule", "location", "contact", "document", "score"}

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
        self._store_dir_candidates = _rag_store_dir_candidates(store_dir)
        self.store_dir = self._store_dir_candidates[0]
        self._writable_store_dir: Path | None = None
        self._failed_store_dirs: set[str] = set()
        self._persistence_warning_logged = False
        self._documents: dict[str, RagArtifact] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def upsert(self, artifact: RagArtifact) -> None:
        with self._lock:
            self._ensure_loaded_locked()
            self._documents[artifact.document_id] = artifact
            payload = json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2)
            last_error = ""
            for _ in range(len(self._store_dir_candidates)):
                store_dir = self._resolve_writable_store_dir_locked()
                if store_dir is None:
                    break

                path = store_dir / f"{_safe_id(artifact.document_id)}.json"
                try:
                    path.write_text(payload, encoding="utf-8")
                    return
                except OSError as exc:
                    last_error = f"path={path} error={exc}"
                    self._failed_store_dirs.add(str(store_dir))
                    self._writable_store_dir = None

            detail = last_error or "No writable RAG store directory is available."
            self._warn_persistence_disabled(f"RAG artifact is kept in memory because store write failed. {detail}")

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
        for root in self._store_dir_candidates:
            if not root.exists():
                continue
            try:
                paths = sorted(root.glob("*.json"))
            except OSError as exc:
                logger.warning("Could not read RAG store directory. root=%s error=%s", root, exc)
                continue
            for path in paths:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    artifact = RagArtifact.from_dict(payload)
                except Exception:
                    continue
                if artifact.document_id:
                    self._documents[artifact.document_id] = artifact

    def _resolve_writable_store_dir_locked(self) -> Path | None:
        if self._writable_store_dir is not None:
            return self._writable_store_dir

        for root in self._store_dir_candidates:
            if str(root) in self._failed_store_dirs:
                continue
            try:
                root.mkdir(parents=True, exist_ok=True)
                _assert_writable_dir(root)
            except OSError as exc:
                logger.warning("RAG store directory is not writable. root=%s error=%s", root, exc)
                self._failed_store_dirs.add(str(root))
                continue
            self.store_dir = root
            self._writable_store_dir = root
            return root

        self._warn_persistence_disabled("No writable RAG store directory is available; artifacts are kept in memory only.")
        return None

    def _warn_persistence_disabled(self, message: str) -> None:
        if self._persistence_warning_logged:
            return
        self._persistence_warning_logged = True
        logger.warning(message)


def _rag_store_dir_candidates(store_dir: Path | None = None) -> list[Path]:
    if store_dir is not None:
        return [store_dir]

    candidates: list[Path] = []
    explicit = os.getenv("RAG_STORE_DIR")
    if explicit and explicit.strip():
        candidates.append(Path(explicit))

    runtime_root = Path(os.getenv("DIFY_RUNTIME_DIR", str(Path.cwd() / ".runtime")))
    candidates.append(runtime_root / "rag-store")

    try:
        candidates.append(Path.home() / ".cache" / APP_RUNTIME_DIR_NAME / "rag-store")
    except RuntimeError:
        pass

    candidates.append(Path(tempfile.gettempdir()) / "dify-rag-store")
    return _unique_paths(candidates)


def _assert_writable_dir(path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path, prefix=".write-test-", encoding="utf-8", delete=True) as probe:
        probe.write("")


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


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
    document_marker_legends = extract_unambiguous_marker_legends(normalized.split("\n"))

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
            document_marker_legends=document_marker_legends,
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
        "marker_legend_count": len(document_marker_legends),
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
    document_marker_legends: dict[str, str] | None = None,
) -> tuple[str, list[StructuredRecord]]:
    lines = text.split("\n")
    local_marker_legends = extract_marker_legends(lines)
    marker_legends = {**(document_marker_legends or {}), **local_marker_legends}
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
            raw_fields = {
                header: clean_cell(row[column_index]) if column_index < len(row) else ""
                for column_index, header in enumerate(headers)
            }
            raw_fields = {key: value for key, value in raw_fields.items() if key and value}
            fields = enrich_fields_with_marker_legends(raw_fields, marker_legends)
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

    plain_marker_records = extract_plain_marker_records(
        lines,
        marker_legends,
        document_id=document_id,
        file_name=file_name,
        page=page,
        section_path=section_path,
        table_index=table_index + 1,
    )
    if plain_marker_records:
        records.extend(plain_marker_records)
        output.append("")
        output.append("표시/범례 기반 구조화 레코드:")
        output.extend(build_record_text(record) for record in plain_marker_records)

    return normalize_markdown("\n".join(output)), records


def extract_plain_marker_records(
    lines: list[str],
    marker_legends: dict[str, str],
    *,
    document_id: str,
    file_name: str,
    page: int | None,
    section_path: str,
    table_index: int,
) -> list[StructuredRecord]:
    if not marker_legends:
        return []

    records = []
    for line_index, line in enumerate(lines, start=1):
        raw_line = clean_cell(line)
        if not raw_line or looks_like_table_row(raw_line) or is_marker_legend_line(raw_line):
            continue
        for symbol in unique_keep_order(MARKER_SYMBOL_RE.findall(raw_line)):
            meaning = marker_legends.get(symbol)
            if not meaning:
                continue
            item = extract_marked_line_item(raw_line, symbol)
            if not item:
                continue
            fields = {
                "항목": item,
                "표시": symbol,
                "표시 의미": meaning,
                "표시 대상 필드": "항목",
                "원문": remove_marker_symbol(raw_line, symbol),
            }
            source_text = " | ".join(fields.values())
            answer_text = row_to_sentence(fields)
            record_type = classify_record(fields, source_text)
            keywords = extract_keywords(" ".join([section_path, source_text, answer_text]), limit=12)
            records.append(
                StructuredRecord(
                    record_id=f"{_safe_id(document_id)}_p{page or 0:03d}_m{table_index:02d}_r{line_index:03d}",
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
            )
    return records


def is_marker_legend_line(line: str) -> bool:
    return parse_marker_legend_line(line) is not None


def parse_marker_legend_line(line: str) -> tuple[str, str] | None:
    cleaned = clean_legend_line(line)
    if not cleaned:
        return None
    for pattern in MARKER_LEGEND_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        symbol = match.group("symbol")
        meaning = clean_marker_meaning(match.group("meaning"))
        if symbol and meaning:
            return symbol, meaning
    return None


def extract_marked_line_item(line: str, symbol: str) -> str:
    before_symbol = line.split(symbol, 1)[0]
    before_symbol = clean_cell(before_symbol)
    before_symbol = strip_plain_table_prefix(before_symbol)
    before_symbol = re.sub(r"^\d+\s+", "", before_symbol)
    return clean_cell(before_symbol)


def strip_plain_table_prefix(value: str) -> str:
    # OCR/plain-text tables sometimes keep a section/category before the marked item.
    # Keep the item part when the prefix looks like a short category label.
    parts = value.split()
    if len(parts) >= 2 and len(parts[0]) <= 6 and not re.search(r"[A-Za-z0-9가-힣]+과$", parts[0]):
        return " ".join(parts[1:])
    return value


def extract_marker_legends(lines: list[str]) -> dict[str, str]:
    legends: dict[str, str] = {}
    for line in lines:
        parsed = parse_marker_legend_line(line)
        if not parsed:
            continue
        symbol, meaning = parsed
        legends[symbol] = meaning
    return legends


def extract_unambiguous_marker_legends(lines: list[str]) -> dict[str, str]:
    meanings_by_symbol: dict[str, list[str]] = {}
    for line in lines:
        parsed = parse_marker_legend_line(line)
        if not parsed:
            continue
        symbol, meaning = parsed
        meanings = meanings_by_symbol.setdefault(symbol, [])
        if normalize_for_match(meaning) not in {normalize_for_match(item) for item in meanings}:
            meanings.append(meaning)

    return {
        symbol: meanings[0]
        for symbol, meanings in meanings_by_symbol.items()
        if len(meanings) == 1
    }


def clean_legend_line(line: str) -> str:
    cleaned = clean_cell(line)
    cleaned = re.sub(r"^[\-*•\s]+", "", cleaned)
    cleaned = re.sub(r"^※\s*", "", cleaned)
    return cleaned.strip()


def clean_marker_meaning(value: str) -> str:
    value = clean_cell(value)
    value = re.sub(r"※.*$", "", value).strip()
    return value.rstrip(".。;；")


def enrich_fields_with_marker_legends(fields: dict[str, str], marker_legends: dict[str, str]) -> dict[str, str]:
    if not fields:
        return {}
    enriched: dict[str, str] = {}
    marker_hits: list[tuple[str, str, str]] = []

    for key, value in fields.items():
        symbols = [
            symbol
            for symbol in unique_keep_order(MARKER_SYMBOL_RE.findall(value))
            if symbol in marker_legends
        ]
        cleaned_value = value
        for symbol in symbols:
            cleaned_value = remove_marker_symbol(cleaned_value, symbol)
            marker_hits.append((symbol, marker_legends[symbol], key))
        enriched[key] = clean_cell(cleaned_value)

    if marker_hits:
        symbols = unique_exact_keep_order(symbol for symbol, _, _ in marker_hits)
        meanings = unique_exact_keep_order(meaning for _, meaning, _ in marker_hits)
        marked_fields = unique_exact_keep_order(field for _, _, field in marker_hits)
        enriched["표시"] = ", ".join(symbols)
        enriched["표시 의미"] = "; ".join(meanings)
        enriched["표시 대상 필드"] = ", ".join(marked_fields)

    return {key: value for key, value in enriched.items() if key and value}


def remove_marker_symbol(value: str, symbol: str) -> str:
    escaped = re.escape(symbol)
    value = re.sub(rf"\s*{escaped}\s*", " ", value)
    return clean_cell(value)


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
    if rows and looks_like_secondary_header_row(rows[0]):
        headers = merge_multilevel_headers(headers, rows[0])
        rows = rows[1:]
    rows = [row for row in rows if any(clean_cell(cell) for cell in row)]
    if not rows:
        return None
    return headers, rows


def looks_like_secondary_header_row(row: list[str]) -> bool:
    cells = [clean_cell(cell) for cell in row]
    non_empty = [cell for cell in cells if cell]
    if len(non_empty) < 2:
        return False
    label_cells = [
        cell
        for cell in non_empty
        if not NUMBER_RE.fullmatch(cell) and not DATE_RE.search(cell) and len(cell) <= 30
    ]
    leading_blanks = 0
    for cell in cells:
        if cell:
            break
        leading_blanks += 1
    return leading_blanks >= 1 and len(label_cells) / len(non_empty) >= 0.75


def merge_multilevel_headers(headers: list[str], subheaders: list[str]) -> list[str]:
    merged = []
    current_parent = ""
    for index, header in enumerate(headers):
        header = clean_cell(header)
        subheader = clean_cell(subheaders[index]) if index < len(subheaders) else ""
        if header and not re.fullmatch(r"Column\s+\d+", header, re.IGNORECASE):
            current_parent = header
        parent = current_parent or header
        if subheader:
            if parent and normalize_for_match(subheader) not in normalize_for_match(parent):
                merged.append(clean_cell(f"{parent} {subheader}"))
            else:
                merged.append(parent or subheader)
        else:
            merged.append(parent or header or f"Column {index + 1}")
    return [clean_cell(header) or f"Column {index}" for index, header in enumerate(merged, start=1)]


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
    if re.search(r"비교|차이|대비|compare|difference|versus| vs\.? ", lower):
        return "comparison"
    if re.search(r"오류|에러|장애|해결|대응|복구|원인|troubleshoot|incident|resolution|error", lower):
        return "troubleshooting"
    if re.search(r"담당|연락처|전화|이메일|문의|owner|contact|phone|email", lower):
        return "contact_lookup"
    if re.search(r"의존|연동|dependency|depends|impact", lower):
        return "dependency_lookup"
    if re.search(r"언제|날짜|일정|기간|마감|접수|발표|시작|종료|기한|일시|예약|보존|폐기일|개정일|주기|date|deadline|schedule|period|retention|rotation|deprecation", lower):
        return "date_lookup"
    if re.search(r"몇|얼마|수량|개수|건수|횟수|인원|정원|모집인원|선발|한도|제한|임계값|기준값|점수|금액|비용|가격|예산|매출|가용성|응답시간|지연시간|비율|율|number|amount|capacity|limit|score|rate|ratio|budget|price|cost", lower):
        return "number_lookup"
    if re.search(r"가능|지원|제공|포함|필수|요건|조건|자격|예외|승인|할 수|모집해|휴학|available|supported|included|required|requirement|condition|eligible", lower):
        return "condition_lookup"
    if re.search(r"목록|리스트|명단|어떤|어느|어디|학과|부서|종류|카테고리|표|항목|list|which|where|table|category", lower):
        return "table_lookup"

    scores = {
        query_type: sum(1 for term in terms if query_type_term_matches(term, lower))
        for query_type, terms in QUERY_TYPE_TERMS.items()
    }
    if NUMBER_RE.search(query) and scores["date_lookup"] == 0:
        scores["number_lookup"] += 1
    if DATE_RE.search(query):
        scores["date_lookup"] += 2
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    return best_type if best_score > 0 else "semantic"


def query_type_term_matches(term: str, lower_query: str) -> bool:
    term_lower = term.lower()
    if len(term_lower) <= 1:
        return bool(re.search(rf"(?<![0-9a-z가-힣]){re.escape(term_lower)}(?![0-9a-z가-힣])", lower_query))
    return term_lower in lower_query


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
    if re.search(r"표|목록|리스트|명단|항목|종류|카테고리|개설|list|table|catalog|category|available|offered", lower):
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
    direct_candidates = select_evidence_matches(ranked, max(limit * 6, 40)) if answerability["answerable"] else []
    direct_answer = build_direct_answer(query, active_query_type, active_entities, direct_candidates)
    context_source_matches = merge_match_lists(direct_candidates, combined)
    context_matches = filter_context_matches_for_direct_answer(context_source_matches, direct_answer)
    return {
        "query": query,
        "query_type": active_query_type,
        "entities": active_entities,
        "sub_queries": plan.get("sub_queries", []),
        "answer_style": plan.get("answer_style", "grounded_explanation"),
        "direct_answer": direct_answer["direct_answer"],
        "answer_items": direct_answer["answer_items"],
        "answer_field": direct_answer["answer_field"],
        "filter_terms": direct_answer["filter_terms"],
        "matches": combined,
        "context": format_lookup_context(context_matches, direct_answer),
        "evidence_items": build_evidence_items(context_matches),
        "diagnostics": {
            "candidate_count": len(ranked),
            "selected_count": len(combined),
            "top_score": ranked[0]["score"] if ranked else 0,
            "average_coverage": round(sum(match["coverage"] for match in combined) / len(combined), 3) if combined else 0,
            "answerability": answerability,
        },
}


def merge_match_lists(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for group in groups:
        for match in group:
            key = (match.get("record_id") or "", match.get("chunk_id") or "", match.get("match_type") or "")
            if key in seen:
                continue
            seen.add(key)
            merged.append(match)
    return merged


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
    haystack_compact = haystack.replace(" ", "")
    matches = []
    for term in terms:
        term_norm = normalize_for_match(term)
        if not term_norm:
            continue
        term_compact = term_norm.replace(" ", "")
        if term_norm in haystack or (term_compact and term_compact in haystack_compact):
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


def filter_context_matches_for_direct_answer(matches: list[dict[str, Any]], direct_answer: dict[str, Any]) -> list[dict[str, Any]]:
    items = direct_answer.get("answer_items") or []
    record_ids = {
        str(item.get("record_id"))
        for item in items
        if isinstance(item, dict) and item.get("record_id")
    }
    if not record_ids:
        return matches
    filtered = [match for match in matches if str(match.get("record_id") or "") in record_ids]
    return filtered or matches


def build_direct_answer(query: str, query_type: str, entities: list[str], matches: list[dict[str, Any]]) -> dict[str, Any]:
    empty = {
        "direct_answer": "",
        "answer_items": [],
        "answer_field": "",
        "filter_terms": [],
        "mode": "",
    }
    answer_style = classify_answer_style(query, query_type)
    should_build_keyed_list = query_type == "table_lookup" or answer_style == "table_or_bullets"
    should_build_boolean_matrix = query_type in {"condition_lookup", "table_lookup"} or answer_style == "table_or_bullets"
    should_build_field_value = query_type in {"number_lookup", "date_lookup", "contact_lookup"} or answer_style == "direct"
    should_build_inverse_list = should_build_keyed_list or looks_like_inverse_value_question(query)
    should_build_marker_meaning = looks_like_marker_meaning_question(query)
    if not should_build_keyed_list and not should_build_boolean_matrix and not should_build_field_value and not should_build_inverse_list and not should_build_marker_meaning:
        return empty

    structured = [
        match
        for match in matches
        if match.get("match_type") == "structured_record" and isinstance(match.get("fields"), dict) and match.get("fields")
    ]
    if not structured:
        return empty
    if stronger_unstructured_evidence_precedes_structured(matches, structured):
        return empty

    if should_build_marker_meaning:
        marker_meaning = build_marker_meaning_answer(query, structured)
        if marker_meaning["direct_answer"]:
            return marker_meaning

    top_group = top_structured_group(structured)
    query_keywords = meaningful_answer_terms([*extract_keywords(query, limit=16), *entities])
    if not query_keywords:
        return empty

    if should_build_inverse_list:
        inverse = build_inverse_value_list_answer(query, query_type, query_keywords, top_group)
        if inverse["direct_answer"]:
            return inverse

    if looks_like_membership_question(query) and (should_build_keyed_list or should_build_boolean_matrix):
        membership = build_membership_answer(query, query_type, query_keywords, top_group)
        if membership["direct_answer"]:
            return membership

    if should_build_field_value:
        field_value = build_field_value_answer(query, query_type, query_keywords, top_group)
        if field_value["direct_answer"]:
            return field_value

    if should_build_keyed_list:
        keyed = build_keyed_list_answer(query, query_type, query_keywords, top_group)
        if keyed["direct_answer"]:
            return keyed

    if should_build_boolean_matrix:
        pivoted = build_boolean_matrix_answer(query, query_keywords, top_group)
        if pivoted["direct_answer"]:
            return pivoted

    return empty


def stronger_unstructured_evidence_precedes_structured(matches: list[dict[str, Any]], structured: list[dict[str, Any]]) -> bool:
    if not matches or not structured:
        return False
    top_match = matches[0]
    if top_match.get("match_type") == "structured_record":
        return False
    top_structured = structured[0]
    top_score = float(top_match.get("score") or 0)
    structured_score = float(top_structured.get("score") or 0)
    top_coverage = float(top_match.get("coverage") or 0)
    structured_coverage = float(top_structured.get("coverage") or 0)
    return top_score >= structured_score * 1.25 and top_coverage > structured_coverage


def looks_like_inverse_value_question(query: str) -> bool:
    if not (extract_number_values(query) or extract_date_values(query)):
        return False
    return bool(re.search(r"\d[\d,.\s]*(?:[A-Za-z가-힣/%$]+(?:\s+[A-Za-z가-힣/%$]+){0,2})?\s*인\s+", query))


def looks_like_marker_meaning_question(query: str) -> bool:
    return bool(MARKER_SYMBOL_RE.search(query)) and bool(
        re.search(r"표시|뜻|의미|범례|legend|indicates|means|denotes", query, re.IGNORECASE)
    )


def build_marker_meaning_answer(query: str, matches: list[dict[str, Any]]) -> dict[str, Any]:
    markers = unique_keep_order(MARKER_SYMBOL_RE.findall(query))
    values = []
    answer_items = []
    for match in sorted(matches, key=match_document_order_key):
        fields = match.get("fields") or {}
        meaning = clean_cell(fields.get("표시 의미", ""))
        if not meaning:
            continue
        field_symbols = unique_keep_order(MARKER_SYMBOL_RE.findall(fields.get("표시", "")))
        if markers and not any(marker in field_symbols for marker in markers):
            continue
        symbol = next((marker for marker in markers if marker in field_symbols), field_symbols[0] if field_symbols else "")
        value = f"{symbol}: {meaning}" if symbol else meaning
        if normalize_for_match(value) in {normalize_for_match(item) for item in values}:
            continue
        values.append(value)
        answer_items.append(
            {
                "value": value,
                "field": "표시 의미",
                "record_id": match.get("record_id", ""),
                "page": match.get("page"),
                "section_path": match.get("section_path", ""),
                "fields": fields,
            }
        )

    if not values:
        return {
            "direct_answer": "",
            "answer_items": [],
            "answer_field": "",
            "filter_terms": [],
            "mode": "",
        }

    return {
        "direct_answer": format_direct_answer_text("표시 의미", values),
        "answer_items": answer_items,
        "answer_field": "표시 의미",
        "filter_terms": markers,
        "mode": "marker_meaning",
    }


def top_structured_group(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not matches:
        return []
    first = matches[0]
    first_key = (first.get("document_id"), first.get("chunk_id"), first.get("section_path"))
    grouped = [
        match
        for match in matches
        if (match.get("document_id"), match.get("chunk_id"), match.get("section_path")) == first_key
    ]
    return grouped or matches


def meaningful_answer_terms(terms: list[str]) -> list[str]:
    result = []
    for term in unique_keep_order(terms):
        cleaned = clean_cell(term)
        norm = normalize_for_match(cleaned)
        if len(norm) < 2:
            continue
        if norm in {normalize_for_match(item) for item in GENERIC_LIST_TERMS}:
            continue
        result.append(cleaned)
    return result


def looks_like_membership_question(query: str) -> bool:
    return bool(
        re.search(
            r"(?:은|는|이|가).*(?:맞아|맞나요|해당|포함|지원|제공|가능|개설|운영|야\?|인가|인가요|있어\?)"
            r"|(?:is|are).*(?:available|supported|included|enabled)",
            query,
            re.IGNORECASE,
        )
    )


def build_membership_answer(
    query: str,
    query_type: str,
    query_terms: list[str],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    keyed = build_keyed_list_answer(query, query_type, query_terms, matches)
    if not keyed["answer_items"]:
        pivoted = build_boolean_matrix_answer(query, query_terms, matches)
        if pivoted["answer_items"]:
            return pivoted
        return {
            "direct_answer": "",
            "answer_items": [],
            "answer_field": "",
            "filter_terms": [],
            "mode": "",
        }

    target_key = keyed["answer_field"]
    listed_values = [
        clean_cell(item.get("value", ""))
        for item in keyed["answer_items"]
        if isinstance(item, dict) and clean_cell(item.get("value", ""))
    ]
    requested_values = requested_values_for_field(query, query_terms, target_key, matches)
    if not requested_values:
        return {
            "direct_answer": "",
            "answer_items": [],
            "answer_field": "",
            "filter_terms": [],
            "mode": "",
        }

    listed_norms = {normalize_for_match(value) for value in listed_values}
    values = []
    answer_items = []
    for requested in requested_values:
        requested_norm = normalize_for_match(requested)
        supported = requested_norm in listed_norms
        value = f"{requested}: {'예' if supported else '아니오'}"
        values.append(value)
        source_item = next(
            (
                item
                for item in keyed["answer_items"]
                if isinstance(item, dict) and normalize_for_match(str(item.get("value") or "")) == requested_norm
            ),
            keyed["answer_items"][0],
        )
        answer_items.append(
            {
                "value": value,
                "field": target_key,
                "record_id": source_item.get("record_id", "") if isinstance(source_item, dict) else "",
                "page": source_item.get("page") if isinstance(source_item, dict) else None,
                "section_path": source_item.get("section_path", "") if isinstance(source_item, dict) else "",
                "fields": source_item.get("fields", {}) if isinstance(source_item, dict) else {},
                "membership_value": requested,
                "supported": supported,
            }
        )

    return {
        "direct_answer": format_direct_answer_text(target_key, values),
        "answer_items": answer_items,
        "answer_field": target_key,
        "filter_terms": keyed.get("filter_terms", []),
        "mode": "membership",
    }


def requested_values_for_field(
    query: str,
    query_terms: list[str],
    target_key: str,
    matches: list[dict[str, Any]],
) -> list[str]:
    values_by_norm: dict[str, str] = {}
    for match in matches:
        fields = match.get("fields") or {}
        for value in split_compound_answer_value(clean_cell(fields.get(target_key, ""))):
            norm = normalize_for_match(value)
            if norm:
                values_by_norm[norm] = value

    requested = []
    query_norm = normalize_for_match(query)
    for norm, value in values_by_norm.items():
        if norm and norm in query_norm:
            requested.append(value)

    if requested:
        return unique_exact_keep_order(requested)

    for term in query_terms:
        term_norm = normalize_for_match(term)
        if term_norm in values_by_norm:
            requested.append(values_by_norm[term_norm])
    return unique_exact_keep_order(requested)


def build_keyed_list_answer(
    query: str,
    query_type: str,
    query_terms: list[str],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    target_key = choose_target_field_key(query, query_terms, matches)
    if not target_key:
        return {
            "direct_answer": "",
            "answer_items": [],
            "answer_field": "",
            "filter_terms": [],
            "mode": "",
        }

    filter_terms = [term for term in query_terms if not term_matches_key(term, target_key)]
    marker_filtered_matches = filter_matches_by_marker_meaning(matches, filter_terms, target_key)
    filtered_matches = marker_filtered_matches or filter_matches_by_row_terms(matches, filter_terms, exclude_keys={target_key})
    if not filtered_matches:
        filtered_matches = matches

    values = []
    answer_items = []
    for match in sorted(filtered_matches, key=match_document_order_key):
        fields = match.get("fields") or {}
        value = clean_cell(fields.get(target_key, ""))
        if not value:
            continue
        for split_value in split_compound_answer_value(value):
            if normalize_for_match(split_value) in {normalize_for_match(item) for item in values}:
                continue
            values.append(split_value)
            answer_items.append(
                {
                    "value": split_value,
                    "field": target_key,
                    "record_id": match.get("record_id", ""),
                    "page": match.get("page"),
                    "section_path": match.get("section_path", ""),
                    "fields": fields,
                    "raw_value": value,
                }
            )

    if not values:
        return {
            "direct_answer": "",
            "answer_items": [],
            "answer_field": "",
            "filter_terms": [],
            "mode": "",
        }

    return {
        "direct_answer": format_direct_answer_text(target_key, values),
        "answer_items": answer_items,
        "answer_field": target_key,
        "filter_terms": filter_terms,
        "mode": "target_field",
    }


def split_compound_answer_value(value: str) -> list[str]:
    cleaned = clean_cell(value)
    if not cleaned:
        return []
    korean_entities = re.findall(
        r"[가-힣A-Za-z0-9._/-]+(?:학과|학부|전공|대학|기관|부서|센터|팀|조직|시스템|서비스|API|DB|Plan|Feature)",
        cleaned,
    )
    if len(korean_entities) >= 2:
        compact_entities = normalize_for_match(" ".join(korean_entities)).replace(" ", "")
        compact_value = normalize_for_match(cleaned).replace(" ", "")
        if compact_entities and compact_entities == compact_value:
            return unique_exact_keep_order(korean_entities)
    return [cleaned]


def choose_target_field_key(query: str, query_terms: list[str], matches: list[dict[str, Any]]) -> str:
    query_norm = normalize_for_match(query)
    scores: Counter[str] = Counter()
    key_by_norm: dict[str, str] = {}
    for match in matches:
        fields = match.get("fields") or {}
        marker_meaning = fields.get("표시 의미", "")
        marker_target_fields = split_field_list(fields.get("표시 대상 필드", ""))
        if marker_meaning and marker_target_fields and any(term_in_text(term, marker_meaning) for term in query_terms):
            for marker_target_field in marker_target_fields:
                if marker_target_field in fields:
                    key_norm = normalize_for_match(marker_target_field)
                    key_by_norm[key_norm] = marker_target_field
                    scores[key_norm] += 12
        for key, value in fields.items():
            key = clean_cell(key)
            if not key:
                continue
            if key in MARKER_METADATA_FIELDS:
                continue
            key_norm = normalize_for_match(key)
            key_by_norm[key_norm] = key
            if key_norm and key_norm in query_norm:
                scores[key_norm] += 8
            for term in query_terms:
                term_norm = normalize_for_match(term)
                if not term_norm:
                    continue
                if term_norm in key_norm or key_norm in term_norm:
                    scores[key_norm] += 4
                    continue
                token_hits = sum(1 for token in term_norm.split() if len(token) >= 2 and token in key_norm)
                scores[key_norm] += token_hits
            value_norm = normalize_for_match(value)
            if value_norm and value_norm in query_norm:
                scores[key_norm] -= 5
    if not scores:
        return ""
    key_norm, score = scores.most_common(1)[0]
    return key_by_norm.get(key_norm, "") if score >= 3 else ""


def match_document_order_key(match: dict[str, Any]) -> tuple[Any, ...]:
    return (
        match.get("page") if match.get("page") is not None else 10**9,
        str(match.get("chunk_id") or ""),
        str(match.get("record_id") or ""),
    )


def split_field_list(value: str) -> list[str]:
    return [clean_cell(part) for part in re.split(r"[,;/]", value or "") if clean_cell(part)]


def build_inverse_value_list_answer(
    query: str,
    query_type: str,
    query_terms: list[str],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    constraints = extract_scalar_constraints(query, query_type)
    if not constraints:
        return {
            "direct_answer": "",
            "answer_items": [],
            "answer_field": "",
            "filter_terms": [],
            "mode": "",
        }

    query_field_terms = [
        term
        for term in query_terms
        if not any(term_in_text(term, value) for value in constraints)
    ]
    values = []
    answer_items = []
    for match in sorted(matches, key=match_document_order_key):
        fields = match.get("fields") or {}
        if not fields:
            continue
        constraint_keys = fields_matching_constraints(fields, constraints, query_field_terms)
        if not constraint_keys:
            continue
        target_key = choose_subject_answer_field(fields, query_terms, exclude_keys=set(constraint_keys))
        value = clean_cell(fields.get(target_key, ""))
        if not value:
            continue
        if normalize_for_match(value) in {normalize_for_match(item) for item in values}:
            continue
        values.append(value)
        answer_items.append(
            {
                "value": value,
                "field": target_key,
                "record_id": match.get("record_id", ""),
                "page": match.get("page"),
                "section_path": match.get("section_path", ""),
                "fields": fields,
                "matched_constraint_fields": constraint_keys,
            }
        )

    if not values:
        return {
            "direct_answer": "",
            "answer_items": [],
            "answer_field": "",
            "filter_terms": [],
            "mode": "",
        }

    answer_field = answer_items[0]["field"] if answer_items else ""
    return {
        "direct_answer": format_direct_answer_text(answer_field, values),
        "answer_items": answer_items,
        "answer_field": answer_field,
        "filter_terms": [*query_field_terms, *constraints],
        "mode": "inverse_value_filter",
    }


def build_field_value_answer(
    query: str,
    query_type: str,
    query_terms: list[str],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    target_key = choose_target_field_key(query, query_terms, matches)
    if not target_key:
        target_key = choose_value_field_from_query(query, query_type, matches)
    if not target_key:
        return {
            "direct_answer": "",
            "answer_items": [],
            "answer_field": "",
            "filter_terms": [],
            "mode": "",
        }

    constraints = extract_scalar_constraints(query, query_type)
    row_filter_terms = [
        term
        for term in query_terms
        if not term_matches_key(term, target_key) and not any(term_in_text(term, value) for value in constraints)
    ]
    row_filter_terms = terms_that_match_any_row_value(row_filter_terms, matches, exclude_keys={target_key})
    filtered_matches = filter_matches_by_row_terms(matches, row_filter_terms, exclude_keys={target_key})
    if not filtered_matches and constraints:
        filtered_matches = [
            match
            for match in matches
            if fields_matching_constraints(match.get("fields") or {}, constraints, row_filter_terms)
        ]
    if not filtered_matches:
        filtered_matches = matches[:1]

    values = []
    answer_items = []
    for match in sorted(filtered_matches, key=match_document_order_key):
        fields = match.get("fields") or {}
        value = clean_cell(fields.get(target_key, ""))
        if not value:
            continue
        subject_key = choose_subject_answer_field(fields, query_terms, exclude_keys={target_key})
        subject_value = clean_cell(fields.get(subject_key, ""))
        display_value = f"{subject_value}: {value}" if subject_value and len(filtered_matches) > 1 else value
        if normalize_for_match(display_value) in {normalize_for_match(item) for item in values}:
            continue
        values.append(display_value)
        answer_items.append(
            {
                "value": display_value,
                "field": target_key,
                "record_id": match.get("record_id", ""),
                "page": match.get("page"),
                "section_path": match.get("section_path", ""),
                "fields": fields,
            }
        )

    if not values:
        return {
            "direct_answer": "",
            "answer_items": [],
            "answer_field": "",
            "filter_terms": [],
            "mode": "",
        }

    return {
        "direct_answer": format_direct_answer_text(target_key, values),
        "answer_items": answer_items,
        "answer_field": target_key,
        "filter_terms": row_filter_terms,
        "mode": "field_value",
    }


def extract_scalar_constraints(query: str, query_type: str) -> list[str]:
    constraints = extract_inverse_value_constraints(query)
    if constraints:
        return constraints
    if query_type in {"number_lookup", "table_lookup", "condition_lookup"} or NUMBER_RE.search(query):
        constraints.extend(extract_number_values(query))
    if query_type in {"date_lookup", "table_lookup", "condition_lookup"} or DATE_RE.search(query):
        constraints.extend(extract_date_values(query))
    return unique_keep_order(constraints)


def extract_inverse_value_constraints(query: str) -> list[str]:
    constraints = []
    pattern = re.compile(
        r"(?P<value>[$€₩]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:[A-Za-z가-힣/%$]+(?:\s+[A-Za-z가-힣/%$]+){0,2})?)\s*인\s+"
    )
    for match in pattern.finditer(query or ""):
        value = clean_cell(match.group("value"))
        if value:
            constraints.append(value)
    return unique_keep_order(constraints)


def terms_that_match_any_row_value(terms: list[str], matches: list[dict[str, Any]], *, exclude_keys: set[str]) -> list[str]:
    useful = []
    for term in terms:
        if any(
            term_in_text(term, value)
            for match in matches
            for key, value in (match.get("fields") or {}).items()
            if key not in exclude_keys and key not in MARKER_METADATA_FIELDS
        ):
            useful.append(term)
    return unique_keep_order(useful)


def fields_matching_constraints(fields: dict[str, str], constraints: list[str], query_field_terms: list[str]) -> list[str]:
    matched_keys = []
    for key, value in fields.items():
        if key in MARKER_METADATA_FIELDS:
            continue
        if not all(scalar_value_in_cell(constraint, value) for constraint in constraints):
            continue
        if query_field_terms and not any(term_matches_key(term, key) for term in query_field_terms):
            if not VALUE_FIELD_RE.search(key):
                continue
        matched_keys.append(key)
    return matched_keys


def scalar_value_in_cell(constraint: str, value: Any) -> bool:
    constraint_norm = normalize_for_match(constraint).replace(" ", "")
    value_norm = normalize_for_match(str(value or "")).replace(" ", "")
    if not constraint_norm or not value_norm:
        return False
    constraint_numbers = extract_number_values(constraint)
    if constraint_numbers:
        value_numbers = extract_number_values(str(value or ""))
        if not any(number in value_numbers for number in constraint_numbers):
            return False
        unit_tokens = [token.lower() for token in re.findall(r"[A-Za-z]{2,}", constraint)]
        if unit_tokens:
            value_text = normalize_for_match(str(value or ""))
            return all(token in value_text for token in unit_tokens)
        return True
    constraint_dates = extract_date_values(constraint)
    if constraint_dates:
        value_dates = extract_date_values(str(value or ""))
        return any(date in value_dates or normalize_for_match(date) in value_norm for date in constraint_dates)
    return constraint_norm in value_norm


def choose_subject_answer_field(fields: dict[str, str], query_terms: list[str], *, exclude_keys: set[str]) -> str:
    preferred_patterns = [
        r"모집\s*단위|학과|학부|전공|항목|대상|이름|명칭|구분|"
        r"plan|feature|api|service|system|name|item|category|type|data\s*type|"
        r"code|error\s*code|id|identifier|key|policy|requirement|owner",
    ]
    for pattern in preferred_patterns:
        for key, value in fields.items():
            if key in exclude_keys or key in MARKER_METADATA_FIELDS:
                continue
            if re.search(pattern, key, re.IGNORECASE) and clean_cell(value):
                return key
    for key, value in fields.items():
        if key in exclude_keys or key in MARKER_METADATA_FIELDS:
            continue
        if clean_cell(value) and not looks_like_numeric_or_date(clean_cell(value)) and not VALUE_FIELD_RE.search(key):
            return key
    for key, value in fields.items():
        if key not in exclude_keys and key not in MARKER_METADATA_FIELDS and clean_cell(value):
            return key
    return next((key for key in fields if key not in exclude_keys), "")


def choose_value_field_from_query(query: str, query_type: str, matches: list[dict[str, Any]]) -> str:
    query_terms = meaningful_answer_terms(extract_keywords(query, limit=16))
    scores: Counter[str] = Counter()
    key_by_norm: dict[str, str] = {}
    for match in matches:
        for key, value in (match.get("fields") or {}).items():
            if key in MARKER_METADATA_FIELDS or not clean_cell(value):
                continue
            key_norm = normalize_for_match(key)
            key_by_norm[key_norm] = key
            if query_type == "number_lookup" and NUMBER_RE.search(value):
                scores[key_norm] += 2
            if query_type == "date_lookup" and DATE_RE.search(value):
                scores[key_norm] += 2
            if VALUE_FIELD_RE.search(key):
                scores[key_norm] += 2
            for term in query_terms:
                if term_matches_key(term, key):
                    scores[key_norm] += 6
    if not scores:
        return ""
    key_norm, score = scores.most_common(1)[0]
    return key_by_norm.get(key_norm, "") if score >= 3 else ""


def term_matches_key(term: str, key: str) -> bool:
    term_norm = normalize_for_match(term)
    key_norm = normalize_for_match(key)
    term_compact = term_norm.replace(" ", "")
    key_compact = key_norm.replace(" ", "")
    return bool(
        term_norm
        and key_norm
        and (
            term_norm in key_norm
            or key_norm in term_norm
            or (term_compact and key_compact and (term_compact in key_compact or key_compact in term_compact))
            or all(token in key_norm for token in term_norm.split() if len(token) >= 2)
        )
    )


def filter_matches_by_row_terms(matches: list[dict[str, Any]], filter_terms: list[str], *, exclude_keys: set[str]) -> list[dict[str, Any]]:
    row_terms = [term for term in filter_terms if len(normalize_for_match(term)) >= 2]
    if not row_terms:
        return matches
    filtered = []
    for match in matches:
        fields = match.get("fields") or {}
        row_text = " ".join(str(value) for key, value in fields.items() if key not in exclude_keys and key != "표시 대상 필드")
        if all(term_in_text(term, row_text) for term in row_terms):
            filtered.append(match)
    return filtered


def filter_matches_by_marker_meaning(matches: list[dict[str, Any]], filter_terms: list[str], target_key: str) -> list[dict[str, Any]]:
    marker_terms = [
        term
        for term in filter_terms
        if len(normalize_for_match(term)) >= 2 and not term_matches_key(term, target_key)
    ]
    if not marker_terms:
        return []
    filtered = []
    for match in matches:
        fields = match.get("fields") or {}
        marker_meaning = fields.get("표시 의미", "")
        if marker_meaning and term_coverage(marker_meaning, marker_terms) >= 0.5:
            filtered.append(match)
    return filtered


def build_boolean_matrix_answer(query: str, query_terms: list[str], matches: list[dict[str, Any]]) -> dict[str, Any]:
    for match in matches:
        fields = match.get("fields") or {}
        if not fields:
            continue
        filter_fields = [
            key
            for key, value in fields.items()
            if any(term_in_text(term, value) for term in query_terms)
        ]
        if not filter_fields:
            continue
        values = []
        answer_items = []
        for key, value in fields.items():
            if key in filter_fields:
                continue
            cell = clean_cell(value)
            if is_positive_cell(cell):
                values.append(key)
                answer_items.append(
                    {
                        "value": key,
                        "field": key,
                        "record_id": match.get("record_id", ""),
                        "page": match.get("page"),
                        "section_path": match.get("section_path", ""),
                        "fields": fields,
                        "cell_value": cell,
                    }
                )
        if values:
            return {
                "direct_answer": format_direct_answer_text("supported_fields", values),
                "answer_items": answer_items,
                "answer_field": "supported_fields",
                "filter_terms": query_terms,
                "mode": "boolean_matrix",
            }
    return {
        "direct_answer": "",
        "answer_items": [],
        "answer_field": "",
        "filter_terms": [],
        "mode": "",
    }


def is_positive_cell(value: str) -> bool:
    cell = clean_cell(value)
    if not cell or NEGATIVE_CELL_RE.search(cell):
        return False
    return bool(POSITIVE_CELL_RE.search(cell))


def term_in_text(term: str, text: Any) -> bool:
    term_norm = normalize_for_match(term)
    text_norm = normalize_for_match(str(text or ""))
    if not term_norm or not text_norm:
        return False
    term_compact = term_norm.replace(" ", "")
    text_compact = text_norm.replace(" ", "")
    if term_norm in text_norm or (term_compact and term_compact in text_compact):
        return True
    tokens = [token for token in term_norm.split() if len(token) >= 2]
    return bool(tokens and all(token in text_norm for token in tokens))


def format_direct_answer_text(answer_field: str, values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"{answer_field}: {values[0]}"
    return f"{answer_field}: " + ", ".join(values)


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
    query_attributes = detect_attribute_concepts(" ".join([query, *entities]))
    evidence_attributes = detect_attribute_concepts(evidence_text)
    missing_strict_attributes = sorted((query_attributes & STRICT_ATTRIBUTE_CONCEPTS) - evidence_attributes)
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

    if missing_strict_attributes:
        return {
            "answerable": False,
            "confidence": 0.25,
            "query_concepts": sorted(query_concepts),
            "evidence_concepts": sorted(evidence_concepts),
            "query_attributes": sorted(query_attributes),
            "evidence_attributes": sorted(evidence_attributes),
            "missing_strict_concepts": missing_strict_concepts,
            "missing_strict_attributes": missing_strict_attributes,
            "reason": "query asks for a specific attribute that is absent from evidence",
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
        "query_attributes": sorted(query_attributes),
        "evidence_attributes": sorted(evidence_attributes),
        "missing_strict_concepts": [],
        "missing_strict_attributes": [],
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


def detect_attribute_concepts(text: str) -> set[str]:
    normalized = normalize_for_match(text)
    concepts = set()
    for concept, terms in ATTRIBUTE_CONCEPT_TERMS.items():
        for term in terms:
            term_norm = normalize_for_match(term)
            if term_norm and term_norm in normalized:
                concepts.add(concept)
                break
    if re.search(r"[$₩€]|(?:^|[\s:])\d[\d,]*(?:원|만원|억원|달러|usd|krw)(?:\s|$)", normalized, re.IGNORECASE):
        concepts.add("price")
    if DATE_RE.search(text or ""):
        concepts.add("schedule")
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
    haystack_compact = haystack.replace(" ", "")
    score = 0.0
    matched_terms = 0
    for term in terms:
        term_norm = normalize_for_match(term)
        if not term_norm:
            continue
        term_compact = term_norm.replace(" ", "")
        if term_norm in haystack or (term_compact and term_compact in haystack_compact):
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
    haystack_compact = haystack.replace(" ", "")
    score = 0.0
    exact_term_matches = 0
    for term in terms:
        term_norm = normalize_for_match(term)
        if not term_norm:
            continue
        term_compact = term_norm.replace(" ", "")
        if term_norm in haystack or (term_compact and term_compact in haystack_compact):
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


def format_lookup_context(matches: list[dict[str, Any]], direct_answer: dict[str, Any] | None = None) -> str:
    blocks = []
    used_chars = 0
    direct_block = format_direct_answer_context(direct_answer or {})
    has_direct_answer = bool(direct_block)
    if direct_block:
        blocks.append(direct_block)
        used_chars += len(direct_block) + 2
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
                f"supporting_context:\n{supporting_context}" if supporting_context and not has_direct_answer else "",
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


def format_direct_answer_context(direct_answer: dict[str, Any]) -> str:
    answer_text = clean_cell(direct_answer.get("direct_answer", ""))
    items = direct_answer.get("answer_items") or []
    if not answer_text or not isinstance(items, list):
        return ""
    values = unique_exact_keep_order(
        clean_cell(item.get("value", ""))
        for item in items
        if isinstance(item, dict) and clean_cell(item.get("value", ""))
    )
    if not values:
        return ""
    lines = [
        "[Direct Answer - complete structured result]",
        f"answer_candidate: {answer_text}",
        f"answer_field: {direct_answer.get('answer_field', '')}",
        f"filter_terms: {', '.join(str(term) for term in direct_answer.get('filter_terms', []))}",
        "complete_values:",
    ]
    lines.extend(f"- {value}" for value in values)
    lines.append("Instruction: include every value in complete_values; do not omit or add values.")
    return "\n".join(lines)


def merge_evidence_context(lookup: dict[str, Any] | str | None, knowledge_result: Any = None) -> dict[str, str]:
    lookup_payload = parse_json_like(lookup)
    if not isinstance(lookup_payload, dict):
        lookup_payload = {}

    structured_context = str(lookup_payload.get("context") or "")
    diagnostics = lookup_payload.get("diagnostics") or {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    knowledge_items = parse_json_like(knowledge_result) or []
    if isinstance(knowledge_items, dict):
        knowledge_items = (
            knowledge_items.get("result")
            or knowledge_items.get("records")
            or knowledge_items.get("data")
            or knowledge_items.get("documents")
            or []
        )
    if not isinstance(knowledge_items, list):
        knowledge_items = [knowledge_items]

    knowledge_blocks = []
    for index, item in enumerate(knowledge_items[:12], start=1):
        if isinstance(item, dict):
            segment = item.get("segment") if isinstance(item.get("segment"), dict) else {}
            content = item.get("content") or item.get("text") or segment.get("content") or ""
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            source = metadata.get("source") or metadata.get("file_name") or item.get("title") or ""
            knowledge_blocks.append(f"[knowledge {index}] source={source}\n{content}")
        else:
            knowledge_blocks.append(f"[knowledge {index}]\n{item}")

    structured_block = (
        "[Structured Lookup - authoritative exact evidence]\n"
        "Use this first for table, list, field, value, number, date, identifier, and policy questions.\n"
        + structured_context
        if structured_context
        else ""
    )
    knowledge_block = (
        "[Knowledge Retrieval - supplementary semantic evidence]\n"
        + "\n\n".join(knowledge_blocks)
        if knowledge_blocks
        else ""
    )
    evidence_context = "\n\n".join(part for part in [structured_block, knowledge_block] if part)
    evidence_priority = (
        "Use Structured Lookup first. Knowledge Retrieval is supplementary and may be empty or less specific."
        if structured_context
        else "Use Knowledge Retrieval only because Structured Lookup is empty."
    )

    return {
        "structured_context": structured_context,
        "knowledge_context": "\n\n".join(knowledge_blocks),
        "evidence_context": evidence_context,
        "evidence_priority": evidence_priority,
        "lookup_diagnostics": json.dumps(diagnostics, ensure_ascii=False),
        "lookup_answerability": json.dumps(diagnostics.get("answerability") or {}, ensure_ascii=False),
    }


def build_grounded_answer_draft(question: str, lookup: dict[str, Any]) -> str:
    direct_answer = clean_cell(lookup.get("direct_answer", "")) if isinstance(lookup, dict) else ""
    answer_items = lookup.get("answer_items") if isinstance(lookup, dict) else []
    if direct_answer and isinstance(answer_items, list) and len(answer_items) > 1:
        values = unique_exact_keep_order(
            clean_cell(item.get("value", ""))
            for item in answer_items
            if isinstance(item, dict) and clean_cell(item.get("value", ""))
        )
        if values:
            return "근거에서 확인되는 전체 목록은 다음과 같습니다.\n" + "\n".join(f"- {value}" for value in values)
    if direct_answer:
        return direct_answer

    matches = lookup.get("matches") if isinstance(lookup, dict) else []
    if not isinstance(matches, list) or not matches:
        return "제공된 문서에는 충분한 근거가 없습니다."

    candidates = []
    for match in matches[:8]:
        if not isinstance(match, dict):
            continue
        candidate = clean_cell(match.get("answer") or match.get("answer_hint") or "")
        if candidate:
            candidates.append(candidate)
    candidates = unique_exact_keep_order(candidates)
    if not candidates:
        return "제공된 문서에는 충분한 근거가 없습니다."

    answer_style = str(lookup.get("answer_style") or classify_answer_style(question, str(lookup.get("query_type") or "")))
    if answer_style == "table_or_bullets" or len(candidates) > 1:
        return "근거에서 확인되는 내용은 다음과 같습니다.\n" + "\n".join(f"- {candidate}" for candidate in candidates)
    return candidates[0]


def parse_json_like(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except Exception:
            return value
    return value


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
        if not date_value_supported(date, evidence_dates):
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

    direct_values = extract_complete_answer_values(evidence_text)
    if direct_values:
        missing_direct_values = [
            value
            for value in direct_values
            if normalize_for_match(value) and not complete_value_supported_by_text(value, answer)
        ]
        if missing_direct_values:
            issues.append(
                {
                    "type": "missing_direct_answer_value",
                    "message": "The answer omits values from the complete structured result.",
                    "values": missing_direct_values[:10],
                }
            )

    if looks_like_inverse_value_question(question) and not is_abstention_answer(answer):
        missing_query_constraints = [
            value
            for value in extract_scalar_constraints(question, query_type)
            if not scalar_value_in_cell(value, answer)
        ]
        if missing_query_constraints:
            issues.append(
                {
                    "type": "missing_query_constraint_value",
                    "message": "The answer does not preserve numeric/date constraints from the question.",
                    "values": missing_query_constraints[:10],
                }
            )

    missing_named_entities = missing_requested_named_entities(question, answer, evidence_text)
    if missing_named_entities:
        issues.append(
            {
                "type": "missing_question_entity",
                "message": "The answer appears to answer a different named entity than the question requested.",
                "terms": missing_named_entities[:10],
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
        "missing_direct_answer_value",
        "missing_query_constraint_value",
        "missing_question_entity",
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


def extract_complete_answer_values(evidence_text: str) -> list[str]:
    values = []
    in_values = False
    for raw_line in evidence_text.splitlines():
        line = raw_line.strip()
        if not line:
            if in_values:
                break
            continue
        if line == "complete_values:":
            in_values = True
            continue
        if in_values:
            if line.startswith("- "):
                value = clean_cell(line[2:])
                if value:
                    values.append(value)
                continue
            break
    return unique_keep_order(values)


def complete_value_supported_by_text(value: str, text: str) -> bool:
    value = clean_cell(value)
    text = clean_cell(text)
    if not value:
        return True
    value_norm = normalize_for_match(value)
    text_norm = normalize_for_match(text)
    if value_norm and value_norm in text_norm:
        return True

    value_dates = extract_date_values(value)
    if value_dates:
        text_dates = extract_date_values(text)
        if not all(date_value_supported(date, text_dates) for date in value_dates):
            return False
        value_times = extract_time_values(value)
        text_times = extract_time_values(text)
        return all(time_value in text_times for time_value in value_times)

    value_numbers = extract_number_values(value)
    if value_numbers:
        text_numbers = extract_number_values(text)
        return all(number in text_numbers for number in value_numbers)

    parts = split_compound_answer_value(value)
    if len(parts) > 1:
        return all(normalize_for_match(part) in text_norm for part in parts)

    return False


def is_abstention_answer(answer: str) -> bool:
    return bool(ABSTENTION_RE.search(answer or ""))


def evidence_has_answer_candidate(question: str, evidence_text: str, query_type: str) -> bool:
    if not evidence_text.strip():
        return False

    question_attributes = detect_attribute_concepts(question)
    for match in ANSWER_CANDIDATE_RE.finditer(evidence_text):
        candidate = clean_cell(match.group(1))
        if candidate and candidate not in {"-", "N/A", "None", "null"}:
            if question_attributes and not (question_attributes & detect_attribute_concepts(candidate)):
                continue
            return True

    if "[Structured Lookup" not in evidence_text:
        return False

    terms = extract_keywords(question, limit=8)
    coverage = term_coverage(evidence_text, terms)
    return query_type in STRUCTURED_LOOKUP_TYPES and coverage >= 0.1


def missing_requested_named_entities(question: str, answer: str, evidence_text: str) -> list[str]:
    entity_pattern = re.compile(
        r"[가-힣A-Za-z0-9._/-]{2,50}(?:학과|학부|전공|대학|부서|팀|시스템|서비스|API|DB|Plan|Feature)"
    )
    requested = unique_keep_order(entity_pattern.findall(question or ""))
    if not requested:
        return []
    answer_norm = normalize_for_match(answer)
    evidence_norm = normalize_for_match(evidence_text)
    answer_entities = unique_keep_order(entity_pattern.findall(answer or ""))
    missing = []
    for entity in requested:
        entity_norm = normalize_for_match(entity)
        if not entity_norm or entity_norm in answer_norm or entity_norm not in evidence_norm:
            continue
        conflicting_entity = any(
            normalize_for_match(candidate) != entity_norm
            and same_entity_suffix(candidate, entity)
            for candidate in answer_entities
        )
        if conflicting_entity:
            missing.append(entity)
    return missing


def same_entity_suffix(left: str, right: str) -> bool:
    suffixes = ["학과", "학부", "전공", "대학", "부서", "팀", "시스템", "서비스", "API", "DB", "Plan", "Feature"]
    left_lower = left.lower()
    right_lower = right.lower()
    return any(left_lower.endswith(suffix.lower()) and right_lower.endswith(suffix.lower()) for suffix in suffixes)


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
    return unique_keep_order(
        normalized
        for match in DATE_RE.finditer(text)
        for normalized in [normalize_date_value(match.group(0))]
        if normalized
    )


def normalize_date_value(value: str) -> str:
    cleaned = clean_cell(value)
    if not cleaned:
        return ""
    if re.fullmatch(r"(?:19|20)\d{2}\s*년?", cleaned):
        year = re.search(r"(?:19|20)\d{2}", cleaned)
        return year.group(0) if year else ""
    if re.search(r"상반기|하반기|1학기|2학기|분기|quarter|Q[1-4]", cleaned, re.IGNORECASE):
        return normalize_for_match(cleaned).replace(" ", "-")

    pattern = re.compile(
        r"(?:(?P<year>(?:19|20)\d{2})\s*[.\-/년]\s*)?"
        r"(?P<month>1[0-2]|0?[1-9])\s*(?:[.\-/월]\s*)"
        r"(?P<day>3[01]|[12]\d|0?[1-9])"
    )
    match = pattern.search(cleaned)
    if not match:
        return normalize_for_match(cleaned)
    year = match.group("year")
    month = int(match.group("month"))
    day = int(match.group("day"))
    date_part = f"{month:02d}-{day:02d}"
    return f"{year}-{date_part}" if year else date_part


def date_value_supported(date: str, evidence_dates: list[str]) -> bool:
    if date in evidence_dates:
        return True
    date_month_day = month_day_part(date)
    if not date_month_day:
        return False
    return any(month_day_part(evidence_date) == date_month_day for evidence_date in evidence_dates)


def month_day_part(date: str) -> str:
    match = re.search(r"(?:(?:19|20)\d{2}-)?(?P<month>\d{2})-(?P<day>\d{2})$", date or "")
    return f"{match.group('month')}-{match.group('day')}" if match else ""


def extract_time_values(text: str) -> list[str]:
    values = []
    for hour, minute in re.findall(r"(?<!\d)([01]?\d|2[0-3])\s*:\s*([0-5]\d)(?!\d)", text or ""):
        values.append(f"{int(hour):02d}:{minute}")
    for period, hour in re.findall(r"(오전|오후)\s*([0-9]{1,2})\s*시", text or ""):
        hour_value = int(hour)
        if period == "오후" and hour_value < 12:
            hour_value += 12
        if period == "오전" and hour_value == 12:
            hour_value = 0
        values.append(f"{hour_value:02d}:00")
    return unique_keep_order(values)


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
        expanded.extend(query_term_variants(term))
        expanded.extend(extract_number_values(term))
        expanded.extend(extract_date_values(term))
        expanded.extend(extract_keywords(term, limit=8))
    return unique_keep_order(expanded)


def query_term_variants(term: str) -> list[str]:
    variants = []
    cleaned = clean_cell(term)
    particle_stripped = strip_query_particle(cleaned)
    if particle_stripped != cleaned:
        variants.append(particle_stripped)
    compact = normalize_for_match(cleaned).replace(" ", "")
    if compact and compact != normalize_for_match(cleaned):
        variants.append(compact)
    return variants


def strip_query_particle(value: str) -> str:
    value = clean_cell(value)
    stripped = re.sub(r"(?:에서|으로|에게|부터|까지|처럼|이고|이며|이면|라면|인가|인지|인가요|이야|인가요)$", "", value)
    stripped = re.sub(r"(?:은|는|이|가|을|를|의|에|로)$", "", stripped)
    stripped = re.sub(r"(?<=\d)(?:명|개|건|회|원|점|일|년)?인$", lambda match: match.group(0)[:-1], stripped)
    return stripped if len(normalize_for_match(stripped)) >= 2 else value


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


def unique_exact_keep_order(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        item = clean_cell(value)
        key = normalize_for_match(item)
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
    raw = str(value or "doc")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    if re.search(r"[^\x00-\x7f]", raw):
        suffix = short_hash(raw, length=8)
        return f"{safe}_{suffix}" if safe else f"doc_{suffix}"
    return safe or "doc"
