from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import get_settings
from .converter import PdfConverter

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - keeps /convert importable before neo4j is installed.
    GraphDatabase = None  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dify OpenDataLoader PDF Converter")

ALLOWED_RELATION_TYPES = {
    "GENERATES",
    "SENT_TO",
    "USES",
    "DEPENDS_ON",
    "BEFORE",
    "AFTER",
    "RELATED_TO",
}
EXPLICIT_ENTITY_TERMS = [
    "Parent-Child Chunker",
    "Knowledge Retrieval",
    "Knowledge Pipeline",
    "OpenDataLoader",
    "Vector DB",
    "Chatflow",
    "FastAPI",
    "Markdown",
    "Neo4j",
    "Docker",
    "Cypher",
    "Dify",
    "HTTP",
    "RAG",
    "LLM",
    "API",
    "PDF",
]
PAGE_SEPARATOR_RE = re.compile(r"^\s*---\s*Page\s+(\d+)\s*---\s*$", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
TOKEN_RE = re.compile(r"\S+")

_neo4j_driver: Any | None = None
_neo4j_schema_ready = False


class ConvertResponse(BaseModel):
    success: bool
    markdown: str


class RagProcessRequest(BaseModel):
    document_id: str
    file_name: str
    markdown: str
    page_metadata: list[dict[str, Any]] = Field(default_factory=list)


class RagProcessResponse(BaseModel):
    document_id: str
    file_name: str
    dify_markdown: str
    graph_json: dict[str, Any]


class RagProcessIngestResponse(BaseModel):
    status: str
    document_id: str
    file_name: str
    dify_markdown: str
    sections: int
    chunks: int
    relations: int


class EntityPayload(BaseModel):
    name: str
    type: str | None = None


class RelationPayload(BaseModel):
    source: str
    type: str
    target: str


class ChunkPayload(BaseModel):
    chunk_id: str
    parent_id: str
    chunk_type: str = "child"
    section_path: str
    page_start: int | None = None
    page_end: int | None = None
    text: str
    text_hash: str
    entities: list[str | EntityPayload] = Field(default_factory=list)
    relations: list[RelationPayload] = Field(default_factory=list)


class SectionPayload(BaseModel):
    section_id: str
    section_path: str
    page_start: int | None = None
    page_end: int | None = None
    chunks: list[ChunkPayload] = Field(default_factory=list)


class GraphIngestRequest(BaseModel):
    document_id: str
    file_name: str
    sections: list[SectionPayload] = Field(default_factory=list)


class GraphIngestResponse(BaseModel):
    status: str
    document_id: str
    sections: int
    chunks: int
    relations: int


class GraphSearchRequest(BaseModel):
    entities: list[str]
    depth: int = 2
    limit: int = 30


class GraphContextItem(BaseModel):
    source: str
    relation: str
    target: str
    chunk_id: str
    parent_id: str
    section_path: str
    page_start: int | None = None
    page_end: int | None = None
    text_hash: str


class GraphSearchResponse(BaseModel):
    graph_context: list[GraphContextItem]
    expanded_keywords: list[str]
    section_paths: list[str]
    chunk_ids: list[str]
    parent_ids: list[str]


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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/graph/health")
async def graph_health() -> dict[str, str]:
    try:
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        return {"status": "ok", "neo4j": "connected"}
    except Exception as exc:
        logger.exception("Neo4j health check failed")
        raise HTTPException(
            status_code=503,
            detail={"status": "error", "neo4j": "disconnected", "error": str(exc)},
        ) from exc


@app.post("/rag/process", response_model=RagProcessResponse)
async def rag_process(request: RagProcessRequest) -> RagProcessResponse:
    try:
        graph_json, dify_markdown = build_rag_artifacts(request)
        return RagProcessResponse(
            document_id=request.document_id,
            file_name=request.file_name,
            dify_markdown=dify_markdown,
            graph_json=graph_json,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("RAG process failed")
        raise HTTPException(status_code=500, detail="RAG processing failed.") from exc


@app.post("/rag/process-ingest", response_model=RagProcessIngestResponse)
async def rag_process_ingest(request: RagProcessRequest) -> RagProcessIngestResponse:
    try:
        graph_json, dify_markdown = build_rag_artifacts(request)
        stats = count_graph_payload(graph_json)
        driver = get_neo4j_driver()
        with driver.session() as session:
            session.execute_write(ingest_graph_tx, graph_json)
        return RagProcessIngestResponse(
            status="ok",
            document_id=request.document_id,
            file_name=request.file_name,
            dify_markdown=dify_markdown,
            sections=stats["sections"],
            chunks=stats["chunks"],
            relations=stats["relations"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("RAG process and Neo4j ingest failed")
        raise HTTPException(status_code=500, detail="RAG process and Neo4j ingest failed.") from exc


@app.post("/graph/ingest", response_model=GraphIngestResponse)
async def graph_ingest(request: GraphIngestRequest) -> GraphIngestResponse:
    try:
        graph_json = model_to_dict(request)
        stats = count_graph_payload(graph_json)
        driver = get_neo4j_driver()
        with driver.session() as session:
            session.execute_write(ingest_graph_tx, graph_json)
        return GraphIngestResponse(
            status="ok",
            document_id=request.document_id,
            sections=stats["sections"],
            chunks=stats["chunks"],
            relations=stats["relations"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Neo4j graph ingest failed")
        raise HTTPException(status_code=500, detail="Neo4j graph ingest failed.") from exc


@app.post("/graph/search", response_model=GraphSearchResponse)
async def graph_search(request: GraphSearchRequest) -> GraphSearchResponse:
    try:
        entities = [entity.strip() for entity in request.entities if entity.strip()]
        if not entities:
            raise HTTPException(status_code=400, detail="entities must contain at least one value.")

        depth = min(max(int(request.depth), 1), 4)
        limit = min(max(int(request.limit), 1), 100)
        driver = get_neo4j_driver()
        with driver.session() as session:
            rows = session.execute_read(search_graph_tx, entities, depth, limit)

        graph_context = [GraphContextItem(**row) for row in rows]
        expanded_keywords = unique_ordered(
            [*entities, *[row.source for row in graph_context], *[row.target for row in graph_context]]
        )
        section_paths = unique_ordered([row.section_path for row in graph_context if row.section_path])
        chunk_ids = unique_ordered([row.chunk_id for row in graph_context if row.chunk_id])
        parent_ids = unique_ordered([row.parent_id for row in graph_context if row.parent_id])

        return GraphSearchResponse(
            graph_context=graph_context,
            expanded_keywords=expanded_keywords,
            section_paths=section_paths,
            chunk_ids=chunk_ids,
            parent_ids=parent_ids,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Neo4j graph search failed")
        raise HTTPException(status_code=500, detail="Neo4j graph search failed.") from exc


@app.on_event("shutdown")
def shutdown_neo4j_driver() -> None:
    global _neo4j_driver
    if _neo4j_driver is not None:
        _neo4j_driver.close()
        _neo4j_driver = None


def sanitize_relation_type(rel_type: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(rel_type or "").strip().upper()).strip("_")
    if normalized in ALLOWED_RELATION_TYPES:
        return normalized
    return "RELATED_TO"


def sha256_text(text: str) -> str:
    normalized = normalize_text_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_heading_sections(markdown: str) -> list[dict[str, Any]]:
    normalized = normalize_markdown(markdown)
    if not normalized:
        return []

    sections: list[dict[str, Any]] = []
    heading_stack: dict[int, str] = {}
    current: dict[str, Any] | None = None
    current_page: int | None = None
    in_fence = False

    def start_section(section_path: str, level: int) -> dict[str, Any]:
        return {
            "section_path": section_path,
            "heading_level": level,
            "lines": [],
        }

    def ensure_section() -> dict[str, Any]:
        nonlocal current
        if current is None:
            current = start_section("Document", 0)
        return current

    def flush_current() -> None:
        nonlocal current
        if current is None:
            return
        blocks = split_line_items_into_blocks(current.get("lines", []))
        text = "\n\n".join(block["text"] for block in blocks if block["text"]).strip()
        has_body = any(
            line["text"].strip() and not HEADING_RE.match(line["text"].strip())
            for line in current.get("lines", [])
        )
        if text and has_body:
            pages = [
                page
                for block in blocks
                for page in (block.get("page_start"), block.get("page_end"))
                if page is not None
            ]
            section = {
                "section_path": current["section_path"],
                "heading_level": current["heading_level"],
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "text": text,
                "blocks": blocks,
            }
            sections.append(section)
        current = None

    for raw_line in normalized.split("\n"):
        line = raw_line.rstrip()
        if not in_fence:
            page_match = PAGE_SEPARATOR_RE.match(line)
            if page_match:
                current_page = int(page_match.group(1))
                continue

            heading_match = HEADING_RE.match(line)
            if heading_match:
                flush_current()
                level = len(heading_match.group(1))
                title = clean_heading_text(heading_match.group(2))
                heading_stack = {key: value for key, value in heading_stack.items() if key < level}
                heading_stack[level] = title
                section_path = " > ".join(heading_stack[key] for key in sorted(heading_stack))
                current = start_section(section_path or title or "Document", level)
                current["lines"].append({"text": line, "page": current_page})
                continue

        section = ensure_section()
        section["lines"].append({"text": line, "page": current_page})
        if FENCE_RE.match(line):
            in_fence = not in_fence

    flush_current()
    return sections


def build_chunks_from_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_tokens = get_int_env("RAG_CHILD_MAX_TOKENS", 330)
    overlap_tokens = get_int_env("RAG_CHILD_OVERLAP_TOKENS", 60)
    max_tokens = min(max(max_tokens, 120), 1000)
    overlap_tokens = min(max(overlap_tokens, 0), max_tokens // 2)

    chunks: list[dict[str, Any]] = []
    for section_index, section in enumerate(sections, start=1):
        document_prefix = section.get("document_prefix") or stable_id_prefix(section.get("document_id", "doc"))
        section_id = section.get("section_id") or f"{document_prefix}_sec_{section_index:03d}"
        parent_id = section.get("parent_id") or f"{section_id}_parent_001"
        section_path = section.get("section_path") or "Document"
        blocks = section.get("blocks") or [
            {
                "text": section.get("text", ""),
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
            }
        ]
        groups = build_child_groups(blocks, max_tokens, overlap_tokens)
        for child_index, group in enumerate(groups, start=1):
            text = normalize_markdown(group["text"])
            if not text:
                continue
            extracted = extract_entities_and_relations(text)
            chunk_id = f"{section_id}_child_{child_index:03d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "section_id": section_id,
                    "chunk_type": "child",
                    "section_path": section_path,
                    "page_start": group.get("page_start") or section.get("page_start"),
                    "page_end": group.get("page_end") or section.get("page_end"),
                    "text": text,
                    "text_hash": sha256_text(text),
                    "entities": extracted["entities"],
                    "relations": extracted["relations"],
                }
            )
    return chunks


def enrich_dify_markdown(chunks: list[dict[str, Any]]) -> str:
    rendered_chunks = []
    for chunk in chunks:
        page_value = format_page_range(chunk.get("page_start"), chunk.get("page_end"))
        header = "\n".join(
            [
                f"[chunk_id: {chunk['chunk_id']}]",
                f"[parent_id: {chunk['parent_id']}]",
                f"[section_path: {chunk.get('section_path', 'Document')}]",
                f"[page: {page_value}]",
                f"[text_hash: {chunk['text_hash']}]",
            ]
        )
        rendered_chunks.append(f"{header}\n\n{chunk.get('text', '').strip()}".strip())
    return "\n\n".join(rendered_chunks).strip()


def extract_entities_and_relations(text: str) -> dict[str, Any]:
    entity_matches = find_entity_matches(text)
    entities = unique_ordered([match["name"] for match in entity_matches])
    relations: list[dict[str, str]] = []
    seen_relations: set[tuple[str, str, str]] = set()

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    for sentence in sentences:
        sentence_matches = find_entity_matches(sentence)
        sentence_entities = unique_ordered([match["name"] for match in sentence_matches])
        if len(sentence_entities) < 2:
            continue

        relation_type = infer_relation_type(sentence)
        if relation_type == "GENERATES":
            source = first_non_format_entity(sentence_entities) or sentence_entities[0]
            target = preferred_target(sentence_entities, ("Markdown", "PDF")) or sentence_entities[-1]
            add_relation(relations, seen_relations, source, relation_type, target)
        elif relation_type in {"SENT_TO", "USES", "DEPENDS_ON", "BEFORE", "AFTER"}:
            for source, target in adjacent_pairs(sentence_entities):
                add_relation(relations, seen_relations, source, relation_type, target)
        else:
            for source, target in adjacent_pairs(sentence_entities[:4]):
                add_relation(relations, seen_relations, source, "RELATED_TO", target)

    if not relations and len(entities) >= 2:
        for source, target in adjacent_pairs(entities[:4]):
            add_relation(relations, seen_relations, source, "RELATED_TO", target)

    return {"entities": entities[:30], "relations": relations[:20]}


def build_graph_json(document_id: str, file_name: str, markdown: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sections = extract_heading_sections(markdown)
    document_prefix = stable_id_prefix(document_id)
    for index, section in enumerate(sections, start=1):
        section_id = f"{document_prefix}_sec_{index:03d}"
        section["document_id"] = document_id
        section["document_prefix"] = document_prefix
        section["section_id"] = section_id
        section["parent_id"] = f"{section_id}_parent_001"

    chunks = build_chunks_from_sections(sections)
    chunks_by_section: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        chunks_by_section.setdefault(chunk["section_id"], []).append(chunk)

    graph_sections = []
    for section in sections:
        section_chunks = []
        for chunk in chunks_by_section.get(section["section_id"], []):
            section_chunks.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "parent_id": chunk["parent_id"],
                    "chunk_type": chunk["chunk_type"],
                    "section_path": chunk["section_path"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "text": chunk["text"],
                    "text_hash": chunk["text_hash"],
                    "entities": chunk["entities"],
                    "relations": chunk["relations"],
                }
            )
        graph_sections.append(
            {
                "section_id": section["section_id"],
                "section_path": section["section_path"],
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
                "chunks": section_chunks,
            }
        )

    return (
        {
            "document_id": document_id,
            "file_name": file_name,
            "sections": graph_sections,
        },
        chunks,
    )


def build_rag_artifacts(request: RagProcessRequest) -> tuple[dict[str, Any], str]:
    if not request.document_id.strip():
        raise HTTPException(status_code=400, detail="document_id is required.")
    if not request.file_name.strip():
        raise HTTPException(status_code=400, detail="file_name is required.")
    if not request.markdown.strip():
        raise HTTPException(status_code=400, detail="markdown is required.")

    graph_json, chunks = build_graph_json(
        request.document_id,
        request.file_name,
        request.markdown,
    )
    return graph_json, enrich_dify_markdown(chunks)


def get_neo4j_driver() -> Any:
    global _neo4j_driver
    if GraphDatabase is None:
        raise RuntimeError("neo4j package is not installed. Run: pip install neo4j")
    if _neo4j_driver is None:
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            raise RuntimeError("NEO4J_PASSWORD environment variable is required.")
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        _neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
        ensure_graph_schema(_neo4j_driver)
    return _neo4j_driver


def ensure_graph_schema(driver: Any) -> None:
    global _neo4j_schema_ready
    if _neo4j_schema_ready:
        return
    statements = [
        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
        "CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    ]
    with driver.session() as session:
        for statement in statements:
            session.run(statement)
    _neo4j_schema_ready = True


def ingest_graph_tx(tx: Any, graph_json: dict[str, Any]) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    tx.run(
        """
        MERGE (document:Document {document_id: $document_id})
        ON CREATE SET document.created_at = $created_at
        SET document.file_name = $file_name
        """,
        document_id=graph_json["document_id"],
        file_name=graph_json.get("file_name", ""),
        created_at=created_at,
    )

    for section in graph_json.get("sections", []):
        tx.run(
            """
            MERGE (section:Section {section_id: $section_id})
            SET section.section_path = $section_path,
                section.page_start = $page_start,
                section.page_end = $page_end
            WITH section
            MATCH (document:Document {document_id: $document_id})
            MERGE (document)-[:CONTAINS]->(section)
            """,
            document_id=graph_json["document_id"],
            section_id=section["section_id"],
            section_path=section.get("section_path", "Document"),
            page_start=section.get("page_start"),
            page_end=section.get("page_end"),
        )

        for chunk in section.get("chunks", []):
            ingest_chunk_tx(tx, section, chunk, created_at)


def ingest_chunk_tx(tx: Any, section: dict[str, Any], chunk: dict[str, Any], created_at: str) -> None:
    text = chunk.get("text", "")
    text_preview = make_text_preview(text)
    tx.run(
        """
        MERGE (chunk:Chunk {chunk_id: $chunk_id})
        ON CREATE SET chunk.created_at = $created_at
        SET chunk.parent_id = $parent_id,
            chunk.chunk_type = $chunk_type,
            chunk.section_path = $section_path,
            chunk.page_start = $page_start,
            chunk.page_end = $page_end,
            chunk.text_hash = $text_hash,
            chunk.text_preview = $text_preview,
            chunk.updated_at = $created_at
        WITH chunk
        MATCH (section:Section {section_id: $section_id})
        MERGE (section)-[:HAS_CHUNK]->(chunk)
        """,
        section_id=section["section_id"],
        chunk_id=chunk["chunk_id"],
        parent_id=chunk["parent_id"],
        chunk_type=chunk.get("chunk_type", "child"),
        section_path=chunk.get("section_path", section.get("section_path", "Document")),
        page_start=chunk.get("page_start"),
        page_end=chunk.get("page_end"),
        text_hash=chunk["text_hash"],
        text_preview=text_preview,
        created_at=created_at,
    )

    entities = normalize_entity_payloads(chunk.get("entities", []))
    relations = [model_to_dict(relation) for relation in chunk.get("relations", [])]
    for relation in relations:
        source = str(relation.get("source", "")).strip()
        target = str(relation.get("target", "")).strip()
        if source:
            entities.setdefault(source, {"name": source, "type": infer_entity_type(source)})
        if target:
            entities.setdefault(target, {"name": target, "type": infer_entity_type(target)})

    for entity in entities.values():
        tx.run(
            """
            MERGE (entity:Entity {name: $name})
            ON CREATE SET entity.type = $type
            SET entity.type = coalesce(entity.type, $type)
            WITH entity
            MATCH (chunk:Chunk {chunk_id: $chunk_id})
            MERGE (chunk)-[:MENTIONS]->(entity)
            """,
            name=entity["name"],
            type=entity.get("type") or infer_entity_type(entity["name"]),
            chunk_id=chunk["chunk_id"],
        )

    for relation in relations:
        source = str(relation.get("source", "")).strip()
        target = str(relation.get("target", "")).strip()
        if not source or not target or source == target:
            continue
        relation_type = sanitize_relation_type(str(relation.get("type", "RELATED_TO")))
        tx.run(
            f"""
            MATCH (source:Entity {{name: $source}})
            MATCH (target:Entity {{name: $target}})
            MERGE (source)-[relation:{relation_type}]->(target)
            ON CREATE SET relation.created_at = $created_at
            SET relation.updated_at = $created_at,
                relation.chunk_ids = CASE
                    WHEN relation.chunk_ids IS NULL THEN [$chunk_id]
                    WHEN $chunk_id IN relation.chunk_ids THEN relation.chunk_ids
                    ELSE relation.chunk_ids + $chunk_id
                END,
                relation.parent_ids = CASE
                    WHEN relation.parent_ids IS NULL THEN [$parent_id]
                    WHEN $parent_id IN relation.parent_ids THEN relation.parent_ids
                    ELSE relation.parent_ids + $parent_id
                END,
                relation.section_paths = CASE
                    WHEN relation.section_paths IS NULL THEN [$section_path]
                    WHEN $section_path IN relation.section_paths THEN relation.section_paths
                    ELSE relation.section_paths + $section_path
                END,
                relation.text_hashes = CASE
                    WHEN relation.text_hashes IS NULL THEN [$text_hash]
                    WHEN $text_hash IN relation.text_hashes THEN relation.text_hashes
                    ELSE relation.text_hashes + $text_hash
                END
            """,
            source=source,
            target=target,
            chunk_id=chunk["chunk_id"],
            parent_id=chunk["parent_id"],
            section_path=chunk.get("section_path", section.get("section_path", "Document")),
            text_hash=chunk["text_hash"],
            created_at=created_at,
        )


def search_graph_tx(tx: Any, entities: list[str], depth: int, limit: int) -> list[dict[str, Any]]:
    query = f"""
    MATCH path = (start:Entity)-[*1..{depth}]-(end:Entity)
    WHERE start.name IN $entities
      AND all(relation IN relationships(path) WHERE type(relation) IN $relation_types)
    UNWIND relationships(path) AS relation
    WITH DISTINCT relation
    WITH startNode(relation) AS source, type(relation) AS relation_type, endNode(relation) AS target, relation
    UNWIND coalesce(relation.chunk_ids, []) AS chunk_id
    MATCH (chunk:Chunk {{chunk_id: chunk_id}})
    RETURN DISTINCT
        source.name AS source,
        relation_type AS relation,
        target.name AS target,
        chunk.chunk_id AS chunk_id,
        chunk.parent_id AS parent_id,
        chunk.section_path AS section_path,
        chunk.page_start AS page_start,
        chunk.page_end AS page_end,
        chunk.text_hash AS text_hash
    LIMIT $limit
    """
    result = tx.run(query, entities=entities, relation_types=sorted(ALLOWED_RELATION_TYPES), limit=limit)
    return [dict(record) for record in result]


def split_line_items_into_blocks(line_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    in_fence = False

    def flush() -> None:
        nonlocal current
        text = "\n".join(item["text"] for item in current).strip()
        if text:
            pages = [item.get("page") for item in current if item.get("page") is not None and item["text"].strip()]
            blocks.append(
                {
                    "text": text,
                    "page_start": min(pages) if pages else None,
                    "page_end": max(pages) if pages else None,
                }
            )
        current = []

    for item in line_items:
        line = item["text"]
        if not in_fence and not line.strip():
            flush()
            continue
        current.append(item)
        if FENCE_RE.match(line):
            in_fence = not in_fence

    flush()
    return blocks


def build_child_groups(blocks: list[dict[str, Any]], max_tokens: int, overlap_tokens: int) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    last_tail: dict[str, Any] | None = None

    def append_group(parts: list[dict[str, Any]]) -> None:
        nonlocal last_tail
        text = "\n\n".join(part["text"] for part in parts if part.get("text")).strip()
        if not text:
            return
        pages = [
            page
            for part in parts
            for page in (part.get("page_start"), part.get("page_end"))
            if page is not None
        ]
        group = {
            "text": text,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
        }
        groups.append(group)
        tail = tail_tokens(text, overlap_tokens)
        last_tail = {
            "text": tail,
            "page_start": group["page_start"],
            "page_end": group["page_end"],
        } if tail else None

    def start_with_overlap() -> tuple[list[dict[str, Any]], int]:
        if not last_tail:
            return [], 0
        return [last_tail.copy()], token_count(last_tail["text"])

    for block in blocks:
        text = normalize_markdown(block.get("text", ""))
        if not text:
            continue
        block = {**block, "text": text}
        block_tokens = token_count(text)
        if block_tokens > max_tokens:
            if current:
                append_group(current)
                current = []
                current_tokens = 0
            for piece in split_large_block(block, max_tokens, overlap_tokens):
                append_group([piece])
            continue

        if not current and groups and last_tail:
            current, current_tokens = start_with_overlap()
        if current and current_tokens + block_tokens > max_tokens:
            append_group(current)
            current, current_tokens = start_with_overlap()
            if current_tokens + block_tokens > max_tokens:
                current = []
                current_tokens = 0

        current.append(block)
        current_tokens += block_tokens

    if current:
        append_group(current)
    return groups


def split_large_block(block: dict[str, Any], max_tokens: int, overlap_tokens: int) -> list[dict[str, Any]]:
    tokens = tokenize(block.get("text", ""))
    if not tokens:
        return []
    step = max(max_tokens - overlap_tokens, 1)
    pieces = []
    start = 0
    while start < len(tokens):
        window = tokens[start : start + max_tokens]
        pieces.append(
            {
                "text": " ".join(window),
                "page_start": block.get("page_start"),
                "page_end": block.get("page_end"),
            }
        )
        if start + max_tokens >= len(tokens):
            break
        start += step
    return pieces


def find_entity_matches(text: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < used_end and end > used_start for used_start, used_end in occupied)

    for term in sorted(EXPLICIT_ENTITY_TERMS, key=len, reverse=True):
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE)
        for match in pattern.finditer(text):
            if overlaps(match.start(), match.end()):
                continue
            matches.append({"name": canonical_entity_name(term), "start": match.start(), "end": match.end()})
            occupied.append((match.start(), match.end()))

    for pattern in (
        re.compile(r"\b[A-Z]{2,}(?:[-_/][A-Za-z0-9]+)*\b"),
        re.compile(r"\b[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]+)+\b"),
        re.compile(r"\b[A-Z][A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)+\b"),
        re.compile(r"(?=[A-Za-z0-9_\-\uac00-\ud7a3]*[A-Za-z])(?=[A-Za-z0-9_\-\uac00-\ud7a3]*[\uac00-\ud7a3])[A-Za-z0-9_\-\uac00-\ud7a3]{2,}"),
    ):
        for match in pattern.finditer(text):
            value = clean_entity_name(match.group(0))
            if not is_entity_candidate(value) or overlaps(match.start(), match.end()):
                continue
            matches.append({"name": value, "start": match.start(), "end": match.end()})
            occupied.append((match.start(), match.end()))

    matches.sort(key=lambda item: (item["start"], item["end"]))
    deduped = []
    seen: set[str] = set()
    for match in matches:
        key = match["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
    return deduped


def infer_relation_type(sentence: str) -> str:
    lowered = sentence.lower()
    if "->" in sentence or "=>" in sentence or "\u2192" in sentence:
        return "BEFORE"
    if re.search(r"\b(generate|generates|generated|produce|produces|return|returns|convert|converts|converted)\b", lowered):
        return "GENERATES"
    if re.search(r"[\uc0dd\uc131\ubc18\ud658\ucd9c\ub825\ubcc0\ud658]", sentence) and re.search(r"\uc0dd\uc131|\ubc18\ud658|\ucd9c\ub825|\ubcc0\ud658", sentence):
        return "GENERATES"
    if re.search(r"\b(send|sends|sent|forward|forwards|pass|passes|deliver|delivers)\b", lowered):
        return "SENT_TO"
    if re.search(r"\uc804\ub2ec|\uc804\uc1a1|\ubcf4\ub0b4|\ub118\uae30", sentence):
        return "SENT_TO"
    if re.search(r"\b(use|uses|used|using)\b", lowered) or re.search(r"\uc0ac\uc6a9|\ud65c\uc6a9|\ub2f4\ub2f9", sentence):
        return "USES"
    if re.search(r"\bdepend|depends|dependency|required|requires\b", lowered) or re.search(r"\uc758\uc874|\ud544\uc694", sentence):
        return "DEPENDS_ON"
    if re.search(r"\bbefore|prior\b", lowered) or re.search(r"\uc774\uc804|\uba3c\uc800", sentence):
        return "BEFORE"
    if re.search(r"\bafter|next|then\b", lowered) or re.search(r"\uc774\ud6c4|\ub2e4\uc74c|\ud6c4", sentence):
        return "AFTER"
    return "RELATED_TO"


def add_relation(
    relations: list[dict[str, str]],
    seen: set[tuple[str, str, str]],
    source: str,
    relation_type: str,
    target: str,
) -> None:
    source = clean_entity_name(source)
    target = clean_entity_name(target)
    relation_type = sanitize_relation_type(relation_type)
    if not source or not target or source == target:
        return
    key = (source.lower(), relation_type, target.lower())
    if key in seen:
        return
    seen.add(key)
    relations.append({"source": source, "type": relation_type, "target": target})


def adjacent_pairs(values: list[str]) -> list[tuple[str, str]]:
    return [(values[index], values[index + 1]) for index in range(len(values) - 1)]


def first_non_format_entity(entities: list[str]) -> str | None:
    format_entities = {"PDF", "Markdown", "JSON", "HTML", "CSV"}
    for entity in entities:
        if entity not in format_entities:
            return entity
    return None


def preferred_target(entities: list[str], preferred: tuple[str, ...]) -> str | None:
    for value in preferred:
        if value in entities:
            return value
    return None


def normalize_entity_payloads(raw_entities: list[Any]) -> dict[str, dict[str, str]]:
    entities: dict[str, dict[str, str]] = {}
    for raw_entity in raw_entities:
        entity = model_to_dict(raw_entity)
        if isinstance(entity, str):
            name = clean_entity_name(entity)
            entity_type = infer_entity_type(name)
        elif isinstance(entity, dict):
            name = clean_entity_name(str(entity.get("name", "")))
            entity_type = str(entity.get("type") or infer_entity_type(name))
        else:
            continue
        if name:
            entities[name] = {"name": name, "type": entity_type}
    return entities


def infer_entity_type(name: str) -> str:
    normalized = name.lower()
    if normalized in {"pdf", "markdown", "json", "html", "csv"}:
        return "FORMAT"
    if normalized in {"opendataloader", "dify", "neo4j", "fastapi", "docker", "cypher"}:
        return "SYSTEM"
    if normalized in {"rag", "llm", "api", "http", "vector db"}:
        return "CONCEPT"
    return "TERM"


def canonical_entity_name(term: str) -> str:
    lookup = {value.lower(): value for value in EXPLICIT_ENTITY_TERMS}
    return lookup.get(term.lower(), term)


def clean_entity_name(value: str) -> str:
    value = re.sub(r"^[^\w\uac00-\ud7a3]+|[^\w\uac00-\ud7a3]+$", "", value.strip())
    value = re.sub(r"\s+", " ", value)
    return value[:80]


def is_entity_candidate(value: str) -> bool:
    if len(value) < 2 or len(value) > 80:
        return False
    if value.lower() in {"the", "and", "for", "with", "from", "this", "that"}:
        return False
    return bool(re.search(r"[A-Za-z\uac00-\ud7a3]", value))


def stable_id_prefix(document_id: Any) -> str:
    raw = str(document_id or "").strip()
    prefix = re.sub(r"[^0-9A-Za-z_\-]+", "_", raw).strip("_")
    if prefix:
        return prefix
    return f"doc_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]}"


def normalize_markdown(value: str) -> str:
    value = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.split("\n")]
    value = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def normalize_text_for_hash(value: str) -> str:
    value = normalize_markdown(value)
    return re.sub(r"[ \t]+", " ", value).strip()


def clean_heading_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip("#")).strip() or "Untitled"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def token_count(text: str) -> int:
    return len(tokenize(text))


def tail_tokens(text: str, count: int) -> str:
    if count <= 0:
        return ""
    tokens = tokenize(text)
    return " ".join(tokens[-count:])


def format_page_range(page_start: Any, page_end: Any) -> str:
    if page_start is None and page_end is None:
        return "unknown"
    if page_start == page_end or page_end is None:
        return str(page_start)
    if page_start is None:
        return str(page_end)
    return f"{page_start}-{page_end}"


def make_text_preview(text: str, limit: int = 400) -> str:
    preview = re.sub(r"\s+", " ", str(text or "")).strip()
    return preview[:limit]


def unique_ordered(values: list[str]) -> list[str]:
    result = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def model_to_dict(value: Any) -> Any:
    if isinstance(value, BaseModel):
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return value.dict()
    return value


def count_graph_payload(graph_json: dict[str, Any]) -> dict[str, int]:
    sections = graph_json.get("sections", [])
    chunks = [chunk for section in sections for chunk in section.get("chunks", [])]
    relations = [relation for chunk in chunks for relation in chunk.get("relations", [])]
    return {"sections": len(sections), "chunks": len(chunks), "relations": len(relations)}


def get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default
