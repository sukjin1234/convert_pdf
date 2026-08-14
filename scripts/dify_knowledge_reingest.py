from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.converter import PdfConverter  # noqa: E402
from app.rag import RagStore, build_rag_artifact, clean_section_path  # noqa: E402
from scripts.dify_knowledge_artifact_sync import (  # noqa: E402
    build_config,
    list_existing_knowledge_documents,
    resolve_store_dir,
)
from scripts.dify_knowledge_pdf_perf_test import (  # noqa: E402
    KnowledgeConfig,
    build_process_rule,
    load_env_file,
    payload_items,
    poll_indexing_status,
    request_json,
)


DOCUMENT_ID_RE = re.compile(r"^\[document_id:\s*([^\]]+)\]", re.MULTILINE)
BRACKET_METADATA_RE = re.compile(r"^\[([a-z_]+):\s*(.*?)\]\s*$")
SOURCE_SECTION_RE = re.compile(r"^(\s*source_section:\s*).*$")
CONTEXT_LINE_RE = re.compile(r"^\s*\[context:\s*file=.*\]\s*$")
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
PLAIN_NUMBERED_HEADING_RE = re.compile(r"^(\d{1,2}[.)])\s+(.+?)\s*$")
ROMAN_HEADING_RE = re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+[.)]\s+")
ARABIC_HEADING_RE = re.compile(r"^\d{1,2}[.)]\s+")


class SourcePdfUnavailable(RuntimeError):
    pass


@dataclass
class ReingestItem:
    document_id: str
    name: str
    source_file_name: str = ""
    source_kind: str = ""
    artifact_document_id: str = ""
    source_bytes: int = 0
    markdown_chars: int = 0
    dify_markdown_chars: int = 0
    chunk_count: int = 0
    record_count: int = 0
    status: str = "pending"
    batch: str = ""
    indexing_statuses: list[str] | None = None
    indexing_seconds: float = 0.0
    error: str = ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download existing Dify Knowledge PDFs, rebuild retrieval Markdown locally, "
            "and reindex the same document IDs through update-by-text."
        )
    )
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--dataset-name", default=os.getenv("DIFY_DATASET_NAME", "ipsi"))
    parser.add_argument("--document-id", action="append", default=[], help="Dify document ID. Repeatable.")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--index-timeout", type=float, default=1800.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--doc-language", default="Korean")
    parser.add_argument("--store-dir", default="")
    parser.add_argument("--apply", action="store_true", help="Update and reindex documents. Default is dry-run.")
    parser.add_argument("--out", default=str(ROOT / ".runtime" / "dify_knowledge_reingest_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    config = build_config(args)
    documents = select_documents(
        list_existing_knowledge_documents(config, timeout=args.timeout),
        document_ids=args.document_id,
        include_disabled=args.include_disabled,
    )
    if not documents:
        raise RuntimeError("No matching Knowledge documents were found.")

    store = RagStore(resolve_store_dir(args.store_dir)) if args.apply else None
    converter = PdfConverter(get_settings())
    items: list[ReingestItem] = []
    started = time.perf_counter()
    for document in documents:
        item = ReingestItem(
            document_id=str(document.get("id") or ""),
            name=document_name(document),
        )
        items.append(item)
        try:
            reingest_document(config, document, item, converter=converter, store=store, args=args)
        except Exception as exc:  # noqa: BLE001 - continue so every selected document is reported.
            item.status = "failed"
            item.error = f"{type(exc).__name__}: {exc}"

    report = {
        "base_url_host": urllib.parse.urlparse(config.base_url).netloc,
        "dataset_id": config.dataset_id,
        "apply": bool(args.apply),
        "summary": {
            "selected": len(items),
            "updated": sum(item.status == "completed" for item in items),
            "validated": sum(item.status == "validated" for item in items),
            "failed": sum(item.status == "failed" for item in items),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "documents": [asdict(item) for item in items],
        "failed": any(item.status == "failed" for item in items),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report, out_path)
    return 1 if report["failed"] else 0


def select_documents(
    documents: list[dict[str, Any]],
    *,
    document_ids: list[str],
    include_disabled: bool,
) -> list[dict[str, Any]]:
    selected_ids = {value.strip() for value in document_ids if value.strip()}
    selected = []
    found_ids: set[str] = set()
    for document in documents:
        document_id = str(document.get("id") or "")
        if not document_id or (selected_ids and document_id not in selected_ids):
            continue
        found_ids.add(document_id)
        if not include_disabled and not document_enabled(document):
            continue
        selected.append(document)
    missing = sorted(selected_ids - found_ids)
    if missing:
        raise RuntimeError(f"Knowledge document IDs were not found: {', '.join(missing)}")
    return sorted(selected, key=lambda value: document_name(value))


def document_enabled(document: dict[str, Any]) -> bool:
    value = document.get("enabled")
    return True if value is None else bool(value)


def document_name(document: dict[str, Any]) -> str:
    return str(document.get("name") or document.get("document_name") or document.get("id") or "document.pdf").strip()


def reingest_document(
    config: KnowledgeConfig,
    document: dict[str, Any],
    item: ReingestItem,
    *,
    converter: PdfConverter,
    store: RagStore | None,
    args: argparse.Namespace,
) -> None:
    current_segments = fetch_all_document_segments(config, item.document_id, timeout=args.timeout)
    artifact_document_id = extract_artifact_document_id(current_segments) or item.document_id
    artifact = None
    try:
        pdf_bytes, source_name = download_document_pdf(config, item.document_id, item.name, timeout=args.timeout)
    except SourcePdfUnavailable:
        source_name = source_file_name_from_segments(current_segments) or item.name
        dify_markdown, chunk_count = rebuild_dify_markdown_from_segments(
            current_segments,
            document_id=artifact_document_id,
            fallback_file_name=source_name,
        )
        item.source_kind = "knowledge_segments"
        item.source_file_name = source_name
        item.source_bytes = sum(len(str(segment.get("content") or "").encode("utf-8")) for segment in current_segments)
        item.markdown_chars = item.source_bytes
        item.dify_markdown_chars = len(dify_markdown)
        item.chunk_count = chunk_count
        item.record_count = dify_markdown.count("- structured_record:")
    else:
        item.source_kind = "pdf"
        item.source_file_name = source_name
        item.source_bytes = len(pdf_bytes)
        markdown = converter.convert_pdf_bytes(pdf_bytes, source_name)
        artifact = build_rag_artifact(markdown, source_name, document_id=artifact_document_id)
        dify_markdown = artifact.dify_markdown
        item.markdown_chars = len(artifact.markdown)
        item.dify_markdown_chars = len(dify_markdown)
        item.chunk_count = len(artifact.chunks)
        item.record_count = len(artifact.records)

    validate_artifact(dify_markdown, item.chunk_count, item.document_id)
    item.artifact_document_id = artifact_document_id
    if not args.apply:
        item.status = "validated"
        return

    payload = build_update_payload(config, item.name, dify_markdown, args.doc_language)
    response = request_json(
        "POST",
        f"{config.base_url}/datasets/{config.dataset_id}/documents/{item.document_id}/update-by-text",
        config.api_key,
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        content_type="application/json; charset=utf-8",
        timeout=args.timeout,
    )
    item.batch = str(response.get("batch") or "")
    if not item.batch:
        raise RuntimeError(f"update-by-text response did not include batch: {response}")

    indexing_started = time.perf_counter()
    indexing = poll_indexing_status(
        config,
        item.batch,
        timeout=args.timeout,
        index_timeout=args.index_timeout,
        poll_interval=args.poll_interval,
    )
    item.indexing_seconds = round(time.perf_counter() - indexing_started, 3)
    item.indexing_statuses = list(indexing.get("statuses") or [])
    if indexing.get("failed"):
        raise RuntimeError(f"Knowledge indexing failed: {item.indexing_statuses}")
    if store is not None and artifact is not None:
        store.upsert(artifact)
    item.status = "completed"


def download_document_pdf(
    config: KnowledgeConfig,
    document_id: str,
    fallback_name: str,
    *,
    timeout: float,
) -> tuple[bytes, str]:
    url = f"{config.base_url}/datasets/{config.dataset_id}/documents/{document_id}/download"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {config.api_key}"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
            content_disposition = response.headers.get("Content-Disposition", "")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404 and "does not have an uploaded file" in text:
            raise SourcePdfUnavailable(text) from exc
        raise RuntimeError(f"HTTP {exc.code}: {text[:1000]}") from exc
    if not content.lstrip().startswith(b"%PDF-"):
        raise SourcePdfUnavailable("Downloaded Knowledge source is text rather than a PDF stream.")
    filename = filename_from_content_disposition(content_disposition) or fallback_name or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    return content, filename


def filename_from_content_disposition(value: str) -> str:
    encoded = re.search(r"filename\*=UTF-8''([^;]+)", value, re.IGNORECASE)
    if encoded:
        return Path(urllib.parse.unquote(encoded.group(1))).name
    plain = re.search(r'filename="?([^";]+)"?', value, re.IGNORECASE)
    return Path(plain.group(1).strip()).name if plain else ""


def fetch_all_document_segments(config: KnowledgeConfig, document_id: str, *, timeout: float) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"page": page, "limit": 100})
        payload = request_json(
            "GET",
            f"{config.base_url}/datasets/{config.dataset_id}/documents/{document_id}/segments?{query}",
            config.api_key,
            timeout=timeout,
        )
        items = [value for value in payload_items(payload) if isinstance(value, dict)]
        segments.extend(items)
        total = int(payload.get("total") or 0)
        if not items or len(items) < 100 or (total and len(segments) >= total):
            return segments
        page += 1


def extract_artifact_document_id(segments: list[dict[str, Any]]) -> str:
    for segment in segments:
        match = DOCUMENT_ID_RE.search(str(segment.get("content") or ""))
        if match:
            return match.group(1).strip()
    return ""


def source_file_name_from_segments(segments: list[dict[str, Any]]) -> str:
    for segment in segments:
        for line in str(segment.get("content") or "").splitlines()[:10]:
            match = BRACKET_METADATA_RE.match(line.strip())
            if match and match.group(1) == "file_name":
                return match.group(2).strip()
    return ""


def rebuild_dify_markdown_from_segments(
    segments: list[dict[str, Any]],
    *,
    document_id: str,
    fallback_file_name: str,
) -> tuple[str, int]:
    rendered_blocks: list[str] = []
    active_page = "unknown"
    active_section = "Untitled"
    active_file_name = fallback_file_name
    active_record_types = "text"
    active_keywords = ""
    previous_declared_section = ""

    for segment in sorted(segments, key=segment_sort_key):
        metadata, body = split_segment_metadata(str(segment.get("content") or ""))
        declared_page = metadata.get("page", "").strip()
        declared_section = clean_section_path(metadata.get("section", ""))
        if declared_page and declared_page != active_page:
            active_page = declared_page
            active_section = declared_section or active_section
        elif declared_section and declared_section != previous_declared_section:
            active_section = declared_section
        if declared_section:
            previous_declared_section = declared_section
        active_file_name = metadata.get("file_name", "").strip() or active_file_name
        active_record_types = metadata.get("record_types", "").strip() or active_record_types
        active_keywords = metadata.get("keywords", "").strip() or active_keywords

        body_lines = [line for line in body.splitlines() if not CONTEXT_LINE_RE.match(line)]
        for block_body, heading in split_body_at_semantic_headings(body_lines):
            if heading:
                active_section = update_section_path(active_section, heading)
            block_body = rewrite_source_sections(block_body, active_section)
            if not block_body.strip():
                continue
            block_index = len(rendered_blocks) + 1
            chunk_id = f"{document_id}_p{safe_page_id(active_page)}_r{block_index:04d}"
            rendered_blocks.append(
                render_recovered_block(
                    block_body,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    file_name=active_file_name,
                    page=active_page,
                    section=active_section,
                    record_types=active_record_types,
                    keywords=active_keywords,
                )
            )

    return "\n\n---\n\n".join(rendered_blocks).strip(), len(rendered_blocks)


def segment_sort_key(segment: dict[str, Any]) -> tuple[int, str]:
    try:
        position = int(segment.get("position") or 0)
    except (TypeError, ValueError):
        position = 0
    return position, str(segment.get("id") or "")


def split_segment_metadata(content: str) -> tuple[dict[str, str], str]:
    metadata: dict[str, str] = {}
    lines = content.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    body_start = 0
    while body_start < len(lines):
        stripped = lines[body_start].strip()
        match = BRACKET_METADATA_RE.match(stripped)
        if match:
            metadata[match.group(1)] = match.group(2)
            body_start += 1
            continue
        if not stripped:
            body_start += 1
            continue
        break
    return metadata, "\n".join(lines[body_start:]).strip()


def split_body_at_semantic_headings(lines: list[str]) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current: list[str] = []
    current_heading = ""
    for index, line in enumerate(lines):
        heading = semantic_heading(lines, index)
        if heading:
            if any(value.strip() for value in current):
                blocks.append(("\n".join(current).strip(), current_heading))
            current = [line]
            current_heading = heading
        else:
            current.append(line)
    if any(value.strip() for value in current):
        blocks.append(("\n".join(current).strip(), current_heading))
    return blocks


def semantic_heading(lines: list[str], index: int) -> str:
    stripped = lines[index].strip()
    markdown = MARKDOWN_HEADING_RE.match(stripped)
    if markdown:
        return markdown.group(2).strip()
    plain = PLAIN_NUMBERED_HEADING_RE.match(stripped)
    if not plain or len(stripped) > 80 or plain.group(2).lstrip().startswith(("-", "※", "○")):
        return ""
    before_blank = index == 0 or not lines[index - 1].strip()
    after_blank = index + 1 >= len(lines) or not lines[index + 1].strip()
    return stripped if before_blank and after_blank else ""


def update_section_path(current_section: str, heading: str) -> str:
    heading = clean_section_path(heading) or "Untitled"
    parts = [part.strip() for part in clean_section_path(current_section).split(" > ") if part.strip()]
    if ROMAN_HEADING_RE.match(heading):
        return heading
    if ARABIC_HEADING_RE.match(heading):
        for index in range(len(parts) - 1, -1, -1):
            if ARABIC_HEADING_RE.match(parts[index]):
                return " > ".join([*parts[:index], heading])
        return " > ".join([*parts, heading]) if parts else heading
    if parts and parts[-1] == heading:
        return " > ".join(parts)
    return " > ".join([*parts, heading]) if parts else heading


def rewrite_source_sections(body: str, section: str) -> str:
    lines = []
    for line in body.splitlines():
        match = SOURCE_SECTION_RE.match(line)
        lines.append(f"{match.group(1)}{section}" if match else line)
    return "\n".join(lines).strip()


def render_recovered_block(
    body: str,
    *,
    document_id: str,
    chunk_id: str,
    file_name: str,
    page: str,
    section: str,
    record_types: str,
    keywords: str,
) -> str:
    paragraphs = [value.strip() for value in re.split(r"\n{2,}", body) if value.strip()]
    contextualized = []
    for index, paragraph in enumerate(paragraphs, start=1):
        paragraph_id = f"{chunk_id}_p{index:03d}"
        context = (
            f"[context: file={file_name} | page={page} | section={section} | "
            f"paragraph={paragraph_id}]"
        )
        contextualized.append(f"{context}\n{paragraph}")
    header = "\n".join(
        [
            f"[document_id: {document_id}]",
            f"[chunk_id: {chunk_id}]",
            f"[file_name: {file_name}]",
            f"[page: {page}]",
            f"[section: {section}]",
            f"[record_types: {record_types or 'text'}]",
            f"[keywords: {keywords}]",
            f"[paragraph_count: {len(contextualized)}]",
        ]
    )
    return f"{header}\n" + "\n\n".join(contextualized)


def safe_page_id(page: str) -> str:
    match = re.search(r"\d+", page)
    return f"{int(match.group(0)):03d}" if match else "000"


def build_update_payload(
    config: KnowledgeConfig,
    name: str,
    dify_markdown: str,
    doc_language: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "text": dify_markdown,
        "process_rule": build_process_rule(config),
        "doc_language": doc_language,
    }
    if config.doc_form:
        payload["doc_form"] = config.doc_form
    return payload


def validate_artifact(dify_markdown: str, chunk_count: int, document_id: str) -> None:
    if not dify_markdown.strip() or not chunk_count:
        raise RuntimeError(f"Converted document {document_id} did not produce retrieval chunks.")
    if "[document_id:" not in dify_markdown or "[context: file=" not in dify_markdown:
        raise RuntimeError(f"Converted document {document_id} is missing retrieval metadata.")


def print_human(report: dict[str, Any], out_path: Path) -> None:
    summary = report["summary"]
    mode = "apply" if report["apply"] else "dry-run"
    print(f"Dify Knowledge PDF reingest ({mode})")
    print(f"- dataset_id: {report['dataset_id']}")
    print(
        f"- selected/updated/validated/failed: {summary['selected']}/"
        f"{summary['updated']}/{summary['validated']}/{summary['failed']}"
    )
    for item in report["documents"]:
        print(
            f"- {item['name']}: {item['status']}, chunks={item['chunk_count']}, "
            f"records={item['record_count']}, indexing={item['indexing_seconds']}s"
        )
        if item["error"]:
            print(f"  error: {item['error']}")
    print(f"- elapsed: {summary['elapsed_seconds']}s")
    print(f"- saved: {out_path}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI should emit a concise failure.
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
