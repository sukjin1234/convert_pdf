from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dify_app_api_diagnostics import stream_chat_message, summarize_stream_events
from scripts.dify_knowledge_segment_audit import extract_page, extract_section, segment_content


CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
OUTLINE_RE = re.compile(
    r"^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+|[IVXLCDM]+|(?:chapter|part)\s+\w+|제\s*\d+\s*(?:장|절|조)|\d+(?:\.\d+)*[.)])\s*",
    re.IGNORECASE,
)
LIST_RE = re.compile(r"^(?:[-*•·※○●□■]|\(?\d+\)?[.)]|[A-Za-z][.)])\s*")
PAGE_REF_TEMPLATES = (
    r"\bp\s*[.]?\s*{page}\b",
    r"\bpage\s*{page}\b",
    r"\b{page}\s*(?:쪽|페이지)\b",
)


@dataclass(frozen=True)
class StructureAnchor:
    page: int
    kind: str
    text: str


@dataclass(frozen=True)
class SegmentView:
    segment_id: str
    page: int | None
    section: str
    content: str


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Domain-neutral PDF boundary, Dify RAG quality, citation, and latency evaluation."
    )
    parser.add_argument("--manifest", default=str(ROOT / "evaluation" / "rag_acceptance_cases.json"))
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--dataset-id", default=os.getenv("DIFY_KNOWLEDGE_DATASET_ID") or os.getenv("DIFY_DATASET_ID") or "")
    parser.add_argument("--dataset-name", default=os.getenv("DIFY_DATASET_NAME") or "")
    parser.add_argument(
        "--document-id",
        action="append",
        default=[],
        metavar="KEY=ARTIFACT_ID",
        help="Bind a manifest document key to the deployed RAG artifact id. Repeatable.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--segment-concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-structure", action="store_true")
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--fail-under-boundary", type=float, default=0.95)
    parser.add_argument("--fail-under-answer", type=float, default=0.95)
    parser.add_argument("--fail-under-citation", type=float, default=0.95)
    parser.add_argument("--max-p95-latency", type=float, default=10.0)
    parser.add_argument("--out", default=str(ROOT / ".runtime" / "rag_acceptance_latest.json"))
    parser.add_argument("--markdown-out", default=str(ROOT / "evaluation" / "rag_acceptance_report.md"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = validate_manifest(manifest, manifest_path)
    explicit_document_ids = parse_document_id_bindings(args.document_id, documents)
    cases = list(manifest.get("cases") or [])
    if args.limit > 0:
        cases = cases[: args.limit]

    base_url = (os.getenv("DIFY_API_BASE_URL") or "").rstrip("/")
    knowledge_key = os.getenv("DIFY_KNOWLEDGE_API_KEY") or ""
    app_key = os.getenv("DIFY_APP_API_KEY") or os.getenv("DIFY_API_KEY") or ""
    missing_chat_bindings = {item["key"] for item in documents} - set(explicit_document_ids)
    knowledge_required = not args.skip_structure or (not args.skip_chat and bool(missing_chat_bindings))
    if (knowledge_required and (not base_url or not knowledge_key)) or (not args.skip_chat and (not base_url or not app_key)):
        print("Missing DIFY_API_BASE_URL or the required API key in the env file.", file=sys.stderr)
        return 2

    started = time.perf_counter()
    dataset_id = args.dataset_id
    knowledge_documents: list[dict[str, Any]] = []
    document_bindings: dict[str, dict[str, Any]] = {
        key: {"artifact_document_id": document_id} for key, document_id in explicit_document_ids.items()
    }
    segments_by_document: dict[str, list[dict[str, Any]]] = {}
    structure_results: list[dict[str, Any]] = []

    if knowledge_required:
        dataset_id, knowledge_documents = resolve_dataset(
            base_url,
            knowledge_key,
            dataset_id=dataset_id,
            dataset_name=args.dataset_name,
            expected_names=[str(item["knowledge_name"]) for item in documents],
            timeout=args.timeout,
        )
        if not dataset_id:
            raise RuntimeError("Could not resolve a Knowledge dataset containing the manifest documents.")
        knowledge_bindings = bind_documents(documents, knowledge_documents)
        for key, binding in knowledge_bindings.items():
            document_bindings.setdefault(key, {}).update(binding)

    if not args.skip_structure:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(len(documents), 2))) as pool:
            future_by_key = {
                pool.submit(
                    fetch_segments_parallel,
                    base_url,
                    knowledge_key,
                    dataset_id,
                    str(document_bindings[item["key"]]["id"]),
                    timeout=args.timeout,
                    workers=args.segment_concurrency,
                ): item["key"]
                for item in documents
            }
            for future, key in list(future_by_key.items()):
                segments_by_document[key] = future.result()

        for item in documents:
            anchors, meaningful_pages = extract_pdf_structure(Path(item["path"]))
            segment_views = [to_segment_view(segment) for segment in segments_by_document[item["key"]]]
            result = evaluate_structure(anchors, meaningful_pages, segment_views)
            result.update(
                {
                    "document_key": item["key"],
                    "source_path": item["path"],
                    "knowledge_name": item["knowledge_name"],
                    "knowledge_document_id": document_bindings[item["key"]]["id"],
                }
            )
            structure_results.append(result)

    if not args.skip_chat:
        for item in documents:
            key = item["key"]
            if document_bindings[key].get("artifact_document_id"):
                continue
            segments = segments_by_document.get(key)
            if segments is None:
                segments = fetch_segment_page(
                    base_url,
                    knowledge_key,
                    dataset_id,
                    str(document_bindings[key]["id"]),
                    timeout=args.timeout,
                )
                segments_by_document[key] = segments
            if not document_bindings[key].get("artifact_document_id"):
                artifact_document_id = extract_artifact_document_id(segments)
                document_bindings[key]["artifact_document_id"] = artifact_document_id or str(
                    document_bindings[key]["id"]
                )

    quality_results: list[dict[str, Any]] = []
    if not args.skip_chat:
        indexed_cases = list(enumerate(cases))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            future_by_index = {
                pool.submit(
                    run_chat_case,
                    case,
                    base_url=base_url,
                    api_key=app_key,
                    document_id=str(document_bindings[str(case["document"])]["artifact_document_id"]),
                    document=next(item for item in documents if item["key"] == case["document"]),
                    timeout=args.timeout,
                ): index
                for index, case in indexed_cases
            }
            ordered: dict[int, dict[str, Any]] = {}
            for future in concurrent.futures.as_completed(future_by_index):
                index = future_by_index[future]
                try:
                    ordered[index] = future.result()
                except Exception as exc:  # noqa: BLE001
                    case = cases[index]
                    ordered[index] = failed_chat_case(case, exc)
            quality_results = [ordered[index] for index, _ in indexed_cases]

    structure_summary = summarize_structure(structure_results)
    quality_summary = summarize_quality(quality_results)
    latency_summary = summarize_latency(quality_results)
    gates = evaluate_gates(
        structure_summary,
        quality_summary,
        latency_summary,
        skip_structure=args.skip_structure,
        skip_chat=args.skip_chat,
        min_boundary=args.fail_under_boundary,
        min_answer=args.fail_under_answer,
        min_citation=args.fail_under_citation,
        max_p95=args.max_p95_latency,
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "base_url_host": urllib.parse.urlparse(base_url).netloc if base_url else "",
        "dataset_id": dataset_id,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "thresholds": {
            "boundary_accuracy": args.fail_under_boundary,
            "answer_accuracy": args.fail_under_answer,
            "citation_accuracy": args.fail_under_citation,
            "p95_latency_seconds": args.max_p95_latency,
        },
        "gates": gates,
        "structure_summary": structure_summary,
        "quality_summary": quality_summary,
        "latency_summary": latency_summary,
        "structure_results": structure_results,
        "quality_results": quality_results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = Path(args.markdown_out)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report, out_path, markdown_path)
    return 0 if gates["passed"] else 1


def validate_manifest(payload: dict[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    documents = []
    seen = set()
    for raw in payload.get("documents") or []:
        key = str(raw.get("key") or "").strip()
        source = str(raw.get("path") or "").strip()
        if not key or not source or key in seen:
            raise ValueError("Each manifest document needs a unique key and a path.")
        path = Path(source)
        if not path.is_absolute():
            path = (manifest_path.parent.parent / path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        seen.add(key)
        documents.append(
            {
                "key": key,
                "path": str(path),
                "knowledge_name": str(raw.get("knowledge_name") or path.name),
            }
        )
    if not documents:
        raise ValueError("The manifest contains no documents.")
    for case in payload.get("cases") or []:
        if case.get("document") not in seen:
            raise ValueError(f"Unknown case document key: {case.get('document')}")
        if not case.get("question"):
            raise ValueError(f"Case {case.get('id')} has no question.")
    return documents


def parse_document_id_bindings(values: list[str], documents: list[dict[str, Any]]) -> dict[str, str]:
    valid_keys = {str(item["key"]) for item in documents}
    bindings: dict[str, str] = {}
    for value in values:
        key, separator, document_id = value.partition("=")
        key = key.strip()
        document_id = document_id.strip()
        if not separator or not key or not document_id:
            raise ValueError(f"Invalid --document-id binding: {value!r}; expected KEY=ARTIFACT_ID")
        if key not in valid_keys:
            raise ValueError(f"Unknown manifest document key in --document-id: {key}")
        bindings[key] = document_id
    return bindings


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip():
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def request_json(method: str, url: str, api_key: str, *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace") or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Dify HTTP {exc.code}: {body}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from {url}.")
    return payload


def payload_items(payload: dict[str, Any]) -> list[Any]:
    for key in ("data", "documents", "segments", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def list_datasets(base_url: str, api_key: str, *, timeout: float) -> list[dict[str, Any]]:
    payload = request_json("GET", f"{base_url}/datasets?{urllib.parse.urlencode({'page': 1, 'limit': 100})}", api_key, timeout=timeout)
    return [item for item in payload_items(payload) if isinstance(item, dict)]


def list_documents(base_url: str, api_key: str, dataset_id: str, *, timeout: float) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"page": page, "limit": 100})
        payload = request_json("GET", f"{base_url}/datasets/{dataset_id}/documents?{query}", api_key, timeout=timeout)
        items = [item for item in payload_items(payload) if isinstance(item, dict)]
        documents.extend(items)
        total = int(payload.get("total") or 0)
        if not items or len(items) < 100 or (total and len(documents) >= total):
            return documents
        page += 1


def resolve_dataset(
    base_url: str,
    api_key: str,
    *,
    dataset_id: str,
    dataset_name: str,
    expected_names: list[str],
    timeout: float,
) -> tuple[str, list[dict[str, Any]]]:
    if dataset_id:
        return dataset_id, list_documents(base_url, api_key, dataset_id, timeout=timeout)
    datasets = list_datasets(base_url, api_key, timeout=timeout)
    if dataset_name:
        datasets = [item for item in datasets if normalize_text(item.get("name")) == normalize_text(dataset_name)]
    expected = {normalize_text(name) for name in expected_names}
    best: tuple[int, str, list[dict[str, Any]]] = (0, "", [])
    for dataset in datasets:
        candidate_id = str(dataset.get("id") or "")
        if not candidate_id:
            continue
        docs = list_documents(base_url, api_key, candidate_id, timeout=timeout)
        actual = {normalize_text(doc.get("name") or doc.get("document_name")) for doc in docs}
        score = len(expected & actual)
        if score > best[0]:
            best = (score, candidate_id, docs)
        if score == len(expected):
            return candidate_id, docs
    return best[1], best[2]


def bind_documents(manifest_documents: list[dict[str, Any]], knowledge_documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    bindings = {}
    for expected in manifest_documents:
        target = normalize_text(expected["knowledge_name"])
        target_stem = normalize_text(Path(expected["knowledge_name"]).stem)
        matches = []
        for candidate in knowledge_documents:
            name = str(candidate.get("name") or candidate.get("document_name") or "")
            normalized = normalize_text(name)
            score = 2 if normalized == target else 1 if normalize_text(Path(name).stem) == target_stem else 0
            if score:
                matches.append((score, candidate))
        if not matches:
            raise RuntimeError(f"Knowledge document not found: {expected['knowledge_name']}")
        bindings[expected["key"]] = max(matches, key=lambda item: item[0])[1]
    return bindings


def fetch_segments_parallel(
    base_url: str,
    api_key: str,
    dataset_id: str,
    document_id: str,
    *,
    timeout: float,
    workers: int,
) -> list[dict[str, Any]]:
    def fetch_page(page: int) -> dict[str, Any]:
        query = urllib.parse.urlencode({"page": page, "limit": 100})
        return request_json(
            "GET",
            f"{base_url}/datasets/{dataset_id}/documents/{document_id}/segments?{query}",
            api_key,
            timeout=timeout,
        )

    first = fetch_page(1)
    total = int(first.get("total") or len(payload_items(first)))
    page_count = max(1, math.ceil(total / 100))
    pages = {1: first}
    if page_count > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            future_by_page = {pool.submit(fetch_page, page): page for page in range(2, page_count + 1)}
            for future in concurrent.futures.as_completed(future_by_page):
                pages[future_by_page[future]] = future.result()
    segments = [item for page in sorted(pages) for item in payload_items(pages[page]) if isinstance(item, dict)]
    return sorted(segments, key=lambda item: int(item.get("position") or item.get("segment_position") or 0))


def fetch_segment_page(
    base_url: str,
    api_key: str,
    dataset_id: str,
    document_id: str,
    *,
    timeout: float,
) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"page": 1, "limit": 100})
    payload = request_json(
        "GET",
        f"{base_url}/datasets/{dataset_id}/documents/{document_id}/segments?{query}",
        api_key,
        timeout=timeout,
    )
    return [item for item in payload_items(payload) if isinstance(item, dict)]


def extract_artifact_document_id(segments: list[dict[str, Any]]) -> str:
    pattern = re.compile(r"^\[document_id:\s*([^\]]+)\]", flags=re.MULTILINE)
    for segment in segments:
        content = str(segment.get("content") or segment.get("text") or "")
        match = pattern.search(content)
        if match:
            return match.group(1).strip()
    return ""


def extract_pdf_structure(path: Path) -> tuple[list[StructureAnchor], set[int]]:
    import fitz

    anchors: list[StructureAnchor] = []
    meaningful_pages: set[int] = set()
    document = fitz.open(path)
    try:
        for page_number, page in enumerate(document, start=1):
            raw_text = clean_text(page.get_text("text"))
            if len(normalize_text(raw_text)) < 24:
                continue
            meaningful_pages.add(page_number)
            page_dict = page.get_text("dict")
            lines: list[tuple[str, float, float]] = []
            font_sizes: list[float] = []
            page_height = float(page.rect.height or 1)
            for block in page_dict.get("blocks") or []:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines") or []:
                    spans = line.get("spans") or []
                    text = clean_text(" ".join(str(span.get("text") or "") for span in spans))
                    if not text:
                        continue
                    bbox = line.get("bbox") or (0, 0, 0, 0)
                    y0 = float(bbox[1])
                    size = max((float(span.get("size") or 0) for span in spans), default=0.0)
                    if page_height * 0.025 < y0 < page_height * 0.965:
                        lines.append((text, size, y0))
                        if size > 0:
                            font_sizes.append(size)
            body_size = statistics.median(font_sizes) if font_sizes else 10.0
            page_anchors: list[StructureAnchor] = []
            for text, size, _ in lines:
                if is_page_noise(text) or len(text) > 180:
                    continue
                if size >= body_size * 1.16 or (OUTLINE_RE.match(text) and size >= body_size * 0.95):
                    page_anchors.append(StructureAnchor(page_number, "heading", text))
                elif LIST_RE.match(text) and len(text) >= 12:
                    page_anchors.append(StructureAnchor(page_number, "list", text))
                elif "=" in text and len(text) >= 10 and re.search(r"[0-9A-Za-z가-힣]", text):
                    page_anchors.append(StructureAnchor(page_number, "formula", text))
            try:
                tables = page.find_tables().tables
            except Exception:
                tables = []
            for table in tables:
                rows = table.extract() or []
                cells = unique_keep_order(
                    clean_text(cell)
                    for row in rows[:2]
                    for cell in (row or [])
                    if clean_text(cell)
                )
                header = " / ".join(cells)
                if len(cells) >= 2 and len(header) >= 8:
                    page_anchors.append(StructureAnchor(page_number, "table", header[:240]))
            anchors.extend(deduplicate_anchors(page_anchors))
    finally:
        document.close()
    return anchors, meaningful_pages


def deduplicate_anchors(anchors: list[StructureAnchor]) -> list[StructureAnchor]:
    result = []
    seen = set()
    per_kind: defaultdict[str, int] = defaultdict(int)
    limits = {"heading": 12, "table": 8, "list": 8, "formula": 8}
    for anchor in anchors:
        key = (anchor.page, anchor.kind, normalize_text(anchor.text))
        if not key[2] or key in seen or per_kind[anchor.kind] >= limits.get(anchor.kind, 8):
            continue
        seen.add(key)
        per_kind[anchor.kind] += 1
        result.append(anchor)
    return result


def to_segment_view(segment: dict[str, Any]) -> SegmentView:
    raw_content = segment_content(segment)
    raw_page = extract_page(raw_content)
    match = re.search(r"\d+", raw_page or "")
    return SegmentView(
        segment_id=str(segment.get("id") or ""),
        page=int(match.group()) if match else None,
        section=clean_text(extract_section(raw_content)),
        content=clean_text(raw_content),
    )


def evaluate_structure(
    anchors: list[StructureAnchor],
    meaningful_pages: set[int],
    segments: list[SegmentView],
) -> dict[str, Any]:
    by_page: defaultdict[int, list[SegmentView]] = defaultdict(list)
    for segment in segments:
        if segment.page is not None:
            by_page[segment.page].append(segment)
    matched = []
    unmatched = []
    kind_totals: defaultdict[str, int] = defaultdict(int)
    kind_matches: defaultdict[str, int] = defaultdict(int)
    for anchor in anchors:
        kind_totals[anchor.kind] += 1
        candidates = by_page.get(anchor.page, [])
        found = any(anchor_matches(anchor, segment) for segment in candidates)
        if found:
            kind_matches[anchor.kind] += 1
            matched.append(anchor)
        else:
            unmatched.append(anchor)
    covered_pages = meaningful_pages & set(by_page)
    valid_segments = [
        segment
        for segment in segments
        if segment.page is not None
        and segment.section
        and normalize_text(segment.section) not in {"untitled", "목차"}
        and len(normalize_text(segment.content)) >= 20
    ]
    anchor_recall = rate(len(matched), len(anchors), empty=1.0)
    page_coverage = rate(len(covered_pages), len(meaningful_pages), empty=1.0)
    segment_precision = rate(len(valid_segments), len(segments), empty=1.0)
    boundary_accuracy = round(anchor_recall * 0.5 + page_coverage * 0.25 + segment_precision * 0.25, 4)
    return {
        "boundary_accuracy": boundary_accuracy,
        "source_boundary_recall": anchor_recall,
        "page_coverage": page_coverage,
        "segment_boundary_precision": segment_precision,
        "anchor_count": len(anchors),
        "matched_anchor_count": len(matched),
        "meaningful_page_count": len(meaningful_pages),
        "covered_page_count": len(covered_pages),
        "segment_count": len(segments),
        "valid_segment_count": len(valid_segments),
        "anchor_kinds": {
            kind: {
                "total": kind_totals[kind],
                "matched": kind_matches[kind],
                "recall": rate(kind_matches[kind], kind_totals[kind], empty=1.0),
            }
            for kind in sorted(kind_totals)
        },
        "unmatched_anchor_samples": [asdict(anchor) for anchor in unmatched[:20]],
    }


def anchor_matches(anchor: StructureAnchor, segment: SegmentView) -> bool:
    haystack = f"{segment.section}\n{segment.content}"
    anchor_compact = normalize_text(anchor.text)
    haystack_compact = normalize_text(haystack)
    if len(anchor_compact) >= 8 and anchor_compact in haystack_compact:
        return True
    anchor_tokens = meaningful_tokens(anchor.text)
    haystack_tokens = set(meaningful_tokens(haystack))
    if not anchor_tokens:
        return False
    overlap = sum(1 for token in anchor_tokens if token in haystack_tokens) / len(anchor_tokens)
    threshold = 0.55 if anchor.kind in {"table", "list"} else 0.65
    return overlap >= threshold


def run_chat_case(
    case: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    document_id: str,
    document: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    inputs = dict(case.get("inputs") or {})
    inputs.setdefault("document_id", document_id)
    user = f"rag-acceptance-{case.get('id')}-{int(time.time() * 1000)}"
    started = time.perf_counter()
    events, elapsed = stream_chat_message(
        base_url=base_url,
        api_key=api_key,
        query=str(case["question"]),
        inputs=inputs,
        user=user,
        timeout=timeout,
    )
    summary = summarize_stream_events(events)
    answer = str(summary.get("streamed_answer") or summary.get("final_answer") or "")
    answer_correct, missing_groups = evaluate_answer(answer, case.get("answer") or {})
    citation_correct, citation_checks = evaluate_citation(
        answer,
        file_name=str(document["knowledge_name"]),
        source=case.get("source") or {},
    )
    failed_nodes = summary.get("failed_nodes") or []
    return {
        "id": case.get("id"),
        "document": case.get("document"),
        "question": case.get("question"),
        "tags": case.get("tags") or [],
        "success": not failed_nodes and bool(answer),
        "answer_correct": answer_correct,
        "citation_correct": citation_correct,
        "passed": answer_correct and citation_correct and not failed_nodes,
        "missing_answer_groups": missing_groups,
        "citation_checks": citation_checks,
        "latency_seconds": round(elapsed or time.perf_counter() - started, 3),
        "node_timings": summary.get("node_timings") or [],
        "deployment_contract": summary.get("deployment_contract") or {},
        "verify_valid": summary.get("verify_valid"),
        "structured_response_chars": summary.get("structured_response_chars") or 0,
        "answer": answer,
        "error": "",
    }


def failed_chat_case(case: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "id": case.get("id"),
        "document": case.get("document"),
        "question": case.get("question"),
        "tags": case.get("tags") or [],
        "success": False,
        "answer_correct": False,
        "citation_correct": False,
        "passed": False,
        "missing_answer_groups": [],
        "citation_checks": {},
        "latency_seconds": 0,
        "node_timings": [],
        "deployment_contract": {},
        "verify_valid": None,
        "structured_response_chars": 0,
        "answer": "",
        "error": f"{type(exc).__name__}: {exc}",
    }


def evaluate_answer(answer: str, expected: dict[str, Any]) -> tuple[bool, list[list[str]]]:
    groups = expected.get("all_of") or []
    missing = []
    for raw_group in groups:
        group = raw_group if isinstance(raw_group, list) else [raw_group]
        aliases = [str(value) for value in group if clean_text(value)]
        if aliases and not any(answer_contains_alias(answer, alias) for alias in aliases):
            missing.append([str(value) for value in group])
    forbidden = [str(value) for value in expected.get("none_of") or [] if clean_text(value)]
    forbidden_found = any(answer_contains_alias(answer, value) for value in forbidden)
    return bool(groups) and not missing and not forbidden_found, missing


def answer_contains_alias(answer: str, alias: str) -> bool:
    cleaned_alias = clean_text(alias)
    if re.fullmatch(r"[+-]?\d[\d,.]*", cleaned_alias):
        numeric_alias = cleaned_alias.replace(",", "")
        numeric_answer = str(answer or "").replace(",", "")
        return bool(re.search(rf"(?<![\d.]){re.escape(numeric_alias)}(?![\d.])", numeric_answer))
    normalized_alias = normalize_text(cleaned_alias)
    if not normalized_alias:
        return cleaned_alias in str(answer or "")
    return bool(normalized_alias and normalized_alias in normalize_text(answer))


def evaluate_citation(answer: str, *, file_name: str, source: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    normalized_answer = normalize_text(answer)
    file_match = normalize_text(file_name) in normalized_answer or normalize_text(Path(file_name).stem) in normalized_answer
    pages = [int(page) for page in source.get("pages") or []]
    page_match = not pages or any(
        re.search(template.format(page=page), answer, flags=re.IGNORECASE) for page in pages for template in PAGE_REF_TEMPLATES
    )
    section_groups = source.get("section_any") or []
    section_match = not section_groups or any(
        all(normalize_text(term) in normalized_answer for term in (group if isinstance(group, list) else [group]))
        for group in section_groups
    )
    region_match = page_match or section_match if (pages and section_groups) else page_match and section_match
    checks = {"file": file_match, "page": page_match, "section": section_match, "region": region_match}
    return file_match and region_match, checks


def summarize_structure(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    weights = [max(1, int(result.get("anchor_count") or 0)) for result in results]
    total_weight = sum(weights)
    return {
        "documents": len(results),
        "boundary_accuracy": round(sum(result["boundary_accuracy"] * weight for result, weight in zip(results, weights)) / total_weight, 4),
        "source_boundary_recall": round(sum(result["source_boundary_recall"] * weight for result, weight in zip(results, weights)) / total_weight, 4),
        "page_coverage": round(sum(result["page_coverage"] for result in results) / len(results), 4),
        "segment_boundary_precision": round(sum(result["segment_boundary_precision"] for result in results) / len(results), 4),
        "anchors": sum(int(result["anchor_count"]) for result in results),
        "segments": sum(int(result["segment_count"]) for result in results),
    }


def summarize_quality(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    successful = [result for result in results if result.get("success")]
    return {
        "total": len(results),
        "successful": len(successful),
        "success_rate": rate(len(successful), len(results)),
        "answer_correct": sum(1 for result in results if result.get("answer_correct")),
        "answer_accuracy": rate(sum(1 for result in results if result.get("answer_correct")), len(results)),
        "citation_correct": sum(1 for result in results if result.get("citation_correct")),
        "citation_accuracy": rate(sum(1 for result in results if result.get("citation_correct")), len(results)),
        "passed": sum(1 for result in results if result.get("passed")),
        "pass_rate": rate(sum(1 for result in results if result.get("passed")), len(results)),
    }


def summarize_latency(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(result["latency_seconds"]) for result in results if result.get("success")]
    stage_values: defaultdict[str, list[float]] = defaultdict(list)
    node_values: defaultdict[str, list[float]] = defaultdict(list)
    response_chars = []
    for result in results:
        if result.get("structured_response_chars"):
            response_chars.append(int(result["structured_response_chars"]))
        case_stages: defaultdict[str, float] = defaultdict(float)
        for timing in result.get("node_timings") or []:
            elapsed = float(timing.get("elapsed_seconds") or 0)
            title = str(timing.get("title") or timing.get("node_type") or "unknown")
            case_stages[classify_node_stage(title, str(timing.get("node_type") or ""))] += elapsed
            node_values[title].append(elapsed)
        for stage, elapsed in case_stages.items():
            stage_values[stage].append(elapsed)
    return {
        "samples": len(latencies),
        "average_seconds": round(statistics.mean(latencies), 3) if latencies else 0,
        "p50_seconds": percentile(latencies, 0.50),
        "p95_seconds": percentile(latencies, 0.95),
        "max_seconds": round(max(latencies), 3) if latencies else 0,
        "average_structured_response_chars": round(statistics.mean(response_chars), 1) if response_chars else 0,
        "stages": {
            stage: {
                "average_seconds": round(statistics.mean(values), 3),
                "p95_seconds": percentile(values, 0.95),
            }
            for stage, values in sorted(stage_values.items())
        },
        "slowest_nodes": sorted(
            (
                {
                    "title": title,
                    "average_seconds": round(statistics.mean(values), 3),
                    "p95_seconds": percentile(values, 0.95),
                }
                for title, values in node_values.items()
            ),
            key=lambda item: item["p95_seconds"],
            reverse=True,
        )[:10],
    }


def classify_node_stage(title: str, node_type: str) -> str:
    value = f"{title} {node_type}".lower()
    if any(term in value for term in ("plan", "classif", "rewrite", "route", "분류")):
        return "classification"
    if any(term in value for term in ("retriev", "knowledge", "lookup", "rerank", "search", "검색", "지식")):
        return "retrieval"
    if any(term in value for term in ("verify", "guard", "검증")):
        return "verification"
    if any(term in value for term in ("answer", "generation", "llm", "답변")):
        return "generation"
    return "orchestration"


def evaluate_gates(
    structure: dict[str, Any],
    quality: dict[str, Any],
    latency: dict[str, Any],
    *,
    skip_structure: bool,
    skip_chat: bool,
    min_boundary: float,
    min_answer: float,
    min_citation: float,
    max_p95: float,
) -> dict[str, Any]:
    checks = {}
    if not skip_structure:
        checks["boundary_accuracy"] = float(structure.get("boundary_accuracy") or 0) >= min_boundary
    if not skip_chat:
        checks["answer_accuracy"] = float(quality.get("answer_accuracy") or 0) >= min_answer
        checks["citation_accuracy"] = float(quality.get("citation_accuracy") or 0) >= min_citation
        checks["latency_p95"] = max_p95 <= 0 or float(latency.get("p95_seconds") or math.inf) <= max_p95
        checks["request_success"] = float(quality.get("success_rate") or 0) >= 0.95
    return {"passed": all(checks.values()), "checks": checks, "failed": [name for name, passed in checks.items() if not passed]}


def render_markdown_report(report: dict[str, Any]) -> str:
    structure = report.get("structure_summary") or {}
    quality = report.get("quality_summary") or {}
    latency = report.get("latency_summary") or {}
    gates = report.get("gates") or {}
    lines = [
        "# RAG acceptance report",
        "",
        f"- Generated: {report.get('created_at')}",
        f"- Dataset: `{report.get('dataset_id')}`",
        f"- Overall gate: {'PASS' if gates.get('passed') else 'FAIL'}",
        f"- Failed gates: {', '.join(gates.get('failed') or []) or 'none'}",
        "",
        "## Metrics",
        "",
        "| Metric | Result | Target |",
        "| --- | ---: | ---: |",
    ]
    if structure:
        lines.append(f"| Boundary accuracy | {structure.get('boundary_accuracy', 0):.2%} | >= {report['thresholds']['boundary_accuracy']:.0%} |")
        lines.append(f"| Source boundary recall | {structure.get('source_boundary_recall', 0):.2%} | diagnostic |")
        lines.append(f"| Page coverage | {structure.get('page_coverage', 0):.2%} | diagnostic |")
    if quality:
        lines.append(f"| Answer accuracy | {quality.get('answer_accuracy', 0):.2%} | >= {report['thresholds']['answer_accuracy']:.0%} |")
        lines.append(f"| Citation accuracy | {quality.get('citation_accuracy', 0):.2%} | >= {report['thresholds']['citation_accuracy']:.0%} |")
        lines.append(f"| End-to-end pass rate | {quality.get('pass_rate', 0):.2%} | diagnostic |")
    if latency:
        lines.append(f"| End-to-end p95 latency | {latency.get('p95_seconds', 0):.3f}s | <= {report['thresholds']['p95_latency_seconds']:.1f}s |")
    lines.extend(["", "## Per-document structure", ""])
    for item in report.get("structure_results") or []:
        lines.append(
            f"- `{item.get('knowledge_name')}`: boundary {item.get('boundary_accuracy', 0):.2%}, "
            f"anchors {item.get('matched_anchor_count')}/{item.get('anchor_count')}, "
            f"pages {item.get('covered_page_count')}/{item.get('meaningful_page_count')}, "
            f"segments {item.get('valid_segment_count')}/{item.get('segment_count')}"
        )
    lines.extend(["", "## Latency bottlenecks", ""])
    for node in latency.get("slowest_nodes") or []:
        lines.append(f"- {node['title']}: average {node['average_seconds']:.3f}s, p95 {node['p95_seconds']:.3f}s")
    failures = [result for result in report.get("quality_results") or [] if not result.get("passed")]
    lines.extend(["", "## Failed quality cases", ""])
    if not failures:
        lines.append("- None")
    for result in failures:
        reasons = []
        if not result.get("answer_correct"):
            reasons.append("answer")
        if not result.get("citation_correct"):
            reasons.append("citation")
        if result.get("error"):
            reasons.append(result["error"])
        lines.append(f"- `{result.get('id')}` ({', '.join(reasons)}): {result.get('question')}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Boundary accuracy is a weighted structural score: 50% source heading/table/list/formula anchor recall, "
            "25% meaningful-page coverage, and 25% valid segment-boundary precision. Answer and citation accuracy are "
            "scored independently. Expected content stays in the manifest; the evaluator contains no domain-specific field logic.",
            "",
        ]
    )
    return "\n".join(lines)


def print_human(report: dict[str, Any], out_path: Path, markdown_path: Path) -> None:
    structure = report.get("structure_summary") or {}
    quality = report.get("quality_summary") or {}
    latency = report.get("latency_summary") or {}
    print("Dify RAG acceptance evaluation")
    if structure:
        print(f"- boundary accuracy: {structure.get('boundary_accuracy', 0):.2%}")
    if quality:
        print(f"- answer/citation: {quality.get('answer_accuracy', 0):.2%}/{quality.get('citation_accuracy', 0):.2%}")
    if latency:
        print(f"- latency avg/p95/max: {latency.get('average_seconds', 0)}s/{latency.get('p95_seconds', 0)}s/{latency.get('max_seconds', 0)}s")
    print(f"- gate: {'PASS' if report['gates']['passed'] else 'FAIL'} ({', '.join(report['gates']['failed']) or 'none'})")
    print(f"- json: {out_path}")
    print(f"- markdown: {markdown_path}")


def normalize_text(value: Any) -> str:
    text = CONTROL_RE.sub(" ", str(value or "")).lower()
    return "".join(TOKEN_RE.findall(text))


def clean_text(value: Any) -> str:
    text = CONTROL_RE.sub(" ", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def meaningful_tokens(value: Any) -> list[str]:
    return unique_keep_order(token.lower() for token in TOKEN_RE.findall(clean_text(value)) if len(token) >= 2)


def unique_keep_order(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def is_page_noise(text: str) -> bool:
    compact = normalize_text(text)
    return bool(re.fullmatch(r"\d{1,3}", compact)) or bool(re.fullmatch(r"\d{4}학년도모집요강\d{0,3}", compact))


def rate(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    return round(numerator / denominator, 4) if denominator else empty


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(quantile * len(ordered)) - 1)
    return round(ordered[rank], 3)


if __name__ == "__main__":
    raise SystemExit(main())
