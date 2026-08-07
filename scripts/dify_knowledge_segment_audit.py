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
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_NAME = "ipsi"
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SECTION_MARKER_RE = re.compile(r"(?m)^\s*\[section:\s*(.*?)\]\s*$")
PAGE_MARKER_RE = re.compile(r"(?m)^\s*\[page:\s*(.*?)\]\s*$")
SOURCE_SECTION_RE = re.compile(r"(?m)^\s*source_section:\s*(.*?)\s*$")
SOURCE_PAGE_RE = re.compile(r"(?m)^\s*source_page:\s*(.*?)\s*$")
MAJOR_OUTLINE_RE = re.compile(
    r"^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+|[IVXLCDM]+|제\s*\d+\s*장|chapter|part)\b",
    re.IGNORECASE,
)
SECTION_OUTLINE_RE = re.compile(r"^(?:\d+\s*[.．]\s+|제\s*\d+\s*(?:절|조)|section\b|\(\s*\d+\s*\)|\d+\s*\))", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Dify Knowledge document segments for PDF parsing metadata quality.")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--dataset-id", default="", help="Defaults to DIFY_KNOWLEDGE_DATASET_ID or DIFY_DATASET_ID.")
    parser.add_argument("--dataset-name", default=os.getenv("DIFY_DATASET_NAME", DEFAULT_DATASET_NAME))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--document-limit", type=int, default=0, help="0 means all documents.")
    parser.add_argument("--segment-limit", type=int, default=0, help="Maximum segments per document. 0 means all.")
    parser.add_argument("--fail-on-issues", action="store_true")
    parser.add_argument(
        "--max-issue-rate",
        type=float,
        default=-1.0,
        help="Fail if issue_rate exceeds this ratio. Use 0 for a strict zero-issue gate; negative disables.",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=-1,
        help="Fail if issue_count exceeds this value. Use 0 for a strict zero-issue gate; negative disables.",
    )
    parser.add_argument(
        "--max-failed-requests",
        type=int,
        default=-1,
        help="Fail if failed segment-list requests exceed this value. Use 0 for a strict request gate; negative disables.",
    )
    parser.add_argument("--out", default=str(ROOT / ".runtime" / "dify_knowledge_segment_audit_latest.json"))
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    base_url = (os.getenv("DIFY_API_BASE_URL") or "").rstrip("/")
    api_key = os.getenv("DIFY_KNOWLEDGE_API_KEY") or ""
    if not base_url or not api_key:
        print("Missing DIFY_API_BASE_URL or DIFY_KNOWLEDGE_API_KEY.", file=sys.stderr)
        return 2

    dataset_id = args.dataset_id or os.getenv("DIFY_KNOWLEDGE_DATASET_ID") or os.getenv("DIFY_DATASET_ID") or ""
    if not dataset_id:
        dataset_id = resolve_dataset_id(base_url, api_key, args.dataset_name, timeout=args.timeout)
    if not dataset_id:
        print("Knowledge dataset id was not found. Pass --dataset-id or set DIFY_KNOWLEDGE_DATASET_ID.", file=sys.stderr)
        return 2

    started = time.perf_counter()
    documents = list_documents(base_url, api_key, dataset_id, timeout=args.timeout)
    if args.document_limit > 0:
        documents = documents[: args.document_limit]

    audits = []
    for document in documents:
        document_id = str(document.get("id") or "")
        if not document_id:
            continue
        segments, fetch_error = fetch_all_segments(
            base_url,
            api_key,
            dataset_id,
            document_id,
            timeout=args.timeout,
            segment_limit=args.segment_limit,
        )
        audits.append(audit_document_segments(document, segments, fetch_error=fetch_error))

    summary = summarize_audits(audits)
    gate_failures = evaluate_gates(
        summary,
        fail_on_issues=args.fail_on_issues,
        max_issue_rate=args.max_issue_rate,
        max_issues=args.max_issues,
        max_failed_requests=args.max_failed_requests,
    )
    report = {
        "base_url_host": urllib.parse.urlparse(base_url).netloc,
        "dataset_id": dataset_id,
        "dataset_name": args.dataset_name,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "summary": summary,
        "documents": audits,
        "failed": bool(summary["failed_requests"] or summary["issue_count"]),
        "gate_failures": gate_failures,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report, out_path)
    return 1 if gate_failures else 0


def resolve_dataset_id(base_url: str, api_key: str, dataset_name: str, *, timeout: float) -> str:
    payload = request_json("GET", f"{base_url}/datasets?{urllib.parse.urlencode({'limit': 100})}", api_key, timeout=timeout)
    for item in payload_items(payload):
        if isinstance(item, dict) and str(item.get("name") or "").strip() == dataset_name:
            return str(item.get("id") or "")
    return ""


def list_documents(base_url: str, api_key: str, dataset_id: str, *, timeout: float) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"page": page, "limit": 100})
        payload = request_json("GET", f"{base_url}/datasets/{dataset_id}/documents?{query}", api_key, timeout=timeout)
        items = [item for item in payload_items(payload) if isinstance(item, dict)]
        documents.extend(items)
        total = safe_int(payload.get("total")) if isinstance(payload, dict) else 0
        if not items or len(items) < 100 or (total and len(documents) >= total):
            return documents
        page += 1


def fetch_all_segments(
    base_url: str,
    api_key: str,
    dataset_id: str,
    document_id: str,
    *,
    timeout: float,
    segment_limit: int = 0,
) -> tuple[list[dict[str, Any]], str]:
    segments: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"page": page, "limit": 100})
        try:
            payload = request_json(
                "GET",
                f"{base_url}/datasets/{dataset_id}/documents/{document_id}/segments?{query}",
                api_key,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            return segments, f"{type(exc).__name__}: {exc}"
        items = [item for item in payload_items(payload) if isinstance(item, dict)]
        segments.extend(items)
        if segment_limit > 0 and len(segments) >= segment_limit:
            return segments[:segment_limit], ""
        total = safe_int(payload.get("total")) if isinstance(payload, dict) else 0
        if not items or len(items) < 100 or (total and len(segments) >= total):
            return segments, ""
        page += 1


def audit_document_segments(document: dict[str, Any], segments: list[dict[str, Any]], *, fetch_error: str = "") -> dict[str, Any]:
    issues = []
    pages = set()
    sections = set()
    total_content_chars = 0
    for position, segment in enumerate(segments, start=1):
        content = segment_content(segment)
        total_content_chars += len(content)
        page = extract_page(content)
        section = extract_section(content)
        if page:
            pages.add(page)
        if section:
            sections.add(clean_metadata_text(section))
        reasons = segment_issue_reasons(content, section, page)
        if reasons:
            issues.append(
                {
                    "segment_id": segment.get("id"),
                    "position": segment.get("position") or segment.get("segment_position") or position,
                    "page": clean_metadata_text(page),
                    "section": clean_metadata_text(section),
                    "reasons": reasons,
                    "preview": compact_text(content, 240),
                }
            )

    reason_counts = Counter(reason for issue in issues for reason in issue["reasons"])
    return {
        "document_id": document.get("id"),
        "document_name": document.get("name") or document.get("document_name"),
        "display_status": document.get("display_status") or document.get("indexing_status"),
        "enabled": document.get("enabled"),
        "segment_count": len(segments),
        "avg_content_chars": round(total_content_chars / len(segments), 1) if segments else 0,
        "pages": len(pages),
        "sections": len(sections),
        "issue_count": len(issues),
        "reason_counts": dict(sorted(reason_counts.items())),
        "issue_samples": issues[:20],
        "fetch_error": fetch_error,
    }


def summarize_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    failed_requests = []
    for audit in audits:
        reason_counts.update(audit.get("reason_counts") or {})
        if audit.get("fetch_error"):
            failed_requests.append({"document_id": audit.get("document_id"), "error": audit.get("fetch_error")})
    issue_count = sum(safe_int(audit.get("issue_count")) for audit in audits)
    segment_count = sum(safe_int(audit.get("segment_count")) for audit in audits)
    return {
        "documents_checked": len(audits),
        "segments_checked": segment_count,
        "issue_count": issue_count,
        "issue_rate": round(issue_count / segment_count, 4) if segment_count else 0,
        "reason_counts": dict(sorted(reason_counts.items())),
        "failed_requests": failed_requests,
    }


def evaluate_gates(
    summary: dict[str, Any],
    *,
    fail_on_issues: bool,
    max_issue_rate: float,
    max_issues: int,
    max_failed_requests: int,
) -> list[str]:
    issue_count = safe_int(summary.get("issue_count"))
    issue_rate = float(summary.get("issue_rate") or 0)
    failed_request_count = len(summary.get("failed_requests") or [])
    failures: list[str] = []

    if fail_on_issues and (issue_count or failed_request_count):
        failures.append(
            f"audit contains {issue_count} issue(s) and {failed_request_count} failed request(s)"
        )
    if max_issue_rate >= 0 and issue_rate > max_issue_rate:
        failures.append(f"issue_rate {issue_rate:.2%} > {max_issue_rate:.2%}")
    if max_issues >= 0 and issue_count > max_issues:
        failures.append(f"issue_count {issue_count} > {max_issues}")
    if max_failed_requests >= 0 and failed_request_count > max_failed_requests:
        failures.append(f"failed_requests {failed_request_count} > {max_failed_requests}")
    return failures


def segment_issue_reasons(content: str, section: str, page: str) -> list[str]:
    reasons = []
    if CONTROL_CHAR_RE.search(content):
        reasons.append("control_character")
    if not content.strip():
        reasons.append("empty_content")
    if len(content.strip()) < 80:
        reasons.append("very_short_content")
    if not page:
        reasons.append("missing_page")
    reasons.extend(section_issue_reasons(section))
    return reasons


def section_issue_reasons(section: str) -> list[str]:
    cleaned = clean_metadata_text(section)
    reasons = []
    if not cleaned or cleaned in {"-", ">", "Untitled", "untitled"}:
        reasons.append("invalid_section")
        return reasons
    if CONTROL_CHAR_RE.search(section):
        reasons.append("control_character")
    if is_table_of_contents_section(cleaned):
        reasons.append("table_of_contents_section")
    if outline_depth_inversion(cleaned):
        reasons.append("outline_depth_inversion")
    if cleaned.startswith(">") or cleaned.endswith(">") or " > > " in f" {cleaned} ":
        reasons.append("invalid_section_separator")
    return reasons


def outline_depth_inversion(section: str) -> bool:
    depths = [outline_depth(part.strip()) for part in section.split(">") if part.strip()]
    depths = [depth for depth in depths if depth is not None]
    return any(right < left for left, right in zip(depths, depths[1:]))


def outline_depth(title: str) -> int | None:
    if MAJOR_OUTLINE_RE.search(title):
        return 1
    if SECTION_OUTLINE_RE.search(title):
        return 2
    return None


def extract_section(content: str) -> str:
    match = SECTION_MARKER_RE.search(content or "")
    if match:
        return match.group(1)
    match = SOURCE_SECTION_RE.search(content or "")
    return match.group(1) if match else ""


def extract_page(content: str) -> str:
    match = PAGE_MARKER_RE.search(content or "")
    if match:
        return match.group(1)
    match = SOURCE_PAGE_RE.search(content or "")
    return match.group(1) if match else ""


def segment_content(segment: dict[str, Any]) -> str:
    for key in ("content", "text", "answer"):
        value = segment.get(key)
        if value:
            return str(value)
    nested = segment.get("segment") if isinstance(segment.get("segment"), dict) else {}
    return str(nested.get("content") or "")


def is_table_of_contents_section(section: str) -> bool:
    compact = re.sub(r"\s+", "", section).lower()
    return compact in {"목차", "contents", "content", "tableofcontents"}


def clean_metadata_text(value: Any) -> str:
    text = str(value or "")
    text = CONTROL_CHAR_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*>\s*", " > ", text)
    return text.strip()


def compact_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def request_json(method: str, url: str, api_key: str, *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {text[:1000]}") from exc


def payload_items(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        value = payload.get("data")
        if isinstance(value, list):
            return value
        for key in ("records", "documents"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    if isinstance(payload, list):
        return payload
    return []


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


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


def print_human(report: dict[str, Any], out_path: Path) -> None:
    summary = report.get("summary") or {}
    print("Dify Knowledge segment audit")
    print(f"- dataset_id: {report.get('dataset_id')}")
    print(
        f"- documents/segments/issues: {summary.get('documents_checked')}/"
        f"{summary.get('segments_checked')}/{summary.get('issue_count')}"
    )
    print(f"- issue_rate: {summary.get('issue_rate'):.2%}")
    print(f"- reason_counts: {summary.get('reason_counts')}")
    print(f"- failed_requests: {len(summary.get('failed_requests') or [])}")
    if report.get("gate_failures"):
        print("- gate_failures:")
        for failure in report["gate_failures"]:
            print(f"  - {failure}")
    print(f"- saved: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
