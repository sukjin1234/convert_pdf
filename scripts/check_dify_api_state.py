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
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IPSI_DATASET_ID = "0fd367d8-5003-440a-92bc-e256784f8507"
SOURCE_MARKER = "\ucc38\uc870 \ubb38\uc11c:"
DEFAULT_KNOWLEDGE_QUERIES = [
    "\ucd1d \uc7a5\ud559\uae08 \uc5bc\ub9c8\uc57c",
    "\uc218\uc2dc1\ucc28 \uc6d0\uc11c\uc811\uc218 \uae30\uac04 \uc54c\ub824\uc918",
    "\uacf5\uac04\uc815\ubcf4 \ud2b9\uc131\ud654 \uc804\ubb38\ub300\ud559\uc740 \uc5b4\ub290 \uae30\uad00\uacfc \uad00\ub828 \uc788\uc5b4?",
]
DEFAULT_KEYWORDS = [
    "\uc7a5\ud559\uae08",
    "\uc6d0\uc11c\uc811\uc218",
    "\uad6d\ud1a0\uad50\ud1b5\ubd80",
]
DEFAULT_SESSION_QUESTIONS = [
    "\ucd1d \uc7a5\ud559\uae08 \uc5bc\ub9c8\uc57c",
    "\ubc29\uae08 \ub2f5\ubcc0\uc5d0\uc11c \ucc38\uc870\ud55c \ubb38\uc11c \uc774\ub984\ub9cc \ub9d0\ud574\uc918",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Dify Knowledge document health, retrieval, and Chatflow session continuity."
    )
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--dataset-name", default="ipsi")
    parser.add_argument(
        "--dataset-id",
        default="",
        help="Knowledge dataset id. Defaults to DIFY_DATASET_ID/DIFY_KNOWLEDGE_DATASET_ID, then the project ipsi id.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    parser.add_argument(
        "--sections-only",
        action="store_true",
        help="Audit section/page metadata without retrieval or Chatflow checks.",
    )
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    base_url = (os.getenv("DIFY_API_BASE_URL") or "").rstrip("/")
    app_key = os.getenv("DIFY_APP_API_KEY") or ""
    knowledge_key = os.getenv("DIFY_KNOWLEDGE_API_KEY") or ""
    if not base_url or not app_key or not knowledge_key:
        print("Missing DIFY_API_BASE_URL, DIFY_APP_API_KEY, or DIFY_KNOWLEDGE_API_KEY.", file=sys.stderr)
        return 2

    dataset_id = (
        args.dataset_id
        or os.getenv("DIFY_DATASET_ID")
        or os.getenv("DIFY_KNOWLEDGE_DATASET_ID")
        or DEFAULT_IPSI_DATASET_ID
    )

    report: dict[str, Any] = {
        "base_url_host": urllib.parse.urlparse(base_url).netloc,
        "dataset": {},
        "documents": [],
        "section_metadata_audit": {},
        "keyword_segments": [],
        "retrieve_checks": [],
        "session_check": {},
    }

    datasets_status, datasets_payload, datasets_error = request_json(
        "GET", f"{base_url}/datasets?limit=100", knowledge_key, timeout=args.timeout
    )
    datasets = payload_items(datasets_payload)
    for dataset in datasets:
        if isinstance(dataset, dict) and str(dataset.get("name") or "").strip() == args.dataset_name:
            dataset_id = str(dataset.get("id") or dataset_id)
            break

    dataset_status, dataset_payload, dataset_error = request_json(
        "GET", f"{base_url}/datasets/{dataset_id}", knowledge_key, timeout=args.timeout
    )
    report["dataset"] = {
        "id": dataset_id,
        "datasets_list_status": datasets_status,
        "datasets_list_error": datasets_error,
        "status": dataset_status,
        "error": dataset_error,
        "name": value_at(dataset_payload, "name"),
        "document_count": value_at(dataset_payload, "document_count"),
        "word_count": value_at(dataset_payload, "word_count"),
    }

    documents_status, documents_payload, documents_error = request_json(
        "GET", f"{base_url}/datasets/{dataset_id}/documents?limit=100", knowledge_key, timeout=args.timeout
    )
    report["documents_status"] = documents_status
    report["documents_error"] = documents_error
    documents = [item for item in payload_items(documents_payload) if isinstance(item, dict)]

    for document in documents:
        document_id = str(document.get("id") or "")
        seg_status, seg_payload, seg_error = request_json(
            "GET",
            f"{base_url}/datasets/{dataset_id}/documents/{document_id}/segments?limit=1",
            knowledge_key,
            timeout=args.timeout,
        )
        report["documents"].append(
            {
                "id": document_id,
                "name": document.get("name"),
                "indexing_status": document.get("indexing_status"),
                "display_status": document.get("display_status"),
                "enabled": document.get("enabled"),
                "archived": document.get("archived"),
                "word_count": document.get("word_count"),
                "doc_form": document.get("doc_form"),
                "stale_error": compact_text(document.get("error") or "", 260),
                "segments_status": seg_status,
                "segments_error": seg_error,
                "segments_total": value_at(seg_payload, "total"),
            }
        )

    section_document_audits = []
    for document in documents:
        if not document.get("enabled") or document.get("archived"):
            continue
        segments, status, error = fetch_all_segments(
            base_url,
            knowledge_key,
            dataset_id,
            str(document.get("id") or ""),
            args.timeout,
        )
        section_document_audits.append(
            audit_segment_sections(
                str(document.get("name") or document.get("id") or ""),
                segments,
                status=status,
                error=error,
            )
        )
    report["section_metadata_audit"] = summarize_section_audits(section_document_audits)

    if args.sections_only:
        section_audit = report["section_metadata_audit"]
        report["health"] = {
            "section_metadata_issues": section_audit.get("issue_count"),
            "section_audit_failed_requests": section_audit.get("failed_requests"),
            "failed": bool(section_audit.get("issue_count") or section_audit.get("failed_requests")),
        }
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print_section_summary(report)
        return 1 if report["health"]["failed"] else 0

    for keyword in DEFAULT_KEYWORDS:
        for document in documents:
            document_id = str(document.get("id") or "")
            query = urllib.parse.urlencode({"limit": 3, "keyword": keyword})
            status, payload, error = request_json(
                "GET",
                f"{base_url}/datasets/{dataset_id}/documents/{document_id}/segments?{query}",
                knowledge_key,
                timeout=args.timeout,
            )
            samples = []
            for segment in payload_items(payload)[:3]:
                if not isinstance(segment, dict):
                    continue
                content = str(segment.get("content") or "")
                samples.append(
                    {
                        "segment_id": segment.get("id"),
                        "section": extract_section(content),
                        "word_count": segment.get("word_count"),
                        "answer_hints": extract_answer_hints(content),
                    }
                )
            report["keyword_segments"].append(
                {
                    "keyword": keyword,
                    "document_name": document.get("name"),
                    "status": status,
                    "error": error,
                    "total": value_at(payload, "total"),
                    "samples": samples,
                }
            )

    for query in DEFAULT_KNOWLEDGE_QUERIES:
        status, payload, error = request_json(
            "POST",
            f"{base_url}/datasets/{dataset_id}/retrieve",
            knowledge_key,
            body={"query": query},
            timeout=args.timeout,
        )
        report["retrieve_checks"].append(
            {
                "query": query,
                "status": status,
                "error": error,
                "message": value_at(payload, "message"),
                "items": retrieval_items(payload),
            }
        )

    report["session_check"] = check_chatflow_session(base_url, app_key, args.timeout)
    report["health"] = summarize_health(report)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_summary(report)

    failed = has_failed_health(report)
    return 1 if failed else 0


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def request_json(
    method: str,
    url: str,
    api_key: str,
    *,
    body: Any | None = None,
    timeout: float,
) -> tuple[int | None, Any, str]:
    data = None
    headers = {"Authorization": f"Bearer {api_key}"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(text) if text else {}, ""
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            payload: Any = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = text
        return exc.code, payload, ""
    except Exception as exc:  # noqa: BLE001 - diagnostic CLI should keep going.
        return None, None, f"{type(exc).__name__}: {exc}"


def fetch_all_segments(
    base_url: str,
    api_key: str,
    dataset_id: str,
    document_id: str,
    timeout: float,
) -> tuple[list[dict[str, Any]], int | None, str]:
    segments: list[dict[str, Any]] = []
    page = 1
    last_status: int | None = None
    while document_id:
        query = urllib.parse.urlencode({"page": page, "limit": 100})
        last_status, payload, error = request_json(
            "GET",
            f"{base_url}/datasets/{dataset_id}/documents/{document_id}/segments?{query}",
            api_key,
            timeout=timeout,
        )
        if error or last_status != 200:
            return segments, last_status, error or compact_text(value_at(payload, "message") or payload)
        items = [item for item in payload_items(payload) if isinstance(item, dict)]
        segments.extend(items)
        total = safe_int(value_at(payload, "total"))
        if not items or (total > 0 and len(segments) >= total) or len(items) < 100:
            break
        page += 1
    return segments, last_status, ""


def audit_segment_sections(
    document_name: str,
    segments: list[dict[str, Any]],
    *,
    status: int | None,
    error: str,
) -> dict[str, Any]:
    issues = []
    pages: set[str] = set()
    sections: set[str] = set()
    for segment in segments:
        content = str(segment.get("content") or "")
        page = extract_metadata_value(content, "page")
        section = extract_section(content)
        if page:
            pages.add(page)
        if section:
            sections.add(clean_metadata_text(section))

        reasons = section_issue_reasons(section)
        if not page:
            reasons.append("missing_page")
        if reasons:
            issues.append(
                {
                    "segment_id": segment.get("id"),
                    "page": page,
                    "section": clean_metadata_text(section),
                    "reasons": sorted(set(reasons)),
                    "content_preview": compact_text(content, 240),
                }
            )

    reason_counts: dict[str, int] = {}
    for issue in issues:
        for reason in issue.get("reasons") or []:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "document_name": document_name,
        "status": status,
        "error": error,
        "segments": len(segments),
        "pages": len(pages),
        "sections": len(sections),
        "issue_count": len(issues),
        "reason_counts": reason_counts,
        "issue_samples": issues[:20],
    }


def summarize_section_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "documents": audits,
        "documents_checked": len(audits),
        "segments_checked": sum(safe_int(item.get("segments")) for item in audits),
        "issue_count": sum(safe_int(item.get("issue_count")) for item in audits),
        "failed_requests": [
            item.get("document_name")
            for item in audits
            if item.get("status") != 200 or item.get("error")
        ],
    }


def extract_metadata_value(content: str, key: str) -> str:
    match = re.search(rf"(?m)^\[{re.escape(key)}:\s*(.*?)\]\s*$", content or "")
    if match:
        return clean_metadata_text(match.group(1))
    fallback_key = {"page": "source_page", "section": "source_section"}.get(key)
    if fallback_key:
        match = re.search(rf"(?m)^\s*{fallback_key}:\s*(.*?)\s*$", content or "")
    return clean_metadata_text(match.group(1)) if match else ""


def clean_metadata_text(value: Any) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def section_issue_reasons(section: str) -> list[str]:
    reasons = []
    cleaned = clean_metadata_text(section)
    compact = re.sub(r"[\s>]+", "", cleaned).lower()
    if not cleaned or compact in {"", "untitled", "\ubaa9\ucc28", "contents", "content"}:
        reasons.append("invalid_section")
    if re.search(r"[\x00-\x1f\x7f]", section or ""):
        reasons.append("control_character")

    previous_depth: int | None = None
    for part in (clean_metadata_text(item) for item in cleaned.split(">")):
        depth = section_outline_depth(part)
        if depth is None:
            continue
        if previous_depth is not None and depth <= previous_depth:
            reasons.append("non_monotonic_outline")
            break
        previous_depth = depth
    return reasons


def section_outline_depth(title: str) -> int | None:
    text = clean_metadata_text(title)
    if re.match(
        r"^(?:[\u2160-\u216b]+|[IVXLCDM]+|(?:chapter|part)\s+(?:\d+|[IVXLCDM]+)|\uc81c\s*\d+\s*\uc7a5)"
        r"\s*(?:[.\uff0e\u3001:\uff1a)\]-]|\s)",
        text,
        re.IGNORECASE,
    ):
        return 1
    decimal = re.match(r"^(?P<number>\d+(?:\.\d+)+)\s*(?:[.\uff0e\u3001:\uff1a)\]-]|\s)", text)
    if decimal:
        return min(len(decimal.group("number").split(".")) + 1, 6)
    if re.match(
        r"^(?:(?:section|\uc808)\s*(?:\d+|[IVXLCDM]+)|\uc81c\s*\d+\s*(?:\uc808|\uc870)|\d+\s*[.\uff0e]\s+)",
        text,
        re.IGNORECASE,
    ):
        return 2
    if re.match(r"^(?:\(\s*\d+\s*\)|\d+\s*\))\s*", text):
        return 3
    return None


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def payload_items(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        for key in ("data", "records", "documents"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    if isinstance(payload, list):
        return payload
    return []


def value_at(payload: Any, key: str) -> Any:
    return payload.get(key) if isinstance(payload, dict) else None


def compact_text(value: Any, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit] + "..."


def extract_section(content: str) -> str:
    marker = "[section: "
    start = content.find(marker)
    if start >= 0:
        start += len(marker)
        end = content.find("]", start)
        if end >= 0:
            return content[start:end].strip()
    match = re.search(r"(?m)^\s*source_section:\s*(.*?)\s*$", content or "")
    return match.group(1).strip() if match else ""


def extract_answer_hints(content: str) -> list[str]:
    hints: list[str] = []
    stop_markers = [
        "\nrecord_type:",
        "\nsource_file:",
        "\n- structured_record:",
        "\n\uc81c\ubaa9:",
        "\n\ub0b4\uc6a9:",
        "\n\uad6c\ubd84:",
        "\n\ud56d\ubaa9:",
        "\n\ubaa8\uc9d1\ub2e8\uc704:",
        "\n\uc804\ud615\uad6c\ubd84:",
    ]
    for part in content.split("answer_hint:")[1:]:
        end = len(part)
        for marker in stop_markers:
            pos = part.find(marker)
            if 0 <= pos < end:
                end = pos
        hint = compact_text(part[:end], 420)
        if hint:
            hints.append(hint)
    return hints[:5]


def retrieval_items(payload: Any) -> list[dict[str, Any]]:
    items = []
    for item in payload_items(payload)[:5]:
        if not isinstance(item, dict):
            continue
        segment = item.get("segment") if isinstance(item.get("segment"), dict) else item
        document = segment.get("document") if isinstance(segment, dict) and isinstance(segment.get("document"), dict) else {}
        items.append(
            {
                "score": item.get("score") or segment.get("score"),
                "document_name": document.get("name") or segment.get("document_name"),
                "document_id": document.get("id") or segment.get("document_id"),
                "segment_id": segment.get("id"),
                "content_preview": compact_text(segment.get("content") or item.get("content") or "", 260),
            }
        )
    return items


def check_chatflow_session(base_url: str, api_key: str, timeout: float) -> dict[str, Any]:
    user = f"codex-session-check-{int(time.time())}"
    conversation_id = ""
    checks = []
    for query in DEFAULT_SESSION_QUESTIONS:
        status, payload, error = request_json(
            "POST",
            f"{base_url}/chat-messages",
            api_key,
            body={
                "inputs": {},
                "query": query,
                "response_mode": "blocking",
                "conversation_id": conversation_id,
                "user": user,
            },
            timeout=timeout,
        )
        answer = str(value_at(payload, "answer") or "")
        received_conversation_id = str(value_at(payload, "conversation_id") or "")
        is_source_followup = is_source_name_followup(query)
        checks.append(
            {
                "query": query,
                "status": status,
                "error": error,
                "message": value_at(payload, "message"),
                "conversation_id_sent": conversation_id,
                "conversation_id_received": received_conversation_id,
                "message_id": value_at(payload, "message_id"),
                "answer_preview": compact_text(answer, 500),
                "is_source_followup": is_source_followup,
                "has_source_summary": SOURCE_MARKER in answer or (
                    is_source_followup and has_source_name_answer(answer)
                ),
            }
        )
        if received_conversation_id:
            conversation_id = received_conversation_id

    history: dict[str, Any] = {}
    variables: dict[str, Any] = {}
    if conversation_id:
        query = urllib.parse.urlencode({"user": user, "conversation_id": conversation_id, "limit": 10})
        status, payload, error = request_json("GET", f"{base_url}/messages?{query}", api_key, timeout=timeout)
        history = {
            "status": status,
            "error": error,
            "message": value_at(payload, "message"),
            "count": len(payload_items(payload)),
        }
        query = urllib.parse.urlencode({"user": user, "limit": 20})
        status, payload, error = request_json(
            "GET", f"{base_url}/conversations/{conversation_id}/variables?{query}", api_key, timeout=timeout
        )
        variables = {
            "status": status,
            "error": error,
            "message": value_at(payload, "message"),
            "count": len(payload_items(payload)),
        }

    return {
        "user": user,
        "checks": checks,
        "conversation_reused": bool(
            len(checks) >= 2
            and checks[0].get("conversation_id_received")
            and checks[1].get("conversation_id_sent") == checks[0].get("conversation_id_received")
            and checks[1].get("conversation_id_received") == checks[0].get("conversation_id_received")
        ),
        "history": history,
        "variables": variables,
    }


def is_source_name_followup(query: str) -> bool:
    return bool(
        re.search(
            r"\ucc38\uc870\ud55c?\s*\ubb38\uc11c|\ubb38\uc11c\s*\uc774\ub984|\ucd9c\ucc98\s*\ubb38\uc11c|source\s*(?:name|document)",
            query,
            re.IGNORECASE,
        )
    )


def has_source_name_answer(answer: str) -> bool:
    text = compact_text(answer, 500)
    if not text:
        return False
    return bool(re.search(r"\.pdf|\ucc38\uc870|\ucd9c\ucc98|\ubb38\uc11c|\uadfc\uac70", text, re.IGNORECASE))


def has_failed_health(report: dict[str, Any]) -> bool:
    return bool(summarize_health(report).get("failed"))


def summarize_health(report: dict[str, Any]) -> dict[str, Any]:
    documents = report.get("documents") or []
    if not documents:
        return {
            "documents_total": 0,
            "documents_enabled": 0,
            "documents_disabled": 0,
            "unhealthy_enabled_documents": [],
            "session_conversation_reused": False,
            "failed": True,
        }

    enabled_documents = [
        document for document in documents if document.get("enabled") and not document.get("archived")
    ]
    unhealthy_enabled_documents = [
        str(document.get("name") or document.get("id") or "")
        for document in enabled_documents
        if not is_usable_document(document)
    ]
    session = report.get("session_check") or {}
    conversation_reused = bool(session.get("conversation_reused"))
    section_audit = report.get("section_metadata_audit") or {}
    section_issue_count = safe_int(section_audit.get("issue_count"))
    section_failed_requests = section_audit.get("failed_requests") or []
    failed = (
        not enabled_documents
        or bool(unhealthy_enabled_documents)
        or not conversation_reused
        or section_issue_count > 0
        or bool(section_failed_requests)
    )
    return {
        "documents_total": len(documents),
        "documents_enabled": len(enabled_documents),
        "documents_disabled": len(documents) - len(enabled_documents),
        "unhealthy_enabled_documents": unhealthy_enabled_documents,
        "section_metadata_issues": section_issue_count,
        "section_audit_failed_requests": section_failed_requests,
        "session_conversation_reused": conversation_reused,
        "failed": failed,
    }


def is_usable_document(document: dict[str, Any]) -> bool:
    if document.get("indexing_status") != "completed" or document.get("display_status") != "available":
        return False
    try:
        return int(document.get("segments_total") or 0) > 0
    except (TypeError, ValueError):
        return False


def print_summary(report: dict[str, Any]) -> None:
    dataset = report.get("dataset") or {}
    health = report.get("health") or summarize_health(report)
    print(f"Dataset: {dataset.get('name')} ({dataset.get('id')})")
    print(
        f"Documents: {health.get('documents_total')}, enabled={health.get('documents_enabled')}, "
        f"disabled={health.get('documents_disabled')}, words={dataset.get('word_count')}"
    )
    for document in report.get("documents") or []:
        stale = " stale_error=present" if document.get("stale_error") else ""
        state = "enabled" if document.get("enabled") and not document.get("archived") else "disabled"
        print(
            f"- {document.get('name')}: {state}, {document.get('indexing_status')}/"
            f"{document.get('display_status')}, segments={document.get('segments_total')}, "
            f"words={document.get('word_count')}{stale}"
        )
    section_audit = report.get("section_metadata_audit") or {}
    print(
        "Section metadata: "
        f"segments={section_audit.get('segments_checked')}, "
        f"issues={section_audit.get('issue_count')}, "
        f"failed_requests={len(section_audit.get('failed_requests') or [])}"
    )
    for audit in section_audit.get("documents") or []:
        if not audit.get("issue_count") and not audit.get("error"):
            continue
        print(
            f"- {audit.get('document_name')}: segments={audit.get('segments')}, "
            f"pages={audit.get('pages')}, sections={audit.get('sections')}, "
            f"issues={audit.get('issue_count')}, reasons={audit.get('reason_counts') or {}}, "
            f"error={audit.get('error') or '-'}"
        )
        for issue in (audit.get("issue_samples") or [])[:5]:
            print(
                f"  page={issue.get('page') or '-'} section={issue.get('section') or '-'} "
                f"reasons={','.join(issue.get('reasons') or [])}"
            )
    print("Retrieval:")
    for check in report.get("retrieve_checks") or []:
        top = (check.get("items") or [{}])[0]
        print(
            f"- {check.get('query')}: status={check.get('status')}, "
            f"top_doc={top.get('document_name')}, score={top.get('score')}"
        )
    session = report.get("session_check") or {}
    print(
        "Session: "
        f"conversation_reused={session.get('conversation_reused')}, "
        f"history_count={value_at(session.get('history'), 'count')}, "
        f"conversation_variables={value_at(session.get('variables'), 'count')}"
    )
    for check in session.get("checks") or []:
        print(
            f"- chat status={check.get('status')} sent={bool(check.get('conversation_id_sent'))} "
            f"received={bool(check.get('conversation_id_received'))} source_line={check.get('has_source_summary')} "
            f"answer={check.get('answer_preview')}"
        )
    if health.get("unhealthy_enabled_documents"):
        print(f"Unhealthy enabled documents: {', '.join(health['unhealthy_enabled_documents'])}")


def print_section_summary(report: dict[str, Any]) -> None:
    dataset = report.get("dataset") or {}
    section_audit = report.get("section_metadata_audit") or {}
    print(f"Dataset: {dataset.get('name')} ({dataset.get('id')})")
    print(
        "Section metadata: "
        f"documents={section_audit.get('documents_checked')}, "
        f"segments={section_audit.get('segments_checked')}, "
        f"issues={section_audit.get('issue_count')}, "
        f"failed_requests={len(section_audit.get('failed_requests') or [])}"
    )
    for audit in section_audit.get("documents") or []:
        print(
            f"- {audit.get('document_name')}: segments={audit.get('segments')}, "
            f"pages={audit.get('pages')}, sections={audit.get('sections')}, "
            f"issues={audit.get('issue_count')}, reasons={audit.get('reason_counts') or {}}, "
            f"error={audit.get('error') or '-'}"
        )
        for issue in (audit.get("issue_samples") or [])[:5]:
            print(
                f"  page={issue.get('page') or '-'} section={issue.get('section') or '-'} "
                f"reasons={','.join(issue.get('reasons') or [])} "
                f"content={issue.get('content_preview') or '-'}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
