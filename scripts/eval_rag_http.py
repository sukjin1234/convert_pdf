from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_rag_quality import (
    CaseResult,
    missing_needles,
    no_forbidden_values,
    print_human_report,
    summarize,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG quality through the running FastAPI HTTP API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", default=str(ROOT / "evaluation" / "rag_quality_cases.json"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--fail-under", type=float, default=0.95)
    parser.add_argument("--skip-ingest", action="store_true", help="Do not push evaluation markdown documents first.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--quiet", action="store_true", help="Only print summary unless failing.")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    cases_path = Path(args.cases)
    payload = json.loads(cases_path.read_text(encoding="utf-8"))

    health = get_json(base_url, "/health")
    if health.get("status") != "ok":
        raise RuntimeError(f"FastAPI health check failed: {health}")

    if not args.skip_ingest:
        for document in payload.get("documents", []):
            response = post_json(
                base_url,
                "/rag/ingest-markdown",
                {
                    "document_id": document["document_id"],
                    "file_name": document.get("file_name") or f"{document['document_id']}.md",
                    "markdown": document["markdown"],
                },
            )
            if not response.get("success"):
                raise RuntimeError(f"Failed to ingest {document['document_id']}: {response.get('error')}")

    results = [
        evaluate_case_http(case, base_url=base_url, limit=args.limit)
        for case in payload.get("cases", [])
    ]
    summary = summarize(results)

    output = {
        "base_url": base_url,
        "cases_path": str(cases_path),
        "summary": summary,
        "results": [result.to_dict() for result in results],
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print_human_report(results, summary, quiet=args.quiet)

    return 0 if summary["pass_rate"] >= args.fail_under else 1


def evaluate_case_http(case: dict[str, Any], *, base_url: str, limit: int) -> CaseResult:
    case_id = case["id"]
    document_id = case["document_id"]
    question = case["question"]
    answerable = bool(case.get("answerable", True))

    plan = post_json(base_url, "/query/plan", {"query": question, "document_id": document_id})
    expected_query_type = str(case.get("expected_query_type") or "")
    query_type_match = None
    if expected_query_type:
        query_type_match = plan.get("query_type") == expected_query_type

    lookup = post_json(
        base_url,
        "/lookup",
        {
            "query": question,
            "document_id": document_id,
            "entities": plan.get("entities") or [],
            "query_type": plan.get("query_type"),
            "limit": limit,
        },
    )
    context = lookup.get("context") or ""
    diagnostics = lookup.get("diagnostics") or {}
    top_score = float(diagnostics.get("top_score") or 0)
    selected_count = int(diagnostics.get("selected_count") or 0)

    required_terms = list(case.get("required_terms") or [])
    expected_values = list(case.get("expected_values") or [])
    forbidden_values = list(case.get("forbidden_values") or [])

    missing_terms = missing_needles(context, required_terms)
    missing_values = missing_needles(context, expected_values)
    evidence_recall = not missing_terms if answerable else True
    exact_value_match = not missing_values if answerable else no_forbidden_values(context, forbidden_values)
    answerable_detection = selected_count > 0 and evidence_recall if answerable else no_forbidden_values(context, forbidden_values)

    unsupported_guard = None
    unsupported_answer = case.get("unsupported_answer")
    if unsupported_answer:
        verification = post_json(
            base_url,
            "/answer/verify",
            {
                "question": question,
                "answer": str(unsupported_answer),
                "evidence": [context],
            },
        )
        unsupported_guard = not bool(verification.get("valid"))

    notes = []
    if query_type_match is False:
        notes.append(f"query_type expected={expected_query_type} actual={plan.get('query_type')}")
    if missing_terms:
        notes.append("missing required terms: " + ", ".join(missing_terms))
    if missing_values:
        notes.append("missing expected values: " + ", ".join(missing_values))
    if answerable and selected_count <= 0:
        notes.append("no evidence selected")
    if not answerable and not answerable_detection:
        notes.append("unanswerable case retrieved forbidden value")
    if unsupported_guard is False:
        notes.append("unsupported answer was not rejected by verifier")

    checks = [evidence_recall, exact_value_match, answerable_detection]
    if query_type_match is not None:
        checks.append(query_type_match)
    if unsupported_guard is not None:
        checks.append(unsupported_guard)

    return CaseResult(
        case_id=case_id,
        document_id=document_id,
        question=question,
        answerable=answerable,
        passed=all(checks),
        query_type=str(plan.get("query_type") or ""),
        expected_query_type=expected_query_type,
        query_type_match=query_type_match,
        evidence_recall=evidence_recall,
        exact_value_match=exact_value_match,
        answerable_detection=answerable_detection,
        unsupported_guard=unsupported_guard,
        top_score=top_score,
        selected_count=selected_count,
        missing_terms=missing_terms,
        missing_values=missing_values,
        notes=notes,
    )


def get_json(base_url: str, path: str) -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url}{path}", method="GET")
    return _send_json_request(request)


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    return _send_json_request(request)


def _send_json_request(request: urllib.request.Request) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {request.full_url}: {body}") from exc
    return json.loads(body) if body else {}


if __name__ == "__main__":
    raise SystemExit(main())
