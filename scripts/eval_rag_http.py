from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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


@dataclass
class TimedCaseResult:
    index: int
    result: CaseResult
    latency_seconds: float
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = self.result.to_dict()
        payload["index"] = self.index
        payload["latency_seconds"] = self.latency_seconds
        payload["error"] = self.error
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG quality through the running FastAPI HTTP API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", default=str(ROOT / "evaluation" / "rag_quality_cases.json"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--fail-under", type=float, default=0.95)
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the case set N times for latency/concurrency smoke tests.")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent HTTP evaluation workers.")
    parser.add_argument("--max-p95-latency", type=float, default=0.0, help="Fail if p95 case latency exceeds this many seconds. 0 disables.")
    parser.add_argument("--max-avg-latency", type=float, default=0.0, help="Fail if average case latency exceeds this many seconds. 0 disables.")
    parser.add_argument("--skip-ingest", action="store_true", help="Do not push evaluation markdown documents first.")
    parser.add_argument("--out", default="", help="Optional path for a JSON report.")
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

    tasks = build_case_tasks(payload.get("cases", []), repeat=args.repeat)
    timed_results = evaluate_cases_http_concurrently(
        tasks,
        base_url=base_url,
        limit=args.limit,
        concurrency=args.concurrency,
    )
    results = [item.result for item in timed_results]
    summary = summarize(results)
    latency_summary = summarize_latency(timed_results)
    latency_gate_failures = evaluate_latency_gates(
        latency_summary,
        max_p95_latency=args.max_p95_latency,
        max_avg_latency=args.max_avg_latency,
    )

    output = {
        "base_url": base_url,
        "cases_path": str(cases_path),
        "repeat": max(1, args.repeat),
        "concurrency": max(1, args.concurrency),
        "summary": summary,
        "latency": latency_summary,
        "gate_failures": latency_gate_failures,
        "results": [result.to_dict() for result in timed_results],
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print_human_report(results, summary, quiet=args.quiet)
        print_latency_report(latency_summary, latency_gate_failures, Path(args.out) if args.out else None)

    return 0 if summary["pass_rate"] >= args.fail_under and not latency_gate_failures else 1


def build_case_tasks(cases: list[dict[str, Any]], *, repeat: int) -> list[tuple[int, dict[str, Any]]]:
    tasks = []
    repeat = max(1, repeat)
    for iteration in range(1, repeat + 1):
        for case in cases:
            case_copy = dict(case)
            case_copy["_iteration"] = iteration
            tasks.append((len(tasks) + 1, case_copy))
    return tasks


def evaluate_cases_http_concurrently(
    tasks: list[tuple[int, dict[str, Any]]],
    *,
    base_url: str,
    limit: int,
    concurrency: int,
) -> list[TimedCaseResult]:
    if not tasks:
        return []
    results: list[TimedCaseResult] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [
            executor.submit(evaluate_case_http_timed, index, case, base_url=base_url, limit=limit)
            for index, case in tasks
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item.index)


def evaluate_case_http_timed(index: int, case: dict[str, Any], *, base_url: str, limit: int) -> TimedCaseResult:
    started = time.perf_counter()
    try:
        result = evaluate_case_http(case, base_url=base_url, limit=limit)
        return TimedCaseResult(index=index, result=result, latency_seconds=round(time.perf_counter() - started, 6))
    except Exception as exc:  # noqa: BLE001 - report per-case HTTP failures in the JSON output.
        latency = round(time.perf_counter() - started, 6)
        error = f"{type(exc).__name__}: {exc}"
        return TimedCaseResult(index=index, result=failed_case_result(case, error), latency_seconds=latency, error=error)


def failed_case_result(case: dict[str, Any], error: str) -> CaseResult:
    return CaseResult(
        case_id=str(case.get("id") or f"case-{case.get('_iteration', 1)}"),
        document_id=str(case.get("document_id") or ""),
        question=str(case.get("question") or ""),
        answerable=bool(case.get("answerable", True)),
        passed=False,
        query_type="",
        expected_query_type=str(case.get("expected_query_type") or ""),
        query_type_match=None,
        evidence_recall=False,
        exact_value_match=False,
        answerable_detection=False,
        unsupported_guard=None,
        top_score=0,
        selected_count=0,
        missing_terms=[],
        missing_values=[],
        notes=[error],
    )


def summarize_latency(results: list[TimedCaseResult]) -> dict[str, Any]:
    latencies = sorted(result.latency_seconds for result in results)
    successes = [result for result in results if result.result.passed and not result.error]
    return {
        "samples": len(latencies),
        "successes": len(successes),
        "avg_seconds": round(sum(latencies) / len(latencies), 6) if latencies else 0,
        "p50_seconds": percentile(latencies, 50),
        "p95_seconds": percentile(latencies, 95),
        "max_seconds": max(latencies) if latencies else 0,
    }


def percentile(values: list[float], percentile_value: int) -> float:
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile_value / 100
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return round(values[lower] * (1 - weight) + values[upper] * weight, 6)


def evaluate_latency_gates(
    latency_summary: dict[str, Any],
    *,
    max_p95_latency: float,
    max_avg_latency: float,
) -> list[str]:
    failures = []
    if max_p95_latency > 0 and latency_summary["p95_seconds"] > max_p95_latency:
        failures.append(f"p95_seconds {latency_summary['p95_seconds']:.3f}s > {max_p95_latency:.3f}s")
    if max_avg_latency > 0 and latency_summary["avg_seconds"] > max_avg_latency:
        failures.append(f"avg_seconds {latency_summary['avg_seconds']:.3f}s > {max_avg_latency:.3f}s")
    return failures


def print_latency_report(summary: dict[str, Any], failures: list[str], out_path: Path | None) -> None:
    print()
    print("HTTP latency")
    print(
        f"samples={summary['samples']} avg={summary['avg_seconds']:.3f}s "
        f"p50={summary['p50_seconds']:.3f}s p95={summary['p95_seconds']:.3f}s "
        f"max={summary['max_seconds']:.3f}s"
    )
    if failures:
        print("latency gate failures:")
        for failure in failures:
            print(f"  - {failure}")
    if out_path:
        print(f"saved={out_path}")


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
    exact_value_match = (not missing_values and no_forbidden_values(context, forbidden_values)) if answerable else no_forbidden_values(context, forbidden_values)
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
