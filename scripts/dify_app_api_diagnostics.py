from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY = "\uc218\uc2dc1\ucc28 \uc6d0\uc11c\uc811\uc218 \uae30\uac04 \uc54c\ub824\uc918"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect deployed Dify App API capabilities and Chatflow streaming RAG signals."
    )
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--inputs-json", default="{}")
    parser.add_argument("--inputs-file", default="", help="Path to a JSON object used as Dify inputs.")
    parser.add_argument("--document-id", default="", help="Convenience value added to inputs.document_id.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--user-prefix", default="dify-diag")
    parser.add_argument("--app-id", default="", help="Optional Dify app id for console read probes.")
    parser.add_argument("--skip-stream", action="store_true")
    parser.add_argument("--skip-console-probe", action="store_true")
    parser.add_argument(
        "--fail-on-stale-deployment",
        action="store_true",
        help="Exit 1 unless deployed Query Plan and Structured Lookup expose the current response contract.",
    )
    parser.add_argument("--out", default=str(ROOT / ".runtime" / "dify_app_api_diagnostics_latest.json"))
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    base_url = (os.getenv("DIFY_API_BASE_URL") or "").rstrip("/")
    api_key = os.getenv("DIFY_APP_API_KEY") or os.getenv("DIFY_API_KEY") or ""
    if not base_url or not api_key:
        print("Missing DIFY_API_BASE_URL or DIFY_APP_API_KEY.", file=sys.stderr)
        return 2

    inputs = load_inputs(args.inputs_json, args.inputs_file)
    if args.document_id:
        inputs["document_id"] = args.document_id
    report: dict[str, Any] = {
        "base_url_host": urllib.parse.urlparse(base_url).netloc,
        "app_api": probe_app_api(base_url, api_key, timeout=min(args.timeout, 30.0)),
        "stream": {},
        "console_probe": {},
    }

    app_id = args.app_id
    if not args.skip_stream:
        user = f"{args.user_prefix}-{int(time.time() * 1000)}"
        events, elapsed = stream_chat_message(
            base_url=base_url,
            api_key=api_key,
            query=args.query,
            inputs=inputs,
            user=user,
            timeout=args.timeout,
        )
        summary = summarize_stream_events(events)
        app_id = app_id or str(nested_get(summary, "workflow_inputs", "sys.app_id") or "")
        report["stream"] = {
            "elapsed_seconds": round(elapsed, 3),
            "event_count": len(events),
            "summary": summary,
            "events": events,
        }

    if not args.skip_console_probe:
        report["console_probe"] = probe_console_api(
            base_url=base_url,
            api_key=api_key,
            app_id=app_id,
            timeout=min(args.timeout, 30.0),
        )

    report["workflow_edit_available"] = workflow_edit_available(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report, out_path)
    contract = nested_get(report, "stream", "summary", "deployment_contract")
    if args.fail_on_stale_deployment and isinstance(contract, dict) and not contract.get("ready"):
        return 1
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


def parse_json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("--inputs-json must be a JSON object.")
    return parsed


def load_inputs(inputs_json: str, inputs_file: str) -> dict[str, Any]:
    if inputs_file:
        return parse_json_object(Path(inputs_file).read_text(encoding="utf-8"))
    return parse_json_object(inputs_json)


def probe_app_api(base_url: str, api_key: str, *, timeout: float) -> dict[str, Any]:
    probes = {}
    for name, path in {
        "info": "/info",
        "parameters": "/parameters",
        "meta": "/meta",
    }.items():
        status, payload = request_json_or_text("GET", f"{base_url}{path}", api_key, timeout=timeout)
        probes[name] = summarize_probe_response(status, payload)
    return probes


def probe_console_api(base_url: str, api_key: str, *, app_id: str = "", timeout: float) -> dict[str, Any]:
    origin = console_origin(base_url)
    paths = ["/console/api/apps"]
    if app_id:
        paths.extend(
            [
                f"/console/api/apps/{app_id}",
                f"/console/api/apps/{app_id}/workflows/draft",
            ]
        )
    probes = {}
    for path in paths:
        status, payload = request_json_or_text("GET", f"{origin}{path}", api_key, timeout=timeout)
        probes[path] = summarize_probe_response(status, payload)
    return probes


def console_origin(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return base_url[:-3]
    marker = "/v1/"
    if marker in base_url:
        return base_url.split(marker, 1)[0]
    return base_url.rstrip("/")


def request_json_or_text(method: str, url: str, api_key: str, *, timeout: float) -> tuple[int | None, Any]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, parse_json_maybe(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, parse_json_maybe(raw)
    except Exception as exc:  # noqa: BLE001
        return None, {"error": f"{type(exc).__name__}: {exc}"}


def parse_json_maybe(raw: str) -> Any:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw[:1000]


def summarize_probe_response(status: int | None, payload: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"status": status}
    if isinstance(payload, dict):
        summary["keys"] = sorted(payload.keys())[:30]
        if status and status >= 400:
            summary["error"] = first_non_empty(payload.get("message"), payload.get("error"), payload.get("code"))
    elif isinstance(payload, str):
        summary["body_preview"] = payload[:300]
    return summary


def stream_chat_message(
    *,
    base_url: str,
    api_key: str,
    query: str,
    inputs: dict[str, Any],
    user: str,
    timeout: float,
) -> tuple[list[dict[str, Any]], float]:
    payload = {
        "inputs": inputs,
        "query": query,
        "response_mode": "streaming",
        "conversation_id": "",
        "user": user,
    }
    request = urllib.request.Request(
        f"{base_url}/chat-messages",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        lines = (line.decode("utf-8", errors="replace") for line in response)
        events = [flatten_stream_event(event) for event in parse_sse_events(lines)]
    return events, time.perf_counter() - started


def parse_sse_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    events = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("data: "):
            continue
        payload = stripped[len("data: ") :]
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            events.append({"event": "parse_error", "raw": payload[:1000]})
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def flatten_stream_event(event: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "event": event.get("event"),
        "task_id": event.get("task_id"),
        "workflow_run_id": event.get("workflow_run_id"),
    }
    data = event.get("data")
    if isinstance(data, dict):
        flat.update(data)
    for key in ("id", "conversation_id", "message_id", "answer", "metadata"):
        if key in event:
            flat[key] = event[key]
    return {key: value for key, value in flat.items() if value is not None}


def summarize_stream_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    workflow_started = first_event(events, "workflow_started")
    message_end = first_event(events, "message_end")
    query_plan = parse_node_body(events, "Query Plan")
    structured_lookup = parse_node_body(events, "Structured Lookup")
    verify_answer = parse_node_body(events, "Verify Answer")
    final_answer = node_output_text(events, "Final Answer")
    rewrite_answer = node_output_text(events, "Rewrite Answer LLM")
    answer_chunks = [str(event.get("answer") or "") for event in events if event.get("event") == "message"]
    workflow_inputs = workflow_started.get("inputs") if isinstance(workflow_started.get("inputs"), dict) else {}
    query_plan_document_id = query_plan.get("document_id") if isinstance(query_plan, dict) else None
    verify_issues = verify_answer.get("issues") if isinstance(verify_answer, dict) else []
    metadata = message_end.get("metadata") if isinstance(message_end.get("metadata"), dict) else {}
    deployment_contract = deployment_contract_status(query_plan, structured_lookup)
    return {
        "workflow_run_id": workflow_started.get("workflow_run_id") or message_end.get("workflow_run_id"),
        "workflow_inputs": workflow_inputs,
        "workflow_input_keys": sorted(workflow_inputs.keys()),
        "document_id_in_workflow_inputs": has_document_id_key(workflow_inputs),
        "query_plan_document_id": query_plan_document_id,
        "structured_direct_answer": structured_lookup.get("direct_answer") if isinstance(structured_lookup, dict) else "",
        "structured_answer_items": structured_lookup.get("answer_items", []) if isinstance(structured_lookup, dict) else [],
        "structured_response_chars": node_response_body_chars(events, "Structured Lookup"),
        "node_timings": node_timing_summary(events),
        "deployment_contract": deployment_contract,
        "final_answer": final_answer,
        "rewrite_answer": rewrite_answer,
        "streamed_answer": "".join(answer_chunks),
        "verify_valid": verify_answer.get("valid") if isinstance(verify_answer, dict) else None,
        "verify_issues": verify_issues,
        "usage": metadata.get("usage") if isinstance(metadata, dict) else {},
        "retriever_resource_count": len(metadata.get("retriever_resources") or []) if isinstance(metadata, dict) else 0,
        "failed_nodes": [
            {
                "title": event.get("title") or event.get("node_title"),
                "error": event.get("error"),
            }
            for event in events
            if event.get("event") == "node_finished" and event.get("status") == "failed"
        ],
    }


def deployment_contract_status(query_plan: Any, structured_lookup: Any) -> dict[str, Any]:
    query_plan = query_plan if isinstance(query_plan, dict) else {}
    structured_lookup = structured_lookup if isinstance(structured_lookup, dict) else {}
    matches = structured_lookup.get("matches") if isinstance(structured_lookup.get("matches"), list) else []
    diagnostics = structured_lookup.get("diagnostics") if isinstance(structured_lookup.get("diagnostics"), dict) else {}
    checks = {
        "knowledge_query": bool(str(query_plan.get("knowledge_query") or "").strip()),
        "knowledge_queries": bool(query_plan.get("knowledge_queries")),
        "lookup_prefilter": "prefilter_limit" in diagnostics and "hydrated_candidate_count" in diagnostics,
        "compact_matches": bool(matches) and all("chunk_text" not in match for match in matches if isinstance(match, dict)),
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "missing": [name for name, passed in checks.items() if not passed],
    }


def node_timing_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timings = []
    for event in events:
        if event.get("event") != "node_finished":
            continue
        elapsed = event.get("elapsed_time")
        if not isinstance(elapsed, (int, float)):
            continue
        timings.append(
            {
                "title": event.get("title") or event.get("node_title") or "",
                "node_type": event.get("node_type") or "",
                "status": event.get("status") or "",
                "elapsed_seconds": round(float(elapsed), 6),
            }
        )
    return timings


def node_response_body_chars(events: list[dict[str, Any]], title: str) -> int:
    for event in events:
        if event.get("event") != "node_finished":
            continue
        if (event.get("title") or event.get("node_title")) != title:
            continue
        outputs = event.get("outputs") if isinstance(event.get("outputs"), dict) else {}
        body = outputs.get("body")
        return len(body) if isinstance(body, str) else 0
    return 0


def first_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any]:
    return next((event for event in events if event.get("event") == event_name), {})


def parse_node_body(events: list[dict[str, Any]], title: str) -> Any:
    for event in events:
        if event.get("event") != "node_finished":
            continue
        if (event.get("title") or event.get("node_title")) != title:
            continue
        outputs = event.get("outputs") if isinstance(event.get("outputs"), dict) else {}
        body = outputs.get("body")
        return parse_json_maybe(body) if isinstance(body, str) else body
    return {}


def node_output_text(events: list[dict[str, Any]], title: str) -> str:
    for event in events:
        if event.get("event") != "node_finished":
            continue
        if (event.get("title") or event.get("node_title")) != title:
            continue
        outputs = event.get("outputs") if isinstance(event.get("outputs"), dict) else {}
        return str(outputs.get("text") or "")
    return ""


def has_document_id_key(inputs: dict[str, Any]) -> bool:
    for key, value in inputs.items():
        if key == "document_id" or key.endswith(".document_id"):
            return bool(str(value or "").strip())
    return False


def workflow_edit_available(report: dict[str, Any]) -> bool:
    console_probe = report.get("console_probe") if isinstance(report.get("console_probe"), dict) else {}
    return any(isinstance(item, dict) and 200 <= int(item.get("status") or 0) < 300 for item in console_probe.values())


def nested_get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def print_human(report: dict[str, Any], out_path: Path) -> None:
    stream = report.get("stream") if isinstance(report.get("stream"), dict) else {}
    summary = stream.get("summary") if isinstance(stream.get("summary"), dict) else {}
    print("Dify App API diagnostics")
    print(f"- host: {report.get('base_url_host')}")
    print(f"- workflow_edit_available: {report.get('workflow_edit_available')}")
    if summary:
        print(f"- workflow_run_id: {summary.get('workflow_run_id')}")
        print(f"- document_id_in_workflow_inputs: {summary.get('document_id_in_workflow_inputs')}")
        print(f"- query_plan_document_id: {summary.get('query_plan_document_id')}")
        print(f"- verify_valid: {summary.get('verify_valid')}")
        contract = summary.get("deployment_contract") if isinstance(summary.get("deployment_contract"), dict) else {}
        print(f"- deployment_contract_ready: {contract.get('ready')}")
        if contract.get("missing"):
            print(f"- stale_contract_fields: {', '.join(contract['missing'])}")
        print(f"- structured_response_chars: {summary.get('structured_response_chars')}")
        timings = summary.get("node_timings") if isinstance(summary.get("node_timings"), list) else []
        slowest = sorted(timings, key=lambda item: float(item.get("elapsed_seconds") or 0), reverse=True)[:5]
        for item in slowest:
            print(f"- node: {item.get('title')} {item.get('elapsed_seconds')}s")
        print(f"- event_count: {stream.get('event_count')}, elapsed: {stream.get('elapsed_seconds')}s")
    print(f"- saved: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
