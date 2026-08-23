from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD_LOOKUP_ID = "1786000000001"
DIRECT_IF_ID = "1786000000002"
DIRECT_STATE_ID = "1786000000003"
DIRECT_ASSIGNER_ID = "1786000000004"
DIRECT_ANSWER_ID = "1786000000005"
LEGACY_PATCH_NODE_IDS = {
    "codex-build-lookup-v1",
    "codex-direct-answer-if-v1",
    "codex-direct-state-v1",
    "codex-direct-assigner-v1",
    "codex-direct-answer-v1",
}


PARSE_QUERY_PLAN_CODE = '''import json

def main(body) -> dict:
    data = json.loads(body) if isinstance(body, str) else body
    entities = data.get("entities") or []
    keywords = data.get("keywords") or []
    expanded_queries = data.get("expanded_queries") or []
    knowledge_queries = data.get("knowledge_queries") or []

    return {
        "query": data.get("query", ""),
        "retrieval_query": data.get("retrieval_query", ""),
        "document_id": data.get("document_id") or "",
        "query_type": data.get("query_type", "semantic"),
        "answer_style": data.get("answer_style", "grounded_explanation"),
        "entities": entities,
        "keywords": keywords,
        "knowledge_query": data.get("knowledge_query") or data.get("query", ""),
        "knowledge_query_secondary": data.get("knowledge_query_secondary") or "",
        "knowledge_queries": knowledge_queries,
        "sub_queries": data.get("sub_queries") or [],
        "expanded_queries": expanded_queries,
        "entities_text": ", ".join(entities),
        "sub_query_text": "\\n".join(data.get("sub_queries") or []),
        "expanded_query_text": "\\n".join(expanded_queries),
    }
'''


BUILD_LOOKUP_CODE = '''import json

def parse_json_like(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, dict, int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default

def main(query="", retrieval_query="", document_id="", entities=None, query_type="semantic", **kwargs) -> dict:
    parsed_entities = parse_json_like(entities, [])
    if isinstance(parsed_entities, str):
        parsed_entities = [part.strip() for part in parsed_entities.split(",") if part.strip()]
    if not isinstance(parsed_entities, list):
        parsed_entities = []

    body = {
        "query": str(query or "").strip(),
        "retrieval_query": str(retrieval_query or "").strip(),
        "document_id": str(document_id or "").strip(),
        "entities": [str(item) for item in parsed_entities if str(item).strip()],
        "query_type": str(query_type or "semantic").strip(),
        "limit": 8,
    }
    return {"lookup_body_json": json.dumps(body, ensure_ascii=False)}
'''


PARSE_MERGE_CODE = '''import json

def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def main(body, status_code=200) -> dict:
    code = int(status_code or 0)
    if code < 200 or code >= 300:
        raise ValueError(f"Merge Evidence Request failed. status_code={code} body={as_text(body)[:500]}")

    data = json.loads(body) if isinstance(body, str) else body
    if not isinstance(data, dict):
        raise ValueError("Merge Evidence Request body is not a JSON object")

    evidence_context = as_text(data.get("evidence_context"))
    if not evidence_context.strip():
        evidence_context = "제공된 구조화/Knowledge 근거가 비어 있습니다. 문서에 충분한 근거가 없다고 답하세요."

    return {
        "structured_context": as_text(data.get("structured_context")),
        "knowledge_context": as_text(data.get("knowledge_context")),
        "evidence_context": evidence_context,
        "evidence_priority": as_text(data.get("evidence_priority")),
        "source_summary": as_text(data.get("source_summary")),
        "answer_contract": as_text(data.get("answer_contract")),
        "answer_contract_text": as_text(data.get("answer_contract_text")),
        "lookup_diagnostics": as_text(data.get("lookup_diagnostics")),
        "lookup_answerability": as_text(data.get("lookup_answerability")),
        "answer_contract_status": as_text(data.get("answer_contract_status")),
        "direct_answer": as_text(data.get("direct_answer")),
        "has_direct_answer": bool(data.get("has_direct_answer", False)),
        "deterministic_answer": as_text(data.get("deterministic_answer")),
    }
'''


STRING_OUTPUT = {"type": "string", "children": None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch and publish the generic Dify RAG Chatflow graph.")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--apply", action="store_true", help="Synchronize the patched draft after saving a backup.")
    parser.add_argument("--publish", action="store_true", help="Publish after synchronizing and verifying the draft.")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.publish and not args.apply:
        parser.error("--publish requires --apply")

    load_env_file(Path(args.env_file))
    base_url = console_origin(required_env("DIFY_CONSOLE_API_BASE_URL"))
    app_id = required_env("DIFY_CONSOLE_APP_ID")
    headers = console_headers()
    draft_url = f"{base_url}/console/api/apps/{app_id}/workflows/draft"

    draft = request_json("GET", draft_url, headers=headers, timeout=args.timeout)
    patched = copy.deepcopy(draft)
    changes = patch_workflow(patched["graph"])
    validate_patched_graph(patched["graph"])

    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "app_id": app_id,
        "workflow_id": draft.get("id"),
        "before_hash": draft.get("hash"),
        "node_count_before": len(draft["graph"].get("nodes", [])),
        "node_count_after": len(patched["graph"].get("nodes", [])),
        "edge_count_before": len(draft["graph"].get("edges", [])),
        "edge_count_after": len(patched["graph"].get("edges", [])),
        "changes": changes,
    }
    if not args.apply:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    backup_path = save_backup(draft, app_id)
    payload = {
        "graph": patched["graph"],
        "features": draft.get("features") or {},
        "hash": draft.get("hash"),
        "conversation_variables": draft.get("conversation_variables") or [],
    }
    synced = request_json("POST", draft_url, payload=payload, headers=headers, timeout=args.timeout)
    verified = request_json("GET", draft_url, headers=headers, timeout=args.timeout)
    validate_patched_graph(verified["graph"])
    summary.update(
        {
            "backup": str(backup_path.relative_to(ROOT)),
            "sync_result": synced.get("result"),
            "after_hash": verified.get("hash"),
            "verified": True,
        }
    )

    if args.publish:
        publish_headers = console_headers(require_csrf=True)
        published = request_json(
            "POST",
            f"{base_url}/console/api/apps/{app_id}/workflows/publish",
            payload={"marked_name": "RAG direct path", "marked_comment": "Generic query preservation and deterministic answer bypass"},
            headers=publish_headers,
            timeout=args.timeout,
        )
        summary["publish_result"] = published.get("result")
        summary["published_at"] = published.get("created_at")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def patch_workflow(graph: dict[str, Any]) -> list[str]:
    nodes = graph.setdefault("nodes", [])
    edges = graph.setdefault("edges", [])
    changes: list[str] = []

    if any(node.get("id") in LEGACY_PATCH_NODE_IDS for node in nodes):
        nodes[:] = [node for node in nodes if node.get("id") not in LEGACY_PATCH_NODE_IDS]
        edges[:] = [
            edge
            for edge in edges
            if edge.get("source") not in LEGACY_PATCH_NODE_IDS and edge.get("target") not in LEGACY_PATCH_NODE_IDS
        ]
        changes.append("migrated generated node ids to Dify-compatible numeric ids")

    start = find_node(nodes, node_type="start")
    rewrite = find_node(nodes, title="Rewrite Standalone Query")
    query_plan = find_node(nodes, title="Query Plan")
    parse_plan = find_node(nodes, title="Parse Query Plan")
    structured = find_node(nodes, title="Structured Lookup")
    knowledge = find_node(nodes, node_type="knowledge-retrieval")
    build_merge = find_node(nodes, title="Build Merge Evidence Body")
    parse_merge = find_node(nodes, title="Parse Merge Evidence")
    final_llm = find_node(nodes, title="Final Answer")
    state_template = find_node(nodes, title="Build Conversation State")
    assigner_template = find_node(nodes, node_type="assigner")
    answer_template = find_node(nodes, node_type="answer")
    if_template = find_node(nodes, node_type="if-else")

    variables = start["data"].setdefault("variables", [])
    if not any(item.get("variable") == "document_id" for item in variables):
        variables.append(
            {
                "label": "document_id",
                "max_length": 256,
                "options": [],
                "required": False,
                "type": "text-input",
                "variable": "document_id",
            }
        )
        changes.append("added Start.document_id")

    query_plan["data"]["body"] = {
        "type": "json",
        "data": [
            {
                "id": "codex-query-plan-body-v1",
                "type": "text",
                "key": "",
                "value": (
                    "{\n"
                    '  "query": {{#sys.query#}},\n'
                    f'  "retrieval_query": {{{{#{rewrite["id"]}.text#}}}},\n'
                    f'  "document_id": {{{{#{start["id"]}.document_id#}}}}\n'
                    "}"
                ),
            }
        ],
    }
    changes.append("preserved original query in Query Plan")

    parse_plan["data"]["code"] = PARSE_QUERY_PLAN_CODE
    parse_plan["data"]["outputs"] = {
        "query": copy.deepcopy(STRING_OUTPUT),
        "retrieval_query": copy.deepcopy(STRING_OUTPUT),
        "document_id": copy.deepcopy(STRING_OUTPUT),
        "query_type": copy.deepcopy(STRING_OUTPUT),
        "answer_style": copy.deepcopy(STRING_OUTPUT),
        "entities": {"type": "array[string]", "children": None},
        "keywords": {"type": "array[string]", "children": None},
        "knowledge_query": copy.deepcopy(STRING_OUTPUT),
        "knowledge_query_secondary": copy.deepcopy(STRING_OUTPUT),
        "knowledge_queries": {"type": "array[string]", "children": None},
        "sub_queries": {"type": "array[string]", "children": None},
        "expanded_queries": {"type": "array[string]", "children": None},
        "entities_text": copy.deepcopy(STRING_OUTPUT),
        "sub_query_text": copy.deepcopy(STRING_OUTPUT),
        "expanded_query_text": copy.deepcopy(STRING_OUTPUT),
    }
    changes.append("expanded Query Plan outputs")

    build_lookup = upsert_cloned_node(nodes, parse_plan, BUILD_LOOKUP_ID)
    build_lookup["position"] = offset_position(parse_plan, 330, 120)
    build_lookup["data"] = {
        "code": BUILD_LOOKUP_CODE,
        "code_language": "python3",
        "variables": [
            selector("query", "sys", "query", "string"),
            selector("retrieval_query", rewrite["id"], "text", "string"),
            selector("document_id", parse_plan["id"], "document_id", "string"),
            selector("entities", parse_plan["id"], "entities", "array[string]"),
            selector("query_type", parse_plan["id"], "query_type", "string"),
        ],
        "outputs": {"lookup_body_json": copy.deepcopy(STRING_OUTPUT)},
        "type": "code",
        "title": "Build Lookup Body",
        "selected": False,
    }
    structured["data"]["body"] = {
        "type": "raw-text",
        "data": [
            {
                "id": "codex-lookup-body-v1",
                "type": "text",
                "key": "",
                "value": f"{{{{#{BUILD_LOOKUP_ID}.lookup_body_json#}}}}",
            }
        ],
    }
    knowledge["data"]["query_variable_selector"] = [parse_plan["id"], "knowledge_query"]
    replace_edge(edges, parse_plan["id"], structured["id"], parse_plan["id"], BUILD_LOOKUP_ID, "code")
    ensure_edge(edges, BUILD_LOOKUP_ID, structured["id"], "code", "http-request")
    remove_edges(edges, structured["id"], knowledge["id"])
    ensure_edge(edges, parse_plan["id"], knowledge["id"], "code", "knowledge-retrieval")
    ensure_edge(edges, structured["id"], build_merge["id"], "http-request", "code")
    changes.append("added typed lookup body and parallelized structured/Knowledge retrieval")

    parse_merge["data"]["code"] = PARSE_MERGE_CODE
    parse_merge["data"]["outputs"].update(
        {
            "answer_contract_status": copy.deepcopy(STRING_OUTPUT),
            "direct_answer": copy.deepcopy(STRING_OUTPUT),
            "has_direct_answer": {"type": "boolean", "children": None},
            "deterministic_answer": copy.deepcopy(STRING_OUTPUT),
        }
    )

    direct_if = upsert_cloned_node(nodes, if_template, DIRECT_IF_ID)
    direct_if["position"] = offset_position(parse_merge, 330, -320)
    direct_if["data"] = {
        "cases": [
            {
                "id": "true",
                "case_id": "true",
                "logical_operator": "and",
                "conditions": [
                    {
                        "id": "codex-has-direct-answer-v1",
                        "varType": "boolean",
                        "variable_selector": [parse_merge["id"], "has_direct_answer"],
                        "comparison_operator": "is",
                        "value": True,
                    }
                ],
            }
        ],
        "type": "if-else",
        "title": "IF Deterministic Direct Answer",
        "selected": False,
    }

    direct_state = upsert_cloned_node(nodes, state_template, DIRECT_STATE_ID)
    direct_state["position"] = offset_position(parse_merge, 660, -480)
    direct_state["data"]["title"] = "Build Conversation State (Direct)"
    direct_state["data"]["variables"] = [
        selector("question", "sys", "query", "string"),
        selector("answer", parse_merge["id"], "deterministic_answer", "string"),
        selector("source_summary", parse_merge["id"], "source_summary", "string"),
    ]

    direct_assigner = upsert_cloned_node(nodes, assigner_template, DIRECT_ASSIGNER_ID)
    direct_assigner["position"] = offset_position(parse_merge, 990, -480)
    direct_assigner["data"]["title"] = "Assign Conversation State (Direct)"
    for item in direct_assigner["data"].get("items", []):
        value = item.get("value")
        if isinstance(value, list) and value:
            value[0] = DIRECT_STATE_ID

    direct_answer = upsert_cloned_node(nodes, answer_template, DIRECT_ANSWER_ID)
    direct_answer["position"] = offset_position(parse_merge, 1320, -480)
    direct_answer["data"] = {
        "variables": [],
        "answer": f"{{{{#{parse_merge['id']}.deterministic_answer#}}}}",
        "type": "answer",
        "title": "Deterministic Direct Answer",
        "selected": False,
    }

    remove_edges(edges, parse_merge["id"], final_llm["id"])
    ensure_edge(edges, parse_merge["id"], DIRECT_IF_ID, "code", "if-else")
    ensure_edge(edges, DIRECT_IF_ID, DIRECT_STATE_ID, "if-else", "code", source_handle="true")
    ensure_edge(edges, DIRECT_IF_ID, final_llm["id"], "if-else", "llm", source_handle="false")
    ensure_edge(edges, DIRECT_STATE_ID, DIRECT_ASSIGNER_ID, "code", "assigner")
    ensure_edge(edges, DIRECT_ASSIGNER_ID, DIRECT_ANSWER_ID, "assigner", "answer")
    changes.append("added deterministic direct-answer bypass with conversation state")
    return changes


def validate_patched_graph(graph: dict[str, Any]) -> None:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    ids = [str(node.get("id")) for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("patched graph contains duplicate node ids")
    known = set(ids)
    for edge in edges:
        if edge.get("source") not in known or edge.get("target") not in known:
            raise ValueError(f"edge refers to a missing node: {edge.get('id')}")
    for node_id in (BUILD_LOOKUP_ID, DIRECT_IF_ID, DIRECT_STATE_ID, DIRECT_ASSIGNER_ID, DIRECT_ANSWER_ID):
        if node_id not in known:
            raise ValueError(f"required patched node is missing: {node_id}")
    start = find_node(nodes, node_type="start")
    if not any(item.get("variable") == "document_id" for item in start["data"].get("variables", [])):
        raise ValueError("Start.document_id is missing")
    parse_merge = find_node(nodes, title="Parse Merge Evidence")
    for output in ("has_direct_answer", "deterministic_answer"):
        if output not in parse_merge["data"].get("outputs", {}):
            raise ValueError(f"Parse Merge Evidence output is missing: {output}")


def selector(variable: str, node_id: str, output: str, value_type: str) -> dict[str, Any]:
    return {"variable": variable, "value_selector": [node_id, output], "value_type": value_type}


def find_node(nodes: list[dict[str, Any]], *, title: str | None = None, node_type: str | None = None) -> dict[str, Any]:
    matches = []
    for node in nodes:
        data = node.get("data") or {}
        if title is not None and data.get("title") != title:
            continue
        if node_type is not None and data.get("type") != node_type:
            continue
        matches.append(node)
    if not matches:
        label = title or node_type or "unknown"
        raise ValueError(f"required workflow node was not found: {label}")
    return matches[0]


def upsert_cloned_node(nodes: list[dict[str, Any]], template: dict[str, Any], node_id: str) -> dict[str, Any]:
    for node in nodes:
        if node.get("id") == node_id:
            return node
    node = copy.deepcopy(template)
    node["id"] = node_id
    node["selected"] = False
    nodes.append(node)
    return node


def offset_position(node: dict[str, Any], dx: float, dy: float) -> dict[str, float]:
    position = node.get("position") or {}
    return {"x": float(position.get("x") or 0) + dx, "y": float(position.get("y") or 0) + dy}


def remove_edges(edges: list[dict[str, Any]], source: str, target: str) -> None:
    edges[:] = [edge for edge in edges if not (edge.get("source") == source and edge.get("target") == target)]


def replace_edge(
    edges: list[dict[str, Any]],
    old_source: str,
    old_target: str,
    new_source: str,
    new_target: str,
    new_target_type: str,
) -> None:
    remove_edges(edges, old_source, old_target)
    ensure_edge(edges, new_source, new_target, "code", new_target_type)


def ensure_edge(
    edges: list[dict[str, Any]],
    source: str,
    target: str,
    source_type: str,
    target_type: str,
    *,
    source_handle: str = "source",
) -> None:
    edge_id = f"{source}-{source_handle}-{target}-target"
    for edge in edges:
        if edge.get("id") == edge_id or (
            edge.get("source") == source
            and edge.get("target") == target
            and edge.get("sourceHandle") == source_handle
        ):
            return
    edges.append(
        {
            "id": edge_id,
            "type": "custom",
            "source": source,
            "target": target,
            "sourceHandle": source_handle,
            "targetHandle": "target",
            "data": {"sourceType": source_type, "targetType": target_type, "isInLoop": False},
            "zIndex": 0,
        }
    )


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise ValueError(f"missing required environment variable: {name}")
    return value


def console_origin(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("DIFY_CONSOLE_API_BASE_URL must be an absolute URL")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def console_headers(*, require_csrf: bool = False) -> dict[str, str]:
    configured = required_env("DIFY_CONSOLE_COOKIE")
    headers = {"Accept": "application/json"}
    cookie_header = ""
    if "=" in configured:
        cookie_header = configured
        headers["Cookie"] = cookie_header
    else:
        headers["Authorization"] = f"Bearer {configured}"

    if require_csrf:
        csrf = (os.getenv("DIFY_CONSOLE_CSRF_TOKEN") or csrf_from_cookie(cookie_header)).strip()
        if not csrf:
            raise ValueError("publishing requires DIFY_CONSOLE_CSRF_TOKEN or a csrf_token cookie")
        headers["X-CSRF-Token"] = csrf
        if not cookie_header:
            headers["Cookie"] = f"csrf_token={csrf}; __Host-csrf_token={csrf}"
    return headers


def csrf_from_cookie(cookie_header: str) -> str:
    for part in cookie_header.split(";"):
        name, separator, value = part.strip().partition("=")
        if separator and name.removeprefix("__Host-") == "csrf_token":
            return value
    return ""


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    body = None
    request_headers = dict(headers)
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"Dify Console API {method} failed with HTTP {exc.code}: {error_body}") from exc
    parsed = json.loads(raw) if raw else {}
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Dify Console API {method} returned a non-object response")
    return parsed


def save_backup(draft: dict[str, Any], app_id: str) -> Path:
    backup_dir = ROOT / ".runtime" / "dify-workflow-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = backup_dir / f"{app_id}-{stamp}.json"
    path.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
