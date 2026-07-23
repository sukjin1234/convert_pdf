from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STRUCTURED_QUERY_TYPES = {
    "number_lookup",
    "date_lookup",
    "table_lookup",
    "condition_lookup",
    "contact_lookup",
    "dependency_lookup",
    "troubleshooting",
    "procedure",
}


@dataclass
class LogFinding:
    severity: str
    type: str
    run_id: str
    query: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "type": self.type,
            "run_id": self.run_id,
            "query": self.query,
            "message": self.message,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Dify RAG eval log JSON files.")
    parser.add_argument("--log-dir", default="dify-rag-eval-logs")
    parser.add_argument("--limit", type=int, default=20, help="Maximum finding rows to print in human mode.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit with 1 when error findings exist.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    records = load_records(log_dir)
    analyzed = []
    findings = []
    for record in records:
        item, item_findings = analyze_record(record)
        analyzed.append(item)
        findings.extend(item_findings)
    summary = summarize(analyzed, findings)
    output = {
        "log_dir": str(log_dir),
        "summary": summary,
        "findings": [finding.to_dict() for finding in findings],
        "records": analyzed,
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print_human_report(summary, findings, limit=args.limit)

    has_errors = any(finding.severity == "error" for finding in findings)
    return 1 if args.fail_on_error and has_errors else 0


def load_records(log_dir: Path) -> list[dict[str, Any]]:
    if not log_dir.exists():
        return []
    records = []
    for path in sorted(log_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            payload["_log_path"] = str(path)
            records.append(payload)
    return records


def analyze_record(record: dict[str, Any]) -> tuple[dict[str, Any], list[LogFinding]]:
    run_id = str(record.get("run_id") or "")
    query = str(record.get("query") or "")
    document_id = record.get("document_id")
    query_plan = record.get("query_plan") if isinstance(record.get("query_plan"), dict) else {}
    nodes = record.get("nodes") if isinstance(record.get("nodes"), dict) else {}
    verify = nodes.get("parse_verify_result") if isinstance(nodes.get("parse_verify_result"), dict) else {}
    lookup = parse_json_like(nodes.get("structured_lookup_body"))
    if not isinstance(lookup, dict):
        lookup = {}
    merge_evidence = nodes.get("merge_evidence") if isinstance(nodes.get("merge_evidence"), dict) else {}
    answerability = merge_evidence.get("lookup_answerability")
    if not isinstance(answerability, dict):
        answerability = parse_json_like(answerability)
    if not isinstance(answerability, dict):
        answerability = {}

    qtype = str(query_plan.get("query_type") or lookup.get("query_type") or "")
    final_answer = str(nodes.get("final_answer") or record.get("final_answer") or "")
    direct_answer = str(lookup.get("direct_answer") or "")
    selected_count = int((lookup.get("diagnostics") or {}).get("selected_count") or 0) if isinstance(lookup.get("diagnostics"), dict) else 0
    valid = verify.get("valid")
    confidence = verify.get("confidence")
    evidence_context = str(merge_evidence.get("evidence_context") or "")
    answer_contract_text = str(merge_evidence.get("answer_contract_text") or "")
    has_answer_contract = "Answer Contract - obey before writing" in f"{answer_contract_text}\n{evidence_context}"

    findings = []
    if not document_id:
        findings.append(
            LogFinding(
                "warning",
                "missing_document_id",
                run_id,
                query,
                "document_id is empty; multi-document lookup can search the wrong artifact.",
            )
        )
    if valid is False:
        findings.append(
            LogFinding(
                "error",
                "verify_failed",
                run_id,
                query,
                issue_text(verify) or "Verify Answer returned valid=false.",
            )
        )
    if qtype in STRUCTURED_QUERY_TYPES and selected_count > 0 and not direct_answer:
        findings.append(
            LogFinding(
                "warning",
                "missing_direct_answer",
                run_id,
                query,
                "Structured query selected evidence but did not produce direct_answer.",
            )
        )
    if qtype in STRUCTURED_QUERY_TYPES and selected_count == 0 and answerability.get("answerable") is not False:
        findings.append(
            LogFinding(
                "warning",
                "no_structured_evidence",
                run_id,
                query,
                "Structured query has no selected lookup evidence.",
            )
        )
    if valid is True and qtype in STRUCTURED_QUERY_TYPES and not direct_answer and selected_count > 0:
        findings.append(
            LogFinding(
                "warning",
                "valid_without_direct_answer",
                run_id,
                query,
                "Verifier passed an answer even though structured direct_answer is empty.",
            )
        )
    if qtype in STRUCTURED_QUERY_TYPES and direct_answer and not has_answer_contract:
        findings.append(
            LogFinding(
                "warning",
                "missing_answer_contract",
                run_id,
                query,
                "Structured Lookup produced a direct_answer, but Merge Evidence did not pass answer_contract_text.",
            )
        )
    if not final_answer.strip():
        findings.append(LogFinding("error", "empty_final_answer", run_id, query, "Final answer is empty."))

    item = {
        "run_id": run_id,
        "created_at": record.get("created_at", ""),
        "query": query,
        "document_id": document_id,
        "query_type": qtype,
        "valid": valid,
        "confidence": confidence,
        "has_direct_answer": bool(direct_answer),
        "has_answer_contract": has_answer_contract,
        "selected_count": selected_count,
        "final_answer_preview": truncate(final_answer.replace("\n", " "), 220),
        "findings": [finding.to_dict() for finding in findings],
    }
    return item, findings


def summarize(records: list[dict[str, Any]], findings: list[LogFinding]) -> dict[str, Any]:
    total = len(records)
    severity_counts = Counter(finding.severity for finding in findings)
    type_counts = Counter(finding.type for finding in findings)
    query_type_counts = Counter(str(record.get("query_type") or "") for record in records)
    return {
        "total": total,
        "valid_true": sum(1 for record in records if record.get("valid") is True),
        "valid_false": sum(1 for record in records if record.get("valid") is False),
        "missing_document_id": sum(1 for record in records if not record.get("document_id")),
        "direct_answer_present": sum(1 for record in records if record.get("has_direct_answer")),
        "answer_contract_present": sum(1 for record in records if record.get("has_answer_contract")),
        "finding_count": len(findings),
        "error_count": severity_counts.get("error", 0),
        "warning_count": severity_counts.get("warning", 0),
        "finding_types": dict(sorted(type_counts.items())),
        "query_types": dict(sorted(query_type_counts.items())),
    }


def print_human_report(summary: dict[str, Any], findings: list[LogFinding], *, limit: int) -> None:
    print("Eval Log Analysis")
    print("=================")
    print(f"total={summary['total']} valid_true={summary['valid_true']} valid_false={summary['valid_false']}")
    print(
        f"missing_document_id={summary['missing_document_id']} "
        f"direct_answer_present={summary['direct_answer_present']} "
        f"answer_contract_present={summary['answer_contract_present']}"
    )
    print(f"errors={summary['error_count']} warnings={summary['warning_count']}")
    print()
    print("Finding Types")
    print("-------------")
    if summary["finding_types"]:
        for key, count in summary["finding_types"].items():
            print(f"{key}: {count}")
    else:
        print("none")
    print()
    print("Findings")
    print("--------")
    if not findings:
        print("none")
        return
    for finding in findings[: max(limit, 0)]:
        print(f"[{finding.severity}] {finding.type} run={finding.run_id} query={finding.query}")
        print(f"  {finding.message}")
    remaining = len(findings) - limit
    if remaining > 0:
        print(f"... {remaining} more")


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


def issue_text(verify: dict[str, Any]) -> str:
    issues = verify.get("issues")
    if isinstance(issues, list) and issues:
        parts = []
        for issue in issues:
            if isinstance(issue, dict):
                parts.append(f"{issue.get('type', 'unknown')}: {issue.get('message', '')}".strip())
            else:
                parts.append(str(issue))
        return "; ".join(parts)
    return str(verify.get("issues_text") or "")


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 15)].rstrip() + "...[truncated]"


if __name__ == "__main__":
    raise SystemExit(main())
