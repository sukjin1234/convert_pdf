from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag import build_rag_artifact, lookup_matches, normalize_for_match, plan_query, verify_answer


@dataclass
class CaseResult:
    case_id: str
    document_id: str
    question: str
    answerable: bool
    passed: bool
    query_type: str
    expected_query_type: str
    query_type_match: bool | None
    evidence_recall: bool
    exact_value_match: bool
    answerable_detection: bool
    unsupported_guard: bool | None
    top_score: float
    selected_count: int
    missing_terms: list[str]
    missing_values: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "document_id": self.document_id,
            "question": self.question,
            "answerable": self.answerable,
            "passed": self.passed,
            "query_type": self.query_type,
            "expected_query_type": self.expected_query_type,
            "query_type_match": self.query_type_match,
            "evidence_recall": self.evidence_recall,
            "exact_value_match": self.exact_value_match,
            "answerable_detection": self.answerable_detection,
            "unsupported_guard": self.unsupported_guard,
            "top_score": self.top_score,
            "selected_count": self.selected_count,
            "missing_terms": self.missing_terms,
            "missing_values": self.missing_values,
            "notes": self.notes,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local RAG retrieval and verification quality.")
    parser.add_argument("--cases", default=str(ROOT / "evaluation" / "rag_quality_cases.json"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--fail-under", type=float, default=0.90)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--quiet", action="store_true", help="Only print summary unless failing.")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    artifacts = {
        document["document_id"]: build_rag_artifact(
            document["markdown"],
            document.get("file_name") or f"{document['document_id']}.md",
            document_id=document["document_id"],
        )
        for document in payload.get("documents", [])
    }

    results = [
        evaluate_case(case, artifacts, limit=args.limit)
        for case in payload.get("cases", [])
    ]
    summary = summarize(results)

    output = {
        "cases_path": str(cases_path),
        "summary": summary,
        "results": [result.to_dict() for result in results],
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print_human_report(results, summary, quiet=args.quiet)

    return 0 if summary["pass_rate"] >= args.fail_under else 1


def evaluate_case(case: dict[str, Any], artifacts: dict[str, Any], *, limit: int) -> CaseResult:
    case_id = case["id"]
    document_id = case["document_id"]
    question = case["question"]
    answerable = bool(case.get("answerable", True))
    artifact = artifacts[document_id]

    plan = plan_query(question, document_id=document_id)
    expected_query_type = str(case.get("expected_query_type") or "")
    query_type_match = None
    if expected_query_type:
        query_type_match = plan["query_type"] == expected_query_type

    lookup = lookup_matches(
        query=question,
        records=artifact.records,
        chunks=artifact.chunks,
        entities=plan["entities"],
        query_type=plan["query_type"],
        limit=limit,
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
        verification = verify_answer(question, str(unsupported_answer), [context])
        unsupported_guard = not verification["valid"]

    notes = []
    if query_type_match is False:
        notes.append(f"query_type expected={expected_query_type} actual={plan['query_type']}")
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
    passed = all(checks)

    return CaseResult(
        case_id=case_id,
        document_id=document_id,
        question=question,
        answerable=answerable,
        passed=passed,
        query_type=plan["query_type"],
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


def missing_needles(text: str, needles: list[str]) -> list[str]:
    haystack = normalize_for_match(text)
    missing = []
    for needle in needles:
        normalized = normalize_for_match(needle)
        if normalized and normalized not in haystack:
            missing.append(needle)
    return missing


def no_forbidden_values(text: str, forbidden_values: list[str]) -> bool:
    return not any(value for value in forbidden_values if not missing_needles(text, [value]))


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    answerable = [result for result in results if result.answerable]
    unanswerable = [result for result in results if not result.answerable]
    typed = [result for result in results if result.query_type_match is not None]
    guarded = [result for result in results if result.unsupported_guard is not None]

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "evidence_recall": rate(answerable, lambda result: result.evidence_recall),
        "exact_value_match": rate(answerable, lambda result: result.exact_value_match),
        "answerable_detection": rate(results, lambda result: result.answerable_detection),
        "query_type_accuracy": rate(typed, lambda result: bool(result.query_type_match)),
        "unsupported_guard": rate(guarded, lambda result: bool(result.unsupported_guard)),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
    }


def rate(items: list[CaseResult], predicate) -> float:
    if not items:
        return 0.0
    return round(sum(1 for item in items if predicate(item)) / len(items), 4)


def print_human_report(results: list[CaseResult], summary: dict[str, Any], *, quiet: bool) -> None:
    print("RAG quality evaluation")
    print("======================")
    print(
        "total={total} passed={passed} failed={failed} pass_rate={pass_rate:.2%}".format(
            **summary
        )
    )
    print(
        "evidence_recall={evidence_recall:.2%} exact_value_match={exact_value_match:.2%} "
        "query_type_accuracy={query_type_accuracy:.2%} unsupported_guard={unsupported_guard:.2%}".format(
            **summary
        )
    )
    print(f"answerable_cases={summary['answerable_cases']} unanswerable_cases={summary['unanswerable_cases']}")

    failures = [result for result in results if not result.passed]
    if not quiet or failures:
        print()
        print("Case results")
        print("------------")
        for result in results:
            if quiet and result.passed:
                continue
            status = "PASS" if result.passed else "FAIL"
            print(
                f"{status} {result.case_id} "
                f"type={result.query_type} score={result.top_score:.2f} selected={result.selected_count}"
            )
            for note in result.notes:
                print(f"  - {note}")


if __name__ == "__main__":
    raise SystemExit(main())
