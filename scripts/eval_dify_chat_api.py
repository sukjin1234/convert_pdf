from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MARKER = "\ucc38\uc870 \ubb38\uc11c:"
DEFAULT_DOC_2026_ID = "de22a9fc-929d-43fb-9b05-3929cd453702"
DEFAULT_DOC_2027_ID = "c7b1cb1b-91a3-4a6e-a64b-a90837445f3f"


@dataclass(frozen=True)
class Case:
    id: str
    query: str
    inputs: dict[str, Any] = field(default_factory=dict)
    expected_contains: tuple[str, ...] = ()
    forbidden_contains: tuple[str, ...] = ()
    require_source: bool = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deployed Dify Chatflow answers through the App API.")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=str(ROOT / ".runtime" / "dify_chat_api_eval_latest.json"))
    parser.add_argument("--doc2026-id", default=os.getenv("DIFY_DOC_2026_ID") or DEFAULT_DOC_2026_ID)
    parser.add_argument("--doc2027-id", default=os.getenv("DIFY_DOC_2027_ID") or DEFAULT_DOC_2027_ID)
    parser.add_argument("--suite", choices=["admission", "mixed", "all"], default="admission")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    base_url = (os.getenv("DIFY_API_BASE_URL") or "").rstrip("/")
    api_key = os.getenv("DIFY_APP_API_KEY") or os.getenv("DIFY_API_KEY") or ""
    if not base_url or not api_key:
        print("Missing DIFY_API_BASE_URL or DIFY_APP_API_KEY.", file=sys.stderr)
        return 2

    cases = build_cases(args.doc2026_id, args.doc2027_id, suite=args.suite)
    results = [run_case(case, base_url=base_url, api_key=api_key, timeout=args.timeout) for case in cases]
    summary = summarize(results)
    output = {"summary": summary, "results": results}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print_human(summary, results, out_path)
    return 0 if summary["failed"] == 0 else 1


def build_cases(doc2026_id: str, doc2027_id: str, *, suite: str) -> list[Case]:
    admission_cases = [
        Case(
            id="date_default",
            query="\uc218\uc2dc1\ucc28 \uc6d0\uc11c\uc811\uc218 \uae30\uac04 \uc54c\ub824\uc918",
            expected_contains=("2026", "2027\ud559\ub144\ub3c4"),
            forbidden_contains=("2025. 9. 8", "2025\ub144 9\uc6d4 8", "document.pdf"),
        ),
        Case(
            id="date_doc2026_filter",
            query="\uc218\uc2dc1\ucc28 \uc6d0\uc11c\uc811\uc218 \uae30\uac04 \uc54c\ub824\uc918",
            inputs={"document_id": doc2026_id},
            expected_contains=("2025", "document.pdf"),
            forbidden_contains=("2027\ud559\ub144\ub3c4",),
        ),
        Case(
            id="date_doc2027_filter",
            query="\uc218\uc2dc1\ucc28 \uc6d0\uc11c\uc811\uc218 \uae30\uac04 \uc54c\ub824\uc918",
            inputs={"document_id": doc2027_id},
            expected_contains=("2026", "2027\ud559\ub144\ub3c4"),
            forbidden_contains=("document.pdf",),
        ),
        Case(
            id="quota_default",
            query="\ucef4\ud4e8\ud130\uc815\ubcf4\uacf5\ud559\uacfc \uc815\uc6d0\ub0b4 \ubaa8\uc9d1\uc815\uc6d0\uc740 \uba87 \uba85\uc774\uc57c?",
            expected_contains=("93",),
            forbidden_contains=("ch01_", "ch03_", "ch04_", "ch05_", "ch06_"),
        ),
        Case(
            id="quota_doc2026_filter",
            query="\ucef4\ud4e8\ud130\uc815\ubcf4\uacf5\ud559\uacfc \uc815\uc6d0\ub0b4 \ubaa8\uc9d1\uc815\uc6d0\uc740 \uba87 \uba85\uc774\uc57c?",
            inputs={"document_id": doc2026_id},
            expected_contains=("93", "document.pdf"),
        ),
        Case(
            id="quota_doc2027_filter",
            query="\ucef4\ud4e8\ud130\uc815\ubcf4\uacf5\ud559\uacfc \uc815\uc6d0\ub0b4 \ubaa8\uc9d1\uc815\uc6d0\uc740 \uba87 \uba85\uc774\uc57c?",
            inputs={"document_id": doc2027_id},
            expected_contains=("2027\ud559\ub144\ub3c4",),
            forbidden_contains=("document.pdf",),
        ),
        Case(
            id="explicit_2027_query",
            query="2027\ud559\ub144\ub3c4 \ucef4\ud4e8\ud130\uc815\ubcf4\uacf5\ud559\uacfc \ubaa8\uc9d1\uc778\uc6d0 \uc54c\ub824\uc918",
            expected_contains=("2027\ud559\ub144\ub3c4",),
            forbidden_contains=("document.pdf",),
        ),
        Case(
            id="scholarship_total_abstains",
            query="\ucd1d \uc7a5\ud559\uae08 \uc5bc\ub9c8\uc57c",
            expected_contains=("\uadfc\uac70",),
            forbidden_contains=("290\uc5b5\uc6d0 423\ub9cc\uc6d0\uc785\ub2c8\ub2e4",),
        ),
    ]

    mixed_cases = [
        Case(
            id="mixed_admission_quota",
            query="\ucef4\ud4e8\ud130\uc815\ubcf4\uacf5\ud559\uacfc \uc815\uc6d0\ub0b4 \ubaa8\uc9d1\uc815\uc6d0\uc740 \uba87 \uba85\uc774\uc57c?",
            expected_contains=("93",),
            forbidden_contains=("ch01_", "ch03_", "ch04_", "ch05_", "ch06_"),
        ),
        Case(
            id="mixed_software_process",
            query="\uc18c\ud504\ud2b8\uc6e8\uc5b4 \uac1c\ubc1c \ud504\ub85c\uc138\uc2a4\ub294 \ubb50\uc57c?",
            expected_contains=("\uc18c\ud504\ud2b8\uc6e8\uc5b4", "ch01_"),
            forbidden_contains=("document.pdf", "\uc9c0\uc6d0\uc790\uaca9", "\ubaa8\uc9d1\uc778\uc6d0"),
        ),
        Case(
            id="mixed_software_requirements",
            query="\uc694\uad6c\ubd84\uc11d \ub2e8\uacc4\uc5d0\uc11c \ubb34\uc5c7\uc744 \ud558\ub294\uc9c0 \uc54c\ub824\uc918",
            expected_contains=("\uc694\uad6c\ubd84\uc11d", "ch04_"),
            forbidden_contains=("document.pdf", "\uc9c0\uc6d0\uc790\uaca9", "\ubaa8\uc9d1", "\uc785\ud559"),
        ),
        Case(
            id="mixed_software_architecture",
            query="\uc544\ud0a4\ud14d\ucc98 \uc124\uacc4\uc640 \ud074\ub798\uc2a4 \uc124\uacc4 \ucc28\uc774 \uc54c\ub824\uc918",
            expected_contains=("\uc544\ud0a4\ud14d\ucc98", "\ud074\ub798\uc2a4", "ch06_"),
            forbidden_contains=("document.pdf", "\ubaa8\uc9d1", "\uc6d0\uc11c\uc811\uc218"),
        ),
    ]

    if suite == "admission":
        return admission_cases
    if suite == "mixed":
        return mixed_cases
    return [*admission_cases, *mixed_cases]


def run_case(case: Case, *, base_url: str, api_key: str, timeout: float) -> dict[str, Any]:
    user = f"dify-chat-eval-{case.id}-{int(time.time() * 1000)}"
    payload = {
        "inputs": case.inputs,
        "query": case.query,
        "response_mode": "blocking",
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
    status: int | None = None
    answer = ""
    error = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            data = json.loads(response.read().decode("utf-8", errors="replace") or "{}")
            answer = str(data.get("answer") or "")
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", errors="replace")[:1000]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    latency = round(time.perf_counter() - started, 3)
    missing = [term for term in case.expected_contains if term not in answer]
    forbidden = [term for term in case.forbidden_contains if term in answer]
    has_source = SOURCE_MARKER in answer
    passed = bool(status and 200 <= status < 300 and not error and not missing and not forbidden)
    if case.require_source and not has_source:
        passed = False
    return {
        "id": case.id,
        "passed": passed,
        "status": status,
        "latency_seconds": latency,
        "inputs": case.inputs,
        "query": case.query,
        "has_source": has_source,
        "missing_expected": missing,
        "forbidden_found": forbidden,
        "answer": answer,
        "error": error,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    failed = sum(1 for result in results if not result.get("passed"))
    latencies = [float(result["latency_seconds"]) for result in results if result.get("status") == 200]
    return {
        "total": total,
        "passed": total - failed,
        "failed": failed,
        "pass_rate": round((total - failed) / total, 4) if total else 0,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "max_latency_seconds": max(latencies) if latencies else 0,
    }


def print_human(summary: dict[str, Any], results: list[dict[str, Any]], out_path: Path) -> None:
    print("Dify Chat API quality evaluation")
    print(f"- total/passed/failed: {summary['total']}/{summary['passed']}/{summary['failed']}")
    print(f"- pass_rate: {summary['pass_rate']:.2%}")
    print(f"- avg/max latency: {summary['avg_latency_seconds']}s/{summary['max_latency_seconds']}s")
    print(f"- saved: {out_path}")
    for result in results:
        status = "PASS" if result.get("passed") else "FAIL"
        print(
            f"[{status}] {result['id']} {result['latency_seconds']}s "
            f"missing={result['missing_expected']} forbidden={result['forbidden_found']}"
        )


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


if __name__ == "__main__":
    raise SystemExit(main())
