from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "evaluation" / "dify_perf_questions.json"


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SendResult:
    index: int
    iteration: int
    question_id: str
    question: str
    ok: bool
    status_code: int | None
    latency_seconds: float
    conversation_id: str
    error: str
    workflow_run_id: str = ""
    time_to_first_token_seconds: float = 0
    total_tokens: int = 0


@dataclass
class ResponseStats:
    conversation_id: str = ""
    workflow_run_id: str = ""
    time_to_first_token_seconds: float = 0
    total_tokens: int = 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send performance-test questions to the Dify Chat Messages API."
    )
    parser.add_argument("--env-file", default=str(ROOT / ".env"), help="Env file containing DIFY_API_BASE_URL and DIFY_APP_API_KEY.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="Question file: JSON, TXT, or Markdown table.")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N questions. 0 means all.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the question set N times.")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent request count.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout seconds.")
    parser.add_argument("--response-mode", choices=["blocking", "streaming"], default="blocking")
    parser.add_argument("--user-prefix", default="dify-perf", help="Prefix for Dify user identifiers.")
    parser.add_argument("--inputs-json", default="{}", help="JSON object for Dify inputs.")
    parser.add_argument("--inputs-file", default="", help="Path to JSON object for Dify inputs.")
    parser.add_argument(
        "--reuse-conversation",
        action="store_true",
        help="Reuse one conversation_id per concurrency slot. Default sends each request as a new conversation.",
    )
    parser.add_argument("--out", default="", help="Optional path for a JSON performance report.")
    parser.add_argument("--json", action="store_true", help="Print the JSON performance report.")
    parser.add_argument(
        "--fail-under-success-rate",
        type=float,
        default=1.0,
        help="Fail if successful request ratio is below this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-p95-latency",
        type=float,
        default=0.0,
        help="Fail if p95 latency exceeds this many seconds. 0 disables this gate.",
    )
    parser.add_argument(
        "--max-avg-latency",
        type=float,
        default=0.0,
        help="Fail if average latency exceeds this many seconds. 0 disables this gate.",
    )
    parser.add_argument("--quiet", action="store_true", help="Print only the final send summary.")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))

    base_url = (os.getenv("DIFY_API_BASE_URL") or "").rstrip("/")
    api_key = os.getenv("DIFY_APP_API_KEY") or os.getenv("DIFY_API_KEY") or ""
    if not base_url:
        print(f"DIFY_API_BASE_URL is missing. Add it to {args.env_file}.", file=sys.stderr)
        return 2
    if not api_key:
        print(f"DIFY_APP_API_KEY is missing. Add it to {args.env_file}.", file=sys.stderr)
        return 2

    inputs = load_inputs(args.inputs_json, args.inputs_file)
    default_document_id = os.getenv("DIFY_DOCUMENT_ID") or os.getenv("DIFY_DEFAULT_DOCUMENT_ID") or ""
    if default_document_id and "document_id" not in inputs:
        inputs["document_id"] = default_document_id
    questions = load_questions(Path(args.questions))
    if args.limit > 0:
        questions = questions[: args.limit]
    if not questions:
        print(f"No questions found in {args.questions}", file=sys.stderr)
        return 2

    repeat = max(1, args.repeat)
    concurrency = max(1, args.concurrency)
    tasks: list[tuple[int, int, int, Question]] = []
    for iteration in range(1, repeat + 1):
        for question in questions:
            index = len(tasks) + 1
            slot = (index - 1) % concurrency
            tasks.append((index, iteration, slot, question))

    started = time.perf_counter()
    print_lock = threading.Lock()

    def print_progress(result: SendResult) -> None:
        if args.quiet:
            return
        status = "OK" if result.ok else "ERR"
        with print_lock:
            print(
                f"[{status}] #{result.index} {result.latency_seconds:.3f}s "
                f"{result.question_id} {result.question[:60]}"
            )

    def run_task(task: tuple[int, int, int, Question], conversation_id: str = "") -> SendResult:
        index, iteration, slot, question = task
        request_inputs = {**inputs, **question.inputs}
        return send_chat_message(
            base_url=base_url,
            api_key=api_key,
            question=question,
            inputs=request_inputs,
            response_mode=args.response_mode,
            conversation_id=conversation_id,
            user=f"{args.user_prefix}-{slot + 1}",
            timeout=args.timeout,
            index=index,
            iteration=iteration,
        )

    results: list[SendResult] = []
    if args.reuse_conversation:
        tasks_by_slot: list[list[tuple[int, int, int, Question]]] = [[] for _ in range(concurrency)]
        for task in tasks:
            tasks_by_slot[task[2]].append(task)

        def run_slot(slot_tasks: list[tuple[int, int, int, Question]]) -> list[SendResult]:
            slot_results: list[SendResult] = []
            conversation_id = ""
            for task in slot_tasks:
                result = run_task(task, conversation_id=conversation_id)
                if result.conversation_id:
                    conversation_id = result.conversation_id
                slot_results.append(result)
                print_progress(result)
            return slot_results

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(run_slot, slot_tasks) for slot_tasks in tasks_by_slot if slot_tasks]
            for future in as_completed(futures):
                results.extend(future.result())
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(run_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print_progress(result)

    elapsed = time.perf_counter() - started
    results.sort(key=lambda item: item.index)
    summary = summarize_results(results, elapsed)
    report = {
        "base_url_host": base_url.split("//")[-1].split("/")[0],
        "question_file": str(Path(args.questions)),
        "repeat": repeat,
        "concurrency": concurrency,
        "response_mode": args.response_mode,
        "reuse_conversation": bool(args.reuse_conversation),
        "summary": summary,
        "results": [asdict(result) for result in results],
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_summary(summary, results, Path(args.out) if args.out else None)

    failed_gate = bool(
        (args.fail_under_success_rate > 0 and summary["success_rate"] < args.fail_under_success_rate)
        or (args.max_p95_latency > 0 and summary["latency_p95_seconds"] > args.max_p95_latency)
        or (args.max_avg_latency > 0 and summary["latency_avg_seconds"] > args.max_avg_latency)
    )
    return 1 if failed_gate else 0


def send_chat_message(
    *,
    base_url: str,
    api_key: str,
    question: Question,
    inputs: dict[str, Any],
    response_mode: str,
    conversation_id: str,
    user: str,
    timeout: float,
    index: int,
    iteration: int,
) -> SendResult:
    payload = {
        "inputs": inputs,
        "query": question.text,
        "response_mode": response_mode,
        "conversation_id": conversation_id,
        "user": user,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat-messages",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )

    started = time.perf_counter()
    status_code: int | None = None
    response_conversation_id = ""
    workflow_run_id = ""
    time_to_first_token_seconds = 0.0
    total_tokens = 0
    error = ""

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = response.status
            if response_mode == "streaming":
                stats = drain_sse_response(response, started_at=started)
                response_conversation_id = stats.conversation_id
                workflow_run_id = stats.workflow_run_id
                time_to_first_token_seconds = stats.time_to_first_token_seconds
                total_tokens = stats.total_tokens
            else:
                response_body = response.read().decode("utf-8")
                data = json.loads(response_body) if response_body else {}
                stats = response_stats_from_json(data)
                response_conversation_id = stats.conversation_id
                workflow_run_id = stats.workflow_run_id
                time_to_first_token_seconds = stats.time_to_first_token_seconds
                total_tokens = stats.total_tokens
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        error_body = exc.read().decode("utf-8", errors="replace")
        error = f"HTTP {exc.code}: {error_body[:500]}"
    except urllib.error.URLError as exc:
        error = f"URL error: {exc.reason}"
    except TimeoutError:
        error = f"Timeout after {timeout}s"
    except Exception as exc:  # noqa: BLE001 - CLI should record unexpected per-request failures.
        error = f"{type(exc).__name__}: {exc}"

    latency_seconds = time.perf_counter() - started
    ok = bool(status_code and 200 <= status_code < 300 and not error)
    return SendResult(
        index=index,
        iteration=iteration,
        question_id=question.id,
        question=question.text,
        ok=ok,
        status_code=status_code,
        latency_seconds=round(latency_seconds, 6),
        conversation_id=response_conversation_id,
        error=error,
        workflow_run_id=workflow_run_id,
        time_to_first_token_seconds=round(time_to_first_token_seconds, 6),
        total_tokens=total_tokens,
    )


def drain_sse_response(response: Any, *, started_at: float) -> ResponseStats:
    stats = ResponseStats()

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data_text = line.removeprefix("data:").strip()
        if data_text == "[DONE]":
            break
        try:
            data = json.loads(data_text)
        except json.JSONDecodeError:
            continue

        if data.get("conversation_id"):
            stats.conversation_id = str(data.get("conversation_id"))
        if data.get("workflow_run_id"):
            stats.workflow_run_id = str(data.get("workflow_run_id"))
        if data.get("event") in {"message", "agent_message"} and data.get("answer") and not stats.time_to_first_token_seconds:
            stats.time_to_first_token_seconds = time.perf_counter() - started_at
        apply_usage_stats(stats, data)

    return stats


def response_stats_from_json(data: dict[str, Any]) -> ResponseStats:
    stats = ResponseStats(
        conversation_id=str(data.get("conversation_id") or ""),
        workflow_run_id=str(data.get("workflow_run_id") or ""),
    )
    apply_usage_stats(stats, data)
    return stats


def apply_usage_stats(stats: ResponseStats, data: dict[str, Any]) -> None:
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else data.get("usage")
    if not isinstance(usage, dict):
        return
    if not stats.total_tokens:
        stats.total_tokens = safe_int(usage.get("total_tokens"))
    usage_ttft = usage.get("time_to_first_token")
    if not stats.time_to_first_token_seconds and usage_ttft is not None:
        try:
            stats.time_to_first_token_seconds = float(usage_ttft)
        except (TypeError, ValueError):
            pass


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def load_inputs(inputs_json: str, inputs_file: str) -> dict[str, Any]:
    if inputs_file:
        value = json.loads(Path(inputs_file).read_text(encoding="utf-8"))
    else:
        value = json.loads(inputs_json or "{}")
    if not isinstance(value, dict):
        raise ValueError("Dify inputs must be a JSON object.")
    return value


def load_questions(path: Path) -> list[Question]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return questions_from_json(payload)
    if suffix in {".txt", ".text"}:
        return [
            Question(id=f"q{line_number}", text=line.strip())
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if suffix in {".md", ".markdown"}:
        return questions_from_markdown(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported questions file type: {path}")


def questions_from_json(payload: Any) -> list[Question]:
    if isinstance(payload, list):
        questions = []
        for index, item in enumerate(payload, start=1):
            if isinstance(item, str):
                questions.append(Question(id=f"q{index}", text=item))
            elif isinstance(item, dict) and item.get("question"):
                item_inputs = item.get("inputs") if isinstance(item.get("inputs"), dict) else {}
                question_inputs = dict(item_inputs)
                if item.get("document_id") and "document_id" not in question_inputs:
                    question_inputs["document_id"] = str(item["document_id"])
                questions.append(
                    Question(
                        id=str(item.get("id") or f"q{index}"),
                        text=str(item["question"]),
                        inputs=question_inputs,
                    )
                )
        return questions

    if isinstance(payload, dict):
        if isinstance(payload.get("questions"), list):
            return questions_from_json(payload["questions"])
        if isinstance(payload.get("cases"), list):
            return questions_from_json(payload["cases"])

    raise ValueError("JSON questions must be a list, {questions: [...]}, or {cases: [{question: ...}]}.")


def questions_from_markdown(markdown: str) -> list[Question]:
    questions: list[Question] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped:
            continue
        columns = [column.strip() for column in stripped.strip("|").split("|")]
        if len(columns) < 2:
            continue
        if columns[0].lower() in {"id", "항목"} or columns[1] == "질문":
            continue
        if columns[1]:
            questions.append(Question(id=columns[0] or f"q{len(questions) + 1}", text=columns[1]))
    return questions


def summarize_results(results: list[SendResult], elapsed_seconds: float) -> dict[str, Any]:
    success_count = sum(1 for result in results if result.ok)
    failed = [result for result in results if not result.ok]
    throughput = len(results) / elapsed_seconds if elapsed_seconds > 0 else 0
    latencies = sorted(result.latency_seconds for result in results if result.ok)
    ttfts = sorted(result.time_to_first_token_seconds for result in results if result.ok and result.time_to_first_token_seconds > 0)
    token_counts = [result.total_tokens for result in results if result.ok and result.total_tokens > 0]
    return {
        "total": len(results),
        "success": success_count,
        "failed": len(failed),
        "success_rate": round(success_count / len(results), 4) if results else 0,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "throughput_requests_per_second": round(throughput, 6),
        "latency_avg_seconds": round(sum(latencies) / len(latencies), 6) if latencies else 0,
        "latency_p50_seconds": percentile(latencies, 50),
        "latency_p95_seconds": percentile(latencies, 95),
        "latency_max_seconds": max(latencies) if latencies else 0,
        "ttft_avg_seconds": round(sum(ttfts) / len(ttfts), 6) if ttfts else 0,
        "ttft_p50_seconds": percentile(ttfts, 50),
        "ttft_p95_seconds": percentile(ttfts, 95),
        "ttft_max_seconds": max(ttfts) if ttfts else 0,
        "total_tokens": sum(token_counts),
        "tokens_avg": round(sum(token_counts) / len(token_counts), 3) if token_counts else 0,
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


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def print_summary(summary: dict[str, Any], results: list[SendResult], out_path: Path | None) -> None:
    failed = [result for result in results if not result.ok]
    print("")
    print("Dify question send summary")
    print(f"- total/success/failed: {summary['total']}/{summary['success']}/{summary['failed']}")
    print(f"- success rate: {summary['success_rate']:.2%}")
    print(f"- elapsed: {summary['elapsed_seconds']:.3f}s")
    print(f"- send throughput: {summary['throughput_requests_per_second']:.3f} req/s")
    print(
        f"- latency avg/p50/p95/max: {summary['latency_avg_seconds']:.3f}s/"
        f"{summary['latency_p50_seconds']:.3f}s/{summary['latency_p95_seconds']:.3f}s/"
        f"{summary['latency_max_seconds']:.3f}s"
    )
    if summary.get("ttft_avg_seconds"):
        print(
            f"- TTFT avg/p50/p95/max: {summary['ttft_avg_seconds']:.3f}s/"
            f"{summary['ttft_p50_seconds']:.3f}s/{summary['ttft_p95_seconds']:.3f}s/"
            f"{summary['ttft_max_seconds']:.3f}s"
        )
    if summary.get("total_tokens"):
        print(f"- tokens total/avg: {summary['total_tokens']}/{summary['tokens_avg']:.1f}")
    if out_path:
        print(f"- saved: {out_path}")
    if failed:
        print("- error samples:")
        for result in failed[:10]:
            print(f"  #{result.index} {result.status_code} {result.error}")


if __name__ == "__main__":
    raise SystemExit(main())
