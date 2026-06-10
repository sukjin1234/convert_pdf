from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAX_INLINE_TEXT_CHARS = 80_000


class EvalLogStore:
    def __init__(self, root_dir: Path | None = None):
        self.root_dir = root_dir or Path(os.getenv("EVAL_LOG_DIR", str(Path(tempfile.gettempdir()) / "dify-rag-eval-logs")))
        self._lock = threading.Lock()

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        run_id = clean_run_id(str(payload.get("run_id") or uuid.uuid4()))
        record = {
            "run_id": run_id,
            "created_at": now.isoformat(),
            "schema_version": 1,
            **payload,
        }
        record["run_id"] = run_id
        record["created_at"] = now.isoformat()
        record["size"] = estimate_json_size(record)

        with self._lock:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            day_path = self.root_dir / f"{now.date().isoformat()}.jsonl"
            run_path = self.root_dir / f"{run_id}.json"
            line = json.dumps(record, ensure_ascii=False, default=str)
            with day_path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
            run_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        return {
            "success": True,
            "run_id": run_id,
            "created_at": record["created_at"],
            "log_path": str(run_path),
            "jsonl_path": str(day_path),
            "size": record["size"],
        }

    def list(self, *, limit: int = 50, date: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        paths = [self.root_dir / f"{date}.jsonl"] if date else sorted(self.root_dir.glob("*.jsonl"), reverse=True)
        records: list[dict[str, Any]] = []
        for path in paths:
            if not path.exists():
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                records.append(summarize_record(record))
                if len(records) >= limit:
                    return records
        return records

    def get(self, run_id: str) -> dict[str, Any] | None:
        path = self.root_dir / f"{clean_run_id(run_id)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    query = first_non_empty(
        record.get("query"),
        nested_get(record, "input", "query"),
        nested_get(record, "sys", "query"),
    )
    final_answer = first_non_empty(
        record.get("final_answer"),
        nested_get(record, "output", "answer"),
        nested_get(record, "answer", "text"),
    )
    verify_valid = first_non_empty(
        nested_get(record, "verify_result", "valid"),
        nested_get(record, "verify", "valid"),
        nested_get(record, "node_status", "draft_answer_valid"),
    )
    return {
        "run_id": record.get("run_id", ""),
        "created_at": record.get("created_at", ""),
        "query": truncate_text(str(query or ""), 300),
        "document_id": record.get("document_id") or nested_get(record, "input", "document_id") or "",
        "final_answer": truncate_text(str(final_answer or ""), 500),
        "verify_valid": verify_valid,
        "size": record.get("size", 0),
    }


def clean_run_id(value: str) -> str:
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in {"-", "_"})
    return cleaned[:96] or str(uuid.uuid4())


def nested_get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 20)].rstrip() + "\n...[truncated]"


def estimate_json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


EVAL_LOG_STORE = EvalLogStore()
