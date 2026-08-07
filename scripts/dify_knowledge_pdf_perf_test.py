from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_NAME = "ipsi"
TERMINAL_INDEXING_STATUSES = {"completed", "error", "paused", "stopped"}


@dataclass(frozen=True)
class KnowledgeConfig:
    base_url: str
    api_key: str
    dataset_id: str
    doc_form: str = ""
    indexing_technique: str = "high_quality"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure PDF upload, parsing, chunking, indexing, and retrieval through the Dify Knowledge API."
    )
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--dataset-id", default="", help="Defaults to DIFY_KNOWLEDGE_DATASET_ID or DIFY_DATASET_ID.")
    parser.add_argument("--dataset-name", default=os.getenv("DIFY_DATASET_NAME", DEFAULT_DATASET_NAME))
    parser.add_argument("--file", default="", help="PDF file to upload. If omitted, a small generated PDF is used.")
    parser.add_argument("--query", default="Codex Knowledge PDF performance smoke test")
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP request timeout seconds.")
    parser.add_argument("--index-timeout", type=float, default=900.0, help="Maximum seconds to wait for indexing.")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--out", default=str(ROOT / ".runtime" / "dify_knowledge_pdf_perf_latest.json"))
    parser.add_argument("--json", action="store_true", help="Print the JSON performance report.")
    parser.add_argument("--keep-document", action="store_true", help="Do not delete the uploaded smoke-test document.")
    parser.add_argument("--max-upload-seconds", type=float, default=0.0, help="0 disables this gate.")
    parser.add_argument("--max-index-seconds", type=float, default=0.0, help="0 disables this gate.")
    parser.add_argument("--allow-segment-miss", action="store_true", help="Do not fail when uploaded PDF text is absent from its segments.")
    parser.add_argument("--fail-on-retrieve-miss", action="store_true")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    config = build_config(args)
    source_path, generated = resolve_pdf_source(args.file)
    started = time.perf_counter()
    document_id = ""
    cleanup_error = ""

    try:
        upload_started = time.perf_counter()
        create_payload = create_document_by_file(config, source_path, query=args.query, timeout=args.timeout)
        upload_seconds = round(time.perf_counter() - upload_started, 6)
        batch = str(create_payload.get("batch") or "")
        document_id = extract_created_document_id(create_payload)
        if not batch:
            raise RuntimeError(f"Create document response did not include batch: {create_payload}")

        indexing_started = time.perf_counter()
        indexing = poll_indexing_status(
            config,
            batch,
            timeout=args.timeout,
            index_timeout=args.index_timeout,
            poll_interval=args.poll_interval,
        )
        index_seconds = round(time.perf_counter() - indexing_started, 6)
        if not document_id:
            document_id = extract_indexed_document_id(indexing)

        segments_started = time.perf_counter()
        segments_payload = fetch_document_segments(config, document_id, timeout=args.timeout) if document_id else {}
        segments_seconds = round(time.perf_counter() - segments_started, 6)
        segment_items = segment_items_from_payload(segments_payload)
        segment_hit = any(args.query.lower() in item.get("content_preview", "").lower() for item in segment_items)

        retrieve_started = time.perf_counter()
        retrieve_payload = retrieve_chunks(config, args.query, timeout=args.timeout)
        retrieve_seconds = round(time.perf_counter() - retrieve_started, 6)
        retrieve_items = retrieval_items(retrieve_payload)
        retrieve_hit = any(args.query.lower().split()[0] in item.get("content_preview", "").lower() for item in retrieve_items)
        failed = bool(
            indexing.get("failed")
            or (args.max_upload_seconds > 0 and upload_seconds > args.max_upload_seconds)
            or (args.max_index_seconds > 0 and index_seconds > args.max_index_seconds)
            or (not args.allow_segment_miss and not segment_hit)
            or (args.fail_on_retrieve_miss and not retrieve_hit)
        )
        report = {
            "dataset_id": config.dataset_id,
            "source_file": str(source_path),
            "generated_source": generated,
            "document_id": document_id,
            "batch": batch,
            "timing": {
                "upload_seconds": upload_seconds,
                "index_seconds": index_seconds,
                "segments_seconds": segments_seconds,
                "retrieve_seconds": retrieve_seconds,
                "total_seconds": round(time.perf_counter() - started, 6),
            },
            "indexing": indexing,
            "segment_hit": segment_hit,
            "segments": segment_items,
            "retrieve_hit": retrieve_hit,
            "retrieve_items": retrieve_items,
            "failed": failed,
        }
    finally:
        if generated:
            source_path.unlink(missing_ok=True)

    if document_id and not args.keep_document:
        cleanup_error = delete_document(config, document_id, timeout=args.timeout)

    report["cleanup"] = {
        "attempted": bool(document_id and not args.keep_document),
        "error": cleanup_error,
    }
    if cleanup_error:
        report["failed"] = True

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report, out_path)
    return 1 if report.get("failed") else 0


def build_config(args: argparse.Namespace) -> KnowledgeConfig:
    base_url = (os.getenv("DIFY_API_BASE_URL") or "").rstrip("/")
    api_key = os.getenv("DIFY_KNOWLEDGE_API_KEY") or ""
    if not base_url:
        raise RuntimeError(f"DIFY_API_BASE_URL is missing. Add it to {args.env_file}.")
    if not api_key:
        raise RuntimeError(f"DIFY_KNOWLEDGE_API_KEY is missing. Add it to {args.env_file}.")

    dataset_id = args.dataset_id or os.getenv("DIFY_KNOWLEDGE_DATASET_ID") or os.getenv("DIFY_DATASET_ID") or ""
    if not dataset_id:
        dataset_id = resolve_dataset_id(base_url, api_key, args.dataset_name, timeout=args.timeout)
    if not dataset_id:
        raise RuntimeError("Knowledge dataset id was not found. Pass --dataset-id or set DIFY_KNOWLEDGE_DATASET_ID.")
    dataset = request_json("GET", f"{base_url}/datasets/{dataset_id}", api_key, timeout=args.timeout)
    return KnowledgeConfig(
        base_url=base_url,
        api_key=api_key,
        dataset_id=dataset_id,
        doc_form=str(dataset.get("doc_form") or ""),
        indexing_technique=str(dataset.get("indexing_technique") or "high_quality"),
    )


def resolve_dataset_id(base_url: str, api_key: str, dataset_name: str, *, timeout: float) -> str:
    query = urllib.parse.urlencode({"limit": 100})
    payload = request_json("GET", f"{base_url}/datasets?{query}", api_key, timeout=timeout)
    for item in payload_items(payload):
        if isinstance(item, dict) and str(item.get("name") or "").strip() == dataset_name:
            return str(item.get("id") or "")
    return ""


def resolve_pdf_source(file_path: str) -> tuple[Path, bool]:
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path, False

    tmp = tempfile.NamedTemporaryFile("wb", prefix="dify-knowledge-pdf-smoke-", suffix=".pdf", delete=False)
    with tmp:
        tmp.write(sample_pdf_bytes())
    return Path(tmp.name), True


def sample_pdf_bytes() -> bytes:
    body = b"BT /F1 18 Tf 72 720 Td (Codex Knowledge PDF performance smoke test) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(body)).encode("ascii") + b" >>\nstream\n" + body + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


def create_document_by_file(config: KnowledgeConfig, path: Path, *, query: str, timeout: float) -> dict[str, Any]:
    data = {
        "indexing_technique": config.indexing_technique or "high_quality",
        "doc_language": "English",
        "process_rule": {"mode": "automatic"},
    }
    if config.doc_form:
        data["doc_form"] = config.doc_form
    fields = {"data": json.dumps(data, ensure_ascii=False)}
    files = {"file": (path.name, path.read_bytes(), "application/pdf")}
    body, content_type = encode_multipart(fields, files)
    return request_json(
        "POST",
        f"{config.base_url}/datasets/{config.dataset_id}/document/create-by-file",
        config.api_key,
        body=body,
        content_type=content_type,
        timeout=timeout,
    )


def poll_indexing_status(
    config: KnowledgeConfig,
    batch: str,
    *,
    timeout: float,
    index_timeout: float,
    poll_interval: float,
) -> dict[str, Any]:
    deadline = time.perf_counter() + index_timeout
    samples = []
    while True:
        payload = request_json(
            "GET",
            f"{config.base_url}/datasets/{config.dataset_id}/documents/{batch}/indexing-status",
            config.api_key,
            timeout=timeout,
        )
        entries = [item for item in payload_items(payload) if isinstance(item, dict)]
        statuses = [str(entry.get("indexing_status") or "") for entry in entries]
        samples.append(
            {
                "elapsed_seconds": round(index_timeout - max(deadline - time.perf_counter(), 0), 3),
                "statuses": statuses,
                "completed_segments": sum(safe_int(entry.get("completed_segments")) for entry in entries),
                "total_segments": sum(safe_int(entry.get("total_segments")) for entry in entries),
            }
        )
        if entries and all(status in TERMINAL_INDEXING_STATUSES for status in statuses):
            return {
                "statuses": statuses,
                "entries": entries,
                "polls": len(samples),
                "samples": samples,
                "failed": any(status != "completed" for status in statuses),
            }
        if time.perf_counter() >= deadline:
            return {"statuses": statuses, "entries": entries, "polls": len(samples), "samples": samples, "failed": True}
        time.sleep(max(poll_interval, 0.5))


def retrieve_chunks(config: KnowledgeConfig, query: str, *, timeout: float) -> dict[str, Any]:
    return request_json(
        "POST",
        f"{config.base_url}/datasets/{config.dataset_id}/retrieve",
        config.api_key,
        body=json.dumps({"query": query}, ensure_ascii=False).encode("utf-8"),
        content_type="application/json; charset=utf-8",
        timeout=timeout,
    )


def fetch_document_segments(config: KnowledgeConfig, document_id: str, *, timeout: float) -> dict[str, Any]:
    query = urllib.parse.urlencode({"limit": 20})
    return request_json(
        "GET",
        f"{config.base_url}/datasets/{config.dataset_id}/documents/{document_id}/segments?{query}",
        config.api_key,
        timeout=timeout,
    )


def delete_document(config: KnowledgeConfig, document_id: str, *, timeout: float) -> str:
    try:
        request_json(
            "DELETE",
            f"{config.base_url}/datasets/{config.dataset_id}/documents/{document_id}",
            config.api_key,
            timeout=timeout,
            allow_empty=True,
        )
        return ""
    except Exception as exc:  # noqa: BLE001 - cleanup failure should be reported in the JSON result.
        return f"{type(exc).__name__}: {exc}"


def request_json(
    method: str,
    url: str,
    api_key: str,
    *,
    body: bytes | None = None,
    content_type: str = "",
    timeout: float,
    allow_empty: bool = False,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        if allow_empty and exc.code == 204:
            return {}
        raise RuntimeError(f"HTTP {exc.code}: {text[:1000]}") from exc


def encode_multipart(
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----codex-dify-{int(time.time() * 1000)}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, (filename, content, content_type) in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode("utf-8"),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def extract_created_document_id(payload: dict[str, Any]) -> str:
    document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    return str(document.get("id") or payload.get("document_id") or "")


def extract_indexed_document_id(indexing: dict[str, Any]) -> str:
    entries = indexing.get("entries") if isinstance(indexing.get("entries"), list) else []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id"):
            return str(entry.get("id"))
    return ""


def retrieval_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for item in payload_items(payload)[:5]:
        if not isinstance(item, dict):
            continue
        segment = item.get("segment") if isinstance(item.get("segment"), dict) else item
        content = str(segment.get("content") or item.get("content") or "")
        items.append(
            {
                "score": item.get("score") or segment.get("score"),
                "segment_id": segment.get("id"),
                "content_preview": compact_text(content, 300),
            }
        )
    return items


def segment_items_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for segment in payload_items(payload)[:20]:
        if not isinstance(segment, dict):
            continue
        items.append(
            {
                "segment_id": segment.get("id"),
                "word_count": segment.get("word_count"),
                "content_preview": compact_text(segment.get("content") or "", 500),
            }
        )
    return items


def payload_items(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        value = payload.get("data")
        if isinstance(value, list):
            return value
        for key in ("records", "documents"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    if isinstance(payload, list):
        return payload
    return []


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def compact_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


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


def print_human(report: dict[str, Any], out_path: Path) -> None:
    timing = report.get("timing") or {}
    indexing = report.get("indexing") or {}
    print("Dify Knowledge PDF performance")
    print(f"- dataset_id: {report.get('dataset_id')}")
    print(f"- document_id: {report.get('document_id')}")
    print(f"- batch: {report.get('batch')}")
    print(
        f"- upload/index/segments/retrieve/total: {timing.get('upload_seconds')}s/"
        f"{timing.get('index_seconds')}s/{timing.get('segments_seconds')}s/"
        f"{timing.get('retrieve_seconds')}s/"
        f"{timing.get('total_seconds')}s"
    )
    print(f"- indexing statuses: {indexing.get('statuses')}, polls={indexing.get('polls')}")
    print(f"- segment_hit: {report.get('segment_hit')}")
    print(f"- retrieve_hit: {report.get('retrieve_hit')}")
    print(f"- cleanup: {report.get('cleanup')}")
    print(f"- saved: {out_path}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI should return a clear one-line failure.
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
