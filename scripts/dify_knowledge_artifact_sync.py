from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag import RagArtifact  # noqa: E402
from scripts.dify_knowledge_pdf_perf_test import (  # noqa: E402
    KnowledgeConfig,
    build_process_rule,
    delete_document,
    encode_multipart,
    extract_created_document_id,
    extract_indexed_document_id,
    load_env_file,
    poll_indexing_status,
    request_json,
)


DEFAULT_DATASET_NAME = "ipsi"


@dataclass
class ArtifactSyncItem:
    document_id: str
    source_file_name: str
    upload_file_name: str
    markdown_chars: int
    chunk_count: int
    record_count: int
    status: str = "ready"
    existing_document_ids: list[str] | None = None
    batch: str = ""
    uploaded_document_id: str = ""
    indexing_statuses: list[str] | None = None
    error: str = ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload cleaned local RAG artifact dify_markdown files to a Dify Knowledge dataset."
    )
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--store-dir", default="", help="Defaults to RAG_STORE_DIR or DIFY_RUNTIME_DIR/rag-store.")
    parser.add_argument("--dataset-id", default="", help="Defaults to DIFY_KNOWLEDGE_DATASET_ID or DIFY_DATASET_ID.")
    parser.add_argument("--dataset-name", default=os.getenv("DIFY_DATASET_NAME", DEFAULT_DATASET_NAME))
    parser.add_argument("--document-id", action="append", default=[], help="Artifact document_id to upload. Repeatable.")
    parser.add_argument("--max-documents", type=int, default=0, help="Use only the first N selected artifacts. 0 means all.")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--index-timeout", type=float, default=900.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--doc-language", default="Korean")
    parser.add_argument("--upload", action="store_true", help="Actually create Knowledge documents. Default is dry-run.")
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="Resolve the Knowledge dataset during dry-run and mark artifacts whose upload file already exists.",
    )
    parser.add_argument(
        "--duplicate-action",
        choices=["skip", "replace", "upload"],
        default="skip",
        help="What to do when an upload_file_name already exists in Knowledge. replace deletes matching docs first.",
    )
    parser.add_argument("--out", default=str(ROOT / ".runtime" / "dify_knowledge_artifact_sync_latest.json"))
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    store_dir = resolve_store_dir(args.store_dir)
    artifacts = load_artifacts(store_dir, args.document_id)
    artifacts = sorted(artifacts, key=lambda artifact: artifact.document_id)
    if args.max_documents > 0:
        artifacts = artifacts[: args.max_documents]

    items = [artifact_sync_item(artifact) for artifact in artifacts]
    failed = not items or any(item.status != "ready" for item in items)
    config: KnowledgeConfig | None = None

    if args.upload or args.check_existing:
        config = build_config(args)
        existing_by_name = existing_documents_by_name(
            list_existing_knowledge_documents(config, timeout=args.timeout)
        )
        apply_duplicate_policy(items, existing_by_name, duplicate_action=args.duplicate_action, upload=args.upload)

    if args.upload:
        for item, artifact in zip(items, artifacts, strict=False):
            if item.status == "skipped_existing":
                continue
            if item.status == "replace_pending":
                replace_existing_documents(config, item, timeout=args.timeout)
                if item.error:
                    failed = True
                    continue
                item.status = "ready"
            if item.status != "ready":
                continue
            upload_item(config, item, artifact, args)
            if item.error:
                failed = True

    report = {
        "base_url_host": urllib.parse.urlparse(config.base_url).netloc if config else "",
        "dataset_id": config.dataset_id if config else args.dataset_id,
        "dataset_name": args.dataset_name,
        "store_dir": str(store_dir),
        "upload": bool(args.upload),
        "summary": summarize_items(items),
        "documents": [asdict(item) for item in items],
        "failed": failed,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report, out_path)
    return 1 if failed else 0


def resolve_store_dir(store_dir: str) -> Path:
    if store_dir:
        return Path(store_dir)
    explicit = os.getenv("RAG_STORE_DIR")
    if explicit and explicit.strip():
        return Path(explicit)
    runtime_root = Path(os.getenv("DIFY_RUNTIME_DIR", str(ROOT / ".runtime")))
    return runtime_root / "rag-store"


def load_artifacts(store_dir: Path, document_ids: list[str]) -> list[RagArtifact]:
    selected = {document_id for document_id in document_ids if document_id}
    if not store_dir.exists():
        return []
    artifacts: list[RagArtifact] = []
    for path in sorted(store_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            artifact = RagArtifact.from_dict(payload)
        except Exception:
            continue
        if selected and artifact.document_id not in selected:
            continue
        artifacts.append(artifact)
    return artifacts


def artifact_sync_item(artifact: RagArtifact) -> ArtifactSyncItem:
    markdown = artifact.dify_markdown.strip()
    status = "ready" if markdown else "skipped_empty_markdown"
    return ArtifactSyncItem(
        document_id=artifact.document_id,
        source_file_name=artifact.file_name,
        upload_file_name=artifact_upload_file_name(artifact),
        markdown_chars=len(markdown),
        chunk_count=len(artifact.chunks),
        record_count=len(artifact.records),
        status=status,
    )


def artifact_upload_file_name(artifact: RagArtifact) -> str:
    base = artifact.document_id or Path(artifact.file_name or "rag-artifact").stem
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", base).strip("._-")
    return f"{safe or 'rag-artifact'}.rag.md"


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


def list_existing_knowledge_documents(config: KnowledgeConfig, *, timeout: float) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"page": page, "limit": 100})
        payload = request_json(
            "GET",
            f"{config.base_url}/datasets/{config.dataset_id}/documents?{query}",
            config.api_key,
            timeout=timeout,
        )
        items = [item for item in payload_items(payload) if isinstance(item, dict)]
        documents.extend(items)
        total = safe_int(payload.get("total")) if isinstance(payload, dict) else 0
        if not items or len(items) < 100 or (total and len(documents) >= total):
            return documents
        page += 1


def existing_documents_by_name(documents: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_name: dict[str, list[str]] = {}
    for document in documents:
        document_id = str(document.get("id") or "")
        if not document_id:
            continue
        for name in existing_document_names(document):
            by_name.setdefault(name, []).append(document_id)
    return by_name


def existing_document_names(document: dict[str, Any]) -> set[str]:
    names = {
        clean_document_name(document.get("name")),
        clean_document_name(document.get("document_name")),
    }
    for key in ("data_source_info", "data_source_detail_dict", "data_source_detail"):
        value = document.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = {}
        if not isinstance(value, dict):
            continue
        names.add(clean_document_name(value.get("upload_file_name")))
        names.add(clean_document_name(value.get("file_name")))
        upload_file = value.get("upload_file") if isinstance(value.get("upload_file"), dict) else {}
        names.add(clean_document_name(upload_file.get("name")))
        names.add(clean_document_name(upload_file.get("filename")))
    return {name for name in names if name}


def clean_document_name(value: Any) -> str:
    return str(value or "").strip()


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def apply_duplicate_policy(
    items: list[ArtifactSyncItem],
    existing_by_name: dict[str, list[str]],
    *,
    duplicate_action: str,
    upload: bool,
) -> None:
    for item in items:
        if item.status != "ready":
            continue
        existing_ids = existing_by_name.get(item.upload_file_name, [])
        if not existing_ids:
            continue
        item.existing_document_ids = existing_ids
        if duplicate_action == "upload":
            continue
        if duplicate_action == "replace":
            item.status = "replace_pending" if upload else "would_replace_existing"
            continue
        item.status = "skipped_existing" if upload else "would_skip_existing"


def replace_existing_documents(config: KnowledgeConfig, item: ArtifactSyncItem, *, timeout: float) -> None:
    errors = []
    for document_id in item.existing_document_ids or []:
        error = delete_document(config, document_id, timeout=timeout)
        if error:
            errors.append(f"{document_id}: {error}")
    if errors:
        item.status = "error"
        item.error = "; ".join(errors)


def upload_item(config: KnowledgeConfig, item: ArtifactSyncItem, artifact: RagArtifact, args: argparse.Namespace) -> None:
    try:
        create_payload = create_document_from_artifact(
            config,
            item.upload_file_name,
            artifact.dify_markdown,
            doc_language=args.doc_language,
            timeout=args.timeout,
        )
        item.batch = str(create_payload.get("batch") or "")
        item.uploaded_document_id = extract_created_document_id(create_payload)
        if not item.batch:
            item.status = "error"
            item.error = f"Create document response did not include batch: {create_payload}"
            return

        indexing = poll_indexing_status(
            config,
            item.batch,
            timeout=args.timeout,
            index_timeout=args.index_timeout,
            poll_interval=args.poll_interval,
        )
        item.indexing_statuses = [str(status) for status in indexing.get("statuses") or []]
        if not item.uploaded_document_id:
            item.uploaded_document_id = extract_indexed_document_id(indexing)
        if indexing.get("failed"):
            item.status = "error"
            item.error = f"Indexing did not complete: {item.indexing_statuses}"
        else:
            item.status = "uploaded"
    except Exception as exc:  # noqa: BLE001 - keep syncing other artifacts and report per-item failure.
        item.status = "error"
        item.error = f"{type(exc).__name__}: {exc}"


def create_document_from_artifact(
    config: KnowledgeConfig,
    file_name: str,
    markdown: str,
    *,
    doc_language: str,
    timeout: float,
) -> dict[str, Any]:
    data = {
        "indexing_technique": config.indexing_technique or "high_quality",
        "doc_language": doc_language,
        "process_rule": build_process_rule(config),
    }
    if config.doc_form:
        data["doc_form"] = config.doc_form
    fields = {"data": json.dumps(data, ensure_ascii=False)}
    files = {"file": (file_name, markdown.encode("utf-8"), "text/markdown; charset=utf-8")}
    body, content_type = encode_multipart(fields, files)
    return request_json(
        "POST",
        f"{config.base_url}/datasets/{config.dataset_id}/document/create-by-file",
        config.api_key,
        body=body,
        content_type=content_type,
        timeout=timeout,
    )


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


def summarize_items(items: list[ArtifactSyncItem]) -> dict[str, Any]:
    return {
        "total": len(items),
        "ready": sum(1 for item in items if item.status == "ready"),
        "uploaded": sum(1 for item in items if item.status == "uploaded"),
        "skipped": sum(1 for item in items if item.status.startswith("skipped") or item.status.startswith("would_skip")),
        "replace_pending": sum(1 for item in items if item.status in {"replace_pending", "would_replace_existing"}),
        "errors": sum(1 for item in items if item.status == "error"),
        "markdown_chars": sum(item.markdown_chars for item in items),
    }


def print_human(report: dict[str, Any], out_path: Path) -> None:
    summary = report.get("summary") or {}
    mode = "upload" if report.get("upload") else "dry-run"
    print("Dify Knowledge artifact sync")
    print(f"- mode: {mode}")
    print(f"- store_dir: {report.get('store_dir')}")
    print(f"- dataset_id: {report.get('dataset_id') or '(not resolved in dry-run)'}")
    print(
        f"- total/ready/uploaded/skipped/replace/errors: {summary.get('total')}/"
        f"{summary.get('ready')}/{summary.get('uploaded')}/{summary.get('skipped')}/"
        f"{summary.get('replace_pending')}/{summary.get('errors')}"
    )
    print(f"- markdown_chars: {summary.get('markdown_chars')}")
    for item in report.get("documents", [])[:10]:
        print(
            f"  - {item.get('document_id')} -> {item.get('upload_file_name')} "
            f"{item.get('status')} chars={item.get('markdown_chars')} "
            f"existing={len(item.get('existing_document_ids') or [])}"
        )
    print(f"- saved: {out_path}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI should return a clear one-line failure.
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
