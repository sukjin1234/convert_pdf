from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.converter import PdfConverter
from app.rag import RagArtifact, RagStore, build_rag_artifact, lookup_matches, normalize_for_match, plan_query
from scripts.evaluate_rag_system import (
    SegmentView,
    evaluate_answer,
    evaluate_structure,
    extract_pdf_structure,
    percentile,
    validate_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate PDF parsing boundaries and local structured retrieval without Dify or an LLM."
    )
    parser.add_argument("--manifest", default=str(ROOT / "evaluation" / "rag_acceptance_cases.json"))
    parser.add_argument("--store-dir", default=str(ROOT / ".runtime" / "local-pdf-rag-store"))
    parser.add_argument("--reuse-artifacts", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", default=str(ROOT / ".runtime" / "local_pdf_rag_evaluation.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = validate_manifest(manifest, manifest_path)
    cases = list(manifest.get("cases") or [])
    if args.limit > 0:
        cases = cases[: args.limit]

    started = time.perf_counter()
    store = RagStore(Path(args.store_dir), refresh_interval_seconds=0)
    artifacts = load_or_build_artifacts(documents, store, reuse=args.reuse_artifacts)
    structure_results = [evaluate_artifact_structure(item, artifacts[item["key"]]) for item in documents]
    retrieval_results = [
        evaluate_retrieval_case(case, artifacts[str(case["document"])], top_k=max(1, args.top_k)) for case in cases
    ]
    report = {
        "manifest": str(manifest_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "structure_summary": summarize_local_structure(structure_results),
        "retrieval_summary": summarize_retrieval(retrieval_results),
        "structure_results": structure_results,
        "retrieval_results": retrieval_results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report, out_path)
    return 0


def load_or_build_artifacts(
    documents: list[dict[str, Any]], store: RagStore, *, reuse: bool
) -> dict[str, RagArtifact]:
    existing = {str(item.get("document_id")): item for item in store.list_documents()} if reuse else {}
    converter = PdfConverter(get_settings())
    artifacts: dict[str, RagArtifact] = {}
    for document in documents:
        key = str(document["key"])
        artifact = store.get_document(key) if key in existing else None
        if artifact is None:
            path = Path(document["path"])
            markdown = converter.convert_pdf_bytes(path.read_bytes(), path.name)
            artifact = build_rag_artifact(markdown, path.name, document_id=key)
            store.upsert(artifact)
        artifacts[key] = artifact
    return artifacts


def evaluate_artifact_structure(document: dict[str, Any], artifact: RagArtifact) -> dict[str, Any]:
    anchors, meaningful_pages = extract_pdf_structure(Path(document["path"]))
    segments = [
        SegmentView(
            chunk.chunk_id,
            chunk.page_start,
            chunk.section_path,
            " ".join(part for part in (chunk.title, chunk.text, chunk.metadata_text) if part),
        )
        for chunk in artifact.chunks
    ]
    result = evaluate_structure(anchors, meaningful_pages, segments)
    result.update(
        {
            "document_key": document["key"],
            "source_path": document["path"],
            "file_name": artifact.file_name,
            "record_count": len(artifact.records),
        }
    )
    return result


def evaluate_retrieval_case(case: dict[str, Any], artifact: RagArtifact, *, top_k: int) -> dict[str, Any]:
    started = time.perf_counter()
    query = str(case["question"])
    plan = plan_query(query, artifact.document_id, query)
    lookup = lookup_matches(
        query=query,
        retrieval_query=str(plan.get("retrieval_query") or ""),
        records=artifact.records,
        chunks=artifact.chunks,
        entities=list(plan.get("entities") or []),
        query_type=str(plan.get("query_type") or "semantic"),
        limit=top_k,
    )
    matches = list(lookup.get("matches") or [])
    evidence_text = str(lookup.get("context") or "")
    direct_answer = str(lookup.get("direct_answer") or "")
    evidence_correct, missing_evidence = evaluate_answer(evidence_text, case.get("answer") or {})
    direct_correct, missing_direct = evaluate_answer(direct_answer, case.get("answer") or {})
    source_ranks = [index for index, match in enumerate(matches, start=1) if match_matches_source(match, case.get("source") or {})]
    first_source_rank = source_ranks[0] if source_ranks else 0
    elapsed = time.perf_counter() - started
    return {
        "id": case.get("id"),
        "document": case.get("document"),
        "question": query,
        "query_type": plan.get("query_type"),
        "answer_style": plan.get("answer_style"),
        "retrieval_pass": bool(evidence_correct and first_source_rank),
        "evidence_answer_recall": evidence_correct,
        "source_recall_at_k": bool(first_source_rank),
        "first_source_rank": first_source_rank,
        "reciprocal_rank": round(1 / first_source_rank, 4) if first_source_rank else 0,
        "direct_answer_correct": direct_correct,
        "has_direct_answer": bool(direct_answer),
        "missing_evidence_groups": missing_evidence,
        "missing_direct_groups": missing_direct,
        "latency_seconds": round(elapsed, 6),
        "direct_answer": direct_answer,
        "top_matches": [compact_match(match) for match in matches[:top_k]],
    }


def match_matches_source(match: dict[str, Any], source: dict[str, Any]) -> bool:
    pages = {int(page) for page in source.get("pages") or []}
    page = match.get("page")
    page_match = not pages or (page is not None and int(page) in pages)
    section = normalize_for_match(match.get("section_path") or match.get("section") or "")
    section_groups = source.get("section_any") or []
    section_match = not section_groups or any(
        all(normalize_for_match(term) in section for term in (group if isinstance(group, list) else [group]))
        for group in section_groups
    )
    return (page_match or section_match) if pages and section_groups else page_match and section_match


def compact_match(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank_score": round(float(match.get("score") or 0), 4),
        "coverage": round(float(match.get("coverage") or 0), 4),
        "match_type": match.get("match_type"),
        "page": match.get("page"),
        "section": match.get("section_path") or match.get("section"),
        "record_type": match.get("record_type"),
        "fields": match.get("fields") or {},
    }


def summarize_local_structure(results: list[dict[str, Any]]) -> dict[str, Any]:
    weights = [max(1, int(item.get("anchor_count") or 0)) for item in results]
    total = sum(weights)
    return {
        "documents": len(results),
        "boundary_accuracy": round(
            sum(float(item["boundary_accuracy"]) * weight for item, weight in zip(results, weights)) / total, 4
        )
        if total
        else 0,
        "source_boundary_recall": round(
            sum(float(item["source_boundary_recall"]) * weight for item, weight in zip(results, weights)) / total, 4
        )
        if total
        else 0,
        "page_coverage": round(statistics.mean(float(item["page_coverage"]) for item in results), 4) if results else 0,
        "segment_boundary_precision": round(
            statistics.mean(float(item["segment_boundary_precision"]) for item in results), 4
        )
        if results
        else 0,
    }


def summarize_retrieval(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    latencies = [float(item["latency_seconds"]) for item in results]
    rate = lambda count: round(count / total, 4) if total else 0
    return {
        "cases": total,
        "retrieval_pass_rate": rate(sum(1 for item in results if item["retrieval_pass"])),
        "evidence_answer_recall": rate(sum(1 for item in results if item["evidence_answer_recall"])),
        "source_recall_at_k": rate(sum(1 for item in results if item["source_recall_at_k"])),
        "mrr": round(statistics.mean(float(item["reciprocal_rank"]) for item in results), 4) if results else 0,
        "direct_answer_coverage": rate(sum(1 for item in results if item["has_direct_answer"])),
        "direct_answer_accuracy": rate(sum(1 for item in results if item["direct_answer_correct"])),
        "average_latency_seconds": round(statistics.mean(latencies), 6) if latencies else 0,
        "p95_latency_seconds": percentile(latencies, 0.95),
    }


def print_human(report: dict[str, Any], out_path: Path) -> None:
    structure = report["structure_summary"]
    retrieval = report["retrieval_summary"]
    print("Local PDF RAG evaluation")
    print(f"- boundary accuracy: {structure['boundary_accuracy']:.2%}")
    print(
        f"- retrieval pass/evidence recall/source recall: {retrieval['retrieval_pass_rate']:.2%}/"
        f"{retrieval['evidence_answer_recall']:.2%}/{retrieval['source_recall_at_k']:.2%}"
    )
    print(f"- MRR/direct accuracy: {retrieval['mrr']:.4f}/{retrieval['direct_answer_accuracy']:.2%}")
    print(f"- local retrieval p95: {retrieval['p95_latency_seconds']}s")
    print(f"- saved: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
