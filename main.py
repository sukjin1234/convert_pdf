"""FastAPI 서버 — PDF to Markdown 변환 API

실행: python main.py
"""

import asyncio
import glob
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager, suppress
from typing import Any, Optional

import cv2
import easyocr
import numpy as np
import opendataloader_pdf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── GPU ──────────────────────────────────────────────────────────
USE_GPU = torch.cuda.is_available()

# ── 환경 변수 ────────────────────────────────────────────────────
HYBRID_PORT          = int(os.environ.get("HYBRID_PORT", "5002"))
HYBRID_URL           = f"http://127.0.0.1:{HYBRID_PORT}"
DEFAULT_OCR_LANG     = os.environ.get("HYBRID_OCR_LANG", "ko,en")
DEFAULT_FORCE_OCR    = os.environ.get("HYBRID_FORCE_OCR", "").strip().lower() in ("1", "true", "yes", "on")
HYBRID_PAGE_CHUNK    = max(1, int(os.environ.get("HYBRID_PAGE_CHUNK", "10")))
HYBRID_RECYCLE_EVERY = max(1, int(os.environ.get("HYBRID_RECYCLE_EVERY", "2")))
HYBRID_WORKERS       = max(1, int(os.environ.get("HYBRID_WORKERS", "4")))
MAX_UPLOAD_MB        = int(os.environ.get("MAX_UPLOAD_MB", "200"))
RAG_TABLE_RECORD_MAX_ROWS = max(1, int(os.environ.get("RAG_TABLE_RECORD_MAX_ROWS", "40")))
RAG_TABLE_CELL_MAX_CHARS = max(32, int(os.environ.get("RAG_TABLE_CELL_MAX_CHARS", "180")))
RAG_TABLE_CONTEXT_COLS = max(0, int(os.environ.get("RAG_TABLE_CONTEXT_COLS", "2")))
OCR_MIN_CONFIDENCE = min(0.99, max(0.0, float(os.environ.get("OCR_MIN_CONFIDENCE", "0.35"))))
OCR_LINE_MERGE_RATIO = min(0.08, max(0.003, float(os.environ.get("OCR_LINE_MERGE_RATIO", "0.012"))))
PDF_TABLE_EXTRACT_TIMEOUT_S = max(10, int(os.environ.get("PDF_TABLE_EXTRACT_TIMEOUT_S", "120")))
USE_PYMUPDF_TABLES = os.environ.get("USE_PYMUPDF_TABLES", "true").strip().lower() in ("1", "true", "yes", "on")
PYMUPDF_SUPPRESS_LOGS = os.environ.get("PYMUPDF_SUPPRESS_LOGS", "true").strip().lower() in ("1", "true", "yes", "on")

# ── 전역 락 (lifespan에서 초기화) ────────────────────────────────
_ocr_lock: asyncio.Lock
_hybrid_lock: asyncio.Lock
_img2table_lock: asyncio.Lock

# ── OCR 리더 싱글톤 ──────────────────────────────────────────────
_ocr_reader: Optional[easyocr.Reader] = None
_ocr_reader_langs: Optional[str] = None

# ── img2table OCR 싱글톤 (표 인식 전용) ─────────────────────────
_img2table_ocr: Optional[Any] = None
_img2table_ocr_langs: Optional[str] = None

# ── 전역 하이브리드 프로세스 (소형 PDF 전용) ─────────────────────
_hybrid_proc: Optional[subprocess.Popen] = None
_hybrid_proc_lang: str = DEFAULT_OCR_LANG
_hybrid_proc_force_ocr: bool = DEFAULT_FORCE_OCR


# ════════════════════════════════════════════════════════════════
# 하이브리드 CLI 유틸
# ════════════════════════════════════════════════════════════════

def _find_hybrid_cli() -> Optional[str]:
    scripts = os.path.dirname(sys.executable)
    for name in ("opendataloader-pdf-hybrid", "opendataloader-pdf-hybrid.exe"):
        path = os.path.join(scripts, name)
        if os.path.isfile(path):
            return path
    return shutil.which("opendataloader-pdf-hybrid")


def _build_hybrid_cmd(cli: str, ocr_lang: str, force_ocr: bool, port: int) -> list[str]:
    cmd = [cli, "--port", str(port), "--ocr-lang", ocr_lang]
    if force_ocr:
        cmd.append("--force-ocr")
    return cmd


def _kill_proc(proc: subprocess.Popen, timeout: float = 10) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _bind_free_port() -> tuple[socket.socket, int]:
    """빈 포트를 바인딩한 소켓과 포트 번호를 반환한다.
    소켓을 닫기 전까지 다른 프로세스가 해당 포트를 점유하지 못한다 (TOCTOU 방지).
    호출자가 프로세스 기동 직전에 소켓을 닫아야 한다."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    return s, int(s.getsockname()[1])


def _is_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except urllib.error.URLError:
        return False


async def _wait_reachable(url: str, proc: subprocess.Popen, timeout_s: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if await asyncio.to_thread(_is_reachable, url):
            return True
        await asyncio.sleep(0.3)
    return False


# ════════════════════════════════════════════════════════════════
# 전역 하이브리드 프로세스 관리 (소형 PDF / 비-청크 모드)
# ════════════════════════════════════════════════════════════════

async def _ensure_hybrid(ocr_lang: str, force_ocr: bool) -> bool:
    """전역 하이브리드 프로세스가 올바른 설정으로 실행 중인지 확인하고 필요 시 재시작.
    락으로 보호되므로 동시 요청이 와도 중복 기동되지 않는다."""
    global _hybrid_proc, _hybrid_proc_lang, _hybrid_proc_force_ocr

    async with _hybrid_lock:
        running = (
            _hybrid_proc is not None
            and _hybrid_proc.poll() is None
            and _hybrid_proc_lang == ocr_lang
            and _hybrid_proc_force_ocr == force_ocr
        )
        if running and await asyncio.to_thread(_is_reachable, HYBRID_URL):
            return True

        # 기존 프로세스 정리
        if _hybrid_proc is not None:
            await asyncio.to_thread(_kill_proc, _hybrid_proc)
            _hybrid_proc = None

        cli = _find_hybrid_cli()
        if cli is None:
            logger.error("[hybrid] opendataloader-pdf-hybrid CLI를 찾을 수 없습니다.")
            return False

        cmd = _build_hybrid_cmd(cli, ocr_lang, force_ocr, HYBRID_PORT)
        logger.info("[hybrid] 시작: %s", " ".join(cmd))
        _hybrid_proc = subprocess.Popen(cmd)

        if await _wait_reachable(HYBRID_URL, _hybrid_proc):
            _hybrid_proc_lang = ocr_lang
            _hybrid_proc_force_ocr = force_ocr
            logger.info("[hybrid] 준비 완료 (%s)", HYBRID_URL)
            return True

        logger.error("[hybrid] 시작 실패 (exit=%s)", _hybrid_proc.poll())
        await asyncio.to_thread(_kill_proc, _hybrid_proc)
        _hybrid_proc = None
        return False


# ════════════════════════════════════════════════════════════════
# 에페메랄(임시) 하이브리드 — 병렬 청크 처리 전용
# ════════════════════════════════════════════════════════════════

async def _start_ephemeral_hybrid(
    cli: str, ocr_lang: str, force_ocr: bool
) -> tuple[subprocess.Popen, str]:
    """빈 포트를 확보하고 임시 하이브리드 프로세스를 시작한다.
    소켓을 닫는 순간과 프로세스가 바인딩하는 순간 사이의 창이 최소화되도록
    소켓을 닫자마자 즉시 프로세스를 띄운다."""
    sock, port = _bind_free_port()
    url = f"http://127.0.0.1:{port}"
    cmd = _build_hybrid_cmd(cli, ocr_lang, force_ocr, port)
    sock.close()  # 닫는 즉시 Popen — 동일 머신 내 충돌 가능성 극히 낮음
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, url


def _pre_extract_chunks(
    in_path: str, tmpdir: str, n_pages: int, chunk_size: int
) -> list[tuple[int, int, str, str]]:
    """원본 PDF를 한 번만 읽어 청크 PDF + 출력 디렉터리를 생성한다.
    블로킹 함수 — 반드시 asyncio.to_thread()로 호출할 것."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(in_path)
    infos: list[tuple[int, int, str, str]] = []
    for start in range(1, n_pages + 1, chunk_size):
        end = min(start + chunk_size - 1, n_pages)
        chunk_pdf = os.path.join(tmpdir, f"_chunk_{start}_{end}.pdf")
        writer = PdfWriter()
        for i in range(start - 1, end):
            writer.add_page(reader.pages[i])
        with open(chunk_pdf, "wb") as f:
            writer.write(f)
        chunk_out = os.path.join(tmpdir, f"out_{start}_{end}")
        os.makedirs(chunk_out, exist_ok=True)
        infos.append((start, end, chunk_pdf, chunk_out))
    return infos


async def _worker_loop(
    worker_id: int,
    chunk_indices: list[int],
    chunk_infos: list[tuple[int, int, str, str]],
    cli: str,
    ocr_lang: str,
    force_ocr: bool,
    kwargs_base: dict,
    n_pages: int,
    results: list[Optional[str]],
    result_out_dirs: list[Optional[str]],
    errors: list[Optional[str]],
) -> None:
    """한 워커가 할당된 청크들을 순서대로 처리한다.
    HYBRID_RECYCLE_EVERY 청크마다 임시 하이브리드를 재시작해 메모리를 해제한다.

    results/errors 공유 리스트는 asyncio 태스크(단일 스레드)에서만 쓰이므로
    별도 락 없이 안전하다."""
    proc: Optional[subprocess.Popen] = None
    eph_url: Optional[str] = None
    done_on_proc = 0

    try:
        for idx in chunk_indices:
            start, end, chunk_pdf, chunk_out = chunk_infos[idx]

            # 프로세스 최초 기동 또는 RECYCLE_EVERY 도달 시 재시작
            if proc is None or done_on_proc >= HYBRID_RECYCLE_EVERY:
                if proc is not None:
                    await asyncio.to_thread(_kill_proc, proc)
                    proc = None
                    await asyncio.sleep(0.3)

                proc, eph_url = await _start_ephemeral_hybrid(cli, ocr_lang, force_ocr)
                logger.info("[worker-%d] 임시 하이브리드 시작 %s (청크 %d/%d~)",
                            worker_id, eph_url, idx + 1, len(chunk_infos))

                if not await _wait_reachable(eph_url, proc):
                    errors[idx] = f"임시 하이브리드 기동 실패 (worker={worker_id}, url={eph_url})"
                    return

                logger.info("[worker-%d] 준비 완료 (%s)", worker_id, eph_url)
                done_on_proc = 0

            chunk_kw = {
                **kwargs_base,
                "input_path": [chunk_pdf],
                "output_dir": chunk_out,
                "hybrid_url": eph_url,
            }

            t0 = time.monotonic()
            md_text: Optional[str] = None
            md_out_dir: Optional[str] = None
            hybrid_ok = True

            # 1) 하이브리드 변환 시도
            try:
                await asyncio.to_thread(opendataloader_pdf.convert, **chunk_kw)
            except Exception as e:
                hybrid_ok = False
                logger.warning("[worker-%d] 하이브리드 실패 (p%d-%d): %s",
                               worker_id, start, end, type(e).__name__)

            done_on_proc += 1
            chunk_stem = _stem(os.path.basename(chunk_pdf))

            if hybrid_ok:
                md_path = _find_md(chunk_out, chunk_stem)
                if md_path:
                    with open(md_path, encoding="utf-8") as f:
                        md_text = f.read()
                    md_out_dir = chunk_out
                # 출력이 너무 짧으면 fallback 대상으로 판정
                expected_pages = end - start + 1
                if (
                    md_text is not None
                    and expected_pages > 1
                    and len(md_text.strip()) < expected_pages * 50
                ):
                    logger.warning("[worker-%d] 출력 너무 짧음 (%d chars), fallback 시도",
                                   worker_id, len(md_text.strip()))
                    hybrid_ok = False

            # 2) Non-hybrid fallback
            if not hybrid_ok:
                fallback_out = chunk_out + "_fb"
                os.makedirs(fallback_out, exist_ok=True)
                fallback_kw = {
                    "input_path": [chunk_pdf],
                    "output_dir": fallback_out,
                    "format": "markdown",
                    "image_output": "external",
                    "image_format": "png",
                    "quiet": True,
                }
                try:
                    await asyncio.to_thread(opendataloader_pdf.convert, **fallback_kw)
                    fb_path = _find_md(fallback_out, chunk_stem)
                    if fb_path:
                        with open(fb_path, encoding="utf-8") as f:
                            fb_text = f.read()
                        if md_text is None or len(fb_text.strip()) > len(md_text.strip()):
                            md_text = fb_text
                            md_out_dir = fallback_out
                            logger.info("[worker-%d] fallback 성공 (p%d-%d, %d chars)",
                                        worker_id, start, end, len(fb_text.strip()))
                except Exception as e2:
                    logger.error("[worker-%d] fallback도 실패 (p%d-%d): %s",
                                 worker_id, start, end, e2)

            elapsed = time.monotonic() - t0
            if md_text is not None:
                results[idx] = md_text
                result_out_dirs[idx] = md_out_dir or chunk_out
                logger.info("[worker-%d] p%d-%d/%d 완료 (%.1fs)",
                            worker_id, start, end, n_pages, elapsed)
            else:
                errors[idx] = f"Markdown 출력 없음 (p{start}-{end}/{n_pages})"

    finally:
        if proc is not None:
            await asyncio.to_thread(_kill_proc, proc)
            logger.info("[worker-%d] 임시 하이브리드 종료", worker_id)


async def _convert_chunks_parallel(
    in_path: str,
    tmpdir: str,
    kwargs_base: dict,
    n_pages: int,
    ocr_lang: str,
    force_ocr: bool,
) -> tuple[str, Optional[JSONResponse]]:
    """청크를 여러 임시 하이브리드 워커로 병렬 변환한다."""
    cli = _find_hybrid_cli()
    if cli is None:
        return "", JSONResponse(
            status_code=503,
            content={"ok": False, "error": "opendataloader-pdf-hybrid CLI를 찾을 수 없습니다."},
        )

    t_total = time.monotonic()

    # 청크 PDF 일괄 추출 (블로킹 I/O → to_thread)
    t0 = time.monotonic()
    chunk_infos = await asyncio.to_thread(
        _pre_extract_chunks, in_path, tmpdir, n_pages, HYBRID_PAGE_CHUNK
    )
    total = len(chunk_infos)
    logger.info("[convert] %d페이지 → %d청크(%dp) 추출 완료 (%.1fs), 워커 %d개",
                n_pages, total, HYBRID_PAGE_CHUNK, time.monotonic() - t0, HYBRID_WORKERS)

    # 라운드로빈 배분
    worker_chunks: list[list[int]] = [[] for _ in range(HYBRID_WORKERS)]
    for i in range(total):
        worker_chunks[i % HYBRID_WORKERS].append(i)

    results: list[Optional[str]] = [None] * total
    result_out_dirs: list[Optional[str]] = [None] * total
    errors: list[Optional[str]] = [None] * total

    # 병렬 변환
    await asyncio.gather(*[
        _worker_loop(
            worker_id=w,
            chunk_indices=indices,
            chunk_infos=chunk_infos,
            cli=cli,
            ocr_lang=ocr_lang,
            force_ocr=force_ocr,
            kwargs_base=kwargs_base,
            n_pages=n_pages,
            results=results,
            result_out_dirs=result_out_dirs,
            errors=errors,
        )
        for w, indices in enumerate(worker_chunks)
        if indices
    ])

    # 전체 실패 체크
    first_err = next((e for e in errors if e), None)
    if first_err and all(r is None for r in results):
        return "", JSONResponse(
            status_code=500,
            content={"ok": False, "error": first_err},
        )

    # 조립 (순차 — EasyOCR은 동일 인스턴스 동시 호출 비권장)
    parts: list[str] = []
    for i, (start, end, _, chunk_out) in enumerate(chunk_infos):
        md = results[i]
        if md is None:
            md = f"<!-- 변환 실패: {errors[i] or 'unknown'} -->"
        else:
            md = await _replace_images_with_ocr(
                md,
                result_out_dirs[i] or chunk_out,
                ocr_lang,
            )
        parts.append(f"\n\n<!-- pages {start}-{end} of {n_pages} -->\n\n{md}")

    logger.info("[convert] 전체 완료: %d페이지, %.1f초", n_pages, time.monotonic() - t_total)
    return "".join(parts).lstrip(), None


# ════════════════════════════════════════════════════════════════
# OCR
# ════════════════════════════════════════════════════════════════

_IMG_RE = re.compile(r"!\[([^\]]*)\]\((.+?\.(?:png|jpe?g|gif|bmp|webp))\)", re.IGNORECASE)
_PAGE_MARKER_RE = re.compile(r"<!--\s*pages\s+(\d+)-(\d+)\s+of\s+(\d+)\s*-->", re.IGNORECASE)
_IMAGE_OCR_MARKER_RE = re.compile(r"<!--\s*OCR:\s*(.*?)\s*-->", re.IGNORECASE)
_IMAGE_TABLE_MARKER_RE = re.compile(r"<!--\s*표:\s*(.*?)\s*-->", re.IGNORECASE)
_PDF_TABLE_MARKER_RE = re.compile(r"<!--\s*pdfplumber 표 \(p([0-9\-]+)\)\s*-->", re.IGNORECASE)
_FAIL_MARKER_RE = re.compile(r"<!--\s*변환 실패:\s*(.*?)\s*-->", re.IGNORECASE)
_MD_TABLE_BLOCK_RE = re.compile(
    r"(?m)(^\|.*\|\s*\n^\|(?:\s*:?-{3,}:?\s*\|)+\s*\n(?:^\|.*\|\s*(?:\n|$))+)"
)


def _compact_text(v: Any) -> str:
    """검색/임베딩 품질을 위해 문자열을 단일 공백 기준으로 정리한다."""
    if v is None:
        return ""
    s = str(v).replace("\xa0", " ").replace("\u200b", " ")
    return re.sub(r"\s+", " ", s).strip()


def _format_ocr_lines_for_markdown(lines: list[str]) -> str:
    """OCR 결과를 중복 제거 + 불릿 목록으로 정규화한다."""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        text = _compact_text(raw)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return "\n".join(f"- {line}" for line in normalized)


def _extract_ocr_lines(reader: easyocr.Reader, img_array: np.ndarray) -> list[str]:
    """좌표/신뢰도 기반으로 OCR 결과를 정렬해 문장 순서를 안정화한다."""
    img_array = _preprocess_for_ocr(img_array)
    lines_with_pos: list[tuple[float, float, str]] = []
    img_h = int(img_array.shape[0]) if hasattr(img_array, "shape") else 1000

    try:
        raw = reader.readtext(img_array, detail=1, paragraph=False)
    except Exception:
        raw = []

    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        bbox = item[0]
        text = _compact_text(item[1])
        conf = float(item[2]) if len(item) > 2 and item[2] is not None else 1.0
        if not text or conf < OCR_MIN_CONFIDENCE:
            continue

        xs: list[float] = []
        ys: list[float] = []
        if isinstance(bbox, (list, tuple)):
            for pt in bbox:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    try:
                        xs.append(float(pt[0]))
                        ys.append(float(pt[1]))
                    except (TypeError, ValueError):
                        continue
        cx = sum(xs) / len(xs) if xs else 0.0
        cy = sum(ys) / len(ys) if ys else float(len(lines_with_pos))
        lines_with_pos.append((cy, cx, text))

    if not lines_with_pos:
        try:
            fallback = reader.readtext(img_array, detail=0, paragraph=True)
        except Exception:
            fallback = []
        return [_compact_text(t) for t in fallback if _compact_text(t)]

    lines_with_pos.sort(key=lambda x: (x[0], x[1]))
    y_tol = max(8.0, img_h * OCR_LINE_MERGE_RATIO)
    merged: list[tuple[float, list[str]]] = []
    for cy, _cx, text in lines_with_pos:
        if not merged or abs(cy - merged[-1][0]) > y_tol:
            merged.append((cy, [text]))
        else:
            merged[-1][1].append(text)

    result: list[str] = []
    for _cy, parts in merged:
        line = _compact_text(" ".join(parts))
        if line:
            result.append(line)
    return result


async def _get_ocr_reader(lang_csv: str) -> easyocr.Reader:
    """OCR 리더 싱글톤. 언어가 바뀔 때만 재초기화.
    _ocr_lock으로 보호하여 동시 요청이 와도 중복 초기화되지 않는다."""
    global _ocr_reader, _ocr_reader_langs
    async with _ocr_lock:
        if _ocr_reader is None or _ocr_reader_langs != lang_csv:
            langs = [lang.strip() for lang in lang_csv.split(",") if lang.strip()]
            logger.info("[ocr] Reader 초기화: %s (gpu=%s)", langs, USE_GPU)
            # easyocr.Reader() 는 무거운 블로킹 호출 → to_thread
            _ocr_reader = await asyncio.to_thread(easyocr.Reader, langs, gpu=USE_GPU)
            _ocr_reader_langs = lang_csv
        return _ocr_reader


def _norm_image_key(path: str) -> str:
    s = path.strip().strip('"').strip("'").replace("\\", "/")
    s = re.sub(r"^\./", "", s)
    return s.lower()


def _build_image_index(out_dir: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    by_rel: dict[str, str] = {}
    by_name: dict[str, list[str]] = {}
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
                abs_path = os.path.join(root, f)
                rel = os.path.relpath(abs_path, out_dir).replace("\\", "/")
                rel_key = _norm_image_key(rel)
                name_key = _norm_image_key(f)
                by_rel.setdefault(rel_key, abs_path)
                by_name.setdefault(name_key, []).append(abs_path)
    return by_rel, by_name


def _resolve_image_path(
    img_path: str,
    out_dir: str,
    by_rel: dict[str, str],
    by_name: dict[str, list[str]],
) -> Optional[str]:
    key = _norm_image_key(img_path)
    if key in by_rel:
        return by_rel[key]

    name_key = _norm_image_key(os.path.basename(img_path))
    candidates = by_name.get(name_key, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    suffix_key = _norm_image_key(img_path)
    matched = [
        p for p in candidates
        if _norm_image_key(os.path.relpath(p, out_dir)).endswith(suffix_key)
    ]
    if len(matched) == 1:
        return matched[0]
    logger.debug("[ocr] 이미지 경로가 모호해 대체를 건너뜀: %s (%d candidates)", img_path, len(candidates))
    return None


def _read_image(path: str) -> Optional[np.ndarray]:
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """OCR 전처리: 그레이스케일 변환 → 소형 이미지 확대 → CLAHE 대비 향상 → 가우시안 노이즈 제거."""
    if img is None:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # 너무 작은 이미지는 확대 (OCR 인식률 향상)
    h, w = gray.shape[:2]
    if min(h, w) < 100:
        scale = max(2.0, 200.0 / min(h, w))
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # 지역 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 가우시안 노이즈 제거
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


async def _replace_images_with_ocr(markdown: str, out_dir: str, lang_csv: str) -> str:
    """Markdown 내 이미지 참조를 OCR 텍스트 또는 Markdown 표로 대체한다.

    처리 우선순위:
      1. img2table로 표 구조 감지 → Markdown 표 형식 반환
      2. EasyOCR로 일반 텍스트 추출 → 텍스트 블록 반환

    re.sub + OCR 추론 모두 블로킹 → asyncio.to_thread 내에서 실행."""
    by_rel, by_name = _build_image_index(out_dir)
    if not by_rel and not by_name:
        return markdown

    # 두 OCR 인스턴스를 병렬 초기화
    reader, img2table_ocr = await asyncio.gather(
        _get_ocr_reader(lang_csv),
        _get_img2table_ocr(lang_csv),
    )

    def _do_ocr_sub() -> str:
        def _replace_image(match: re.Match) -> str:
            alt, img_path = match.group(1), match.group(2)
            alt = _compact_text(alt).replace("--", "-")
            abs_path = _resolve_image_path(img_path, out_dir, by_rel, by_name)
            if abs_path is None or not os.path.isfile(abs_path):
                return match.group(0)

            # 1) 표 구조 감지 (img2table)
            if img2table_ocr is not None:
                table_md = _image_to_markdown_table(abs_path, img2table_ocr)
                if table_md:
                    logger.debug("[table] 이미지 표 변환 성공: %s", os.path.basename(img_path))
                    return f"\n\n<!-- 표: {alt} -->\n\n{table_md}\n"

            # 2) 일반 텍스트 OCR (EasyOCR)
            try:
                img_array = _read_image(abs_path)
                if img_array is None:
                    return match.group(0)
                ocr_lines = _extract_ocr_lines(reader, img_array)
                text = _format_ocr_lines_for_markdown(ocr_lines).strip()
            except Exception:
                return match.group(0)
            return f"\n\n<!-- OCR: {alt} -->\n\n{text}\n" if text else match.group(0)

        return _IMG_RE.sub(_replace_image, markdown)

    return await asyncio.to_thread(_do_ocr_sub)


# ════════════════════════════════════════════════════════════════
# 표 인식 — 이미지 표 (img2table) + 디지털 표 (pdfplumber)
# ════════════════════════════════════════════════════════════════

async def _get_img2table_ocr(lang_csv: str) -> Optional[Any]:
    """img2table용 EasyOCR 싱글톤. 기존 _ocr_reader와 별도 인스턴스를 유지한다.
    img2table이 설치되지 않은 경우 None을 반환한다."""
    global _img2table_ocr, _img2table_ocr_langs
    try:
        from img2table.ocr import EasyOCR as Img2TableEasyOCR  # noqa: F401
    except ImportError:
        return None

    async with _img2table_lock:
        if _img2table_ocr is None or _img2table_ocr_langs != lang_csv:
            langs = [lang.strip() for lang in lang_csv.split(",") if lang.strip()]
            logger.info("[table] img2table EasyOCR 초기화: %s (gpu=%s)", langs, USE_GPU)
            from img2table.ocr import EasyOCR as Img2TableEasyOCR
            _img2table_ocr = await asyncio.to_thread(
                Img2TableEasyOCR, lang=langs, gpu=USE_GPU
            )
            _img2table_ocr_langs = lang_csv
        return _img2table_ocr


def _norm_cell(v) -> str:
    """셀 값을 Markdown에 안전한 단일 행 문자열로 정규화한다."""
    if v is None:
        return ""
    s = str(v)
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.IGNORECASE)  # <br> → 공백
    s = s.replace("\n", " ").replace("\r", "").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("|", "\\|")
    return s


def _rows_to_markdown(headers: list[str], data_rows: list[list[str]]) -> str:
    """정규화된 헤더·데이터 행 → Markdown 표 문자열.

    1. 완전히 빈 행 제거
    2. 다중 헤더 감지·합성 (헤더에 빈 열이 있고 첫 데이터 행이 그 자리를 채우면 서브헤더로 간주)
    """
    data_rows = [r for r in data_rows if any(r)]
    if not data_rows:
        return ""

    empty_h_pos = [i for i, h in enumerate(headers) if not h]
    if empty_h_pos:
        first = data_rows[0]
        filled_at_empty = sum(1 for i in empty_h_pos if i < len(first) and first[i])
        if filled_at_empty >= max(1, len(empty_h_pos) // 2):
            merged: list[str] = []
            for i, h in enumerate(headers):
                sub = first[i] if i < len(first) else ""
                merged.append(f"{h} {sub}" if h and sub else h or sub)
            headers = merged
            data_rows = [r for r in data_rows[1:] if any(r)]

    if not data_rows:
        return ""

    n = len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in data_rows:
        padded = row + [""] * (n - len(row))
        lines.append("| " + " | ".join(padded[:n]) + " |")
    return "\n".join(lines)


def _df_to_markdown(df) -> str:
    """pandas DataFrame → Markdown 표 문자열.
    정수형 컬럼 인덱스 → 첫 행 헤더 승격 후 _rows_to_markdown 위임."""
    if df is None or df.empty:
        return ""

    cols = list(df.columns)
    if all(isinstance(c, int) for c in cols) and len(df) > 0:
        candidate = [_norm_cell(v) for v in df.iloc[0]]
        if any(candidate):
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = candidate
            cols = candidate

    headers = [_norm_cell(c) for c in cols]
    all_rows = [[_norm_cell(v) for v in row] for _, row in df.iterrows()]
    return _rows_to_markdown(headers, all_rows)


def _image_to_markdown_table(img_path: str, img2table_ocr) -> Optional[str]:
    """이미지에서 표를 감지해 Markdown 표 문자열로 반환한다.
    표가 없거나 감지 실패 시 None을 반환한다. (블로킹 함수)"""
    try:
        from img2table.document import Image as Img2TableImage

        doc = Img2TableImage(src=img_path)
        extracted = doc.extract_tables(
            ocr=img2table_ocr,
            implicit_rows=True,       # 명시적 구분선 없는 행도 감지
            implicit_columns=True,    # 명시적 구분선 없는 열도 감지
            borderless_tables=True,   # 테두리 없는 표도 감지
            min_confidence=60,        # 노이즈 오인식 방지를 위해 상향
        )

        if not extracted:
            return None

        parts: list[str] = []
        for table in extracted:
            df = table.df
            if df is None or df.empty:
                continue
            # 너무 작은 결과(1행 1열)는 표가 아닌 노이즈로 판단
            if df.shape[0] < 1 or df.shape[1] < 2:
                continue
            md = _df_to_markdown(df)
            if md:
                parts.append(md)

        return "\n\n".join(parts) if parts else None

    except ImportError:
        return None
    except Exception as e:
        logger.debug("[table] 이미지 표 추출 실패 (%s): %s", os.path.basename(img_path), e)
        return None


def _raw_table_to_markdown(raw_table: list[list[Optional[str]]]) -> Optional[str]:
    """pdfplumber raw 표 (리스트의 리스트) → Markdown 표 문자열.
    pdfplumber는 병합 셀(rowspan/colspan)을 None으로 반환한다."""
    if not raw_table or len(raw_table) < 2:
        return None
    header = [_norm_cell(v) for v in raw_table[0]]
    if len(header) < 2:
        return None
    data_rows = [[_norm_cell(v) for v in row] for row in raw_table[1:]]
    md = _rows_to_markdown(header, data_rows)
    return md or None


def _configure_pymupdf_logging(fitz_mod: Any) -> None:
    """PyMuPDF가 stderr에 내보내는 경고/에러를 가능한 범위에서 억제한다."""
    if not PYMUPDF_SUPPRESS_LOGS:
        return

    tools = getattr(fitz_mod, "TOOLS", None)
    if tools is None:
        return

    for name in ("mupdf_display_errors", "set_mupdf_display_errors"):
        fn = getattr(tools, name, None)
        if not callable(fn):
            continue
        try:
            fn(False)
        except TypeError:
            try:
                fn(on=False)
            except Exception:
                pass
        except Exception:
            pass

    for name in ("mupdf_display_warnings", "set_mupdf_display_warnings"):
        fn = getattr(tools, name, None)
        if not callable(fn):
            continue
        try:
            fn(False)
        except TypeError:
            try:
                fn(on=False)
            except Exception:
                pass
        except Exception:
            pass


def _pdf_tables_with_pymupdf(pdf_path: str) -> dict[int, list[str]]:
    """pymupdf(fitz)로 PDF 표를 페이지별로 추출한다. (블로킹 함수)

    pymupdf >= 1.23의 find_tables()는 병합 셀을 자동으로 채워주므로
    rowspan/colspan이 있는 복잡한 표에서 pdfplumber보다 정확하다."""
    result: dict[int, list[str]] = {}
    if not USE_PYMUPDF_TABLES:
        logger.info("[table] USE_PYMUPDF_TABLES=false — pymupdf 표 추출 비활성화")
        return result

    try:
        import fitz  # pymupdf
    except ImportError:
        logger.debug("[table] pymupdf 미설치, pdfplumber로 fallback")
        return result

    try:
        _configure_pymupdf_logging(fitz)
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            try:
                finder = page.find_tables()
            except AttributeError:
                # pymupdf < 1.23 — find_tables 미지원
                logger.debug("[table] pymupdf 버전이 낮아 find_tables 미지원 (p%d)", page_num)
                break
            except Exception as e_find:
                logger.debug("[table] pymupdf find_tables 실패 (p%d): %s", page_num, e_find)
                continue

            if not finder.tables:
                continue

            page_tables: list[str] = []
            for table in finder.tables:
                try:
                    df = table.to_pandas()
                except Exception:
                    continue
                if df is None or df.empty or df.shape[1] < 2:
                    continue
                md = _df_to_markdown(df)
                if md:
                    page_tables.append(md)

            if page_tables:
                result[page_num] = page_tables
                logger.info("[table] p%d: pymupdf 표 %d개", page_num, len(page_tables))

        doc.close()
    except Exception as e:
        logger.warning("[table] pymupdf 추출 실패: %s", e)

    return result


def _pdf_tables_with_pdfplumber(pdf_path: str) -> dict[int, list[str]]:
    """pdfplumber로 디지털 PDF 표를 페이지별로 추출한다. (블로킹 함수)
    pymupdf가 없거나 실패했을 때의 fallback으로 사용된다."""
    result: dict[int, list[str]] = {}
    try:
        import pdfplumber
    except ImportError:
        logger.debug("[table] pdfplumber 미설치")
        return result

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # 1차: 완전한 격자선 기반 (정확도 가장 높음)
                tables = page.extract_tables({
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "lines_strict",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 3,
                    "min_words_vertical": 1,
                    "min_words_horizontal": 1,
                })
                # 2차: 일반 선 기반 (점선·불완전 선 포함)
                if not tables:
                    tables = page.extract_tables({
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "snap_tolerance": 5,
                        "join_tolerance": 5,
                        "edge_min_length": 3,
                        "min_words_vertical": 1,
                        "min_words_horizontal": 1,
                    })
                # 3차: 텍스트 정렬 기반 (격자선 없는 표)
                if not tables:
                    tables = page.extract_tables({
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 3,
                    })

                page_tables: list[str] = []
                for raw in (tables or []):
                    md = _raw_table_to_markdown(raw)
                    if md:
                        page_tables.append(md)

                if page_tables:
                    result[page_num] = page_tables
                    logger.info("[table] p%d: pdfplumber 표 %d개", page_num, len(page_tables))

    except Exception as e:
        logger.warning("[table] pdfplumber 추출 실패: %s", e)

    return result


def _pdf_tables_to_markdown(pdf_path: str) -> dict[int, list[str]]:
    """PDF 디지털 표를 추출한다. pymupdf와 pdfplumber 결과를 페이지 단위로 병합한다. (블로킹 함수)

    페이지별로 두 추출기의 표를 합치되, 동일 표 fingerprint는 중복 제거한다.
    두 라이브러리 모두 미설치이면 빈 딕트를 반환한다."""
    pymupdf_result = _pdf_tables_with_pymupdf(pdf_path)
    plumber_result = _pdf_tables_with_pdfplumber(pdf_path)

    if not plumber_result:
        return pymupdf_result
    if not pymupdf_result:
        return plumber_result

    merged: dict[int, list[str]] = {}
    pages = sorted(set(pymupdf_result) | set(plumber_result))
    for page in pages:
        combined = list(pymupdf_result.get(page, [])) + list(plumber_result.get(page, []))
        page_seen: set[str] = set()
        page_tables: list[str] = []
        for tbl in combined:
            fp = re.sub(r"\s+", "", tbl).lower()
            if fp in page_seen:
                continue
            page_seen.add(fp)
            page_tables.append(tbl)
        if page_tables:
            merged[page] = page_tables
    return merged


def _inject_pdf_tables(markdown: str, pdf_tables: dict[int, list[str]]) -> str:
    """pdfplumber로 추출한 표를 Markdown에 주입한다.

    opendataloader가 만든 페이지 구분 주석(<!-- pages N-M of T -->)을 기준으로
    해당 페이지 범위에 포함된 디지털 표를 삽입한다.
    주석이 없으면 전체 Markdown 끝에 표를 추가한다."""
    if not pdf_tables:
        return markdown

    PAGE_RE = re.compile(r"(<!-- pages (\d+)-(\d+) of \d+ -->)")

    def _tables_for_range(start: int, end: int) -> list[str]:
        tables: list[str] = []
        for p in range(start, end + 1):
            tables.extend(pdf_tables.get(p, []))
        return tables

    # 페이지 구분 주석이 있는 경우 — 청크 모드
    if PAGE_RE.search(markdown):
        def _replace_chunk(m: re.Match) -> str:
            start, end = int(m.group(2)), int(m.group(3))
            tables = _tables_for_range(start, end)
            if not tables:
                return m.group(0)
            injected = "\n\n".join(tables)
            return f"{m.group(0)}\n\n<!-- pdfplumber 표 (p{start}-{end}) -->\n\n{injected}"
        return PAGE_RE.sub(_replace_chunk, markdown)

    # 페이지 구분 주석이 없는 경우 — 전체 표를 끝에 추가
    all_tables: list[str] = []
    for page_num in sorted(pdf_tables):
        for tbl in pdf_tables[page_num]:
            all_tables.append(f"<!-- pdfplumber 표 (p{page_num}) -->\n\n{tbl}")

    if all_tables:
        markdown += "\n\n" + "\n\n".join(all_tables)

    return markdown


def _dedup_pdf_tables(markdown: str) -> str:
    """인접한 동일 표 블록만 제거한다.

    같은 표가 문서의 다른 페이지에서 반복되는 정상 케이스는 보존하고,
    동일 위치에서 두 경로가 중복 생성한 연속 블록만 제거한다."""
    out_parts: list[str] = []
    last_end = 0
    prev_fp: Optional[str] = None

    for match in _MD_TABLE_BLOCK_RE.finditer(markdown):
        gap = markdown[last_end:match.start()]
        out_parts.append(gap)

        table_md = match.group(1)
        fp = re.sub(r"\s+", "", table_md).lower()
        only_space_between = not re.sub(r"\s+", "", gap)

        if prev_fp == fp and only_space_between:
            # 바로 앞 표와 동일하고 사이가 공백뿐이면 중복으로 본다.
            pass
        else:
            out_parts.append(table_md)
            prev_fp = fp

        last_end = match.end()

    out_parts.append(markdown[last_end:])
    return "".join(out_parts)


def _split_markdown_row(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    cells = re.split(r"(?<!\\)\|", row)
    return [cell.strip().replace("\\|", "|") for cell in cells]


def _table_records_for_embedding(table_md: str) -> list[str]:
    """Markdown 표를 행 단위 key=value 레코드로 직렬화한다."""
    lines = [ln.strip() for ln in table_md.splitlines() if ln.strip()]
    if len(lines) < 3:
        return []

    headers = [_compact_text(cell) for cell in _split_markdown_row(lines[0])]
    if len(headers) < 2:
        return []

    body = lines[2:]
    records: list[str] = []
    carry_context = [""] * len(headers)
    for row_idx, row_line in enumerate(body, start=1):
        cells = [_compact_text(cell) for cell in _split_markdown_row(row_line)]
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[: len(headers)]

        pairs: list[str] = []
        for col_idx, (header, value) in enumerate(zip(headers, cells), start=1):
            if value:
                carry_context[col_idx - 1] = value
            effective = value
            inherited = False
            if (
                not effective
                and col_idx <= RAG_TABLE_CONTEXT_COLS
                and carry_context[col_idx - 1]
            ):
                effective = carry_context[col_idx - 1]
                inherited = True

            if not effective:
                continue
            key = header or f"col{col_idx}"
            if len(effective) > RAG_TABLE_CELL_MAX_CHARS:
                effective = effective[: RAG_TABLE_CELL_MAX_CHARS - 3].rstrip() + "..."
            if inherited:
                pairs.append(f"{key}={effective} (inherited)")
            else:
                pairs.append(f"{key}={effective}")

        if pairs:
            records.append(f"- row {row_idx}: " + "; ".join(pairs))

        if len(records) >= RAG_TABLE_RECORD_MAX_ROWS:
            rest = max(0, len(body) - row_idx)
            if rest:
                records.append(f"- ... {rest} more rows omitted")
            break

    return records


def _augment_tables_for_embedding(markdown: str) -> str:
    """표 바로 아래에 레코드형 텍스트를 추가해 임베딩 검색 적중률을 높인다."""
    seen_tables: set[str] = set()

    def _replace_table(match: re.Match) -> str:
        table_md = match.group(1).rstrip()
        fingerprint = re.sub(r"\s+", " ", table_md).strip().lower()
        if fingerprint in seen_tables:
            return match.group(1)
        seen_tables.add(fingerprint)

        records = _table_records_for_embedding(table_md)
        if not records:
            return match.group(1)
        return (
            f"{table_md}\n\n"
            "#### Table Records (for retrieval)\n"
            + "\n".join(records)
            + "\n"
        )

    return _MD_TABLE_BLOCK_RE.sub(_replace_table, markdown)


def _rewrite_markers_for_embedding(markdown: str) -> str:
    """내부 HTML 주석 마커를 임베딩 친화적인 섹션 헤더로 변환한다."""
    markdown = _PAGE_MARKER_RE.sub(
        lambda m: f"\n\n## Pages {m.group(1)}-{m.group(2)} of {m.group(3)}\n", markdown
    )
    markdown = _IMAGE_TABLE_MARKER_RE.sub(
        lambda m: f"\n\n### Image Table Extraction: {_compact_text(m.group(1)) or 'unnamed'}\n",
        markdown,
    )
    markdown = _IMAGE_OCR_MARKER_RE.sub(
        lambda m: f"\n\n### Image OCR Text: {_compact_text(m.group(1)) or 'unnamed'}\n",
        markdown,
    )
    markdown = _PDF_TABLE_MARKER_RE.sub(
        lambda m: f"\n\n### Digital PDF Table: page {m.group(1)}\n",
        markdown,
    )
    markdown = _FAIL_MARKER_RE.sub(
        lambda m: f"\n\n> Conversion warning: {_compact_text(m.group(1)) or 'unknown'}\n",
        markdown,
    )
    return markdown


def _markdown_quality_report(markdown: str) -> dict[str, Any]:
    lines = markdown.splitlines()
    table_records = sum(1 for ln in lines if ln.strip().startswith("- row "))
    report = {
        "chars": len(markdown),
        "lines": len(lines),
        "table_blocks": len(_MD_TABLE_BLOCK_RE.findall(markdown)),
        "table_record_rows": table_records,
        "image_ocr_sections": len(re.findall(r"^### Image OCR Text:", markdown, flags=re.MULTILINE)),
        "image_table_sections": len(re.findall(r"^### Image Table Extraction:", markdown, flags=re.MULTILINE)),
        "digital_table_sections": len(re.findall(r"^### Digital PDF Table:", markdown, flags=re.MULTILINE)),
        "warnings": len(re.findall(r"^> Conversion warning:", markdown, flags=re.MULTILINE)),
    }
    report["low_signal"] = report["chars"] < 800 or (
        report["table_blocks"] == 0 and report["image_ocr_sections"] == 0
    )
    return report


def _prepare_markdown_for_embedding(
    markdown: str, filename: str, n_pages: Optional[int]
) -> str:
    """최종 Markdown을 RAG 임베딩에 유리한 형태로 정규화한다."""
    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = _rewrite_markers_for_embedding(normalized)
    normalized = _augment_tables_for_embedding(normalized)

    meta = [
        "# Document Metadata",
        f"- file_name: {filename}",
        f"- total_pages: {n_pages if n_pages is not None else 'unknown'}",
    ]

    combined = "\n".join(meta) + "\n\n" + normalized
    combined = re.sub(r"\n{3,}", "\n\n", combined).strip() + "\n"
    return combined


async def _cancel_task(task: Optional[asyncio.Task]) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def _safe_wait_pdf_tables(task: Optional[asyncio.Task]) -> dict[int, list[str]]:
    if task is None:
        return {}
    try:
        return await asyncio.wait_for(task, timeout=PDF_TABLE_EXTRACT_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("[table] 디지털 표 추출 타임아웃(%ss) — 표 주입 생략", PDF_TABLE_EXTRACT_TIMEOUT_S)
        return {}
    except Exception as e:
        logger.warning("[table] 디지털 표 추출 실패 — 표 주입 생략: %s", e)
        return {}


# ════════════════════════════════════════════════════════════════
# PDF 유틸
# ════════════════════════════════════════════════════════════════

def _pdf_page_count(path: str) -> Optional[int]:
    """블로킹 함수 — 반드시 asyncio.to_thread()로 호출할 것."""
    try:
        from pypdf import PdfReader
        with open(path, "rb") as f:
            n = len(PdfReader(f).pages)
        logger.info("[page_count] pypdf → %d페이지", n)
        return n
    except ImportError:
        logger.error("[page_count] pypdf 미설치 — pip install pypdf")
        return None
    except Exception as e:
        logger.warning("[page_count] pypdf 실패: %s", e)

    # regex fallback
    try:
        with open(path, "rb") as f:
            raw = f.read()
        count = len(re.findall(rb"/Type\s*/Page(?!s)", raw))
        if count > 0:
            logger.info("[page_count] regex fallback → %d페이지", count)
            return count
    except Exception as e2:
        logger.warning("[page_count] regex fallback 실패: %s", e2)

    return None


def _find_md(directory: str, stem: str) -> Optional[str]:
    paths = glob.glob(os.path.join(directory, f"{stem}.md"))
    if paths:
        return paths[0]
    paths = glob.glob(os.path.join(directory, "*.md"))
    return paths[0] if paths else None


def _stem(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


# ════════════════════════════════════════════════════════════════
# FastAPI lifespan
# ════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ocr_lock, _hybrid_lock, _img2table_lock

    # 락은 실행 중인 이벤트 루프에서 초기화해야 한다
    _ocr_lock = asyncio.Lock()
    _hybrid_lock = asyncio.Lock()
    _img2table_lock = asyncio.Lock()

    logger.info(
        "[startup] GPU=%s CHUNK=%dp RECYCLE=%d WORKERS=%d MAX_UPLOAD=%dMB PYMUPDF=%s SUPPRESS_LOGS=%s",
        USE_GPU,
        HYBRID_PAGE_CHUNK,
        HYBRID_RECYCLE_EVERY,
        HYBRID_WORKERS,
        MAX_UPLOAD_MB,
        USE_PYMUPDF_TABLES,
        PYMUPDF_SUPPRESS_LOGS,
    )

    yield

    # 종료 시 전역 하이브리드 정리
    if _hybrid_proc is not None and _hybrid_proc.poll() is None:
        logger.info("[shutdown] 하이브리드 프로세스 종료 중…")
        _kill_proc(_hybrid_proc)


# ════════════════════════════════════════════════════════════════
# FastAPI 앱
# ════════════════════════════════════════════════════════════════

app = FastAPI(title="PDF to Markdown API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root():
    return {"ok": True, "message": "PDF to Markdown API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/convert")
async def convert(
    file: UploadFile = File(..., description="변환할 PDF 파일"),
    use_hybrid: bool = Form(False, description="하이브리드 모드 사용 여부 (표·이미지 분석 강화)"),
    ocr_lang: str = Form(
        "ko,en",
        description="OCR 언어 (쉼표 구분, 예: ko,en / ja,en / ch_sim,en)",
    ),
    force_ocr: Optional[bool] = Form(
        None,
        description="전 페이지 OCR 강제 여부 (미지정 시 환경변수 HYBRID_FORCE_OCR 사용)",
    ),
):
    # ── 입력 검증 ────────────────────────────────────────────────
    filename = (file.filename or "document.pdf").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=".pdf 파일만 업로드할 수 있습니다.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"파일 크기가 {MAX_UPLOAD_MB}MB를 초과합니다.",
        )

    eff_force_ocr = force_ocr if force_ocr is not None else DEFAULT_FORCE_OCR
    eff_ocr_lang = ocr_lang.strip()

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.pdf")
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(out_dir)

        with open(in_path, "wb") as f:
            f.write(data)

        # 페이지 수 확인 (블로킹 → to_thread)
        n_pages = await asyncio.to_thread(_pdf_page_count, in_path)

        will_chunk = (
            use_hybrid
            and n_pages is not None
            and n_pages > HYBRID_PAGE_CHUNK
        )

        logger.info(
            "[convert] file=%s pages=%s hybrid=%s chunk=%s force_ocr=%s lang=%s",
            filename, n_pages, use_hybrid, will_chunk, eff_force_ocr, eff_ocr_lang,
        )

        kwargs: dict = {
            "input_path": [in_path],
            "output_dir": out_dir,
            "format": "markdown",
            "image_output": "external",
            "image_format": "png",
            "quiet": True,
        }
        if use_hybrid:
            kwargs["hybrid"] = "docling-fast"
            kwargs["hybrid_url"] = HYBRID_URL
            kwargs["hybrid_fallback"] = True

        markdown = ""
        quality: dict[str, Any] = {}
        pdf_tables_task: Optional[asyncio.Task] = None

        try:
            # pdfplumber 디지털 표 추출 (변환과 병렬로 실행)
            pdf_tables_task = asyncio.create_task(
                asyncio.to_thread(_pdf_tables_to_markdown, in_path)
            )

            if will_chunk:
                markdown, err_resp = await _convert_chunks_parallel(
                    in_path=in_path,
                    tmpdir=tmpdir,
                    kwargs_base=kwargs,
                    n_pages=n_pages,
                    ocr_lang=eff_ocr_lang,
                    force_ocr=eff_force_ocr,
                )
                if err_resp is not None:
                    return err_resp

            else:
                if use_hybrid:
                    ready = await _ensure_hybrid(eff_ocr_lang, eff_force_ocr)
                    if not ready:
                        raise HTTPException(
                            status_code=503,
                            detail=f"하이브리드 백엔드({HYBRID_URL})를 시작할 수 없습니다.",
                        )

                # 블로킹 변환 → to_thread
                await asyncio.to_thread(opendataloader_pdf.convert, **kwargs)

                md_path = _find_md(out_dir, _stem("input.pdf"))
                if not md_path:
                    return JSONResponse(
                        status_code=500,
                        content={"ok": False, "error": "Markdown 출력 파일을 찾을 수 없습니다."},
                    )
                with open(md_path, encoding="utf-8") as f:
                    markdown = f.read()
                markdown = await _replace_images_with_ocr(markdown, out_dir, eff_ocr_lang)

            # pdfplumber 표 결과를 기다린 뒤 Markdown에 주입
            pdf_tables = await _safe_wait_pdf_tables(pdf_tables_task)
            if pdf_tables:
                markdown = _inject_pdf_tables(markdown, pdf_tables)
                logger.info("[convert] pdfplumber 표 %d페이지분 주입 완료",
                            len(pdf_tables))
                markdown = _dedup_pdf_tables(markdown)

            markdown = _prepare_markdown_for_embedding(
                markdown=markdown,
                filename=filename,
                n_pages=n_pages,
            )
            quality = _markdown_quality_report(markdown)

        except HTTPException:
            raise  # 400/503 등은 그대로 전파 (except Exception에 삼켜지지 않도록)
        except Exception as e:
            logger.exception("[convert] 변환 실패: %s", filename)
            return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
        finally:
            await _cancel_task(pdf_tables_task)

    return {
        "ok": True,
        "filename": filename,
        "pages": n_pages,
        "markdown": markdown,
        "quality": quality,
        "hybrid": use_hybrid,
        "ocr_lang": eff_ocr_lang,
        **({"force_ocr": eff_force_ocr, "page_chunk": HYBRID_PAGE_CHUNK} if use_hybrid else {}),
    }


if __name__ == "__main__":
    logger.info(
        "[GPU] CUDA: %s%s",
        USE_GPU,
        f" ({torch.cuda.get_device_name(0)})" if USE_GPU else "",
    )
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
