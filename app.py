"""
PDF → Markdown 변환 서버 (Dify 외부 파서 연동용)

사용 모델:
  - LLM      : exaon3.5:latest
  - Embedding: qwen3-embedding:latest

엔드포인트:
  GET  /               상태 확인
  GET  /health         헬스체크
  POST /convert        PDF 변환 시작 → job_id 즉시 반환
  GET  /status/{id}    변환 상태/결과 조회
  POST /convert/sync   동기 변환 (Dify 직접 연동용)

변환 모드:
  Hybrid 사이드카  opendataloader-pdf-hybrid 프로세스풀로 청크 병렬 변환
  ODL fallback     사이드카 실패 시 ODL JVM 직접 호출
  pymupdf fallback ODL도 실패하면 텍스트만 추출

환경변수:
  HYBRID_WORKERS      병렬 사이드카 수     (기본: 2)
  HYBRID_PAGE_CHUNK   청크당 페이지 수      (기본: 10)
  HYBRID_BASE_PORT    사이드카 시작 포트    (기본: 5100)
  HYBRID_EXE          사이드카 실행파일 경로 (기본: 자동 탐색)
  HYBRID_OCR_LANG     사이드카 OCR 언어     (기본: ko,en)
  MAX_UPLOAD_MB       업로드 상한 MB        (기본: 200)
"""

import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import requests as _requests
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# ── 선택적 의존성 ──────────────────────────────────────────────
try:
    from docling_core.types.doc import DoclingDocument
    _HAS_DOCLING = True
except ImportError:
    _HAS_DOCLING = False
    logging.warning("docling_core 미설치 — 사이드카 JSON→Markdown 변환 불가")

try:
    import opendataloader_pdf
    _HAS_ODL = True
except ImportError:
    _HAS_ODL = False
    logging.warning("opendataloader_pdf 미설치 — ODL fallback 비활성화")

try:
    import pdfplumber
    _HAS_PLUMBER = True
except ImportError:
    _HAS_PLUMBER = False

try:
    import easyocr
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False

try:
    import fitz as _fitz
    _HAS_FITZ = True
except ImportError:
    _fitz = None  # type: ignore[assignment]
    _HAS_FITZ = False

# ── 설정 ───────────────────────────────────────────────────────
MAX_UPLOAD_MB     = int(os.getenv("MAX_UPLOAD_MB", "200"))
OCR_LANG          = os.getenv("OCR_LANG", "ko,en")
LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")
HYBRID_WORKERS    = int(os.getenv("HYBRID_WORKERS", "2"))
HYBRID_PAGE_CHUNK = int(os.getenv("HYBRID_PAGE_CHUNK", "10"))
HYBRID_BASE_PORT  = int(os.getenv("HYBRID_BASE_PORT", "5100"))
HYBRID_OCR_LANG   = os.getenv("HYBRID_OCR_LANG", "ko,en")

# 사이드카 실행파일 탐색: 같은 conda 환경의 Scripts 디렉터리
_python_dir = Path(sys.executable).parent
_default_exe = str(_python_dir / "Scripts" / "opendataloader-pdf-hybrid.exe")
if not Path(_default_exe).exists():
    # 환경에 따라 python.exe가 Scripts 안에 있는 경우
    _default_exe = str(_python_dir / "opendataloader-pdf-hybrid.exe")
HYBRID_EXE = os.getenv("HYBRID_EXE", _default_exe)

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Job 저장소 ─────────────────────────────────────────────────
_jobs: dict = {}

# ── 사이드카 프로세스 풀 ───────────────────────────────────────
_sidecar_procs: list[subprocess.Popen] = []
_sidecar_queue: Optional[asyncio.Queue] = None  # asyncio.Queue[int] of available ports

# ── EasyOCR 싱글톤 ─────────────────────────────────────────────
_ocr_reader: Optional[object] = None
_ocr_lock: Optional[asyncio.Lock] = None
_ocr_langs: str = ""


async def _get_ocr_reader(langs: str = OCR_LANG):
    global _ocr_reader, _ocr_langs
    if not _HAS_OCR:
        return None
    async with _ocr_lock:  # type: ignore[arg-type]
        if _ocr_reader is None or _ocr_langs != langs:
            lang_list = [l.strip() for l in langs.split(",")]
            log.info("EasyOCR 초기화: %s", lang_list)
            _ocr_reader = await asyncio.to_thread(
                easyocr.Reader, lang_list, gpu=True, verbose=False
            )
            _ocr_langs = langs
    return _ocr_reader


# ══════════════════════════════════════════════════════════════
# 사이드카 프로세스 관리
# ══════════════════════════════════════════════════════════════

def _start_sidecar(port: int, ocr_lang: str) -> subprocess.Popen:
    """단일 사이드카 프로세스 기동"""
    exe = HYBRID_EXE
    if not Path(exe).exists():
        raise FileNotFoundError(f"사이드카 실행파일 없음: {exe}")
    cmd = [exe, "--port", str(port), "--ocr-lang", ocr_lang]
    log.info("사이드카 기동: port=%d  cmd=%s", port, " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    return proc


def _wait_sidecar_ready(port: int, timeout: int = 180) -> bool:
    """사이드카 /health 가 ok 반환할 때까지 대기. 타임아웃 초과 시 False."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _requests.get(url, timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ok":
                log.info("사이드카 port=%d 준비 완료", port)
                return True
        except Exception:
            pass
        time.sleep(2)
    log.error("사이드카 port=%d 타임아웃 (%ds)", port, timeout)
    return False


async def _init_sidecars():
    """서버 시작 시 HYBRID_WORKERS개 사이드카를 기동하고 큐에 등록.
    HYBRID_WORKERS=0 이면 사이드카를 시작하지 않음 (ODL 전용 모드).
    ODL과 사이드카(Docling)는 메모리를 공유하므로 동시 실행 시 충돌.
    사이드카가 필요하면 HYBRID_WORKERS>=1로 설정하되, ODL이 메모리 부족으로
    실패할 수 있음.
    """
    global _sidecar_procs, _sidecar_queue

    _sidecar_queue = asyncio.Queue()

    if HYBRID_WORKERS == 0:
        log.info("HYBRID_WORKERS=0 — 사이드카 비활성화 (ODL 전용)")
        return

    if not Path(HYBRID_EXE).exists():
        log.warning("사이드카 실행파일 없음 — ODL/pymupdf fallback 전용 모드")
        return

    _sidecar_queue = asyncio.Queue()
    ports = [HYBRID_BASE_PORT + i for i in range(HYBRID_WORKERS)]

    # 모든 프로세스 동시 기동
    for port in ports:
        try:
            proc = _start_sidecar(port, HYBRID_OCR_LANG)
            _sidecar_procs.append(proc)
        except Exception as e:
            log.error("사이드카 기동 실패 port=%d: %s", port, e)

    # 병렬로 헬스 대기 (블로킹이므로 to_thread)
    async def _await_port(port: int):
        ok = await asyncio.to_thread(_wait_sidecar_ready, port, 180)
        if ok:
            await _sidecar_queue.put(port)
        else:
            log.error("사이드카 port=%d 준비 실패 — 이 포트는 사용 안 함", port)

    await asyncio.gather(*[_await_port(p) for p in ports])
    ready = _sidecar_queue.qsize()
    log.info("사이드카 준비 완료: %d/%d", ready, HYBRID_WORKERS)


def _stop_sidecars():
    for proc in _sidecar_procs:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _sidecar_procs.clear()
    log.info("사이드카 프로세스 종료 완료")


# ── Lifespan ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _ocr_lock
    _ocr_lock = asyncio.Lock()
    log.info(
        "서버 시작 — ODL=%s docling=%s pdfplumber=%s pymupdf=%s "
        "WORKERS=%d CHUNK=%d BASE_PORT=%d",
        _HAS_ODL, _HAS_DOCLING, _HAS_PLUMBER, _HAS_FITZ,
        HYBRID_WORKERS, HYBRID_PAGE_CHUNK, HYBRID_BASE_PORT,
    )
    await _init_sidecars()
    yield
    _stop_sidecars()
    log.info("서버 종료")


app = FastAPI(
    title="PDF → Markdown 변환 서버",
    description="Dify 사내 RAG용 PDF 파싱 백엔드 (Hybrid 사이드카 병렬)",
    version="4.0.0",
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════════
# 페이지 범위 계산
# ══════════════════════════════════════════════════════════════

def _make_page_ranges(pdf_path: str, chunk_size: int) -> list[tuple[int, int]]:
    """1-based 페이지 범위 목록 반환. fitz 없으면 단일 전체 범위."""
    if _HAS_FITZ:
        doc = _fitz.open(pdf_path)
        total = len(doc)
        doc.close()
    else:
        return [(1, 9999)]

    ranges = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size - 1, total - 1)
        ranges.append((start + 1, end + 1))   # 1-based
    return ranges


def _extract_pages_to_tmp(pdf_path: str, page_start: int, page_end: int) -> str:
    """
    fitz로 page_start~page_end(1-based) 페이지만 추출해 임시 PDF 생성.
    반환: 임시 파일 경로 (호출자가 삭제 책임).
    """
    src = _fitz.open(pdf_path)
    dst = _fitz.open()
    for i in range(page_start - 1, min(page_end, len(src))):
        dst.insert_pdf(src, from_page=i, to_page=i)
    with tempfile.NamedTemporaryFile(suffix=".pdf", prefix="chunk_", delete=False) as f:
        tmp = f.name
    dst.save(tmp)
    dst.close()
    src.close()
    return tmp



# ══════════════════════════════════════════════════════════════
# 사이드카 표 추출 (표 품질 향상용)
# ══════════════════════════════════════════════════════════════

def _extract_md_tables(markdown: str) -> list[str]:
    """Markdown 텍스트에서 `|---|` 구분선이 있는 표 블록만 추출."""
    tables: list[str] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if line.strip().startswith("|"):
            current.append(line)
        else:
            if len(current) >= 2 and any("---" in r for r in current):
                tables.append("\n".join(current))
            current = []
    if len(current) >= 2 and any("---" in r for r in current):
        tables.append("\n".join(current))
    # 데이터 셀이 있는 표만 (모두 빈 셀인 표 제외)
    return [t for t in tables if re.search(r"\|\s*\S+\s*\|", t)]


def _extract_sidecar_tables_sync(port: int, pdf_path: str,
                                  page_start: int, page_end: int) -> list[str]:
    """소형 청크 PDF를 사이드카에 보내 Markdown 표 목록 반환."""
    if not (_HAS_FITZ and _HAS_DOCLING):
        return []
    chunk_pdf = _extract_pages_to_tmp(pdf_path, page_start, page_end)
    try:
        url = f"http://localhost:{port}/v1/convert/file"
        with open(chunk_pdf, "rb") as f:
            resp = _requests.post(
                url,
                files={"files": ("chunk.pdf", f, "application/pdf")},
                timeout=300,
            )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") not in ("success", "partial_success"):
            return []
        json_content = data["document"]["json_content"]
        doc = DoclingDocument.model_validate(json_content)
        md = doc.export_to_markdown()
        return _extract_md_tables(md)
    except Exception as e:
        log.warning("사이드카 표 추출 실패 (p%d~%d): %s", page_start, page_end, e)
        return []
    finally:
        try:
            os.unlink(chunk_pdf)
        except OSError:
            pass


async def _sidecar_tables_chunk(pdf_path: str, page_start: int, page_end: int) -> list[str]:
    """큐에서 포트를 빌려 청크 표 추출. 포트는 항상 반납."""
    assert _sidecar_queue is not None
    port = await _sidecar_queue.get()
    try:
        return await asyncio.to_thread(
            _extract_sidecar_tables_sync, port, pdf_path, page_start, page_end
        )
    except Exception as e:
        log.warning("사이드카 표 청크 실패: %s", e)
        return []
    finally:
        await _sidecar_queue.put(port)


# ══════════════════════════════════════════════════════════════
# ODL 단일 청크 변환 (fallback)
# ══════════════════════════════════════════════════════════════

def _run_odl_full(pdf_path: str) -> str:
    """ODL로 전체 PDF를 한 번에 변환 (pages 파라미터 없음, 가장 안정적)."""
    if not _HAS_ODL:
        return ""
    out_dir = tempfile.mkdtemp(prefix="odl_full_")
    try:
        opendataloader_pdf.convert(
            pdf_path,
            output_dir=out_dir,
            format="markdown",
            quiet=True,
        )
        md_files = list(Path(out_dir).glob("*.md"))
        if md_files:
            return md_files[0].read_text(encoding="utf-8")
    except Exception as e:
        log.error("ODL 전체 변환 실패: %s", e)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    return ""


def _run_odl_full_to_dir(pdf_path: str, out_dir: str) -> None:
    """ODL 전체 변환 결과를 지정된 디렉터리에 저장 (OCR용 이미지 추출 목적)."""
    if not _HAS_ODL:
        return
    try:
        opendataloader_pdf.convert(
            pdf_path,
            output_dir=out_dir,
            format="markdown",
            quiet=True,
        )
    except Exception as e:
        log.error("ODL OCR용 변환 실패: %s", e)


# ══════════════════════════════════════════════════════════════
# pymupdf fallback
# ══════════════════════════════════════════════════════════════

def _fallback_pages(pdf_path: str, start: int, end: Optional[int]) -> str:
    if not _HAS_FITZ:
        return ""
    try:
        doc = _fitz.open(pdf_path)
        end_page = end if end is not None else len(doc) - 1
        parts = []
        for i in range(start, end_page + 1):
            text = doc[i].get_text("markdown").strip() or doc[i].get_text("text").strip()
            if text:
                parts.append(f"## 페이지 {i + 1}\n\n{text}")
        doc.close()
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        log.error("pymupdf fallback 실패: %s", e)
        return ""


async def _run_odl_or_fallback(pdf_path: str) -> str:
    """ODL 전체 변환 → 실패 시 pymupdf 전체 fallback."""
    if _HAS_ODL:
        md = await asyncio.to_thread(_run_odl_full, pdf_path)
        if md and len(md.strip()) >= 30:
            return md
    log.warning("ODL 실패 → pymupdf 전체 fallback")
    return await asyncio.to_thread(_fallback_pages, pdf_path, 0, None)


# ══════════════════════════════════════════════════════════════
# pdfplumber 표 추출
# ══════════════════════════════════════════════════════════════

def _table_to_markdown(table: list) -> str:
    if not table:
        return ""
    rows = []
    for r, row in enumerate(table):
        cells = [str(c or "").replace("\n", " ").strip() for c in row]
        rows.append("| " + " | ".join(cells) + " |")
        if r == 0:
            rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(rows)


def _extract_tables_plumber(pdf_path: str) -> dict:
    result: dict = {}
    if not _HAS_PLUMBER:
        return result
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                tables = (
                    page.extract_tables({
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                    })
                    or page.extract_tables({
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                    })
                )
                if not tables:
                    continue
                md_tables = [_table_to_markdown(t) for t in tables if t]
                md_tables = [t for t in md_tables if t.strip()]
                if md_tables:
                    result[i] = md_tables
    except Exception as e:
        log.warning("pdfplumber 실패: %s", e)
    return result


# ══════════════════════════════════════════════════════════════
# EasyOCR (선택적)
# ══════════════════════════════════════════════════════════════

def _ocr_image_dir(img_dir: str, reader) -> dict:
    texts: dict = {}
    if not os.path.isdir(img_dir):
        return texts
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    img_files = sorted(f for f in os.listdir(img_dir)
                       if Path(f).suffix.lower() in exts)
    log.info("이미지 OCR: %d개", len(img_files))
    for fname in img_files:
        try:
            result = reader.readtext(os.path.join(img_dir, fname), detail=0)
            text = " ".join(result).strip()
            if text:
                texts[fname] = text
        except Exception as e:
            log.debug("OCR 실패 (%s): %s", fname, e)
    return texts


# ══════════════════════════════════════════════════════════════
# Markdown 후처리
# ══════════════════════════════════════════════════════════════

_IMG_REF = re.compile(r"!\[([^\]]*)\]\(([^)]+/(imageFile\d+\.[a-zA-Z]+))\)")


def _replace_images_with_ocr(markdown: str, ocr_map: dict) -> str:
    def _sub(m: re.Match) -> str:
        text = ocr_map.get(m.group(3), "")
        return f"\n> **[이미지 텍스트]** {text}\n" if text else ""
    return _IMG_REF.sub(_sub, markdown)


def _remove_image_refs(markdown: str) -> str:
    return _IMG_REF.sub("", markdown)


def _inject_plumber_tables(markdown: str, tables: dict) -> str:
    if not tables:
        return markdown
    extras = [
        f"\n\n**[표 — 페이지 {p}]**\n\n{tbl}"
        for p in sorted(tables)
        for tbl in tables[p]
    ]
    return markdown + "\n" + "\n".join(extras) if extras else markdown


def _clean_markdown(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


# ══════════════════════════════════════════════════════════════
# 메인 변환 파이프라인
# ══════════════════════════════════════════════════════════════

async def _do_convert(pdf_path: str, ocr_lang: str,
                      use_ocr: bool, use_hybrid: bool = False) -> str:
    """
    변환 파이프라인:
      ODL 전체 실행  ──┐
      pdfplumber 표  ──┼── 병렬
      사이드카 표*   ──┘   (* use_hybrid=True 시)

      ODL 완료 후: 이미지 정리 → 표 주입(사이드카 우선, plumber 보완)
    """
    sidecar_ready = _sidecar_queue is not None and _sidecar_queue.qsize() > 0
    log.info("변환 시작 (hybrid_table=%s 사이드카=%s ODL=%s)",
             use_hybrid, sidecar_ready, _HAS_ODL)

    # 1. ODL 전체 변환 (pages 파라미터 없음 — 가장 안정적)
    odl_task = asyncio.create_task(_run_odl_or_fallback(pdf_path))

    # 2. pdfplumber 표 추출 (항상 병렬)
    plumber_task = asyncio.create_task(
        asyncio.to_thread(_extract_tables_plumber, pdf_path)
    )

    # 3. use_hybrid=True이면 사이드카 표 추출도 병렬 시작
    sidecar_table_task: Optional[asyncio.Future] = None
    if use_hybrid and sidecar_ready:
        page_ranges = await asyncio.to_thread(
            _make_page_ranges, pdf_path, HYBRID_PAGE_CHUNK
        )
        log.info("사이드카 표 추출 병렬 시작 (%d청크)", len(page_ranges))
        sidecar_table_task = asyncio.ensure_future(asyncio.gather(*[
            _sidecar_tables_chunk(pdf_path, p_start, p_end)
            for p_start, p_end in page_ranges
        ]))

    # 모든 태스크 완료 대기
    markdown = await odl_task
    plumber_tables = await plumber_task
    sidecar_table_lists: list[list[str]] = []
    if sidecar_table_task is not None:
        sidecar_table_lists = list(await sidecar_table_task)

    # 4. 이미지 처리
    if use_ocr and _HAS_OCR and _HAS_ODL:
        log.info("OCR 모드: 이미지 디렉터리에서 OCR")
        out_dir = tempfile.mkdtemp(prefix="odl_ocr_")
        try:
            await asyncio.to_thread(_run_odl_full_to_dir, pdf_path, out_dir)
            img_dirs = [d for d in Path(out_dir).iterdir()
                        if d.is_dir() and d.name.endswith("_images")]
            if img_dirs:
                reader = await _get_ocr_reader(ocr_lang)
                ocr_map = await asyncio.to_thread(
                    _ocr_image_dir, str(img_dirs[0]), reader
                )
                markdown = _replace_images_with_ocr(markdown, ocr_map)
            else:
                markdown = _remove_image_refs(markdown)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
    else:
        markdown = _remove_image_refs(markdown)

    # 5. 표 주입: 사이드카 표(고품질) 우선, pdfplumber 보완
    all_sidecar_tables = [t for chunk in sidecar_table_lists for t in chunk]
    if all_sidecar_tables:
        log.info("사이드카 표 %d개 주입", len(all_sidecar_tables))
        extras = "\n\n".join(
            f"**[Docling 추출 표 {i+1}]**\n\n{t}"
            for i, t in enumerate(all_sidecar_tables)
        )
        markdown = markdown + "\n\n---\n\n## 표 목록 (Docling)\n\n" + extras
    else:
        markdown = _inject_plumber_tables(markdown, plumber_tables)

    return _clean_markdown(markdown)


# ══════════════════════════════════════════════════════════════
# 백그라운드 작업 실행기
# ══════════════════════════════════════════════════════════════

async def _run_job(job_id: str, pdf_path: str, ocr_lang: str,
                   use_ocr: bool, use_hybrid: bool):
    _jobs[job_id]["status"] = "running"
    try:
        markdown = await _do_convert(pdf_path, ocr_lang, use_ocr, use_hybrid)
        _jobs[job_id].update({"status": "done", "markdown": markdown})
        log.info("Job %s 완료: %d자", job_id, len(markdown))
    except Exception as e:
        log.error("Job %s 실패: %s", job_id, e)
        _jobs[job_id].update({"status": "error", "error": str(e)})
    finally:
        try:
            os.unlink(pdf_path)
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════
# API 엔드포인트
# ══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    sidecar_ready = _sidecar_queue.qsize() if _sidecar_queue else 0
    return {
        "service": "PDF→Markdown 변환 서버",
        "version": "4.0.0",
        "config": {
            "workers": HYBRID_WORKERS,
            "chunk_pages": HYBRID_PAGE_CHUNK,
            "base_port": HYBRID_BASE_PORT,
        },
        "backends": {
            "sidecar_ready": sidecar_ready,
            "opendataloader": _HAS_ODL,
            "docling_core": _HAS_DOCLING,
            "pdfplumber": _HAS_PLUMBER,
            "easyocr": _HAS_OCR,
            "pymupdf": _HAS_FITZ,
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/convert", summary="PDF 변환 시작 (비동기, 권장)")
async def convert_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ocr_lang: str = Form(OCR_LANG),
    use_ocr: bool = Form(False),
    use_hybrid: bool = Form(False),
):
    """
    PDF 변환을 백그라운드에서 시작하고 job_id를 즉시 반환.
    use_hybrid=true: Docling 사이드카 사용 (고품질, 느림)
    use_hybrid=false: ODL 병렬 청크 사용 (기본, 빠름)
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 허용됩니다.")
    content = await file.read()
    size_mb = len(content) / (1024 ** 2)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(status_code=413,
                            detail=f"파일 크기 {size_mb:.1f}MB > 상한 {MAX_UPLOAD_MB}MB")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="pdf_") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "pending",
        "filename": file.filename,
        "size_mb": round(size_mb, 2),
        "markdown": "",
        "error": "",
    }
    background_tasks.add_task(_run_job, job_id, tmp_path, ocr_lang, use_ocr, use_hybrid)
    log.info("Job %s 등록: %s (%.1f MB) hybrid=%s", job_id, file.filename, size_mb, use_hybrid)
    return JSONResponse({"job_id": job_id, "status": "pending"}, status_code=202)


@app.get("/status/{job_id}", summary="변환 상태 조회")
async def get_status(job_id: str):
    """status: pending | running | done | error"""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' 없음")
    return JSONResponse(job)


@app.post("/convert/sync", summary="PDF 변환 (동기, Dify 직접 연동)")
async def convert_sync(
    file: UploadFile = File(...),
    ocr_lang: str = Form(OCR_LANG),
    use_ocr: bool = Form(False),
    use_hybrid: bool = Form(False),
):
    """동기 변환. Dify 외부 파서 연동 시 사용."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 허용됩니다.")
    content = await file.read()
    size_mb = len(content) / (1024 ** 2)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(status_code=413,
                            detail=f"파일 크기 {size_mb:.1f}MB > 상한 {MAX_UPLOAD_MB}MB")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="pdf_") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    log.info("동기 변환 시작: %s (%.1f MB) hybrid=%s", file.filename, size_mb, use_hybrid)
    try:
        markdown = await _do_convert(tmp_path, ocr_lang, use_ocr, use_hybrid)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    log.info("동기 변환 완료: %d자", len(markdown))
    return JSONResponse({
        "filename": file.filename,
        "size_mb": round(size_mb, 2),
        "markdown": markdown,
        "length": len(markdown),
    })


# ══════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level=LOG_LEVEL.lower(),
        timeout_keep_alive=600,
    )
