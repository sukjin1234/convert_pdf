"""FastAPI 서버 — 실행: python main.py"""

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
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import easyocr
import numpy as np
import opendataloader_pdf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

USE_GPU = torch.cuda.is_available()

# ── 이미지 OCR ───────────────────────────────────────────────
_ocr_reader: Optional[easyocr.Reader] = None
_ocr_reader_langs: Optional[str] = None
_IMG_RE = re.compile(r"!\[([^\]]*)\]\((.+?\.(?:png|jpe?g|gif|bmp|webp))\)", re.IGNORECASE)

# ── 하이브리드 백엔드 설정 ───────────────────────────────────
HYBRID_PORT = int(os.environ.get("HYBRID_PORT", "5002"))
HYBRID_URL = f"http://127.0.0.1:{HYBRID_PORT}"
HYBRID_OCR_LANG = os.environ.get("HYBRID_OCR_LANG", "ko,en")
HYBRID_FORCE_OCR = os.environ.get("HYBRID_FORCE_OCR", "").strip().lower() in (
    "1", "true", "yes", "on",
)
# 한 청크에 넣을 페이지 수. bad_alloc이 ~28페이지에서 나므로 10이면 안전.
HYBRID_PAGE_CHUNK = max(1, int(os.environ.get("HYBRID_PAGE_CHUNK", "10")))
# 같은 임시 하이브리드에서 이만큼 청크를 처리한 뒤 재시작 (메모리 해제)
HYBRID_RECYCLE_EVERY = max(1, int(os.environ.get("HYBRID_RECYCLE_EVERY", "2")))
# 동시에 띄울 임시 하이브리드 프로세스 수 (GPU 메모리에 따라 조절)
HYBRID_WORKERS = max(1, int(os.environ.get("HYBRID_WORKERS", "2")))

_hybrid_proc: Optional[subprocess.Popen] = None
_hybrid_child_force_ocr: bool = HYBRID_FORCE_OCR


# ── 하이브리드 프로세스 유틸 ──────────────────────────────────

def _find_hybrid_cli() -> Optional[str]:
    scripts = os.path.dirname(sys.executable)
    for name in ("opendataloader-pdf-hybrid", "opendataloader-pdf-hybrid.exe"):
        path = os.path.join(scripts, name)
        if os.path.isfile(path):
            return path
    return shutil.which("opendataloader-pdf-hybrid")


def _build_hybrid_cmd(cli: str, force_ocr: bool, port: int = HYBRID_PORT) -> list[str]:
    cmd = [cli, "--port", str(port), "--ocr-lang", HYBRID_OCR_LANG]
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


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _is_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen(url.rstrip("/") + "/", timeout=timeout)
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
        if _is_reachable(url):
            return True
        await asyncio.sleep(0.25)
    return False


# ── 전역 하이브리드 (비-청크 모드 전용) ──────────────────────

async def _restart_hybrid(force_ocr: bool) -> None:
    global _hybrid_proc, _hybrid_child_force_ocr

    if _hybrid_proc is not None:
        _kill_proc(_hybrid_proc)
        _hybrid_proc = None

    cli = _find_hybrid_cli()
    if cli is None:
        return

    cmd = _build_hybrid_cmd(cli, force_ocr)
    print(f"[hybrid] 백엔드 재시작 중 (force_ocr={force_ocr})… {' '.join(cmd)}", flush=True)
    _hybrid_proc = subprocess.Popen(cmd)
    if await _wait_reachable(HYBRID_URL, _hybrid_proc):
        _hybrid_child_force_ocr = force_ocr
        print(f"[hybrid] 재시작 완료 ({HYBRID_URL})", flush=True)
    elif _hybrid_proc.poll() is not None:
        print(f"[hybrid] 재시작 실패 (exit {_hybrid_proc.returncode})", flush=True)
        _hybrid_proc = None
    else:
        print("[hybrid] 재시작 시간 초과", flush=True)


# ── 병렬 임시 하이브리드 변환 ─────────────────────────────────

def _pre_extract_chunks(
    in_path: str, tmpdir: str, n_pages: int, chunk_size: int,
) -> list[tuple[int, int, str, str]]:
    """원본 PDF를 한 번만 읽어 모든 청크 PDF + 출력 디렉터리를 미리 생성한다."""
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
    force_ocr: bool,
    kwargs_base: dict,
    n_pages: int,
    results: list[Optional[str]],
    errors: list[Optional[str]],
) -> None:
    """한 워커가 할당된 청크들을 순서대로 처리한다.
    HYBRID_RECYCLE_EVERY 청크마다 임시 하이브리드를 재시작한다."""
    proc: Optional[subprocess.Popen] = None
    eph_url: Optional[str] = None
    done_on_proc = 0

    try:
        for idx in chunk_indices:
            start, end, chunk_pdf, chunk_out = chunk_infos[idx]

            if proc is None or done_on_proc >= HYBRID_RECYCLE_EVERY:
                if proc is not None:
                    _kill_proc(proc)
                    proc = None
                    await asyncio.sleep(0.3)
                port = _pick_free_port()
                eph_url = f"http://127.0.0.1:{port}"
                cmd = _build_hybrid_cmd(cli, force_ocr, port)
                print(
                    f"[worker-{worker_id}] 임시 하이브리드 시작 port={port} "
                    f"(청크 {idx+1}/{len(chunk_infos)}부터)",
                    flush=True,
                )
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if not await _wait_reachable(eph_url, proc):
                    errors[idx] = f"임시 하이브리드 기동 실패 (worker={worker_id}, port={port})"
                    return
                print(f"[worker-{worker_id}] 준비 완료 ({eph_url})", flush=True)
                done_on_proc = 0

            chunk_kw = {
                **kwargs_base,
                "input_path": [chunk_pdf],
                "output_dir": chunk_out,
                "hybrid_url": eph_url,
            }

            t0 = time.monotonic()
            hybrid_ok = True
            try:
                await asyncio.to_thread(lambda kw=chunk_kw: opendataloader_pdf.convert(**kw))
            except Exception as e:
                hybrid_ok = False
                print(
                    f"[worker-{worker_id}] 하이브리드 실패 (페이지 {start}-{end}), "
                    f"non-hybrid fallback 시도: {type(e).__name__}",
                    flush=True,
                )

            done_on_proc += 1
            chunk_stem = _stem(os.path.basename(chunk_pdf))

            # 하이브리드 성공 시에도 출력이 짧으면 partial_success → fallback 대상
            md_text: Optional[str] = None
            if hybrid_ok:
                md_path = _find(chunk_out, f"{chunk_stem}.md") or _find(chunk_out, "*.md")
                if md_path:
                    with open(md_path, encoding="utf-8") as f:
                        md_text = f.read()
                expected_pages = end - start + 1
                if md_text is not None and expected_pages > 1 and len(md_text.strip()) < expected_pages * 50:
                    print(
                        f"[worker-{worker_id}] 출력이 짧음 ({len(md_text.strip())} chars, "
                        f"예상 {expected_pages * 50}+), non-hybrid fallback 시도",
                        flush=True,
                    )
                    hybrid_ok = False

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
                    await asyncio.to_thread(lambda kw=fallback_kw: opendataloader_pdf.convert(**kw))
                    fb_md_path = _find(fallback_out, f"{chunk_stem}.md") or _find(fallback_out, "*.md")
                    if fb_md_path:
                        with open(fb_md_path, encoding="utf-8") as f:
                            fb_text = f.read()
                        if md_text is None or len(fb_text.strip()) > len(md_text.strip()):
                            md_text = fb_text
                            print(
                                f"[worker-{worker_id}] fallback 성공 (페이지 {start}-{end}, "
                                f"{len(fb_text.strip())} chars)",
                                flush=True,
                            )
                except Exception as e2:
                    print(f"[worker-{worker_id}] fallback도 실패: {e2}", flush=True)

            elapsed = time.monotonic() - t0
            if md_text is not None:
                results[idx] = md_text
                print(
                    f"[worker-{worker_id}] 페이지 {start}-{end}/{n_pages} 완료 "
                    f"({elapsed:.1f}초)",
                    flush=True,
                )
            else:
                errors[idx] = f"Markdown 출력 없음 (페이지 {start}-{end}/{n_pages})"
    finally:
        if proc is not None:
            _kill_proc(proc)
            print(f"[worker-{worker_id}] 임시 하이브리드 종료", flush=True)


async def _convert_chunks_parallel(
    in_path: str,
    tmpdir: str,
    kwargs_base: dict,
    n_pages: int,
    force_ocr: bool,
    ocr_lang: str,
) -> tuple[str, Optional[JSONResponse]]:
    """청크를 여러 임시 하이브리드 워커로 병렬 변환한다."""
    cli = _find_hybrid_cli()
    if cli is None:
        return "", JSONResponse(
            status_code=503,
            content={"ok": False, "error": "opendataloader-pdf-hybrid CLI를 찾을 수 없습니다."},
        )

    t_total = time.monotonic()

    # 1) 청크 PDF 일괄 추출
    t0 = time.monotonic()
    chunk_infos = _pre_extract_chunks(in_path, tmpdir, n_pages, HYBRID_PAGE_CHUNK)
    total = len(chunk_infos)
    print(
        f"[convert] {n_pages}페이지 → {total}청크({HYBRID_PAGE_CHUNK}p) 추출 완료 "
        f"({time.monotonic() - t0:.1f}초), 워커 {HYBRID_WORKERS}개 병렬",
        flush=True,
    )

    # 2) 라운드로빈으로 워커에 청크 배분
    worker_chunks: list[list[int]] = [[] for _ in range(HYBRID_WORKERS)]
    for i in range(total):
        worker_chunks[i % HYBRID_WORKERS].append(i)

    results: list[Optional[str]] = [None] * total
    errors: list[Optional[str]] = [None] * total

    # 3) 병렬 변환
    tasks = [
        _worker_loop(
            worker_id=w,
            chunk_indices=indices,
            chunk_infos=chunk_infos,
            cli=cli,
            force_ocr=force_ocr,
            kwargs_base=kwargs_base,
            n_pages=n_pages,
            results=results,
            errors=errors,
        )
        for w, indices in enumerate(worker_chunks)
        if indices
    ]
    await asyncio.gather(*tasks)

    # 4) 에러 체크
    first_err = next((e for e in errors if e), None)
    if first_err and all(r is None for r in results):
        return "", JSONResponse(
            status_code=500,
            content={"ok": False, "error": first_err},
        )

    # 5) OCR + 조립 (순차 — EasyOCR은 thread-safe 아님)
    parts: list[str] = []
    for i, (start, end, _, chunk_out) in enumerate(chunk_infos):
        md = results[i]
        if md is None:
            md = f"<!-- 변환 실패: {errors[i] or 'unknown'} -->"
        else:
            md = _replace_images_with_ocr(md, chunk_out, ocr_lang)
        parts.append(
            f"\n\n<!-- opendataloader: PDF pages {start}-{end} of {n_pages} -->\n\n{md}"
        )

    elapsed_total = time.monotonic() - t_total
    print(f"[convert] 전체 변환 완료: {n_pages}페이지, {elapsed_total:.1f}초", flush=True)
    return "".join(parts).lstrip(), None


# ── lifespan ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _hybrid_proc, _hybrid_child_force_ocr

    if _is_reachable(HYBRID_URL):
        print(f"[hybrid] 이미 실행 중 ({HYBRID_URL})", flush=True)
    else:
        cli = _find_hybrid_cli()
        if cli is None:
            print("[hybrid] opendataloader-pdf-hybrid 를 찾을 수 없습니다.", flush=True)
        else:
            cmd = _build_hybrid_cmd(cli, HYBRID_FORCE_OCR)
            print(f"[hybrid] 백엔드 시작 (force_ocr={HYBRID_FORCE_OCR}): {' '.join(cmd)}", flush=True)
            _hybrid_proc = subprocess.Popen(cmd)
            if await _wait_reachable(HYBRID_URL, _hybrid_proc):
                _hybrid_child_force_ocr = HYBRID_FORCE_OCR
                print(f"[hybrid] 준비 완료 ({HYBRID_URL})", flush=True)
            elif _hybrid_proc.poll() is not None:
                print(f"[hybrid] 프로세스 종료됨 (exit {_hybrid_proc.returncode})", flush=True)
                _hybrid_proc = None
            else:
                print("[hybrid] 시간 초과 — 백엔드 미준비", flush=True)

    yield

    if _hybrid_proc is not None and _hybrid_proc.poll() is None:
        print("[hybrid] 백엔드 종료 중…", flush=True)
        _kill_proc(_hybrid_proc)
        _hybrid_proc = None


# ── FastAPI 앱 ────────────────────────────────────────────────
app = FastAPI(title="PDF to Markdown API", version="0.1.0", lifespan=lifespan)


@app.get("/")
def root():
    return {"ok": True, "message": "정상 동작 중입니다."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/convert")
async def convert(
    file: UploadFile = File(..., description="변환할 PDF 파일"),
    use_hybrid: bool = Form(False, description="하이브리드 모드 사용 여부 (표·이미지 분석 강화)"),
    hybrid_url: str = Form(
        "http://127.0.0.1:5002",
        description="하이브리드 백엔드 URL",
    ),
    ocr_lang: str = Form(
        "ko,en",
        description="OCR 언어 (쉼표 구분, 예: ko,en / ja,en / ch_sim,en). 변경 시 하이브리드 서버가 자동 재시작됩니다.",
    ),
    hybrid_force_ocr: Optional[bool] = Form(
        None,
        description=(
            "하이브리드 force-ocr. 생략 시 환경변수 HYBRID_FORCE_OCR. "
            "True면 전 페이지 OCR(느리고 RAM 많이 씀). 바뀌면 하이브리드 재시작."
        ),
    ),
):
    filename = (file.filename or "document.pdf").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=".pdf 파일만 업로드할 수 있습니다.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    eff_force_ocr = hybrid_force_ocr if hybrid_force_ocr is not None else HYBRID_FORCE_OCR

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.pdf")
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(out_dir)

        with open(in_path, "wb") as f:
            f.write(data)

        n_pages = _pdf_page_count(in_path)
        will_chunk = (
            use_hybrid
            and HYBRID_PAGE_CHUNK > 0
            and n_pages is not None
            and n_pages > HYBRID_PAGE_CHUNK
        )

        print(
            f"[convert] use_hybrid={use_hybrid} pages={n_pages} "
            f"chunk={HYBRID_PAGE_CHUNK} workers={HYBRID_WORKERS} "
            f"force_ocr={eff_force_ocr} will_chunk={will_chunk}",
            flush=True,
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
            kwargs["hybrid_url"] = hybrid_url
            kwargs["hybrid_fallback"] = True

        stem = _stem("input.pdf")
        markdown: str

        try:
            if will_chunk:
                if use_hybrid:
                    lang_changed = ocr_lang.strip() != HYBRID_OCR_LANG
                    if lang_changed:
                        _update_ocr_lang(ocr_lang.strip())

                markdown, err_resp = await _convert_chunks_parallel(
                    in_path=in_path,
                    tmpdir=tmpdir,
                    kwargs_base=kwargs,
                    n_pages=n_pages,
                    force_ocr=eff_force_ocr,
                    ocr_lang=ocr_lang,
                )
                if err_resp is not None:
                    return err_resp
            else:
                if use_hybrid:
                    lang_changed = ocr_lang.strip() != HYBRID_OCR_LANG
                    if lang_changed:
                        _update_ocr_lang(ocr_lang.strip())
                    if (
                        eff_force_ocr != _hybrid_child_force_ocr
                        or lang_changed
                        or not _is_reachable(hybrid_url)
                    ):
                        await _restart_hybrid(eff_force_ocr)
                    if not _is_reachable(hybrid_url):
                        raise HTTPException(
                            status_code=503,
                            detail=f"하이브리드 백엔드({hybrid_url})에 연결할 수 없습니다.",
                        )

                opendataloader_pdf.convert(**kwargs)
                md_path = _find(out_dir, f"{stem}.md") or _find(out_dir, "*.md")
                if not md_path:
                    return JSONResponse(
                        status_code=500,
                        content={"ok": False, "error": "Markdown 출력 파일을 찾을 수 없습니다."},
                    )
                with open(md_path, encoding="utf-8") as f:
                    markdown = f.read()
                markdown = _replace_images_with_ocr(markdown, out_dir, ocr_lang)
        except Exception as e:
            return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

    out: dict = {
        "ok": True,
        "filename": filename,
        "markdown": markdown,
        "hybrid": use_hybrid,
        "ocr_lang": ocr_lang,
    }
    if use_hybrid:
        out["hybrid_force_ocr"] = eff_force_ocr
        out["hybrid_page_chunk"] = HYBRID_PAGE_CHUNK
    return out


# ── OCR / 이미지 ─────────────────────────────────────────────

def _get_ocr_reader(lang_csv: str) -> easyocr.Reader:
    global _ocr_reader, _ocr_reader_langs
    if _ocr_reader is None or _ocr_reader_langs != lang_csv:
        langs = [l.strip() for l in lang_csv.split(",") if l.strip()]
        _ocr_reader = easyocr.Reader(langs, gpu=USE_GPU)
        _ocr_reader_langs = lang_csv
    return _ocr_reader


def _build_image_index(out_dir: str) -> dict[str, str]:
    index: dict[str, str] = {}
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
                index[f] = os.path.join(root, f)
    return index


def _replace_images_with_ocr(markdown: str, out_dir: str, lang_csv: str) -> str:
    index = _build_image_index(out_dir)
    if not index:
        return markdown

    def _ocr_image(match: re.Match) -> str:
        alt, img_path = match.group(1), match.group(2)
        basename = os.path.basename(img_path)
        abs_path = index.get(basename)
        if abs_path is None or not os.path.isfile(abs_path):
            return match.group(0)
        try:
            img_array = _read_image(abs_path)
            if img_array is None:
                return match.group(0)
            reader = _get_ocr_reader(lang_csv)
            results = reader.readtext(img_array, detail=0)
            text = "\n".join(results).strip()
        except Exception:
            return match.group(0)
        if not text:
            return match.group(0)
        return f"\n\n<!-- OCR: {alt} -->\n{text}\n"

    return _IMG_RE.sub(_ocr_image, markdown)


def _read_image(path: str) -> Optional[np.ndarray]:
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ── PDF 유틸 ─────────────────────────────────────────────────

def _update_ocr_lang(lang: str) -> None:
    global HYBRID_OCR_LANG
    HYBRID_OCR_LANG = lang


def _pdf_page_count(path: str) -> Optional[int]:
    try:
        from pypdf import PdfReader
    except ImportError:
        print("[page_count] pypdf 미설치 — pip install pypdf", flush=True)
        return None

    try:
        with open(path, "rb") as f:
            n = len(PdfReader(f).pages)
        print(f"[page_count] pypdf → {n}페이지", flush=True)
        return n
    except Exception as e:
        print(f"[page_count] pypdf 실패: {e}", flush=True)

    try:
        with open(path, "rb") as f:
            raw = f.read()
        import re as _re
        count = len(_re.findall(rb"/Type\s*/Page(?!s)", raw))
        if count > 0:
            print(f"[page_count] regex fallback → {count}페이지", flush=True)
            return count
    except Exception as e2:
        print(f"[page_count] regex fallback 실패: {e2}", flush=True)

    return None


def _find(directory: str, pattern: str) -> Optional[str]:
    paths = glob.glob(os.path.join(directory, pattern))
    return paths[0] if paths else None


def _stem(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


if __name__ == "__main__":
    print(f"[GPU] CUDA 사용: {USE_GPU}", flush=True)
    if USE_GPU:
        print(f"[GPU] 디바이스: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"[설정] CHUNK={HYBRID_PAGE_CHUNK}p RECYCLE={HYBRID_RECYCLE_EVERY} "
        f"WORKERS={HYBRID_WORKERS}",
        flush=True,
    )
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
