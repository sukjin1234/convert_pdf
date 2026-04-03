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

# ── 하이브리드 백엔드 자동 관리 ──────────────────────────────
HYBRID_PORT = int(os.environ.get("HYBRID_PORT", "5002"))
HYBRID_URL = f"http://127.0.0.1:{HYBRID_PORT}"
HYBRID_OCR_LANG = os.environ.get("HYBRID_OCR_LANG", "ko,en")
HYBRID_FORCE_OCR = os.environ.get("HYBRID_FORCE_OCR", "").strip().lower() in (
    "1", "true", "yes", "on",
)
HYBRID_PAGE_CHUNK = 1
# 임시 하이브리드 프로세스를 N청크마다 재시작해서 메모리(std::bad_alloc) 누적을 끊는다.
HYBRID_RECYCLE_EVERY = max(1, int(os.environ.get("HYBRID_RECYCLE_EVERY", "10")))

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


# ── 임시(ephemeral) 하이브리드로 물리 청크 변환 ──────────────

async def _convert_chunks_with_ephemeral(
    in_path: str,
    tmpdir: str,
    kwargs_base: dict,
    n_pages: int,
    eff_page_chunk: int,
    force_ocr: bool,
    ocr_lang: str,
) -> tuple[str, Optional[JSONResponse]]:
    """1-page 청크마다 잘린 PDF를 임시 하이브리드 서버로 변환한다.
    HYBRID_RECYCLE_EVERY 청크마다 프로세스를 kill→재시작해서 네이티브 메모리 누적을 끊는다.
    """
    cli = _find_hybrid_cli()
    if cli is None:
        return "", JSONResponse(
            status_code=503,
            content={"ok": False, "error": "opendataloader-pdf-hybrid CLI를 찾을 수 없습니다."},
        )

    chunk_ranges = [
        (s, min(s + eff_page_chunk - 1, n_pages))
        for s in range(1, n_pages + 1, eff_page_chunk)
    ]
    total = len(chunk_ranges)
    print(f"[hybrid-eph] 총 {n_pages}페이지 → {total}청크, 매 {HYBRID_RECYCLE_EVERY}청크마다 재활용", flush=True)

    parts: list[str] = []
    proc: Optional[subprocess.Popen] = None
    eph_url: Optional[str] = None
    done_on_proc = 0

    try:
        for idx, (start, end) in enumerate(chunk_ranges):
            if proc is None or done_on_proc >= HYBRID_RECYCLE_EVERY:
                if proc is not None:
                    _kill_proc(proc)
                    proc = None
                    await asyncio.sleep(0.5)
                port = _pick_free_port()
                eph_url = f"http://127.0.0.1:{port}"
                cmd = _build_hybrid_cmd(cli, force_ocr, port)
                print(f"[hybrid-eph] 임시 하이브리드 시작 port={port} (청크 {idx+1}/{total}부터)", flush=True)
                proc = subprocess.Popen(cmd)
                if not await _wait_reachable(eph_url, proc):
                    code = proc.poll()
                    return "", JSONResponse(
                        status_code=503,
                        content={
                            "ok": False,
                            "error": f"임시 하이브리드 기동 실패 (exit={code}, port={port})",
                        },
                    )
                print(f"[hybrid-eph] 준비 완료 ({eph_url})", flush=True)
                done_on_proc = 0

            chunk_pdf_name = f"_chunk_{start}_{end}.pdf"
            chunk_pdf = os.path.join(tmpdir, chunk_pdf_name)
            chunk_stem = _stem(chunk_pdf_name)
            try:
                _extract_pdf_pages(in_path, chunk_pdf, start, end)
            except Exception as ex:
                return "", JSONResponse(
                    status_code=500,
                    content={"ok": False, "error": f"PDF 분할 실패 (페이지 {start}-{end}): {ex}"},
                )

            chunk_out = os.path.join(tmpdir, f"out_{start}_{end}")
            os.makedirs(chunk_out, exist_ok=True)

            chunk_kw = {
                **kwargs_base,
                "input_path": [chunk_pdf],
                "output_dir": chunk_out,
                "hybrid_url": eph_url,
            }
            opendataloader_pdf.convert(**chunk_kw)
            done_on_proc += 1

            md_path = _find(chunk_out, f"{chunk_stem}.md") or _find(chunk_out, "*.md")
            if not md_path:
                return "", JSONResponse(
                    status_code=500,
                    content={
                        "ok": False,
                        "error": f"Markdown 출력 없음 (페이지 {start}-{end}/{n_pages})",
                    },
                )

            with open(md_path, encoding="utf-8") as f:
                chunk_md = f.read()
            chunk_md = _replace_images_with_ocr(chunk_md, chunk_out, ocr_lang)
            parts.append(
                f"\n\n<!-- opendataloader: PDF pages {start}-{end} of {n_pages} -->\n\n"
                f"{chunk_md}"
            )
            print(f"[hybrid-eph] 페이지 {start}-{end}/{n_pages} 완료 ({idx+1}/{total})", flush=True)
    finally:
        if proc is not None:
            _kill_proc(proc)
            print("[hybrid-eph] 임시 하이브리드 종료", flush=True)

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
        safe_name = "input.pdf"
        in_path = os.path.join(tmpdir, safe_name)
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
            f"[convert] use_hybrid={use_hybrid} pages={n_pages} chunk={HYBRID_PAGE_CHUNK} "
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

        stem = _stem(safe_name)
        markdown: str

        try:
            if will_chunk:
                # ── 물리 청크: 임시 하이브리드로 N페이지씩 ──
                if use_hybrid:
                    lang_changed = ocr_lang.strip() != HYBRID_OCR_LANG
                    if lang_changed:
                        _update_ocr_lang(ocr_lang.strip())

                markdown, err_resp = await _convert_chunks_with_ephemeral(
                    in_path=in_path,
                    tmpdir=tmpdir,
                    kwargs_base=kwargs,
                    n_pages=n_pages,
                    eff_page_chunk=HYBRID_PAGE_CHUNK,
                    force_ocr=eff_force_ocr,
                    ocr_lang=ocr_lang,
                )
                if err_resp is not None:
                    return err_resp
            else:
                # ── 단일 변환 (페이지 적거나 non-hybrid) ──
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

    # 방법 1: pypdf
    try:
        with open(path, "rb") as f:
            n = len(PdfReader(f).pages)
        print(f"[page_count] pypdf → {n}페이지", flush=True)
        return n
    except Exception as e:
        print(f"[page_count] pypdf 실패: {e}", flush=True)

    # 방법 2: 바이너리에서 /Type /Page 개수 세기 (근사치)
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


def _extract_pdf_pages(src_pdf: str, dst_pdf: str, page_start_1: int, page_end_1: int) -> None:
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(src_pdf)
    writer = PdfWriter()
    for i in range(page_start_1 - 1, page_end_1):
        writer.add_page(reader.pages[i])
    with open(dst_pdf, "wb") as f:
        writer.write(f)


def _find(directory: str, pattern: str) -> Optional[str]:
    paths = glob.glob(os.path.join(directory, pattern))
    return paths[0] if paths else None


def _stem(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


if __name__ == "__main__":
    print(f"[GPU] CUDA 사용: {USE_GPU}", flush=True)
    if USE_GPU:
        print(f"[GPU] 디바이스: {torch.cuda.get_device_name(0)}", flush=True)
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
