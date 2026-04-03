"""FastAPI 서버 — 실행: python main.py"""

import asyncio
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import easyocr
import numpy as np
import opendataloader_pdf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ── 이미지 OCR ───────────────────────────────────────────────
_ocr_reader: Optional[easyocr.Reader] = None
_ocr_reader_langs: Optional[str] = None
_IMG_RE = re.compile(r"!\[([^\]]*)\]\((.+?\.(?:png|jpe?g|gif|bmp|webp))\)", re.IGNORECASE)

# ── 하이브리드 백엔드 자동 관리 ──────────────────────────────
HYBRID_PORT = int(os.environ.get("HYBRID_PORT", "5002"))
HYBRID_URL = f"http://127.0.0.1:{HYBRID_PORT}"
HYBRID_OCR_LANG = os.environ.get("HYBRID_OCR_LANG", "ko,en")
# --force-ocr 는 모든 페이지를 OCR → RAM 폭증, 긴 PDF에서 std::bad_alloc 유발 가능.
# 필요할 때만: HYBRID_FORCE_OCR=1
HYBRID_FORCE_OCR = os.environ.get("HYBRID_FORCE_OCR", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# 하이브리드: PDF를 N페이지씩 나눠 변환 후 합침(std::bad_alloc·메모리 누적 완화). 0이면 끔.
HYBRID_PAGE_CHUNK = int(os.environ.get("HYBRID_PAGE_CHUNK", "15"))
_hybrid_proc: Optional[subprocess.Popen] = None
# 자식으로 띄운 하이브리드가 마지막으로 기동될 때 적용된 force-ocr (요청별로 다르면 재시작)
_hybrid_child_force_ocr: bool = HYBRID_FORCE_OCR


def _build_hybrid_cmd(cli: str, force_ocr: bool) -> list[str]:
    cmd = [cli, "--port", str(HYBRID_PORT), "--ocr-lang", HYBRID_OCR_LANG]
    if force_ocr:
        cmd.append("--force-ocr")
    return cmd


async def _restart_hybrid(force_ocr: bool) -> None:
    """하이브리드 백엔드를 force_ocr 설정에 맞춰 다시 띄운다."""
    global _hybrid_proc, _hybrid_child_force_ocr

    if _hybrid_proc is not None and _hybrid_proc.poll() is None:
        _hybrid_proc.terminate()
        try:
            _hybrid_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _hybrid_proc.kill()
        _hybrid_proc = None

    cli = _find_hybrid_cli()
    if cli is None:
        return

    cmd = _build_hybrid_cmd(cli, force_ocr)
    print(
        f"[hybrid] 백엔드 재시작 중 (force_ocr={force_ocr})… {' '.join(cmd)}",
        flush=True,
    )
    _hybrid_proc = subprocess.Popen(cmd)
    for _ in range(60):
        await asyncio.sleep(0.5)
        if _is_reachable(HYBRID_URL):
            _hybrid_child_force_ocr = force_ocr
            print(f"[hybrid] 재시작 완료 ({HYBRID_URL})", flush=True)
            return
        if _hybrid_proc.poll() is not None:
            print(f"[hybrid] 재시작 실패 (exit {_hybrid_proc.returncode})", flush=True)
            _hybrid_proc = None
            return
    print("[hybrid] 재시작 시간 초과", flush=True)


def _find_hybrid_cli() -> Optional[str]:
    scripts = os.path.dirname(sys.executable)
    for name in ("opendataloader-pdf-hybrid", "opendataloader-pdf-hybrid.exe"):
        path = os.path.join(scripts, name)
        if os.path.isfile(path):
            return path
    return shutil.which("opendataloader-pdf-hybrid")


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
            print(
                f"[hybrid] 백엔드 시작 (force_ocr={HYBRID_FORCE_OCR}): {' '.join(cmd)}",
                flush=True,
            )
            _hybrid_proc = subprocess.Popen(cmd)
            for _ in range(60):
                await asyncio.sleep(0.5)
                if _is_reachable(HYBRID_URL):
                    _hybrid_child_force_ocr = HYBRID_FORCE_OCR
                    print(f"[hybrid] 준비 완료 ({HYBRID_URL})", flush=True)
                    break
                if _hybrid_proc.poll() is not None:
                    print(f"[hybrid] 프로세스 종료됨 (exit {_hybrid_proc.returncode})", flush=True)
                    _hybrid_proc = None
                    break
            else:
                print("[hybrid] 시간 초과 — 백엔드 미준비", flush=True)

    yield

    if _hybrid_proc is not None and _hybrid_proc.poll() is None:
        print("[hybrid] 백엔드 종료 중…", flush=True)
        _hybrid_proc.terminate()
        try:
            _hybrid_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _hybrid_proc.kill()
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
    hybrid_page_chunk: int = Form(
        -1,
        description=(
            "물리 청크 크기. -1=환경변수 HYBRID_PAGE_CHUNK, 0=청크 끔, 1 이상=N페이지씩 잘라 변환"
        ),
    ),
):
    filename = (file.filename or "document.pdf").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=".pdf 파일만 업로드할 수 있습니다.")

    if hybrid_page_chunk < -1:
        raise HTTPException(
            status_code=400,
            detail="hybrid_page_chunk는 -1(환경변수), 0(끔), 또는 1 이상만 가능합니다.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    eff_hybrid_force_ocr = (
        hybrid_force_ocr if hybrid_force_ocr is not None else HYBRID_FORCE_OCR
    )
    eff_page_chunk = (
        HYBRID_PAGE_CHUNK if hybrid_page_chunk == -1 else hybrid_page_chunk
    )

    if use_hybrid:
        lang_changed = ocr_lang.strip() != HYBRID_OCR_LANG
        if lang_changed:
            _update_ocr_lang(ocr_lang.strip())
        if (
            eff_hybrid_force_ocr != _hybrid_child_force_ocr
            or lang_changed
            or not _is_reachable(hybrid_url)
        ):
            await _restart_hybrid(eff_hybrid_force_ocr)

        if not _is_reachable(hybrid_url):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"하이브리드 백엔드({hybrid_url})에 연결할 수 없습니다. "
                    "자동 재시작도 실패했습니다."
                ),
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, filename)
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(out_dir)

        with open(in_path, "wb") as f:
            f.write(data)

        n_pages_probe = _pdf_page_count(in_path)
        will_physical_chunk = (
            use_hybrid
            and eff_page_chunk > 0
            and n_pages_probe is not None
            and n_pages_probe > eff_page_chunk
        )
        logger.info(
            "[convert] use_hybrid=%s eff_page_chunk=%s eff_force_ocr=%s pypdf_pages=%s → 물리청크=%s",
            use_hybrid,
            eff_page_chunk,
            eff_hybrid_force_ocr,
            n_pages_probe,
            will_physical_chunk,
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

        stem = _stem(filename)
        markdown: str
        try:
            did_chunk = False
            if use_hybrid and eff_page_chunk > 0:
                n_pages = n_pages_probe
                if n_pages is not None and n_pages > eff_page_chunk:
                    did_chunk = True
                    logger.info(
                        "[convert] 하이브리드 물리 청크 시작: 총 %s페이지 → %s페이지짜리 잘린 PDF로 순차 변환",
                        n_pages,
                        eff_page_chunk,
                    )
                    parts: list[str] = []
                    for start in range(1, n_pages + 1, eff_page_chunk):
                        end = min(start + eff_page_chunk - 1, n_pages)
                        chunk_pdf_name = f"_chunk_{start}_{end}.pdf"
                        chunk_pdf = os.path.join(tmpdir, chunk_pdf_name)
                        chunk_stem = _stem(chunk_pdf_name)
                        try:
                            _extract_pdf_pages(in_path, chunk_pdf, start, end)
                        except Exception as ex:
                            return JSONResponse(
                                status_code=500,
                                content={
                                    "ok": False,
                                    "error": (
                                        f"PDF 분할 실패 (페이지 {start}-{end}): {ex}. "
                                        "암호 PDF이거나 손상된 파일일 수 있습니다."
                                    ),
                                },
                            )
                        chunk_out = os.path.join(tmpdir, f"out_{start}_{end}")
                        os.makedirs(chunk_out, exist_ok=True)
                        chunk_kw = {
                            **kwargs,
                            "input_path": [chunk_pdf],
                            "output_dir": chunk_out,
                        }
                        opendataloader_pdf.convert(**chunk_kw)
                        md_path = _find(chunk_out, f"{chunk_stem}.md") or _find(
                            chunk_out, "*.md"
                        )
                        if not md_path:
                            return JSONResponse(
                                status_code=500,
                                content={
                                    "ok": False,
                                    "error": (
                                        f"Markdown 출력 없음 (페이지 {start}-{end}/{n_pages}). "
                                        "hybrid_page_chunk를 더 줄여 보세요."
                                    ),
                                },
                            )
                        with open(md_path, encoding="utf-8") as f:
                            chunk_md = f.read()
                        chunk_md = _replace_images_with_ocr(
                            chunk_md, chunk_out, ocr_lang
                        )
                        parts.append(
                            f"\n\n<!-- opendataloader: PDF pages {start}-{end} of {n_pages} -->\n\n"
                            f"{chunk_md}"
                        )
                    markdown = "".join(parts).lstrip()

            if not did_chunk:
                opendataloader_pdf.convert(**kwargs)
                md_path = _find(out_dir, f"{stem}.md") or _find(out_dir, "*.md")
                if not md_path:
                    return JSONResponse(
                        status_code=500,
                        content={
                            "ok": False,
                            "error": "Markdown 출력 파일을 찾을 수 없습니다.",
                        },
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
        out["hybrid_force_ocr"] = eff_hybrid_force_ocr
        out["hybrid_page_chunk"] = eff_page_chunk
    return out


def _get_ocr_reader(lang_csv: str) -> easyocr.Reader:
    global _ocr_reader, _ocr_reader_langs
    if _ocr_reader is None or _ocr_reader_langs != lang_csv:
        langs = [l.strip() for l in lang_csv.split(",") if l.strip()]
        _ocr_reader = easyocr.Reader(langs, gpu=False)
        _ocr_reader_langs = lang_csv
    return _ocr_reader


def _build_image_index(out_dir: str) -> dict[str, str]:
    """출력 폴더 안의 모든 이미지를 {파일명: 절대경로} 로 인덱싱한다.
    Java CLI가 한국어 디렉터리명을 깨뜨려도 파일명(imageFileN.png)은 ASCII라 살아남는다."""
    index: dict[str, str] = {}
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
                index[f] = os.path.join(root, f)
    return index


def _replace_images_with_ocr(
    markdown: str, out_dir: str, lang_csv: str,
) -> str:
    """마크다운 안의 ![alt](path) 를 이미지 OCR 결과 텍스트로 치환한다."""
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
    """cv2.imread는 Windows에서 깨진 유니코드 경로를 못 읽으므로
    Python으로 바이너리를 읽고 cv2.imdecode로 변환한다."""
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _update_ocr_lang(lang: str) -> None:
    global HYBRID_OCR_LANG
    HYBRID_OCR_LANG = lang


def _pdf_page_count(path: str) -> Optional[int]:
    """PDF 총 페이지 수(하이브리드 청크 분할용). 실패 시 None."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        with open(path, "rb") as f:
            reader = PdfReader(f)
            return len(reader.pages)
    except Exception:
        return None


def _extract_pdf_pages(src_pdf: str, dst_pdf: str, page_start_1: int, page_end_1: int) -> None:
    """원본 PDF에서 [page_start_1, page_end_1] 구간(1-based, 닫힌 구간)만 잘라 새 파일로 저장.

    하이브리드(docling) 경로는 CLI ``--pages``만으로는 전체 파일이 서버에 올라가
    메모리가 그대로 쓰이는 경우가 있어, 실제로 페이지를 잘라 보낸다.
    """
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


def _is_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen(url.rstrip("/") + "/", timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except urllib.error.URLError:
        return False


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
