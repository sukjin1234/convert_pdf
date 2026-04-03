"""FastAPI 서버 — 실행: python main.py"""

import asyncio
import glob
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

# ── 이미지 OCR ───────────────────────────────────────────────
_ocr_reader: Optional[easyocr.Reader] = None
_ocr_reader_langs: Optional[str] = None
_IMG_RE = re.compile(r"!\[([^\]]*)\]\((.+?\.(?:png|jpe?g|gif|bmp|webp))\)", re.IGNORECASE)

# ── 하이브리드 백엔드 자동 관리 ──────────────────────────────
HYBRID_PORT = int(os.environ.get("HYBRID_PORT", "5002"))
HYBRID_URL = f"http://127.0.0.1:{HYBRID_PORT}"
HYBRID_OCR_LANG = os.environ.get("HYBRID_OCR_LANG", "ko,en")
_hybrid_proc: Optional[subprocess.Popen] = None


def _build_hybrid_cmd(cli: str) -> list[str]:
    return [cli, "--port", str(HYBRID_PORT), "--ocr-lang", HYBRID_OCR_LANG, "--force-ocr"]


async def _restart_hybrid() -> None:
    """하이브리드 백엔드가 죽었을 때 자동으로 다시 띄운다."""
    global _hybrid_proc

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

    cmd = _build_hybrid_cmd(cli)
    print(f"[hybrid] 백엔드 재시작 중… {' '.join(cmd)}", flush=True)
    _hybrid_proc = subprocess.Popen(cmd)
    for _ in range(60):
        await asyncio.sleep(0.5)
        if _is_reachable(HYBRID_URL):
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
    global _hybrid_proc

    if _is_reachable(HYBRID_URL):
        print(f"[hybrid] 이미 실행 중 ({HYBRID_URL})", flush=True)
    else:
        cli = _find_hybrid_cli()
        if cli is None:
            print("[hybrid] opendataloader-pdf-hybrid 를 찾을 수 없습니다.", flush=True)
        else:
            cmd = _build_hybrid_cmd(cli)
            print(f"[hybrid] 백엔드 시작: {' '.join(cmd)}", flush=True)
            _hybrid_proc = subprocess.Popen(cmd)
            for _ in range(60):
                await asyncio.sleep(0.5)
                if _is_reachable(HYBRID_URL):
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
):
    filename = (file.filename or "document.pdf").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=".pdf 파일만 업로드할 수 있습니다.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    if use_hybrid:
        lang_changed = ocr_lang.strip() != HYBRID_OCR_LANG
        if lang_changed:
            _update_ocr_lang(ocr_lang.strip())
            await _restart_hybrid()
        elif not _is_reachable(hybrid_url):
            await _restart_hybrid()

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

        try:
            opendataloader_pdf.convert(**kwargs)
        except Exception as e:
            return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

        md_path = _find(out_dir, f"{_stem(filename)}.md") or _find(out_dir, "*.md")
        if not md_path:
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": "Markdown 출력 파일을 찾을 수 없습니다."},
            )

        with open(md_path, encoding="utf-8") as f:
            markdown = f.read()

        markdown = _replace_images_with_ocr(markdown, out_dir, ocr_lang)

    return {
        "ok": True,
        "filename": filename,
        "markdown": markdown,
        "hybrid": use_hybrid,
        "ocr_lang": ocr_lang,
    }


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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
