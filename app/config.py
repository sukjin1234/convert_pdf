from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_csv(name: str) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _env_int_csv(name: str) -> tuple[int, ...]:
    return tuple(int(value) for value in _env_csv(name))


def _default_ocr_server_devices() -> tuple[str, ...]:
    explicit = _env_csv("ODL_OCR_SERVER_DEVICES")
    if explicit:
        return explicit
    if not _env_bool("ODL_OCR_AUTO_DETECT_CUDA", True):
        return ()

    max_devices = _env_int("ODL_OCR_AUTO_DETECT_MAX_GPUS", 3)
    if max_devices <= 0:
        return ()
    return _detect_cuda_device_ids(max_devices)


def _detect_cuda_device_ids(max_devices: int) -> tuple[str, ...]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if result.returncode != 0:
        return ()

    devices = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if value:
            devices.append(value)
        if len(devices) >= max_devices:
            break
    return tuple(devices)


def _default_opendataloader_cli() -> str:
    explicit = os.getenv("ODL_CLI")
    if explicit:
        return explicit

    executable_dir = Path(sys.executable).resolve().parent
    candidates = []
    if os.name == "nt":
        candidates.append(executable_dir / "Scripts" / "opendataloader-pdf.exe")
    candidates.append(executable_dir / "opendataloader-pdf")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "opendataloader-pdf"


def _default_ocr_server_cli() -> str:
    explicit = os.getenv("ODL_OCR_SERVER_CLI")
    if explicit:
        return explicit

    executable_dir = Path(sys.executable).resolve().parent
    candidates = []
    if os.name == "nt":
        candidates.append(executable_dir / "Scripts" / "opendataloader-pdf-hybrid.exe")
    candidates.append(executable_dir / "opendataloader-pdf-hybrid")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "opendataloader-pdf-hybrid"


def _default_ocr_hybrid_url() -> str:
    explicit = os.getenv("ODL_OCR_HYBRID_URL")
    if explicit:
        return explicit
    host = os.getenv("ODL_OCR_SERVER_HOST", "127.0.0.1")
    port = _env_int("ODL_OCR_SERVER_PORT", 5003)
    return f"http://{host}:{port}"


def _default_opendataloader_jar() -> str | None:
    explicit = os.getenv("ODL_JAR")
    if explicit:
        return explicit

    executable_dir = Path(sys.executable).resolve().parent
    candidates = [
        executable_dir / "Lib" / "site-packages" / "opendataloader_pdf" / "jar" / "opendataloader-pdf-cli.jar",
        executable_dir.parent / "Lib" / "site-packages" / "opendataloader_pdf" / "jar" / "opendataloader-pdf-cli.jar",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


@dataclass(frozen=True)
class Settings:
    opendataloader_cli: str = _default_opendataloader_cli()
    opendataloader_jar: str | None = _default_opendataloader_jar()
    hybrid_backend: str = "docling-fast"
    hybrid_url: str = os.getenv("ODL_HYBRID_URL", "http://localhost:5002")
    hybrid_mode: str = os.getenv("ODL_HYBRID_MODE", "auto").lower()
    hybrid_timeout_ms: int = _env_int("ODL_HYBRID_TIMEOUT_MS", 300_000)
    conversion_timeout_seconds: int = _env_int("ODL_CONVERSION_TIMEOUT_SECONDS", 360)
    tmp_root: Path = Path(os.getenv("ODL_TMP_ROOT", str(Path(tempfile.gettempdir()) / "dify-opendataloader")))
    max_pdf_bytes: int = _env_int("ODL_MAX_PDF_BYTES", 80 * 1024 * 1024)
    require_pdf_signature: bool = _env_bool("ODL_REQUIRE_PDF_SIGNATURE", True)
    table_method: str = os.getenv("ODL_TABLE_METHOD", "cluster")
    reading_order: str = "xycut"
    use_struct_tree: bool = _env_bool("ODL_USE_STRUCT_TREE", False)
    qpdf_repair_pdf_on_failure: bool = _env_bool("ODL_QPDF_REPAIR_PDF_ON_FAILURE", True)
    repair_pdf_on_failure: bool = _env_bool("ODL_REPAIR_PDF_ON_FAILURE", True)
    rasterize_pdf_on_failure: bool = _env_bool("ODL_RASTERIZE_PDF_ON_FAILURE", True)
    rasterize_dpi: int = _env_int("ODL_RASTERIZE_DPI", 180)
    native_text_layer_first: bool = _env_bool("ODL_NATIVE_TEXT_LAYER_FIRST", True)
    ocr_server_cli: str = _default_ocr_server_cli()
    ocr_server_host: str = os.getenv("ODL_OCR_SERVER_HOST", "127.0.0.1")
    ocr_server_port: int = _env_int("ODL_OCR_SERVER_PORT", 5003)
    ocr_server_ports: tuple[int, ...] = _env_int_csv("ODL_OCR_SERVER_PORTS")
    ocr_server_devices: tuple[str, ...] = _default_ocr_server_devices()
    ocr_server_workers: int = _env_int("ODL_OCR_SERVER_WORKERS", 0)
    ocr_hybrid_urls: tuple[str, ...] = _env_csv("ODL_OCR_HYBRID_URLS")
    ocr_hybrid_url: str = _default_ocr_hybrid_url()
    ocr_lang: str = os.getenv("ODL_OCR_LANG", "ko,en")
    ocr_enrich_picture_description: bool = _env_bool("ODL_OCR_ENRICH_PICTURE_DESCRIPTION", True)
    ocr_filter_decorative_images: bool = _env_bool("ODL_OCR_FILTER_DECORATIVE_IMAGES", True)
    ocr_server_start_timeout_seconds: int = _env_int("ODL_OCR_SERVER_START_TIMEOUT_SECONDS", 60)
    ocr_server_shutdown_timeout_seconds: int = _env_int("ODL_OCR_SERVER_SHUTDOWN_TIMEOUT_SECONDS", 10)

    def validate(self) -> None:
        if self.hybrid_backend.lower() == "off":
            raise ValueError("Hybrid conversion must stay enabled.")
        if self.hybrid_mode not in {"auto", "full"}:
            raise ValueError("ODL_HYBRID_MODE must be either 'auto' or 'full'.")
        if self.hybrid_timeout_ms <= 0:
            raise ValueError("ODL_HYBRID_TIMEOUT_MS must be positive.")
        if self.conversion_timeout_seconds <= 0:
            raise ValueError("ODL_CONVERSION_TIMEOUT_SECONDS must be positive.")
        if self.max_pdf_bytes <= 0:
            raise ValueError("ODL_MAX_PDF_BYTES must be positive.")
        if self.rasterize_dpi <= 0:
            raise ValueError("ODL_RASTERIZE_DPI must be positive.")
        if self.ocr_server_port <= 0:
            raise ValueError("ODL_OCR_SERVER_PORT must be positive.")
        if any(port <= 0 for port in self.ocr_server_ports):
            raise ValueError("ODL_OCR_SERVER_PORTS values must be positive.")
        if self.ocr_server_workers < 0:
            raise ValueError("ODL_OCR_SERVER_WORKERS must not be negative.")
        if self.ocr_server_start_timeout_seconds <= 0:
            raise ValueError("ODL_OCR_SERVER_START_TIMEOUT_SECONDS must be positive.")
        if self.ocr_server_shutdown_timeout_seconds <= 0:
            raise ValueError("ODL_OCR_SERVER_SHUTDOWN_TIMEOUT_SECONDS must be positive.")


def get_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings
