from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


APP_RUNTIME_DIR_NAME = "dify-rag-api"


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


def _runtime_root() -> Path:
    explicit = os.getenv("DIFY_RUNTIME_DIR")
    if explicit and explicit.strip():
        return Path(explicit)
    return Path.cwd() / ".runtime"


def _default_opendataloader_tmp_root() -> Path:
    explicit = os.getenv("ODL_TMP_ROOT")
    if explicit and explicit.strip():
        return Path(explicit)
    return _runtime_root() / "opendataloader"


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
    tmp_root: Path = _default_opendataloader_tmp_root()
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
    embedded_image_ocr_enabled: bool = _env_bool("ODL_EMBEDDED_IMAGE_OCR_ENABLED", True)
    embedded_image_ocr_max_images: int = _env_int("ODL_EMBEDDED_IMAGE_OCR_MAX_IMAGES", 24)
    embedded_image_ocr_batch_size: int = _env_int("ODL_EMBEDDED_IMAGE_OCR_BATCH_SIZE", 2)
    embedded_image_ocr_min_pixels: int = _env_int("ODL_EMBEDDED_IMAGE_OCR_MIN_PIXELS", 12_000)
    embedded_image_ocr_max_image_bytes: int = _env_int("ODL_EMBEDDED_IMAGE_OCR_MAX_IMAGE_BYTES", 8 * 1024 * 1024)
    embedded_image_ocr_pdf_max_side: int = _env_int("ODL_EMBEDDED_IMAGE_OCR_PDF_MAX_SIDE", 1600)
    resource_retry_enabled: bool = _env_bool("ODL_RESOURCE_RETRY_ENABLED", True)
    resource_retry_interval_seconds: int = _env_int("ODL_RESOURCE_RETRY_INTERVAL_SECONDS", 30)
    resource_retry_max_interval_seconds: int = _env_int("ODL_RESOURCE_RETRY_MAX_INTERVAL_SECONDS", 300)
    resource_retry_max_attempts: int = _env_int("ODL_RESOURCE_RETRY_MAX_ATTEMPTS", 0)

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
        if self.embedded_image_ocr_max_images <= 0:
            raise ValueError("ODL_EMBEDDED_IMAGE_OCR_MAX_IMAGES must be positive.")
        if self.embedded_image_ocr_batch_size <= 0:
            raise ValueError("ODL_EMBEDDED_IMAGE_OCR_BATCH_SIZE must be positive.")
        if self.embedded_image_ocr_min_pixels < 0:
            raise ValueError("ODL_EMBEDDED_IMAGE_OCR_MIN_PIXELS must be zero or positive.")
        if self.embedded_image_ocr_max_image_bytes <= 0:
            raise ValueError("ODL_EMBEDDED_IMAGE_OCR_MAX_IMAGE_BYTES must be positive.")
        if self.embedded_image_ocr_pdf_max_side <= 0:
            raise ValueError("ODL_EMBEDDED_IMAGE_OCR_PDF_MAX_SIDE must be positive.")
        if self.resource_retry_interval_seconds <= 0:
            raise ValueError("ODL_RESOURCE_RETRY_INTERVAL_SECONDS must be positive.")
        if self.resource_retry_max_interval_seconds <= 0:
            raise ValueError("ODL_RESOURCE_RETRY_MAX_INTERVAL_SECONDS must be positive.")
        if self.resource_retry_max_interval_seconds < self.resource_retry_interval_seconds:
            raise ValueError("ODL_RESOURCE_RETRY_MAX_INTERVAL_SECONDS must be greater than or equal to ODL_RESOURCE_RETRY_INTERVAL_SECONDS.")
        if self.resource_retry_max_attempts < 0:
            raise ValueError("ODL_RESOURCE_RETRY_MAX_ATTEMPTS must be zero or positive.")


def get_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings
