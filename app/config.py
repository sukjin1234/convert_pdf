from __future__ import annotations

import os
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

    def validate(self) -> None:
        cli_name = Path(self.opendataloader_cli).name.lower()
        if cli_name.startswith("opendataloader-pdf-hybrid"):
            raise ValueError(
                "ODL_CLI must point to opendataloader-pdf, not opendataloader-pdf-hybrid. "
                "Start opendataloader-pdf-hybrid once as the shared backend."
            )
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


def get_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings
