"""Discover resume files and read them once (bytes + SHA-256)."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from src.config import MAX_PDF_BYTES
from src.models import ParseStatus, ResumeDocument

log = logging.getLogger(__name__)


def discover_resumes(input_dir: Path) -> list[ResumeDocument]:
    """List files in a stable (case-insensitive filename) order.

    Non-PDF files are kept in the list with an UNSUPPORTED status so they show up in
    the report instead of silently disappearing. Dotfiles (.gitkeep) are ignored.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    paths = sorted(
        (p for p in input_dir.iterdir() if p.is_file() and not p.name.startswith(".")),
        key=lambda p: p.name.lower(),
    )
    docs: list[ResumeDocument] = []
    for path in paths:
        doc = ResumeDocument(filename=path.name, path=str(path))
        if path.suffix.lower() != ".pdf":
            doc.status = ParseStatus.UNSUPPORTED
            doc.error = f"Unsupported file type '{path.suffix or 'none'}' (PDF required)"
        docs.append(doc)
    return docs


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_and_hash(doc: ResumeDocument) -> Optional[bytes]:
    """Read the file once and set `doc.file_hash`. Returns None (and marks the doc failed) on error."""
    try:
        path = Path(doc.path)
        size = path.stat().st_size
        if size > MAX_PDF_BYTES:
            raise ValueError(f"file is {size} bytes, over the {MAX_PDF_BYTES}-byte limit")
        data = path.read_bytes()
    except (OSError, ValueError) as exc:
        doc.status = ParseStatus.FAILED
        doc.error = f"Could not read file: {exc}"
        log.warning("%s: %s", doc.filename, doc.error)
        return None
    doc.file_hash = sha256_bytes(data)
    return data
