"""PDF -> normalized text, plus hyperlink targets (resumes often hide GitHub links)."""
from __future__ import annotations

import io
import logging
import re
import unicodedata
from dataclasses import dataclass, field

from pypdf import PdfReader

from src.config import MAX_PDF_PAGES

log = logging.getLogger(__name__)


class PdfParseError(Exception):
    """The file is not a readable PDF."""


@dataclass
class ParsedPdf:
    text: str
    links: list[str] = field(default_factory=list)
    page_count: int = 0


# Bullet glyphs that extractors emit as odd code points (DEL, Symbol-font private use).
_BULLETS = {0x7F: "•", 0xF0B7: "•", 0xF0A7: "•", 0xF076: "•", 0xF0D8: "•", 0x25AA: "•", 0x2023: "•"}
_CONTROL = {c: None for c in range(32) if c not in (9, 10, 13)}


def normalize_text(text: str) -> str:
    """NFKC-normalize (fixes ligatures), unify bullets, collapse spaces per line, cap blank runs."""
    text = unicodedata.normalize("NFKC", text).translate({**_CONTROL, **_BULLETS})
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\u00a0]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def parse_pdf(data: bytes) -> ParsedPdf:
    """Extract text and links. Raises PdfParseError for anything unreadable."""
    if b"%PDF" not in data[:1024]:
        raise PdfParseError("Not a PDF (missing %PDF header)")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PdfParseError("PDF is password-protected")
        pages = list(reader.pages)
        chunks: list[str] = []
        links: list[str] = []
        for index, page in enumerate(pages[:MAX_PDF_PAGES]):
            try:
                chunks.append(page.extract_text() or "")
            except Exception as exc:  # one bad page must not lose the others
                log.debug("page %d text extraction failed: %s", index, exc)
            links.extend(_page_links(page))
        return ParsedPdf(normalize_text("\n".join(chunks)), links, len(pages))
    except PdfParseError:
        raise
    except Exception as exc:
        raise PdfParseError(f"Unreadable PDF: {exc}") from exc


def _page_links(page) -> list[str]:
    found: list[str] = []
    try:
        for annot in page.get("/Annots") or []:
            action = annot.get_object().get("/A")
            if action is None:
                continue
            uri = action.get_object().get("/URI")
            if uri:
                found.append(str(uri))
    except Exception as exc:  # links are a bonus, never a failure
        log.debug("link extraction failed: %s", exc)
    return found
