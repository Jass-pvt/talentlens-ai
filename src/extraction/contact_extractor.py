"""Name, email and GitHub username extraction."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_GITHUB = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38})(?![A-Za-z0-9-])",
    re.I,
)
_RESERVED = {"orgs", "sponsors", "topics", "features", "about", "login", "settings",
             "marketplace", "explore", "pricing", "apps", "collections", "events", "notifications"}
_NOT_NAMES = {"resume", "curriculum vitae", "cv", "profile", "summary", "contact", "curriculum"}


def extract_email(text: str, links: list[str]) -> Optional[str]:
    for link in links:
        if link.lower().startswith("mailto:"):
            m = _EMAIL.search(link)
            if m:
                return m.group(0)
    m = _EMAIL.search(text)
    return m.group(0) if m else None


def extract_github(text: str, links: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Return (normalized_url, username). Hyperlink targets win over visible text."""
    for source in (" ".join(links), text):
        for m in _GITHUB.finditer(source):
            username = m.group(1)
            if username.lower() not in _RESERVED:
                return f"https://github.com/{username}", username
    return None, None


def extract_name(text: str, fallback_filename: str) -> str:
    """First plausible name-looking line near the top; else a prettified filename."""
    for line in text.split("\n")[:8]:
        segment = re.split(r"\s*[|•·,/]\s*|\s{2,}", line.strip())[0].strip()
        words = segment.split()
        if not 2 <= len(words) <= 4 or len(segment) > 40:
            continue
        if segment.lower() in _NOT_NAMES or re.search(r"[\d@:]|http", segment):
            continue
        if all(re.fullmatch(r"[A-Za-z][A-Za-z.'\-]*", w) for w in words) and all(
            w[0].isupper() for w in words
        ):
            return segment.title() if segment.isupper() else segment
    stem = re.sub(r"[_\-]+", " ", Path(fallback_filename).stem).strip()
    return stem.title() or "Unknown Candidate"
