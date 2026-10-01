"""Split resume text into sections and sections into project/job blocks.

Heuristics, not magic: headings must consist only of known heading words, so a line
like "Project Management" is never mistaken for a heading. Unknown headings are
absorbed into the current section rather than dropping content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Heading vocabulary -> section. A heading may only contain these words.
_NEUTRAL = {"technical", "personal", "academic", "key", "selected", "notable", "side", "open",
            "source", "relevant", "professional", "industry", "and", "&", "of", "my", "core"}
_WORD_SECTION = {
    "projects": "projects", "project": "projects", "contributions": "projects",
    "hackathons": "projects", "portfolio": "projects",
    "experience": "experience", "internship": "experience", "internships": "experience",
    "employment": "experience", "work": "experience", "history": "experience",
    "skills": "skills", "skill": "skills", "technologies": "skills", "tools": "skills",
    "tech": "skills", "stack": "skills", "competencies": "skills",
    "education": "education", "qualifications": "education", "coursework": "education",
    "background": "education",
    "summary": "summary", "profile": "summary", "objective": "summary", "about": "summary",
    "me": "summary", "career": "summary",
    "achievements": "other", "certifications": "other", "awards": "other", "links": "other",
    "contact": "other", "interests": "other", "publications": "other",
    "extracurricular": "other", "activities": "other", "training": "other",
    "courses": "other", "responsibility": "other", "positions": "other",
}
_PRIORITY = ["projects", "experience", "skills", "education", "summary", "other"]

BULLET = re.compile(r"^\s*(?:[•●▪◦‣▫■·\-\*–—]|\d{1,2}[.)])\s+")
_DATE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I
)
_TERMINAL = (".", "!", "?", ";", ":")


@dataclass
class Block:
    """One project or job entry: an optional title line plus bullet/sentence items."""

    section: str
    items: list[str] = field(default_factory=list)
    has_title: bool = False

    @property
    def title(self) -> str:
        if self.has_title and self.items:
            head = re.split(r"\s*[|–—]\s*|\s+-\s+|\s*\(", self.items[0])[0].strip()
            return head[:80] or "Untitled"
        return "Untitled"

    @property
    def text(self) -> str:
        return "\n".join(self.items)


@dataclass
class Sections:
    parts: dict[str, str] = field(default_factory=dict)

    def get(self, name: str) -> str:
        return self.parts.get(name, "")

    @property
    def structured(self) -> bool:
        """True if at least one of the sections that carry evidence was found."""
        return any(self.parts.get(n) for n in ("skills", "projects", "experience"))


def classify_heading(line: str) -> Optional[str]:
    """Return the section name if `line` is a heading, else None."""
    cleaned = re.sub(r"[^A-Za-z& ]", " ", line).strip()
    words = cleaned.lower().split()
    if not words or len(words) > 5 or line.rstrip().endswith("."):
        return None
    if not all(w in _WORD_SECTION or w in _NEUTRAL for w in words):
        return None
    found = {_WORD_SECTION[w] for w in words if w in _WORD_SECTION}
    for name in _PRIORITY:
        if name in found:
            return name
    return None


def split_sections(text: str) -> Sections:
    """Group lines under the most recent recognized heading ("header" before the first)."""
    buckets: dict[str, list[str]] = {"header": []}
    current = "header"
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            buckets[current].append("")
            continue
        section = classify_heading(line)
        inline = ""
        if section is None and ":" in line:  # "Skills: Python, FastAPI"
            left, _, right = line.partition(":")
            section, inline = classify_heading(left), right.strip()
        if section is not None:
            current = section
            buckets.setdefault(current, [])
            if inline:
                buckets[current].append(inline)
            continue
        buckets[current].append(line)
    return Sections({k: "\n".join(v).strip() for k, v in buckets.items() if "".join(v).strip()})


def _looks_like_new_entry(line: str) -> bool:
    """A non-bullet line that starts a new project/job (pipe-separated or dated)."""
    return "|" in line or (bool(_DATE.search(line)) and len(line.split()) <= 14)


def split_blocks(text: str, section: str) -> list[Block]:
    """Split a section into entries. Wrapped lines are re-joined into one item."""
    blocks: list[Block] = []
    cur: Optional[Block] = None
    last_was_bullet = False
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            cur = None  # blank line ends an entry
            continue
        is_bullet = bool(BULLET.match(line))
        content = BULLET.sub("", line, count=1)
        if cur is None:
            cur = Block(section=section)
            blocks.append(cur)
            cur.items.append(content)
            cur.has_title = not is_bullet
            last_was_bullet = is_bullet
            continue
        if is_bullet:
            cur.items.append(content)
            last_was_bullet = True
            continue
        prev = cur.items[-1]
        wrapped = not prev.endswith(_TERMINAL) and not _looks_like_new_entry(line)
        if wrapped or line[0].islower():
            cur.items[-1] = f"{prev} {line}"  # continuation of the previous item
        elif last_was_bullet and _looks_like_new_entry(line):
            cur = Block(section=section, items=[line], has_title=True)
            blocks.append(cur)
            last_was_bullet = False
        elif _looks_like_new_entry(line) and len(cur.items) > 0 and not cur.has_title:
            cur = Block(section=section, items=[line], has_title=True)
            blocks.append(cur)
            last_was_bullet = False
        else:
            cur.items.append(line)
            last_was_bullet = False
    return blocks


def scannable_blocks(sections: Sections, full_text: str) -> list[Block]:
    """Blocks that count as project/job evidence.

    Structured resumes use the Projects and Experience sections. If no evidence-bearing
    section was recognized at all, fall back to the whole document (section="unstructured")
    so unusual layouts are not auto-rejected; evidence from it gets reduced confidence.
    """
    if sections.structured:
        blocks: list[Block] = []
        for name in ("projects", "experience"):
            blocks.extend(split_blocks(sections.get(name), name))
        return blocks
    return split_blocks(full_text, "unstructured")
