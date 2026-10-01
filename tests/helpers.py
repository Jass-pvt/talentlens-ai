"""Test utilities: build tiny PDFs without any external fixtures."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def make_pdf(path: Path, lines: list[str], link: Optional[str] = None) -> Path:
    """Write a one-page PDF with one text line per entry (blank entries = blank lines)."""
    c = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in lines:
        c.setFont("Helvetica", 10)
        c.drawString(40, y, line)
        y -= 14
        if y < 40:
            c.showPage()
            y = 800
    if link:
        c.linkURL(link, (40, 810, 200, 830), relative=0)
    c.save()
    return Path(path)


def make_blank_pdf(path: Path) -> Path:
    c = canvas.Canvas(str(path), pagesize=A4)
    c.showPage()
    c.save()
    return Path(path)


STRONG = [
    "Asha Verma", "asha@example.com | github.com/asha-verma-dev", "",
    "SKILLS", "Python, FastAPI, PostgreSQL, Docker, LangGraph, React", "",
    "PROJECTS",
    "Policy Assistant | Python, LangGraph, FastAPI",
    "• Built a stateful RAG pipeline with document chunking, embeddings, FAISS retrieval,",
    "and tool calling; evaluated answers with RAGAS.",
    "• Deployed async FastAPI service with PostgreSQL on AWS using Docker and wrote pytest suites.",
]
