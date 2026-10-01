"""Generate synthetic sample resumes (fictional people) into samples/resumes/.

Dev-only helper (needs reportlab). Run from the repo root:
    python scripts/make_sample_resumes.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tests.helpers import make_blank_pdf, make_pdf  # noqa: E402

OUT = ROOT / "samples" / "resumes"

RESUMES: dict[str, tuple[list[str], str | None]] = {
    "01_asha_verma_strong_rag.pdf": ([
        "Asha Verma", "asha.verma@example.com", "",
        "SKILLS", "Python, FastAPI, PostgreSQL, Redis, Docker, AWS, LangGraph, React, pytest", "",
        "PROJECTS",
        "Policy Assistant | Python, LangGraph, FastAPI",
        "• Built a stateful RAG pipeline with document chunking, embeddings, FAISS retrieval,",
        "reranking and tool calling; evaluated answer quality with RAGAS on a golden set.",
        "• Implemented async FastAPI endpoints backed by PostgreSQL and Redis caching with retry",
        "and error handling; deployed with Docker on AWS.",
        "",
        "Support Triage Agent | Python, LangChain",
        "• Developed a multi-agent workflow with conversation state and function calling for ticket routing.",
        "", "EXPERIENCE", "Backend Intern | Acme Labs | Jun 2025 - Aug 2025",
        "• Wrote unit tests and integration tests, added rate limiting and structured logging to REST APIs.",
        "", "EDUCATION", "B.Tech Computer Science, 2026",
    ], "https://github.com/asha-verma-dev"),
    "02_bruno_thin_wrapper.pdf": ([
        "Bruno Castillo", "bruno@example.com", "",
        "SKILLS", "Python, Flask, OpenAI", "",
        "PROJECTS",
        "Text Summarizer | Python, OpenAI API",
        "• Built a text summarizer that sends user text to the OpenAI API and prints the response.",
        "",
        "Joke Bot | Python, ChatGPT",
        "• Created a chatbot using the OpenAI API with a custom prompt.",
    ], None),
    "03_chitra_python_only.pdf": ([
        "Chitra Nair", "chitra@example.com", "",
        "SKILLS", "Python, Django, MySQL, Git", "",
        "PROJECTS", "Library Manager | Python, Django",
        "• Built a CRUD web app with Django and MySQL for managing books.",
    ], None),
    "04_dev_ai_skills_only.pdf": ([
        "Dev Malhotra", "dev@example.com", "",
        "SKILLS", "Python, LLM, RAG, LangChain, Vector databases", "",
        "PROJECTS", "Portfolio Website | HTML, CSS",
        "• Built a static personal website with HTML and CSS.",
    ], None),
    "05_eli_js_react_only.pdf": ([
        "Eli Thompson", "eli@example.com", "",
        "SKILLS", "JavaScript, React, Node.js, Express, MongoDB", "",
        "PROJECTS", "Shop UI | React, Node.js",
        "• Built a React storefront with a Node.js and Express backend.",
    ], None),
    "06_farah_python_js_ai.pdf": ([
        "Farah Khan", "farah@example.com", "",
        "SKILLS", "Python, JavaScript, TypeScript, React, Next.js, Docker", "",
        "PROJECTS", "Doc Chat | Python, Next.js, LangChain",
        "• Built a RAG chatbot using LangChain and FAISS with a Next.js frontend, deployed with Docker.",
    ], "https://github.com/farah-khan-sample"),
    "07_blank.pdf": ([], None),
}


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name, (lines, link) in RESUMES.items():
        if lines:
            make_pdf(OUT / name, lines, link)
        else:
            make_blank_pdf(OUT / name)
    shutil.copy(OUT / "01_asha_verma_strong_rag.pdf", OUT / "08_asha_verma_duplicate.pdf")
    (OUT / "09_corrupted.pdf").write_bytes(b"%PDF-1.4\nthis is not a real pdf\x00\x01garbage")
    (OUT / "10_notes.txt").write_text("not a resume\n")
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
