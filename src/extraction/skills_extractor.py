"""Canonical skill vocabulary scan (used for reports and rejected-candidate context)."""
from __future__ import annotations

import re

_VOCAB: dict[str, str] = {
    "Python": r"\bpython3?\b", "JavaScript": r"\bjavascript\b", "TypeScript": r"\btypescript\b",
    "Java": r"\bjava\b(?!\s*script)", "C++": r"c\+\+", "C#": r"c#", "Go": r"\bgolang\b",
    "Rust": r"\brust\b", "SQL": r"\bsql\b", "React": r"\breact(?:\.?js)?\b",
    "Next.js": r"\bnext\.?js\b", "Node.js": r"\bnode\.?js\b", "Express": r"\bexpress(?:\.?js)?\b",
    "FastAPI": r"\bfastapi\b", "Django": r"\bdjango\b", "Flask": r"\bflask\b",
    "PostgreSQL": r"\bpostgres(?:ql)?\b", "MySQL": r"\bmysql\b", "MongoDB": r"\bmongo(?:db)?\b",
    "Redis": r"\bredis\b", "Docker": r"\bdocker\b", "Kubernetes": r"\bkubernetes\b|\bk8s\b",
    "AWS": r"\baws\b", "GCP": r"\bgcp\b|google cloud", "Azure": r"\bazure\b",
    "Terraform": r"\bterraform\b", "LangChain": r"\blangchain\b", "LangGraph": r"\blanggraph\b",
    "LlamaIndex": r"llama[- ]?index", "OpenAI": r"\bopenai\b", "RAG": r"\brag\b",
    "LLM": r"\bllms?\b", "FAISS": r"\bfaiss\b", "Chroma": r"\bchroma(?:db)?\b",
    "Pinecone": r"\bpinecone\b", "PyTorch": r"\bpytorch\b", "TensorFlow": r"\btensorflow\b",
    "Pandas": r"\bpandas\b", "NumPy": r"\bnumpy\b", "scikit-learn": r"scikit-?learn|\bsklearn\b",
    "Git": r"\bgit\b", "Linux": r"\blinux\b", "GraphQL": r"\bgraphql\b", "Tailwind": r"\btailwind",
    "Kafka": r"\bkafka\b", "Celery": r"\bcelery\b", "pytest": r"\bpytest\b",
    "HTML/CSS": r"\bhtml\b|\bcss\b", "Vue": r"\bvue(?:\.?js)?\b", "Angular": r"\bangular\b",
}
_COMPILED = {name: re.compile(pat, re.I) for name, pat in _VOCAB.items()}


def extract_skills(text: str) -> list[str]:
    """Canonical skills mentioned anywhere in the resume, in vocabulary order."""
    return [name for name, pat in _COMPILED.items() if pat.search(text)]
