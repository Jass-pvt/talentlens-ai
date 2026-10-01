"""Configuration: environment variables plus the fixed scoring model."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

# Official 100-point model. Do not change without changing the assignment.
WEIGHTS: dict[str, int] = {
    "ai_project_depth": 40,
    "python_backend": 30,
    "cloud_fullstack": 15,
    "github": 10,
    "engineering_depth": 5,
}
assert sum(WEIGHTS.values()) == 100, "scoring weights must total 100"

# Resumes are untrusted input: bound what we are willing to parse.
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 15
MAX_LLM_INPUT_CHARS = 6000


@dataclass(frozen=True)
class Config:
    github_token: Optional[str] = None
    github_timeout: float = 8.0
    github_workers: int = 4
    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: Optional[str] = None
    llm_timeout: float = 30.0

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Config":
        """Build config from the environment (loading .env if python-dotenv exists)."""
        if env is None:
            try:
                from dotenv import load_dotenv

                load_dotenv()  # never overrides variables that are already set
            except ImportError:  # optional convenience only
                pass
            env = os.environ

        api_key = (
            env.get("LLM_API_KEY")
            or env.get("GEMINI_API_KEY")
            or env.get("GOOGLE_API_KEY")
            or None
        )

        return cls(
            github_token=env.get("GITHUB_TOKEN") or None,
            github_timeout=_float(env.get("GITHUB_TIMEOUT"), 8.0),
            llm_provider=(env.get("LLM_PROVIDER") or "").strip().lower(),
            llm_model=(env.get("LLM_MODEL") or "").strip(),
            llm_api_key=api_key,
        )


def _float(raw: Optional[str], default: float) -> float:
    try:
        value = float(raw) if raw else default
    except ValueError:
        return default
    return value if value > 0 else default