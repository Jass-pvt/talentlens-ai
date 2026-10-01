"""Validated shape of the LLM's structured output. Anything off-schema is rejected."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


class SchemaError(ValueError):
    """The model returned something that does not match the schema."""


@dataclass
class LLMEvidence:
    claim: str
    evidence: str
    confidence: float


@dataclass
class ProjectAnalysis:
    project_depth: int  # 0-40, advisory only
    technologies: list[str] = field(default_factory=list)
    evidence: list[LLMEvidence] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: object) -> "ProjectAnalysis":
        if not isinstance(data, dict):
            raise SchemaError("expected a JSON object")
        depth = data.get("project_depth")
        if isinstance(depth, bool) or not isinstance(depth, (int, float)):
            raise SchemaError("project_depth must be a number")
        evidence = []
        for item in data.get("evidence") or []:
            if not isinstance(item, dict) or not isinstance(item.get("evidence"), str):
                continue  # drop malformed items rather than trusting them
            conf = item.get("confidence", 0.5)
            conf = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else 0.5
            evidence.append(LLMEvidence(str(item.get("claim", ""))[:120], item["evidence"][:400],
                                        max(0.0, min(1.0, conf))))
        return cls(
            project_depth=int(max(0, min(40, round(depth)))),
            technologies=[str(t)[:40] for t in (data.get("technologies") or [])][:20],
            evidence=evidence[:10],
            concerns=[str(c)[:200] for c in (data.get("concerns") or [])][:6],
        )


def grounded(snippet: str, source_text: str) -> bool:
    """True if `snippet` really occurs in the resume (whitespace/case-insensitive).

    Guards against the model inventing evidence that the resume does not contain.
    """
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    needle = norm(snippet)
    return len(needle) >= 12 and needle in norm(source_text)
