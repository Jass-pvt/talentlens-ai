"""Hard eligibility rules. Pure Python: no LLM, no network, no randomness.

A candidate is eligible only if BOTH hold:
  A. Python evidence in skills, summary, projects or experience.
  B. AI/agentic evidence inside a project or job entry (a skills-list-only mention
     is not enough: the assignment asks for meaningful evidence).
Other languages (JavaScript, Java, React...) never disqualify anyone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.models import EvidenceLevel
from src.screening.evidence import EvidenceAnalysis

_CONF = {EvidenceLevel.NONE: 0.0, EvidenceLevel.KEYWORD_ONLY: 0.45, EvidenceLevel.PROJECT_MENTION: 0.65,
         EvidenceLevel.IMPLEMENTATION: 0.85, EvidenceLevel.DEEP_IMPLEMENTATION: 0.95}


@dataclass
class EligibilityDecision:
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0


def check_eligibility(analysis: EvidenceAnalysis) -> EligibilityDecision:
    python_level = analysis.backend.level("python")
    ai_blocks = analysis.ai.ai_blocks
    ai_project_level = max((b.levels["core"] for b in ai_blocks), default=EvidenceLevel.NONE)
    ai_any_level = analysis.ai.level("core")

    reasons: list[str] = []
    if python_level == EvidenceLevel.NONE:
        reasons.append("No evidence of Python stack")
    if ai_project_level == EvidenceLevel.NONE:
        if ai_any_level == EvidenceLevel.NONE:
            reasons.append("No AI/agentic project evidence")
        else:
            reasons.append("AI/agentic terms appear only in the skills list or summary, "
                           "with no project or experience evidence")

    if reasons:
        # Confidence in a rejection is lower when the evidence was borderline.
        borderline = max(python_level, ai_any_level) > EvidenceLevel.NONE
        return EligibilityDecision(False, reasons, 0.75 if borderline else 0.95)
    confidence = round((_CONF[python_level] + _CONF[ai_project_level]) / 2, 2)
    return EligibilityDecision(True, [], confidence)
