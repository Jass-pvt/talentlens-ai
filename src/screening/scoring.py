"""The official 100-point model, computed deterministically and explained.

AI project depth (40) is scored per project/job entry, not from the skills list:
    base(evidence level) + sum(sub-signal weights)   capped at 40
Other families sum signal weights scaled by how strong the evidence for each signal is.
An LLM may nudge AI depth by at most +/-LLM_MAX_ADJUSTMENT points and never touches eligibility.
"""
from __future__ import annotations

from typing import Optional

from src.ai.schemas import ProjectAnalysis
from src.config import WEIGHTS
from src.models import (
    CandidateResult, EvidenceLevel, GitHubSummary, Penalty, ScoreBreakdown, ScoreResult,
)
from src.screening.evidence import (
    AI_DEPTH_SIGNALS, AI_SPECS, BACKEND_SPECS, CLOUD_SPECS, ENGINEERING_SPECS,
    BlockAnalysis, EvidenceAnalysis, FamilyAnalysis, SignalSpec, best_ai_block,
)

L = EvidenceLevel
LLM_MAX_ADJUSTMENT = 6
THIN_PENALTY = -10
THIN_REASON = "AI project appears to be a thin API wrapper with limited implementation evidence"

_AI_BASE = {L.PROJECT_MENTION: 5, L.IMPLEMENTATION: 11, L.DEEP_IMPLEMENTATION: 14}
_AI_MULT = {L.PROJECT_MENTION: 0.5, L.IMPLEMENTATION: 1.0, L.DEEP_IMPLEMENTATION: 1.0}
_LEVEL_MULT = {L.NONE: 0.0, L.KEYWORD_ONLY: 0.35, L.PROJECT_MENTION: 0.6,
               L.IMPLEMENTATION: 0.9, L.DEEP_IMPLEMENTATION: 1.0}
_AI_WEIGHTS = {s.name: s.weight for s in AI_SPECS if s.name in AI_DEPTH_SIGNALS}
_LABELS = {"retrieval": "retrieval/RAG", "embeddings": "embeddings", "vector_search": "vector search",
           "tool_calling": "tool calling", "state": "state management", "orchestration": "orchestration",
           "evaluation": "evaluation", "business_logic": "real data/business logic"}


def _round(x: float) -> int:
    return int(x + 0.5)


def block_level(block: BlockAnalysis) -> EvidenceLevel:
    """Evidence level of a block's AI usage (deep if it shows >=3 related sub-signals)."""
    core = block.levels.get("core", L.NONE)
    return L.DEEP_IMPLEMENTATION if block.deep and core >= L.IMPLEMENTATION else core


def ai_block_points(block: BlockAnalysis) -> float:
    level = block_level(block)
    if level < L.PROJECT_MENTION:
        return 0.0
    signal_sum = sum(_AI_WEIGHTS[s] for s in block.depth_signals())
    return min(WEIGHTS["ai_project_depth"], _AI_BASE[level] + _AI_MULT[level] * signal_sum)


def score_ai_depth(analysis: EvidenceAnalysis) -> tuple[int, Optional[BlockAnalysis]]:
    """Best project counts fully, a second strong one adds a quarter (breadth bonus)."""
    ranked = sorted(analysis.ai.ai_blocks, key=ai_block_points, reverse=True)
    if not ranked:
        return 0, None
    points = ai_block_points(ranked[0])
    if len(ranked) > 1:
        points += 0.25 * ai_block_points(ranked[1])
    return min(WEIGHTS["ai_project_depth"], _round(points)), best_ai_block(analysis)


def score_family(family: FamilyAnalysis, specs: tuple[SignalSpec, ...], cap: int) -> int:
    total = sum(s.weight * _LEVEL_MULT[family.level(s.name)] for s in specs)
    return min(cap, _round(total))


def detect_thin_projects(analysis: EvidenceAnalysis) -> Optional[Penalty]:
    """Penalize only when EVERY AI project is a bare hosted-API wrapper.

    Using an API is fine. The penalty keys on missing depth: no retrieval, data
    processing, state, tools, workflow, evaluation or business logic anywhere.
    """
    ai_blocks = [b for b in analysis.ai.ai_blocks if block_level(b) >= L.PROJECT_MENTION]
    if ai_blocks and all(b.is_thin for b in ai_blocks):
        return Penalty(THIN_PENALTY, THIN_REASON)
    return None


def score_candidate(analysis: EvidenceAnalysis, llm: Optional[ProjectAnalysis] = None) -> ScoreResult:
    ai_det, best = score_ai_depth(analysis)
    ai_points, adjustment = ai_det, 0
    if llm is not None:
        adjustment = max(-LLM_MAX_ADJUSTMENT, min(LLM_MAX_ADJUSTMENT, llm.project_depth - ai_det))
        ai_points = max(0, min(WEIGHTS["ai_project_depth"], ai_det + adjustment))

    breakdown = ScoreBreakdown(
        ai_project_depth=ai_points,
        python_backend=score_family(analysis.backend, BACKEND_SPECS, WEIGHTS["python_backend"]),
        cloud_fullstack=score_family(analysis.cloud, CLOUD_SPECS, WEIGHTS["cloud_fullstack"]),
        engineering_depth=score_family(analysis.engineering, ENGINEERING_SPECS, WEIGHTS["engineering_depth"]),
    )
    penalties: list[Penalty] = []
    thin = detect_thin_projects(analysis)
    if thin:
        penalties.append(thin)
    result = ScoreResult(breakdown=breakdown, penalties=penalties, llm_adjustment=adjustment)
    result.strengths, result.concerns = _explain(analysis, best, thin is not None)
    if llm is not None:
        result.concerns.extend(c for c in llm.concerns if c not in result.concerns)
    finalize(result)
    result.why_this_score = _why(analysis, best, result)
    return result


def finalize(result: ScoreResult) -> None:
    """total = sum(categories) + penalties, clamped to [0, 100]."""
    raw = result.breakdown.subtotal() + sum(p.penalty for p in result.penalties)
    result.total_score = max(0, min(100, raw))


def apply_github(result: ScoreResult, summary: GitHubSummary) -> None:
    """Add the (0-10) GitHub score after eligibility; never affects eligibility."""
    result.breakdown.github = min(WEIGHTS["github"], summary.score)
    if summary.status == "ok" and summary.relevant_maintained:
        result.strengths.append("Maintained Python/AI repositories on GitHub")
    elif summary.status not in ("ok", "disabled"):
        reason = {"missing": "No GitHub profile linked in resume"}.get(
            summary.status, f"GitHub data unavailable ({summary.status})")
        result.concerns.append(reason)
    finalize(result)


def rank_candidates(results: list[CandidateResult]) -> list[CandidateResult]:
    """Sort eligible candidates by total desc; ties: AI depth, Python/backend, engineering, name."""
    eligible = [r for r in results if r.eligible and r.score is not None]
    eligible.sort(key=lambda r: (
        -r.score.total_score,
        -r.score.breakdown.ai_project_depth,
        -r.score.breakdown.python_backend,
        -r.score.breakdown.engineering_depth,
        r.profile.candidate_name.casefold(),
        r.profile.source_filename.casefold(),
    ))
    for position, r in enumerate(eligible, start=1):
        r.rank = position
    return eligible


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

def _names(family: FamilyAnalysis, signals: list[str], minimum: EvidenceLevel) -> list[str]:
    return [s for s in signals if family.level(s) >= minimum]


def _explain(analysis: EvidenceAnalysis, best: Optional[BlockAnalysis], thin: bool) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    concerns: list[str] = []
    if best is not None:
        labels = [_LABELS[s] for s in AI_DEPTH_SIGNALS if s in best.signals]
        if labels and block_level(best) >= L.IMPLEMENTATION:
            strengths.append(f"AI project '{best.title}': " + ", ".join(labels))
        missing = set(AI_DEPTH_SIGNALS) - best.signals
        if "evaluation" in missing:
            concerns.append("No evaluation methodology described for AI projects")
        if {"tool_calling", "state"} <= missing:
            concerns.append("Limited agentic evidence (no tool calling or state management)")
    if thin:
        concerns.append(THIN_REASON)
    backend = _names(analysis.backend, ["framework", "async", "database", "redis", "rest_api", "testing"], L.IMPLEMENTATION)
    if backend:
        strengths.append("Backend implementation evidence: " + ", ".join(backend))
    else:
        concerns.append("Backend skills mostly listed rather than demonstrated in projects")
    cloud = _names(analysis.cloud, ["docker", "cloud", "kubernetes", "deployment", "frontend"], L.IMPLEMENTATION)
    if cloud:
        strengths.append("Deployment/full-stack evidence: " + ", ".join(cloud))
    else:
        concerns.append("No implemented cloud, deployment or full-stack evidence")
    if analysis.engineering.level("testing") < L.PROJECT_MENTION:
        concerns.append("No testing evidence in projects or experience")
    if analysis.unstructured:
        concerns.append("Resume layout not recognized; evidence taken from unstructured text")
    return strengths, concerns


def _why(analysis: EvidenceAnalysis, best: Optional[BlockAnalysis], result: ScoreResult) -> str:
    b = result.breakdown
    if best is None:
        return "No AI project evidence was found."
    labels = [_LABELS[s] for s in AI_DEPTH_SIGNALS if s in best.signals]
    parts = [f"Strongest evidence came from '{best.title}' ({block_level(best).name.replace('_', ' ').lower()})"
             + (f" showing {', '.join(labels)}" if labels else "")]
    backend = _names(analysis.backend, ["framework", "async", "database", "redis"], L.IMPLEMENTATION)
    if backend:
        parts.append("with backend work in " + ", ".join(backend))
    text = " ".join(parts) + f". AI depth {b.ai_project_depth}/40, Python/backend {b.python_backend}/30."
    for p in result.penalties:
        text += f" Penalty {p.penalty}: {p.reason}."
    if result.llm_adjustment:
        text += f" LLM adjusted AI depth by {result.llm_adjustment:+d}."
    return text
