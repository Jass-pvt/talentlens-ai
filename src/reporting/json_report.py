"""Machine-readable output."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import WEIGHTS
from src.models import CandidateResult, to_jsonable
from src.pipeline import BatchResult


def _evidence(result: CandidateResult) -> list[dict]:
    return [{"category": e.category, "skill": e.skill, "section": e.section, "source": e.source,
             "level": e.level.name, "confidence": e.confidence, "evidence_text": e.evidence_text}
            for e in result.evidence]


def _eligible(result: CandidateResult) -> dict:
    s, gh = result.score, result.github
    return {
        "rank": result.rank,
        "candidate_name": result.profile.candidate_name,
        "email": result.profile.email,
        "github_url": result.profile.github_url,
        "source_filename": result.profile.source_filename,
        "eligible": True,
        "total_score": s.total_score,
        "score_breakdown": to_jsonable(s.breakdown),
        "penalties": to_jsonable(s.penalties),
        "why_this_score": s.why_this_score,
        "evidence": _evidence(result),
        "strengths": s.strengths,
        "concerns": s.concerns,
        "confidence": result.confidence,
        "matched_skills": result.profile.skills,
        "github": to_jsonable(gh) if gh else None,
        "llm_used": result.llm_used,
    }


def _rejected(result: CandidateResult) -> dict:
    return {
        "candidate_name": result.profile.candidate_name,
        "source_filename": result.profile.source_filename,
        "eligible": False,
        "rejection_reasons": result.rejection_reasons,
        "matched_skills": result.profile.skills,
        "confidence": result.confidence,
    }


def build_report(batch: BatchResult) -> dict:
    docs = batch.documents
    return {
        "batch_summary": batch.summary,
        "ranked_candidates": [_eligible(r) for r in batch.ranked],
        "rejected_candidates": [_rejected(r) for r in batch.rejected],
        "failed_resumes": [{"filename": d.filename, "status": d.status.value, "error": d.error}
                           for d in docs if d.status.value in ("failed", "empty", "unsupported_format")],
        "duplicates": [{"filename": d.filename, "duplicate_of": d.duplicate_of, "file_hash": d.file_hash}
                       for d in docs if d.status.value == "duplicate"],
        "processing": [{"filename": d.filename, "file_hash": d.file_hash, "status": d.status.value}
                       for d in docs],
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "llm_provider": batch.llm_provider,
            "github_enrichment": batch.github_enabled,
            "score_weights": WEIGHTS,
        },
    }


def write_json(report: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
