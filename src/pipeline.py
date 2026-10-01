"""Orchestrates: ingest -> parse -> eligibility -> (LLM) -> score -> GitHub -> rank.

Each resume is isolated: any failure is recorded against that file and the batch continues.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.ai.adapter import LLMAdapter
from src.ai.schemas import ProjectAnalysis, grounded
from src.config import MAX_LLM_INPUT_CHARS, Config
from src.enrichment.github import GitHubClient, github_evidence
from src.extraction.pdf_parser import PdfParseError, parse_pdf
from src.extraction.profile_builder import build_profile
from src.extraction.section_extractor import split_sections
from src.ingestion.deduplication import DuplicateTracker
from src.ingestion.loader import discover_resumes, read_and_hash
from src.models import (
    CandidateResult, Evidence, EvidenceLevel, GitHubSummary, ParseStatus, ResumeDocument,
)
from src.screening.eligibility import check_eligibility
from src.screening.evidence import EvidenceAnalysis, analyze, evidence_for_family
from src.screening.scoring import apply_github, rank_candidates, score_candidate

log = logging.getLogger(__name__)
ProgressFn = Callable[[str, int, int], None]


@dataclass
class BatchResult:
    documents: list[ResumeDocument] = field(default_factory=list)
    results: list[CandidateResult] = field(default_factory=list)
    ranked: list[CandidateResult] = field(default_factory=list)
    rejected: list[CandidateResult] = field(default_factory=list)
    llm_provider: Optional[str] = None
    github_enabled: bool = True

    def count(self, status: ParseStatus) -> int:
        return sum(1 for d in self.documents if d.status == status)

    @property
    def summary(self) -> dict[str, int]:
        failed = self.count(ParseStatus.FAILED) + self.count(ParseStatus.EMPTY) + self.count(ParseStatus.UNSUPPORTED)
        return {
            "total_resumes": len(self.documents),
            "successfully_parsed": self.count(ParseStatus.PARSED),
            "eligible": len(self.ranked),
            "rejected": len(self.rejected),
            "duplicates": self.count(ParseStatus.DUPLICATE),
            "failed_or_unreadable": failed,
        }


def run_pipeline(
    input_dir: Path,
    config: Config,
    *,
    adapter: Optional[LLMAdapter] = None,
    github_client: Optional[GitHubClient] = None,
    enable_github: bool = True,
    on_progress: Optional[ProgressFn] = None,
) -> BatchResult:
    progress: ProgressFn = on_progress or (lambda stage, done, total: None)
    batch = BatchResult(llm_provider=adapter.name if adapter else None, github_enabled=enable_github)
    batch.documents = discover_resumes(Path(input_dir))
    total = len(batch.documents)
    progress("discovered", total, total)

    # 1) One pass per file: read+hash, dedupe, parse, build profile, run hard eligibility.
    tracker = DuplicateTracker()
    pending: list[tuple[CandidateResult, EvidenceAnalysis, str]] = []
    for done, doc in enumerate(batch.documents, start=1):
        try:
            item = _ingest(doc, tracker)
            if item is not None:
                pending.append(item)
        except Exception as exc:  # last-resort isolation: one bad file never stops the batch
            doc.status, doc.error = ParseStatus.FAILED, f"Unexpected error: {type(exc).__name__}: {exc}"
            log.exception("%s: unexpected failure", doc.filename)
        progress("parsing", done, total)

    candidates = [c for c, _, _ in pending]
    progress("eligibility", len(pending), len(pending))
    eligible_items = [(c, a, t) for c, a, t in pending if c.eligible]

    # 2) Optional semantic layer + scoring, eligible candidates only.
    for result, analysis, text in eligible_items:
        try:
            llm = _semantic(adapter, result, analysis, text)
            result.llm_used = llm is not None
            result.score = score_candidate(analysis, llm)
            result.evidence = _collect_evidence(analysis, llm, text, result.profile.candidate_name)
            result.confidence["project_analysis"] = _project_confidence(analysis, llm)
        except Exception as exc:
            _fail(batch, result, f"Scoring error: {type(exc).__name__}: {exc}")
            log.exception("%s: scoring failed", result.profile.source_filename)
    eligible = [c for c, _, _ in eligible_items if c.score is not None]

    # 3) GitHub enrichment after eligibility: no API calls for rejected candidates.
    _enrich(batch, eligible, config, github_client, enable_github, progress)

    batch.results = [c for c in candidates if c.eligible is False or c.score is not None]
    batch.ranked = rank_candidates(batch.results)
    batch.rejected = sorted((c for c in batch.results if not c.eligible),
                            key=lambda c: (c.profile.candidate_name.casefold(), c.profile.source_filename))
    return batch


def _ingest(doc: ResumeDocument, tracker: DuplicateTracker):
    if doc.status == ParseStatus.UNSUPPORTED:
        return None
    data = read_and_hash(doc)
    if data is None:
        return None
    original = tracker.check(doc.file_hash, doc.filename)
    if original:
        doc.status, doc.duplicate_of = ParseStatus.DUPLICATE, original
        return None
    try:
        parsed = parse_pdf(data)
    except PdfParseError as exc:
        doc.status, doc.error = ParseStatus.FAILED, str(exc)
        log.warning("%s: %s", doc.filename, exc)
        return None
    doc.text, doc.links, doc.page_count = parsed.text, parsed.links, parsed.page_count
    if not doc.text.strip():
        doc.status = ParseStatus.EMPTY
        doc.error = "No extractable text (blank or scanned/image-only PDF)"
        return None
    doc.status = ParseStatus.PARSED

    sections = split_sections(doc.text)
    profile = build_profile(doc, sections)
    analysis = analyze(sections, doc.text)
    decision = check_eligibility(analysis)
    profile.python_evidence = [e for e in evidence_for_family(analysis, "backend")
                               if e.skill in ("python", "framework", "async")]
    profile.backend_evidence = [e for e in evidence_for_family(analysis, "backend")
                                if e.skill not in ("python", "framework", "async")]
    profile.ai_evidence = evidence_for_family(analysis, "ai")
    profile.cloud_evidence = evidence_for_family(analysis, "cloud")
    profile.engineering_evidence = evidence_for_family(analysis, "engineering")
    result = CandidateResult(
        profile=profile, eligible=decision.eligible, rejection_reasons=decision.reasons,
        confidence={"eligibility": decision.confidence},
    )
    return result, analysis, doc.text


def _fail(batch: BatchResult, result: CandidateResult, message: str) -> None:
    for doc in batch.documents:
        if doc.filename == result.profile.source_filename:
            doc.status, doc.error = ParseStatus.FAILED, message
    result.score = None


def _semantic(adapter: Optional[LLMAdapter], result: CandidateResult, analysis: EvidenceAnalysis,
              text: str) -> Optional[ProjectAnalysis]:
    """LLM failure -> None -> deterministic scoring. Never raises."""
    if adapter is None:
        return None
    project_text = "\n\n".join(b.text for b in analysis.ai.ai_blocks)[:MAX_LLM_INPUT_CHARS]
    if not project_text:
        return None
    try:
        return adapter.extract_project_evidence(result.profile.candidate_name, project_text)
    except Exception as exc:
        log.warning("%s: LLM failed (%s); using deterministic fallback",
                    result.profile.source_filename, exc)
        return None


def _collect_evidence(analysis: EvidenceAnalysis, llm: Optional[ProjectAnalysis], text: str,
                      name: str) -> list[Evidence]:
    """Evidence behind the score: top entries per category, plus grounded LLM quotes."""
    items: list[Evidence] = []
    for key in ("ai", "backend", "cloud", "engineering"):
        items.extend(_merge_same_text(evidence_for_family(analysis, key))[:4])
    if llm is not None:
        for e in llm.evidence:
            if grounded(e.evidence, text):  # drop anything the resume does not actually say
                items.append(Evidence(e.claim or "llm_claim", "projects", e.evidence, round(e.confidence, 2),
                                      EvidenceLevel.IMPLEMENTATION, "llm", "ai_project_depth"))
    return items


def _merge_same_text(evidence: list[Evidence]) -> list[Evidence]:
    """Signals proven by the same sentence become one entry ('skill' lists them all)."""
    merged: dict[tuple, Evidence] = {}
    for e in evidence:  # already strongest-first
        key = (e.section, e.evidence_text)
        if key in merged:
            merged[key].skill += f", {e.skill}"
        else:
            merged[key] = Evidence(e.skill, e.section, e.evidence_text, e.confidence, e.level, e.source, e.category)
    return list(merged.values())


def _project_confidence(analysis: EvidenceAnalysis, llm: Optional[ProjectAnalysis]) -> float:
    levels = [b.levels["core"] for b in analysis.ai.ai_blocks]
    base = {EvidenceLevel.PROJECT_MENTION: 0.6, EvidenceLevel.IMPLEMENTATION: 0.8,
            EvidenceLevel.DEEP_IMPLEMENTATION: 0.9}.get(max(levels, default=EvidenceLevel.NONE), 0.5)
    return round(min(0.95, base + (0.05 if llm is not None else 0.0)), 2)


def _enrich(batch: BatchResult, eligible: list[CandidateResult], config: Config,
            client: Optional[GitHubClient], enabled: bool, progress: ProgressFn) -> None:
    summaries: dict[str, GitHubSummary] = {}
    if enabled and eligible:
        client = client or GitHubClient(config.github_token, config.github_timeout)
        usernames = [c.profile.github_username for c in eligible]
        try:
            summaries = client.fetch_many(usernames, config.github_workers)
        except Exception as exc:  # defensive: enrichment must never kill the batch
            log.warning("GitHub enrichment aborted: %s", exc)
    for done, cand in enumerate(eligible, start=1):
        username = cand.profile.github_username
        if enabled:
            summary = summaries.get(username) if username else None
            summary = summary or GitHubSummary(status="missing" if not username else "error", username=username)
        else:
            summary = GitHubSummary(status="disabled", username=username)
        cand.github = summary
        cand.confidence["github"] = {"ok": 1.0}.get(summary.status, 0.0)
        cand.evidence.extend(github_evidence(summary))
        apply_github(cand.score, summary)
        progress("github", done, len(eligible))
