"""Domain models shared across the pipeline. Plain dataclasses, no I/O."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional


class EvidenceLevel(IntEnum):
    """How convincingly a resume demonstrates a skill (ordered, comparable)."""

    NONE = 0
    KEYWORD_ONLY = 1  # listed in skills / summary
    PROJECT_MENTION = 2  # named in a project or job, no implementation verb
    IMPLEMENTATION = 3  # "built / implemented ... X"
    DEEP_IMPLEMENTATION = 4  # implemented, alongside several related sub-signals


class ParseStatus(str, Enum):
    PENDING = "pending"
    PARSED = "parsed"
    EMPTY = "empty"
    FAILED = "failed"
    DUPLICATE = "duplicate"
    UNSUPPORTED = "unsupported_format"


@dataclass
class Evidence:
    """One explainable piece of evidence, traceable to a resume section."""

    skill: str
    section: str
    evidence_text: str
    confidence: float
    level: EvidenceLevel = EvidenceLevel.KEYWORD_ONLY
    source: str = "resume"  # resume | github | llm
    category: str = ""  # ai | python_backend | cloud_fullstack | engineering | github


@dataclass
class ResumeDocument:
    filename: str
    path: str
    file_hash: Optional[str] = None
    status: ParseStatus = ParseStatus.PENDING
    text: str = ""
    links: list[str] = field(default_factory=list)
    page_count: int = 0
    error: Optional[str] = None
    duplicate_of: Optional[str] = None


@dataclass
class ProjectEntry:
    title: str
    text: str
    section: str  # projects | experience


@dataclass
class CandidateProfile:
    candidate_name: str
    source_filename: str
    parse_status: ParseStatus = ParseStatus.PARSED
    email: Optional[str] = None
    github_url: Optional[str] = None
    github_username: Optional[str] = None
    skills: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    experience: list[str] = field(default_factory=list)
    projects: list[ProjectEntry] = field(default_factory=list)
    python_evidence: list[Evidence] = field(default_factory=list)
    ai_evidence: list[Evidence] = field(default_factory=list)
    backend_evidence: list[Evidence] = field(default_factory=list)
    cloud_evidence: list[Evidence] = field(default_factory=list)
    engineering_evidence: list[Evidence] = field(default_factory=list)


@dataclass
class GitHubSummary:
    """Outcome of a GitHub lookup. `status` is always set; failures never raise."""

    status: str  # ok | missing | invalid_url | not_found | rate_limited | timeout | error
    username: Optional[str] = None
    public_repos: Optional[int] = None
    recent_repos: int = 0  # own repos pushed in the last 90 days
    maintained_repos: int = 0  # own, non-archived repos pushed in the last year
    python_repos: int = 0
    ai_repos: int = 0
    relevant_maintained: list[str] = field(default_factory=list)
    activity_score: int = 0  # 0-5
    repos_score: int = 0  # 0-5
    error: Optional[str] = None
    from_cache: bool = False

    @property
    def score(self) -> int:
        return self.activity_score + self.repos_score


@dataclass
class Penalty:
    penalty: int  # negative number
    reason: str


@dataclass
class ScoreBreakdown:
    ai_project_depth: int = 0
    python_backend: int = 0
    cloud_fullstack: int = 0
    github: int = 0
    engineering_depth: int = 0

    def subtotal(self) -> int:
        return (
            self.ai_project_depth
            + self.python_backend
            + self.cloud_fullstack
            + self.github
            + self.engineering_depth
        )


@dataclass
class ScoreResult:
    breakdown: ScoreBreakdown
    penalties: list[Penalty] = field(default_factory=list)
    total_score: int = 0
    why_this_score: str = ""
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    llm_adjustment: int = 0


@dataclass
class CandidateResult:
    profile: CandidateProfile
    eligible: bool
    rejection_reasons: list[str] = field(default_factory=list)
    score: Optional[ScoreResult] = None
    github: Optional[GitHubSummary] = None
    evidence: list[Evidence] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)
    llm_used: bool = False
    rank: Optional[int] = None


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses/enums/sets into JSON-safe primitives."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, IntEnum):
        return obj.name
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(to_jsonable(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
