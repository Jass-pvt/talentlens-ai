"""Plain-text terminal UX: progress bars on stderr-free stdout, ranked summary."""
from __future__ import annotations

import sys
from collections import Counter

from src.pipeline import BatchResult

RULE = "─" * 40
_LABELS = {"parsing": "Parsing", "eligibility": "Eligibility", "github": "GitHub enrichment"}


def bar(done: int, total: int, width: int = 20) -> str:
    filled = width if total == 0 else int(width * done / total)
    return "█" * filled + "░" * (width - filled)


class ProgressReporter:
    """Callable progress sink: in-place bars on a TTY, one line per finished stage otherwise."""

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdout
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()

    def __call__(self, stage: str, done: int, total: int) -> None:
        if stage == "discovered":
            print(f"Resumes discovered: {total}\n", file=self.stream)
            return
        label = _LABELS.get(stage, stage)
        line = f"{label:<20} {bar(done, total)} {done}/{total}"
        if self.tty:
            end = "\n" if done >= total else "\r"
            print(line, end=end, file=self.stream, flush=True)
        elif done >= total:
            print(line, file=self.stream)


def render_header(input_dir: str) -> str:
    return f"AI Resume Screening\n{RULE}\n\nInput: {input_dir}"


def render_report(batch: BatchResult, top: int = 10) -> str:
    s = batch.summary
    out = ["", f"Eligible: {s['eligible']}", f"Rejected: {s['rejected']}",
           f"Duplicates: {s['duplicates']}", f"Failed: {s['failed_or_unreadable']}", ""]
    out += ["TOP CANDIDATES", RULE, ""]
    if not batch.ranked:
        out.append("No eligible candidates.")
    for r in batch.ranked[:top]:
        sc, b = r.score, r.score.breakdown
        out.append(f"#{r.rank:<3}{r.profile.candidate_name:<28}{sc.total_score}/100")
        for label, val, cap in (("AI/RAG", b.ai_project_depth, 40), ("Python", b.python_backend, 30),
                                ("Cloud", b.cloud_fullstack, 15), ("GitHub", b.github, 10),
                                ("Engineering", b.engineering_depth, 5)):
            out.append(f"    {label:<12}{val:>3}/{cap}")
        for p in sc.penalties:
            out.append(f"    Penalty     {p.penalty:>3}  {p.reason}")
        out.append("")
        out += [f"    ✓ {x}" for x in sc.strengths[:4]]
        out += [f"    ! {x}" for x in sc.concerns[:2]]
        out.append("")
    if len(batch.ranked) > top:
        out.append(f"... {len(batch.ranked) - top} more in the JSON output\n")
    if batch.rejected:
        out += ["REJECTED (reasons)", RULE]
        tally = Counter(reason for r in batch.rejected for reason in r.rejection_reasons)
        out += [f"  {n:>3} × {reason}" for reason, n in tally.most_common()]
        out.append("")
    problems = [d for d in batch.documents if d.status.value in ("failed", "empty", "unsupported_format")]
    if problems:
        out += ["UNREADABLE / SKIPPED", RULE]
        out += [f"  {d.filename}: {d.error}" for d in problems]
        out.append("")
    return "\n".join(out)
