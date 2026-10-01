"""ResumeDocument -> CandidateProfile (structured fields only; evidence is added later)."""
from __future__ import annotations

from src.extraction.contact_extractor import extract_email, extract_github, extract_name
from src.extraction.section_extractor import Sections, split_blocks
from src.extraction.skills_extractor import extract_skills
from src.models import CandidateProfile, ParseStatus, ProjectEntry, ResumeDocument


def build_profile(doc: ResumeDocument, sections: Sections) -> CandidateProfile:
    url, username = extract_github(doc.text, doc.links)
    projects = [ProjectEntry(b.title, b.text, "projects") for b in split_blocks(sections.get("projects"), "projects")]
    experience = [item for b in split_blocks(sections.get("experience"), "experience") for item in b.items]
    return CandidateProfile(
        candidate_name=extract_name(doc.text, doc.filename),
        source_filename=doc.filename,
        parse_status=ParseStatus.PARSED,
        email=extract_email(doc.text, doc.links),
        github_url=url,
        github_username=username,
        skills=extract_skills(doc.text),
        education=[ln for ln in sections.get("education").split("\n") if ln][:6],
        experience=experience[:20],
        projects=projects,
    )
