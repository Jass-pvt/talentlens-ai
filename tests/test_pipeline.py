import json
from unittest.mock import MagicMock

import requests

from src.ai.adapter import LLMAdapter, LLMError
from src.ai.schemas import ProjectAnalysis
from src.config import Config
from src.enrichment.github import GitHubClient
from src.models import ParseStatus
from src.pipeline import run_pipeline
from src.reporting.json_report import build_report, write_json
from src.reporting.terminal import render_report
from tests.helpers import STRONG, make_blank_pdf, make_pdf

PYTHON_ONLY = ["Chitra Nair", "SKILLS", "Python, Django", "PROJECTS", "Blog | Python", "• Built a blog with Django."]
JS_ONLY = ["Eli T", "SKILLS", "JavaScript, React", "PROJECTS", "Shop | React", "• Built a store in React."]
CFG = Config()


def no_network_client():
    session = MagicMock()
    session.get.side_effect = requests.ConnectionError("offline")
    return GitHubClient(session=session), session


def build_dir(tmp_path):
    make_pdf(tmp_path / "a_strong.pdf", STRONG)
    make_pdf(tmp_path / "b_python_only.pdf", PYTHON_ONLY)
    make_pdf(tmp_path / "c_js.pdf", JS_ONLY)
    make_blank_pdf(tmp_path / "d_blank.pdf")
    (tmp_path / "e_broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    (tmp_path / "f_copy_of_strong.pdf").write_bytes((tmp_path / "a_strong.pdf").read_bytes())
    (tmp_path / "g_notes.txt").write_text("hi")
    return tmp_path


def test_broken_resumes_do_not_stop_the_batch(tmp_path):
    client, _ = no_network_client()
    batch = run_pipeline(build_dir(tmp_path), CFG, github_client=client)
    status = {d.filename: d.status for d in batch.documents}
    assert status["e_broken.pdf"] == ParseStatus.FAILED
    assert status["d_blank.pdf"] == ParseStatus.EMPTY
    assert status["g_notes.txt"] == ParseStatus.UNSUPPORTED
    assert status["f_copy_of_strong.pdf"] == ParseStatus.DUPLICATE
    assert [r.profile.candidate_name for r in batch.ranked] == ["Asha Verma"]
    assert {r.profile.source_filename for r in batch.rejected} == {"b_python_only.pdf", "c_js.pdf"}


def test_batch_summary_adds_up(tmp_path):
    client, _ = no_network_client()
    s = run_pipeline(build_dir(tmp_path), CFG, github_client=client).summary
    assert s == {"total_resumes": 7, "successfully_parsed": 3, "eligible": 1, "rejected": 2,
                 "duplicates": 1, "failed_or_unreadable": 3}
    assert s["total_resumes"] == s["successfully_parsed"] + s["duplicates"] + s["failed_or_unreadable"]


def test_github_failure_does_not_reject_or_crash(tmp_path):
    make_pdf(tmp_path / "a.pdf", STRONG, link="https://github.com/asha-verma-dev")
    client, _ = no_network_client()
    batch = run_pipeline(tmp_path, CFG, github_client=client)
    top = batch.ranked[0]
    assert top.eligible and top.github.status == "error" and top.score.breakdown.github == 0
    assert top.confidence["github"] == 0.0


def test_missing_github_does_not_reject(tmp_path):
    make_pdf(tmp_path / "a.pdf", [ln for ln in STRONG if "github.com" not in ln])
    client, session = no_network_client()
    batch = run_pipeline(tmp_path, CFG, github_client=client)
    assert batch.ranked[0].github.status == "missing" and session.get.call_count == 0


def test_rejected_candidates_trigger_no_github_calls(tmp_path):
    make_pdf(tmp_path / "py.pdf", PYTHON_ONLY + ["github.com/someone"])
    client, session = no_network_client()
    run_pipeline(tmp_path, CFG, github_client=client)
    assert session.get.call_count == 0


class BrokenLLM(LLMAdapter):
    name = "broken"

    def extract_project_evidence(self, candidate_name, project_text):
        raise LLMError("boom")


class OptimisticLLM(LLMAdapter):
    name = "optimist"

    def extract_project_evidence(self, candidate_name, project_text):
        return ProjectAnalysis(project_depth=40, concerns=["No latency numbers"])


def test_llm_failure_falls_back_to_deterministic_scores(tmp_path):
    make_pdf(tmp_path / "a.pdf", STRONG)
    client, _ = no_network_client()
    plain = run_pipeline(tmp_path, CFG, github_client=client)
    client2, _ = no_network_client()
    broken = run_pipeline(tmp_path, CFG, adapter=BrokenLLM(), github_client=client2)
    assert broken.ranked[0].score.total_score == plain.ranked[0].score.total_score
    assert broken.ranked[0].llm_used is False


def test_llm_can_never_change_eligibility(tmp_path):
    make_pdf(tmp_path / "py.pdf", PYTHON_ONLY)
    client, _ = no_network_client()
    batch = run_pipeline(tmp_path, CFG, adapter=OptimisticLLM(), github_client=client)
    assert batch.ranked == [] and len(batch.rejected) == 1


def test_llm_concerns_are_merged_when_available(tmp_path):
    make_pdf(tmp_path / "a.pdf", STRONG)
    client, _ = no_network_client()
    batch = run_pipeline(tmp_path, CFG, adapter=OptimisticLLM(), github_client=client)
    assert batch.ranked[0].llm_used and "No latency numbers" in batch.ranked[0].score.concerns


def test_json_report_is_valid_and_complete(tmp_path):
    client, _ = no_network_client()
    batch = run_pipeline(build_dir(tmp_path), CFG, github_client=client)
    out = tmp_path.parent / (tmp_path.name + "_out") / "results.json"
    write_json(build_report(batch), out)
    data = json.loads(out.read_text())
    top = data["ranked_candidates"][0]
    assert set(top["score_breakdown"]) == {"ai_project_depth", "python_backend", "cloud_fullstack",
                                           "github", "engineering_depth"}
    assert top["evidence"] and top["why_this_score"] and "strengths" in top and "concerns" in top
    assert set(top["confidence"]) == {"eligibility", "project_analysis", "github"}
    rej = data["rejected_candidates"][0]
    assert rej["eligible"] is False and rej["rejection_reasons"] and "matched_skills" in rej
    assert data["duplicates"][0]["duplicate_of"] == "a_strong.pdf"
    assert len(data["processing"]) == 7 and all(p["status"] for p in data["processing"])


def test_terminal_report_renders(tmp_path):
    client, _ = no_network_client()
    text = render_report(run_pipeline(build_dir(tmp_path), CFG, github_client=client))
    assert "TOP CANDIDATES" in text and "Asha Verma" in text and "/100" in text


def test_empty_folder_is_handled(tmp_path):
    batch = run_pipeline(tmp_path, CFG, enable_github=False)
    assert batch.summary["total_resumes"] == 0 and "No eligible candidates." in render_report(batch)


def test_missing_input_directory_raises_clear_error(tmp_path):
    try:
        run_pipeline(tmp_path / "nope", CFG)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass
