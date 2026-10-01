from src.ai.schemas import ProjectAnalysis, grounded
from src.config import WEIGHTS
from src.models import (
    CandidateProfile, CandidateResult, EvidenceLevel, GitHubSummary, Penalty, ScoreBreakdown, ScoreResult,
)
from src.screening.evidence import analyze_text
from src.screening.scoring import (
    LLM_MAX_ADJUSTMENT, THIN_PENALTY, apply_github, finalize, rank_candidates, score_candidate,
)


def ai_score(bullet, skills="Python"):
    text = f"Jane\nSKILLS\n{skills}\nPROJECTS\nApp | Python\n• {bullet}\n"
    return score_candidate(analyze_text(text)).breakdown.ai_project_depth


def test_weights_are_the_official_100_points():
    assert WEIGHTS == {"ai_project_depth": 40, "python_backend": 30, "cloud_fullstack": 15,
                       "github": 10, "engineering_depth": 5}
    assert sum(WEIGHTS.values()) == 100


def test_ai_depth_increases_with_each_subsignal():
    rag = ai_score("Built a RAG chatbot.")
    emb = ai_score("Built a RAG chatbot using embeddings.")
    tools = ai_score("Built a RAG chatbot with tool calling.")
    state = ai_score("Built a RAG chatbot with tool calling and stateful conversation memory.")
    full = ai_score("Built a RAG chatbot with tool calling, stateful conversation memory, and RAGAS evaluation.")
    assert rag < emb < state < full
    assert rag < tools < state < full


def test_langgraph_in_skills_alone_does_not_give_max_points():
    text = "Jane\nSKILLS\nPython, LangGraph, LangChain, RAG, LLM\nPROJECTS\nSite | HTML\n• Built a static site.\n"
    assert score_candidate(analyze_text(text)).breakdown.ai_project_depth == 0


def test_category_caps_and_total_never_exceed_limits():
    stuffed = ("• Built and deployed a stateful multi-agent RAG system with embeddings, FAISS, tool calling, "
               "RAGAS evaluation, async FastAPI, PostgreSQL, Redis caching, Docker, AWS, Kubernetes, Terraform, "
               "React, Next.js, pytest, retry, rate limiting, Prometheus logging, Celery queues and CI/CD.\n")
    text = "Jane\nSKILLS\nPython\nPROJECTS\nBig | Python\n" + stuffed
    b = score_candidate(analyze_text(text)).breakdown
    assert b.ai_project_depth <= 40 and b.python_backend <= 30
    assert b.cloud_fullstack <= 15 and b.engineering_depth <= 5
    apply = ScoreResult(breakdown=b)
    apply_github(apply, GitHubSummary(status="ok", activity_score=5, repos_score=5))
    assert apply.breakdown.github == 10 and apply.total_score <= 100


def test_total_is_sum_of_categories_plus_penalties_and_clamped():
    r = ScoreResult(breakdown=ScoreBreakdown(10, 10, 5, 3, 2), penalties=[Penalty(-10, "x")])
    finalize(r)
    assert r.total_score == 20
    r2 = ScoreResult(breakdown=ScoreBreakdown(1, 0, 0, 0, 0), penalties=[Penalty(-10, "x")])
    finalize(r2)
    assert r2.total_score == 0


def test_thin_api_wrapper_is_penalized_transparently():
    text = "Jane\nSKILLS\nPython\nPROJECTS\nSummarizer | Python, OpenAI API\n• Built a summarizer that sends text to the OpenAI API and prints the reply.\n"
    r = score_candidate(analyze_text(text))
    assert r.penalties and r.penalties[0].penalty == THIN_PENALTY
    assert "thin API wrapper" in r.penalties[0].reason
    assert r.total_score == max(0, r.breakdown.subtotal() + THIN_PENALTY)


def test_using_an_api_is_not_penalized_when_there_is_real_depth():
    text = ("Jane\nSKILLS\nPython\nPROJECTS\nInvoice Bot | Python, OpenAI API\n"
            "• Built an invoice extraction service that validates OpenAI API output against a JSON schema "
            "and stores results in PostgreSQL.\n")
    assert score_candidate(analyze_text(text)).penalties == []


def test_one_strong_project_offsets_a_thin_one():
    text = ("Jane\nSKILLS\nPython\nPROJECTS\nJoke Bot | OpenAI\n• Built a chatbot with the OpenAI API.\n\n"
            "Doc QA | LangChain\n• Built a RAG pipeline with embeddings and FAISS retrieval.\n")
    assert score_candidate(analyze_text(text)).penalties == []


def _result(name, total, ai, py, eng, filename=None):
    sc = ScoreResult(breakdown=ScoreBreakdown(ai, py, 0, 0, eng), total_score=total)
    return CandidateResult(profile=CandidateProfile(name, filename or f"{name}.pdf"), eligible=True, score=sc)


def test_ranking_is_deterministic_with_tie_breakers():
    a = _result("Zed", 80, ai=30, py=20, eng=3)
    b = _result("Amy", 80, ai=35, py=15, eng=1)   # wins on AI depth
    c = _result("Bob", 80, ai=30, py=25, eng=1)   # beats Zed on python
    d = _result("Cat", 80, ai=30, py=25, eng=2)   # beats Bob on engineering
    e = _result("Abe", 80, ai=30, py=25, eng=2)   # ties Cat: alphabetical
    f = _result("Low", 50, ai=40, py=30, eng=5)
    ranked = rank_candidates([a, b, c, d, e, f])
    assert [r.profile.candidate_name for r in ranked] == ["Amy", "Abe", "Cat", "Bob", "Zed", "Low"]
    assert [r.rank for r in ranked] == [1, 2, 3, 4, 5, 6]
    assert [r.profile.candidate_name for r in rank_candidates(list(reversed([a, b, c, d, e, f])))] == \
           [r.profile.candidate_name for r in ranked]


def test_rejected_candidates_are_never_ranked():
    rejected = CandidateResult(profile=CandidateProfile("No", "no.pdf"), eligible=False)
    assert rank_candidates([rejected]) == []


def test_llm_adjustment_is_bounded_and_cannot_remove_penalty():
    text = "Jane\nSKILLS\nPython\nPROJECTS\nDoc QA | LangChain\n• Built a RAG pipeline with embeddings.\n"
    analysis = analyze_text(text)
    base = score_candidate(analysis).breakdown.ai_project_depth
    high = score_candidate(analysis, ProjectAnalysis(project_depth=40)).breakdown.ai_project_depth
    low = score_candidate(analysis, ProjectAnalysis(project_depth=0)).breakdown.ai_project_depth
    assert high - base <= LLM_MAX_ADJUSTMENT and base - low <= LLM_MAX_ADJUSTMENT


def test_llm_schema_clamps_and_rejects_garbage():
    p = ProjectAnalysis.from_dict({"project_depth": 999, "technologies": ["Python"],
                                   "evidence": [{"claim": "c", "evidence": "x" * 10, "confidence": 7}, "junk"]})
    assert p.project_depth == 40 and p.evidence[0].confidence == 1.0 and len(p.evidence) == 1
    for bad in ("text", {"project_depth": "high"}, None):
        try:
            ProjectAnalysis.from_dict(bad)
            raise AssertionError("expected SchemaError")
        except ValueError:
            pass


def test_grounding_rejects_invented_quotes():
    resume = "Built a stateful RAG pipeline with   FAISS retrieval."
    assert grounded("stateful rag pipeline with faiss", resume)
    assert not grounded("Led a team of 40 engineers at Google", resume)
