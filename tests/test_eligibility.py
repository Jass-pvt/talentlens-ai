import pytest

from src.screening.eligibility import check_eligibility
from src.screening.evidence import analyze_text

PY_PROJECT = "PROJECTS\nDoc Bot | Python\n• Built a RAG chatbot with LangChain and FAISS in Python.\n"


def decide(text):
    return check_eligibility(analyze_text(text))


def test_python_plus_ai_is_eligible():
    d = decide("Jane Doe\nSKILLS\nPython, FastAPI\n" + PY_PROJECT)
    assert d.eligible and d.reasons == []


def test_python_only_is_rejected():
    d = decide("Jane Doe\nSKILLS\nPython, Django\nPROJECTS\nBlog | Python\n• Built a blog with Django.\n")
    assert not d.eligible
    assert d.reasons == ["No AI/agentic project evidence"]


def test_ai_only_is_rejected():
    d = decide("Jane Doe\nSKILLS\nJava\nPROJECTS\nDoc Bot | LangChain\n• Built a RAG chatbot with LangChain and FAISS.\n")
    assert not d.eligible
    assert d.reasons == ["No evidence of Python stack"]


def test_javascript_react_only_is_rejected_with_both_reasons():
    d = decide("Jane Doe\nSKILLS\nJavaScript, React, Node.js\nPROJECTS\nShop | React\n• Built a storefront in React.\n")
    assert not d.eligible
    assert len(d.reasons) == 2


def test_python_javascript_and_ai_is_eligible():
    d = decide("Jane Doe\nSKILLS\nPython, JavaScript, React, Next.js\n"
               "PROJECTS\nChat | Next.js, LangChain\n• Built a RAG assistant using LangChain, with a Next.js UI.\n")
    assert d.eligible


def test_ai_keywords_only_in_skills_do_not_qualify():
    d = decide("Jane Doe\nSKILLS\nPython, LLM, RAG, LangGraph\nPROJECTS\nSite | HTML\n• Built a static site.\n")
    assert not d.eligible
    assert "only in the skills list" in d.reasons[0]


def test_python_in_education_or_certifications_is_not_evidence():
    d = decide("Jane Doe\nEDUCATION\nCompleted Python for Everybody course\n"
               "PROJECTS\nDoc Bot | LangChain\n• Built a RAG chatbot with LangChain.\n")
    assert not d.eligible and "No evidence of Python stack" in d.reasons


def test_unstructured_resume_is_not_auto_rejected():
    d = decide("Jane Doe\nI built a RAG chatbot in Python using LangChain and FAISS for my college.")
    assert d.eligible


def test_decision_is_deterministic():
    text = "Jane Doe\nSKILLS\nPython\n" + PY_PROJECT
    assert decide(text) == decide(text)


@pytest.mark.parametrize("line", ["Used LangChain.", "Implemented a RAG pipeline with embeddings and FAISS."])
def test_any_project_level_ai_evidence_passes_hard_filter(line):
    d = decide(f"Jane Doe\nSKILLS\nPython\nPROJECTS\nApp | Python\n• {line}\n")
    assert d.eligible
