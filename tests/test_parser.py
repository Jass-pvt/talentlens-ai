from src.extraction.contact_extractor import extract_email, extract_github, extract_name
from src.extraction.pdf_parser import PdfParseError, parse_pdf
from src.extraction.section_extractor import classify_heading, split_blocks, split_sections
from tests.helpers import STRONG, make_blank_pdf, make_pdf


def test_valid_pdf_is_parsed_and_normalized(tmp_path):
    parsed = parse_pdf(make_pdf(tmp_path / "a.pdf", STRONG).read_bytes())
    assert "Asha Verma" in parsed.text and "RAG pipeline" in parsed.text
    assert parsed.page_count == 1 and "  " not in parsed.text


def test_empty_pdf_yields_empty_text(tmp_path):
    assert parse_pdf(make_blank_pdf(tmp_path / "b.pdf").read_bytes()).text == ""


def test_malformed_pdf_raises_parse_error():
    for blob in (b"%PDF-1.4\nnot really a pdf", b"hello world", b""):
        try:
            parse_pdf(blob)
            raise AssertionError("expected PdfParseError")
        except PdfParseError:
            pass


def test_hyperlink_target_is_extracted(tmp_path):
    parsed = parse_pdf(make_pdf(tmp_path / "l.pdf", ["Jane Doe"], link="https://github.com/janedoe").read_bytes())
    assert "https://github.com/janedoe" in parsed.links
    assert extract_github("Jane Doe", parsed.links) == ("https://github.com/janedoe", "janedoe")


def test_missing_email_and_github_are_none_not_errors():
    assert extract_email("Jane Doe\nPython developer", []) is None
    assert extract_github("Jane Doe\nPython developer", []) == (None, None)


def test_email_and_github_found_in_text():
    text = "Jane Doe | jane.doe@mail.example.com | github.com/jane-doe/rag-bot"
    assert extract_email(text, []) == "jane.doe@mail.example.com"
    assert extract_github(text, []) == ("https://github.com/jane-doe", "jane-doe")


def test_reserved_github_paths_are_not_usernames():
    assert extract_github("see github.com/features and github.com/sponsors", []) == (None, None)


def test_name_extraction_and_filename_fallback():
    assert extract_name("JANE DOE\njane@x.com", "x.pdf") == "Jane Doe"
    assert extract_name("jane@x.com\n+91 99999 99999", "candidate_17.pdf") == "Candidate 17"


def test_heading_detection_does_not_fire_on_normal_lines():
    assert classify_heading("PROJECTS") == "projects"
    assert classify_heading("Technical Skills") == "skills"
    assert classify_heading("Work Experience") == "experience"
    assert classify_heading("Project Management") is None
    assert classify_heading("Built a project with Python.") is None


def test_inline_skills_heading_and_unknown_headings_are_kept():
    s = split_sections("Jane\nSkills: Python, FastAPI\nProjects\nBot | Python\n• Built X\nOpen Source Contributions\n• Built Y")
    assert "Python, FastAPI" in s.get("skills")
    assert "Built Y" in s.get("projects")


def test_wrapped_bullets_are_rejoined_and_entries_split_on_titles():
    blocks = split_blocks("Bot | Python\n• Built a RAG pipeline with\nembeddings and FAISS.\nOther | Go\n• Built Z", "projects")
    assert len(blocks) == 2
    assert blocks[0].items[1] == "Built a RAG pipeline with embeddings and FAISS."


def test_bullet_glyphs_are_normalized_so_entries_split_correctly(tmp_path):
    parsed = parse_pdf(make_pdf(tmp_path / "bl.pdf", ["Bot | Python", "• Built X", "• Built Y"]).read_bytes())
    assert "\x7f" not in parsed.text and "• Built X" in parsed.text
    blocks = split_blocks(parsed.text, "projects")
    assert len(blocks) == 1 and blocks[0].items == ["Bot | Python", "Built X", "Built Y"]
