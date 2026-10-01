from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import requests

from src.enrichment.github import GitHubClient, summarize_repos

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def iso(days_ago):
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


class Resp:
    def __init__(self, code=200, data=None, headers=None, text=""):
        self.status_code, self._data, self.headers, self.text = code, data, headers or {}, text

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


def repo(name, days_ago, lang="Python", fork=False, desc="", archived=False):
    return {"name": name, "pushed_at": iso(days_ago), "language": lang, "fork": fork,
            "description": desc, "topics": [], "archived": archived}


def client(*responses_or_exc):
    session = MagicMock()
    session.get.side_effect = list(responses_or_exc)
    return GitHubClient(token="t", timeout=3, session=session, now=lambda: NOW), session


def ok_pair(repos):
    return [Resp(200, {"public_repos": len(repos)}), Resp(200, repos)]


def test_success_scores_activity_and_relevant_repos():
    repos = [repo("rag-bot", 5, desc="RAG chatbot"), repo("api", 20), repo("old", 400), repo("fork", 2, fork=True)]
    c, s = client(*ok_pair(repos))
    g = c.fetch("janedoe")
    assert g.status == "ok" and g.recent_repos == 2 and g.python_repos == 3
    assert g.relevant_maintained == ["rag-bot", "api"]
    assert g.activity_score == 3 and g.repos_score == 4 and g.score == 7
    assert s.get.call_args.kwargs["timeout"] == 3  # always uses a timeout


def test_score_is_capped_at_ten():
    repos = [repo(f"ai-{i}", i + 1, desc="llm agent") for i in range(8)]
    g = summarize_repos("u", 8, repos, NOW)
    assert g.score == 10


def test_inactive_profile_scores_zero():
    g = summarize_repos("u", 1, [repo("old", 800)], NOW)
    assert g.status == "ok" and g.score == 0


def test_404_user_not_found():
    c, _ = client(Resp(404))
    g = c.fetch("ghost-user")
    assert g.status == "not_found" and g.score == 0


def test_rate_limit_is_detected_and_short_circuits_later_calls():
    c, s = client(Resp(403, headers={"X-RateLimit-Remaining": "0"}))
    assert c.fetch("alice").status == "rate_limited"
    assert c.fetch("bob").status == "rate_limited"
    assert s.get.call_count == 1  # second user never hit the network


def test_timeout_and_network_errors_do_not_raise():
    c, _ = client(requests.Timeout("slow"))
    assert c.fetch("alice").status == "timeout"
    c2, _ = client(requests.ConnectionError("down"))
    assert c2.fetch("alice").status == "error"


def test_invalid_json_and_server_error_do_not_raise():
    c, _ = client(Resp(200, None))
    assert c.fetch("alice").status == "error"
    c2, _ = client(Resp(500))
    assert c2.fetch("alice").status == "error"


def test_missing_and_invalid_usernames_make_no_requests():
    c, s = client()
    assert c.fetch(None).status == "missing"
    for bad in ("../etc/passwd", "a b", "-start", "x" * 40, "name/with/slash"):
        assert c.fetch(bad).status == "invalid_url"
    assert s.get.call_count == 0


def test_same_username_is_fetched_once_even_with_different_case():
    c, s = client(*ok_pair([repo("a", 3)]))
    first = c.fetch("JaneDoe")
    second = c.fetch("janedoe")
    assert s.get.call_count == 2          # one profile + one repos call, total
    assert not first.from_cache and second.from_cache


def test_fetch_many_dedupes_usernames_case_insensitively():
    c, s = client(*ok_pair([repo("a", 3)]))
    out = c.fetch_many(["Jane", "jane", None, "Jane"], workers=3)
    assert set(out) == {"Jane", "jane"} and s.get.call_count == 2


def test_empty_public_profile_is_ok_with_zero_score():
    c, _ = client(*ok_pair([]))
    g = c.fetch("emptyuser")
    assert g.status == "ok" and g.public_repos == 0 and g.score == 0


def test_unexpected_exception_is_contained():
    c, _ = client(RuntimeError("bug"))
    g = c.fetch("alice")
    assert g.status == "error" and "RuntimeError" in g.error
