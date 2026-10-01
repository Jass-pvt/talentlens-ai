"""Public GitHub enrichment (max 10 points). Every failure becomes a status, never an exception.

Two API calls per unique username (profile + up to 100 most recently pushed repos).
Results are cached in memory, and a rate-limit response short-circuits further calls in the run.
"""
from __future__ import annotations

import dataclasses
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional

import requests

from src.enrichment.cache import MemoryCache
from src.models import Evidence, EvidenceLevel, GitHubSummary

log = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"
# GitHub's own username rules; also prevents path tricks like "../" in the request URL.
USERNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")
_AI_REPO = re.compile(r"\b(?:llm|rag|agents?|agentic|langchain|langgraph|openai|embeddings?|vector|gpt|ai|"
                      r"machine[- ]learning|ml|nlp|chatbot|transformers?)\b", re.I)
_ACTIVITY_POINTS = [(5, 5), (3, 4), (2, 3), (1, 2)]  # (min recent repos, points)
_REPO_POINTS = {0: 0, 1: 2, 2: 3, 3: 4}


class _GitHubError(Exception):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


class GitHubClient:
    def __init__(self, token: Optional[str] = None, timeout: float = 8.0,
                 session: Optional[requests.Session] = None,
                 now: Optional[Callable[[], datetime]] = None) -> None:
        self._timeout = timeout
        self._session = session or requests.Session()
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._headers = {"Accept": "application/vnd.github+json", "User-Agent": "ai-resume-screening"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self.cache: MemoryCache[GitHubSummary] = MemoryCache()
        self._rate_limited = threading.Event()
        self.api_calls = 0

    # -- public API ---------------------------------------------------------
    def fetch(self, username: Optional[str]) -> GitHubSummary:
        if not username:
            return GitHubSummary(status="missing")
        if not USERNAME_RE.match(username):
            return GitHubSummary(status="invalid_url", username=username, error="invalid GitHub username")
        key = username.lower()
        cached = self.cache.get(key)
        if cached is not None:
            return dataclasses.replace(cached, from_cache=True)
        if self._rate_limited.is_set():
            return GitHubSummary(status="rate_limited", username=username,
                                 error="GitHub rate limit hit earlier in this run")
        try:
            summary = self._summarize(username)
        except _GitHubError as exc:
            log.warning("GitHub lookup for %s failed: %s (%s)", username, exc.status, exc)
            summary = GitHubSummary(status=exc.status, username=username, error=str(exc))
        except Exception as exc:  # unexpected shape/bug: degrade, never take down the batch
            log.warning("GitHub lookup for %s hit an unexpected error: %r", username, exc)
            summary = GitHubSummary(status="error", username=username, error=f"unexpected: {type(exc).__name__}")
        self.cache.set(key, summary)
        return summary

    def fetch_many(self, usernames: Iterable[Optional[str]], workers: int = 4) -> dict[str, GitHubSummary]:
        """Fetch each distinct (case-insensitive) username once, with bounded concurrency."""
        spellings: dict[str, list[str]] = {}
        for name in usernames:
            if name:
                spellings.setdefault(name.lower(), []).append(name)
        if not spellings:
            return {}
        keys = sorted(spellings)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            fetched = dict(zip(keys, pool.map(lambda k: self.fetch(spellings[k][0]), keys)))
        return {name: fetched[key] for key, names in spellings.items() for name in names}

    # -- internals ----------------------------------------------------------
    def _get(self, path: str, params: Optional[dict] = None):
        self.api_calls += 1
        try:
            resp = self._session.get(API_ROOT + path, headers=self._headers, params=params,
                                     timeout=self._timeout)
        except requests.Timeout as exc:
            raise _GitHubError("timeout", f"timed out after {self._timeout}s") from exc
        except requests.RequestException as exc:
            raise _GitHubError("error", f"network error: {type(exc).__name__}") from exc
        code = resp.status_code
        if code == 404:
            raise _GitHubError("not_found", "GitHub user not found")
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if code == 429 or (code == 403 and (remaining == "0" or "rate limit" in (resp.text or "").lower())):
            self._rate_limited.set()
            raise _GitHubError("rate_limited", "GitHub API rate limit exceeded")
        if code != 200:
            raise _GitHubError("error", f"unexpected HTTP {code}")
        try:
            return resp.json()
        except ValueError as exc:
            raise _GitHubError("error", "invalid JSON from GitHub") from exc

    def _summarize(self, username: str) -> GitHubSummary:
        user = self._get(f"/users/{username}")
        repos = self._get(f"/users/{username}/repos",
                          {"per_page": 100, "sort": "pushed", "type": "owner"})
        if not isinstance(user, dict) or not isinstance(repos, list):
            raise _GitHubError("error", "unexpected response shape")
        return summarize_repos(username, user.get("public_repos"), repos, self._now())


def _pushed(repo: dict) -> Optional[datetime]:
    raw = repo.get("pushed_at")
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")) if raw else None
    except (AttributeError, ValueError):
        return None


def summarize_repos(username: str, public_repos: Optional[int], repos: list, now: datetime) -> GitHubSummary:
    """Pure function: repo list -> counts and 0-10 score (easy to unit test)."""
    own = [r for r in repos if isinstance(r, dict) and not r.get("fork")]
    recent = [r for r in own if (p := _pushed(r)) and now - p <= timedelta(days=90)]
    maintained = [r for r in own if not r.get("archived") and (p := _pushed(r)) and now - p <= timedelta(days=365)]

    def is_ai(r: dict) -> bool:
        blob = " ".join([r.get("name") or "", r.get("description") or "", " ".join(r.get("topics") or [])])
        return bool(_AI_REPO.search(blob))

    python_repos = [r for r in own if r.get("language") == "Python"]
    ai_repos = [r for r in own if is_ai(r)]
    relevant = [r for r in maintained if r.get("language") == "Python" or is_ai(r)]

    activity = next((pts for minimum, pts in _ACTIVITY_POINTS if len(recent) >= minimum), 0)
    repo_pts = _REPO_POINTS.get(len(relevant), 5)
    if any(is_ai(r) for r in relevant) and repo_pts < 5:
        repo_pts += 1  # a maintained AI/ML repo is the most relevant signal
    return GitHubSummary(
        status="ok", username=username, public_repos=public_repos, recent_repos=len(recent),
        maintained_repos=len(maintained), python_repos=len(python_repos), ai_repos=len(ai_repos),
        relevant_maintained=[r.get("name", "") for r in relevant[:5]],
        activity_score=activity, repos_score=repo_pts,
    )


def github_evidence(summary: GitHubSummary) -> list[Evidence]:
    if summary.status != "ok":
        return []
    items = [Evidence("github_activity", "github",
                      f"{summary.recent_repos} own repositories pushed in the last 90 days "
                      f"({summary.public_repos if summary.public_repos is not None else '?'} public total)",
                      1.0, EvidenceLevel.IMPLEMENTATION, "github", "github")]
    if summary.relevant_maintained:
        items.append(Evidence("github_relevant_repos", "github",
                              "Maintained Python/AI repositories: " + ", ".join(summary.relevant_maintained),
                              1.0, EvidenceLevel.IMPLEMENTATION, "github", "github"))
    return items
