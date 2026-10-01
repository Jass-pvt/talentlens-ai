"""Provider-independent LLM interface plus concrete providers (Anthropic & Gemini).

The pipeline only sees `LLMAdapter`. Failure policy (enforced by the caller):
    LLM failure -> deterministic fallback -> continue
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

import requests

from src.ai.prompts import SYSTEM_PROMPT, build_user_prompt
from src.ai.schemas import ProjectAnalysis, SchemaError
from src.config import Config

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Any failure talking to, or parsing output from, a provider."""


class LLMAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def extract_project_evidence(self, candidate_name: str, project_text: str) -> ProjectAnalysis:
        """Return validated structured analysis or raise LLMError."""


class AnthropicAdapter(LLMAdapter):
    name = "anthropic"
    _URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str, timeout: float = 30.0,
                 session: Optional[requests.Session] = None) -> None:
        self._api_key, self._model, self._timeout = api_key, model, timeout
        self._session = session or requests.Session()

    def extract_project_evidence(self, candidate_name: str, project_text: str) -> ProjectAnalysis:
        payload = {
            "model": self._model,
            "max_tokens": 1000,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_user_prompt(candidate_name, project_text)}],
        }
        headers = {"x-api-key": self._api_key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        try:
            resp = self._session.post(self._URL, headers=headers, json=payload, timeout=self._timeout)
            if resp.status_code != 200:
                raise LLMError(f"HTTP {resp.status_code}")
            body = resp.json()
            text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
            return ProjectAnalysis.from_dict(_loads_json(text))
        except LLMError:
            raise
        except (requests.RequestException, ValueError, SchemaError, AttributeError, TypeError) as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc


class GeminiAdapter(LLMAdapter):
    name = "gemini"
    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str, timeout: float = 30.0,
                 session: Optional[requests.Session] = None) -> None:
        self._api_key, self._model, self._timeout = api_key, model, timeout
        self._session = session or requests.Session()

    def extract_project_evidence(self, candidate_name: str, project_text: str) -> ProjectAnalysis:
        model_name = self._model.replace("models/", "")
        url = f"{self._BASE_URL}/{model_name}:generateContent"
        params = {"key": self._api_key}
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": build_user_prompt(candidate_name, project_text)}]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        try:
            resp = self._session.post(url, headers=headers, params=params, json=payload, timeout=self._timeout)
            if resp.status_code != 200:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text}")
            body = resp.json()
            candidates = body.get("candidates", [])
            if not candidates:
                raise LLMError("No response candidates returned by Gemini API")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts if "text" in p)
            return ProjectAnalysis.from_dict(_loads_json(text))
        except LLMError:
            raise
        except (requests.RequestException, ValueError, SchemaError, AttributeError, TypeError, KeyError) as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc


def _loads_json(text: str) -> object:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


def build_adapter(config: Config) -> Optional[LLMAdapter]:
    """Return an adapter, or None to run deterministic-only."""
    provider = config.llm_provider
    if not provider or provider == "none":
        return None

    if provider == "anthropic":
        if not (config.llm_api_key and config.llm_model):
            log.warning("LLM_PROVIDER=anthropic needs LLM_API_KEY and LLM_MODEL; running without LLM")
            return None
        return AnthropicAdapter(config.llm_api_key, config.llm_model, config.llm_timeout)

    if provider in ("gemini", "google"):
        model = config.llm_model or "gemini-2.5-flash"
        if not config.llm_api_key:
            log.warning("LLM_PROVIDER=gemini needs LLM_API_KEY; running without LLM")
            return None
        return GeminiAdapter(config.llm_api_key, model, config.llm_timeout)

    log.warning("Unknown LLM_PROVIDER %r; running without LLM", provider)
    return None