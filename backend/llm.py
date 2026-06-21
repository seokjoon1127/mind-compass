"""OpenAI integration layer.

All calls are server-side only. The API key comes from settings (loaded from .env)
and is NEVER logged or returned to the client. Every call uses temperature 0 and
JSON mode, with one retry on malformed JSON.
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from . import prompts
from .config import settings

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.has_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing or malformed. Put it in the .env file."
            )
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def _chat_json(system: str, user: str, max_attempts: int = 2) -> dict[str, Any]:
    """Call the chat API expecting a JSON object. Retries once if parsing fails."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_err: Exception | None = None
    for _ in range(max_attempts):
        resp = client().chat.completions.create(
            model=settings.openai_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = resp.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            last_err = exc
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": "That was not valid JSON. Reply with ONLY a valid JSON object.",
                }
            )
    raise RuntimeError(f"LLM did not return valid JSON: {last_err}")


# --------------------------------------------------------------------------- #
# Public functions — return plain dicts/lists; the orchestrator builds models.
# --------------------------------------------------------------------------- #

def extract_options(context: str) -> dict[str, Any]:
    system, user = prompts.extract_options(context)
    data = _chat_json(system, user)
    return {
        "context_summary": str(data.get("context_summary", "")).strip(),
        "options": data.get("options", []) or [],
    }


def decide_factors(options: list[dict], context_summary: str) -> list[dict]:
    system, user = prompts.decide_factors(options, context_summary)
    data = _chat_json(system, user)
    return data.get("factors", []) or []


def decompose_indicators(abstract_factors: list[dict], context_summary: str) -> list[dict]:
    if not abstract_factors:
        return []
    system, user = prompts.decompose_indicators(abstract_factors, context_summary)
    data = _chat_json(system, user)
    return data.get("indicators", []) or []


def verbalize_pairs(factors: list[dict], context_summary: str) -> list[dict]:
    if len(factors) < 2:
        return []
    system, user = prompts.verbalize_pairs(factors, context_summary)
    data = _chat_json(system, user)
    return data.get("pairs", []) or []


def score_options(options: list[dict], factors: list[dict], context_summary: str) -> dict[str, Any]:
    system, user = prompts.score_options(options, factors, context_summary)
    data = _chat_json(system, user)
    return data.get("scores", {}) or {}


def decompose_question(
    factor_a: dict, factor_b: dict, context_summary: str,
    asked_so_far: list[str], depth: int,
) -> dict[str, Any]:
    system, user = prompts.decompose_question(factor_a, factor_b, context_summary, asked_so_far, depth)
    data = _chat_json(system, user)
    return {
        "question": str(data.get("question", "")).strip(),
        "options": data.get("options", []) or [],
    }


def generate_report(result_data: dict[str, Any]) -> dict[str, Any]:
    system, user = prompts.generate_report(result_data)
    data = _chat_json(system, user)
    return {
        "report": str(data.get("report", "")).strip(),
        "next_info": data.get("next_info", []) or [],
    }
