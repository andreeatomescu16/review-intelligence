"""
extractor.py

Responsible for calling the local Ollama LLM to extract highlights (positive
points) and pain points (negative points) from a single hotel review text.

Structured JSON output mode is used (`"format": "json"`) to avoid brittle
regex/string parsing — small models are prone to adding prose around JSON when
not constrained.
"""

import asyncio
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Maximum number of retry attempts after the initial try
_MAX_RETRIES = 2

# Base delay in seconds for exponential backoff (1s → 2s)
_BACKOFF_BASE_SECONDS = 1.0

# Ollama model used for extraction
_MODEL = "llama3.2:1b"

# System prompt instructs the model to return ONLY valid JSON with exactly the
# two required keys. Being explicit about key names and value types reduces
# hallucination of extra keys with small models.
_SYSTEM_PROMPT = (
    "You are a hotel review analyst. "
    "Given a hotel review, extract ALL positive points as highlights and ALL "
    "negative points as pain points. "
    "Return ONLY valid JSON — no prose, no markdown, no code fences — with "
    "exactly two keys:\n"
    '  "highlights": a JSON array of strings, each a concise positive point\n'
    '  "pain_points": a JSON array of strings, each a concise negative point\n'
    "If a category has no items, return an empty array for that key."
)

_EMPTY_RESULT: dict[str, list[str]] = {"highlights": [], "pain_points": []}


async def extract_from_review(
    review: str,
    ollama_host: str,
) -> dict[str, list[str]]:
    """
    Send a single review to Ollama and return extracted highlights and pain
    points.

    Args:
        review:      The raw review text to analyse.
        ollama_host: Base URL of the Ollama service, e.g. http://ollama:11434.

    Returns:
        A dict with keys "highlights" and "pain_points", each a list of
        concise string items. Returns empty lists on unrecoverable failure.
    """
    url = f"{ollama_host}/api/chat"

    # Request payload — "format": "json" instructs Ollama to enforce that the
    # model's output is valid JSON before returning it.
    payload: dict[str, Any] = {
        "model": _MODEL,
        "format": "json",
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": review},
        ],
    }

    last_exception: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            # Exponential backoff: 1s before the 2nd attempt, 2s before the 3rd
            backoff_seconds = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Retrying extract_from_review (attempt %d/%d) after %.1fs backoff.",
                attempt + 1,
                _MAX_RETRIES + 1,
                backoff_seconds,
            )
            await asyncio.sleep(backoff_seconds)

        try:
            # Use a generous timeout: small models can be slow on first tokens
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()

            response_body = response.json()

            # Ollama chat response structure:
            # { "message": { "role": "assistant", "content": "<JSON string>" }, ... }
            raw_content: str = response_body["message"]["content"]

            parsed = json.loads(raw_content)

            # Validate that the expected keys are present and are lists
            highlights = parsed.get("highlights", [])
            pain_points = parsed.get("pain_points", [])

            if not isinstance(highlights, list) or not isinstance(pain_points, list):
                raise ValueError(
                    f"Unexpected types: highlights={type(highlights)}, "
                    f"pain_points={type(pain_points)}"
                )

            # Coerce each element to string to guard against numeric/null items
            return {
                "highlights": [str(item) for item in highlights],
                "pain_points": [str(item) for item in pain_points],
            }

        except json.JSONDecodeError as exc:
            logger.warning(
                "JSON parsing failed for review excerpt '%s...': %s",
                review[:60],
                exc,
            )
            last_exception = exc
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning(
                "Request or response error on attempt %d for review excerpt '%s...': %s",
                attempt + 1,
                review[:60],
                exc,
            )
            last_exception = exc

    logger.error(
        "All %d attempts failed for review excerpt '%s...'. Last error: %s",
        _MAX_RETRIES + 1,
        review[:60],
        last_exception,
    )
    return _EMPTY_RESULT
