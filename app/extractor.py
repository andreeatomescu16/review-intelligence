"""
extractor.py

Calls the local Ollama LLM to extract highlights and pain points from a
single hotel review using a single combined call.

The system prompt uses an inline example (not message-pair few-shot) because
for llama3.2:1b, shorter context in a single system message produces more
reliable output than spreading examples across multiple messages.
"""

import asyncio
import json
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_BACKOFF_BASE_SECONDS = 1.0
_MODEL = "llama3.2:1b"

_SYSTEM_PROMPT = (
    "You are a hotel review analyst. "
    "Given a hotel review, extract ALL positive points as highlights and ALL "
    "negative points as pain points. "
    "When you see contrast words like 'but', 'although', 'however', 'though': "
    "the part BEFORE is a highlight, the part AFTER is a pain point. "
    "Example: 'Staff were friendly, although check-in was slow' → "
    "highlights: ['Friendly staff'], pain_points: ['Slow check-in']. "
    "Return ONLY valid JSON — no prose, no markdown, no code fences — with "
    "exactly two keys:\n"
    '  "highlights": a JSON array of strings, each a concise positive point\n'
    '  "pain_points": a JSON array of strings, each a concise negative point\n'
    "If a category has no items, return an empty array for that key."
)

_EMPTY_RESULT: dict[str, list[str]] = {"highlights": [], "pain_points": []}

_LEADING_MARKER = re.compile(
    r"^(but|however|although|though|yet|while|despite)[,\s]+",
    re.IGNORECASE,
)


def _parse_items(raw: list) -> list[str]:
    """Flatten and clean extracted items, handling nested lists and non-strings."""
    result = []
    for item in raw:
        if isinstance(item, list):
            for sub in item:
                text = _LEADING_MARKER.sub("", str(sub)).strip()
                if text:
                    result.append(text)
        else:
            text = _LEADING_MARKER.sub("", str(item)).strip()
            if text:
                result.append(text)
    return result


async def extract_from_review(
    review: str,
    ollama_host: str,
) -> dict[str, list[str]]:
    url = f"{ollama_host}/api/chat"
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
            backoff = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Retrying extract_from_review (attempt %d/%d) after %.1fs.",
                attempt + 1, _MAX_RETRIES + 1, backoff,
            )
            await asyncio.sleep(backoff)

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()

            raw_content: str = response.json()["message"]["content"]
            parsed = json.loads(raw_content)

            highlights = parsed.get("highlights", [])
            pain_points = parsed.get("pain_points", [])

            if not isinstance(highlights, list) or not isinstance(pain_points, list):
                raise ValueError(
                    f"Unexpected types: highlights={type(highlights)}, "
                    f"pain_points={type(pain_points)}"
                )

            return {
                "highlights": _parse_items(highlights),
                "pain_points": _parse_items(pain_points),
            }

        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error for '%s...': %s", review[:60], exc)
            last_exception = exc
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning(
                "Request error on attempt %d for '%s...': %s",
                attempt + 1, review[:60], exc,
            )
            last_exception = exc

    logger.error(
        "All attempts failed for '%s...'. Last error: %s", review[:60], last_exception
    )
    return _EMPTY_RESULT
