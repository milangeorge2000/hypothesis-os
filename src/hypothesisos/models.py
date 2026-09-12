"""Model interface. The developer brings their own LLM; we only need text in/out."""
from __future__ import annotations

import json
import re
from typing import Protocol


class Model(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        """Return raw text completion."""
        ...


def complete_json(model: Model, system: str, user: str, max_tokens: int = 800) -> dict:
    """Ask for JSON, tolerate markdown fences and trailing prose."""
    text = model.complete(system, user, max_tokens=max_tokens)
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Model did not return JSON: {text[:300]}")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Model JSON must be an object")
    return obj


class HeuristicModel:
    """Offline deterministic stand-in. Used for tests and when no API key exists.

    It does NOT call any LLM. Engines fall back to templates when they detect
    this class, so results are fully reproducible.
    """

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        return '{"note": "heuristic model: engines use templates instead"}'
