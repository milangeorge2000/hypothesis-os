"""Model providers. One-line setup for every major LLM API, stdlib only.

    from hypothesisos import create_model

    model = create_model()                      # auto-detect from env vars
    model = create_model("anthropic")           # explicit provider
    model = create_model("openai", model="gpt-4o-mini", api_key="sk-...")

Keys are read from the environment by default and never logged.
"""
from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .models import HeuristicModel


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover
        return None


def _post(url: str, payload: Dict[str, Any], headers: Dict[str, str],
          timeout: float) -> Dict[str, Any]:
    """POST JSON with retry on rate limits / transient 5xx. Returns parsed body."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json", **headers},
                                 method="POST")
    last: Optional[Exception] = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 ** attempt * 2)
                continue
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                detail = ""
            raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    raise last  # pragma: no cover


class OpenAICompatibleModel:
    """Any OpenAI-style /chat/completions endpoint (OpenAI, Groq, OpenRouter,
    Together, Fireworks, vLLM, LiteLLM proxy, ...)."""

    completions_path = "/chat/completions"

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 45):
        if not api_key or "replace" in api_key:
            raise ValueError(f"Missing API key for {type(self).__name__}")
        self._key = api_key
        self._model = model
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        resp = _post(
            self._base + self.completions_path,
            {"model": self._model,
             "messages": [{"role": "system", "content": system},
                          {"role": "user", "content": user}],
             "max_tokens": max_tokens, "temperature": 0.2},
            {"Authorization": "Bearer " + self._key},
            self._timeout)
        try:
            return resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response shape: {str(resp)[:300]}") from exc


class OpenAIModel(OpenAICompatibleModel):
    """api_key=$OPENAI_API_KEY, model=$OPENAI_MODEL (default gpt-4o-mini)."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: float = 45):
        super().__init__(
            api_key or os.getenv("OPENAI_API_KEY", ""),
            model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout)


class GroqModel(OpenAICompatibleModel):
    """api_key=$GROQ_API_KEY, model=$GROQ_MODEL (default llama-3.3-70b-versatile)."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: float = 45):
        super().__init__(
            api_key or os.getenv("GROQ_API_KEY", ""),
            model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            timeout)


class OpenRouterModel(OpenAICompatibleModel):
    """api_key=$OPENROUTER_API_KEY, model=$OPENROUTER_MODEL."""

    def __init__(self, api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: float = 45):
        super().__init__(
            api_key or os.getenv("OPENROUTER_API_KEY", ""),
            model or os.getenv("OPENROUTER_MODEL", "x-ai/grok-4.3"),
            base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout)


class AnthropicModel:
    """api_key=$ANTHROPIC_API_KEY, model=$ANTHROPIC_MODEL (default haiku)."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: float = 45):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key or "replace" in api_key:
            raise ValueError("Missing API key for AnthropicModel ($ANTHROPIC_API_KEY)")
        self._key = api_key
        self._model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        self._base = (base_url or os.getenv("ANTHROPIC_BASE_URL",
                                            "https://api.anthropic.com")).rstrip("/")
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        resp = _post(
            self._base + "/v1/messages",
            {"model": self._model, "max_tokens": max_tokens,
             "system": system,
             "messages": [{"role": "user", "content": user}]},
            {"x-api-key": self._key, "anthropic-version": "2023-06-01"},
            self._timeout)
        try:
            return "".join(b.get("text", "") for b in resp["content"]
                           if isinstance(b, dict) and b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response shape: {str(resp)[:300]}") from exc


class GeminiModel:
    """api_key=$GEMINI_API_KEY (or $GOOGLE_API_KEY), model=$GEMINI_MODEL."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 timeout: float = 45):
        api_key = (api_key or os.getenv("GEMINI_API_KEY", "")
                   or os.getenv("GOOGLE_API_KEY", ""))
        if not api_key or "replace" in api_key:
            raise ValueError("Missing API key for GeminiModel ($GEMINI_API_KEY)")
        self._key = api_key
        self._model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        resp = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent",
            {"system_instruction": {"parts": [{"text": system}]},
             "contents": [{"parts": [{"text": user}]}],
             "generationConfig": {"temperature": 0.2,
                                  "maxOutputTokens": max_tokens}},
            {"x-goog-api-key": self._key},
            self._timeout)
        try:
            parts = resp["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response shape: {str(resp)[:300]}") from exc


class OllamaModel:
    """Local models via Ollama. No key. host=$OLLAMA_HOST, model=$OLLAMA_MODEL."""

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None,
                 timeout: float = 120):
        self._model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        self._host = (host or os.getenv("OLLAMA_HOST",
                                        "http://localhost:11434")).rstrip("/")
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        resp = _post(
            self._host + "/api/chat",
            {"model": self._model, "stream": False,
             "messages": [{"role": "system", "content": system},
                          {"role": "user", "content": user}],
             "options": {"temperature": 0.2, "num_predict": max_tokens}},
            {}, self._timeout)
        try:
            return resp["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response shape: {str(resp)[:300]}") from exc


_PROVIDERS = {
    "openai": OpenAIModel,
    "groq": GroqModel,
    "openrouter": OpenRouterModel,
    "anthropic": AnthropicModel,
    "claude": AnthropicModel,
    "gemini": GeminiModel,
    "google": GeminiModel,
    "ollama": OllamaModel,
    "local": OllamaModel,
    "heuristic": HeuristicModel,
    "offline": HeuristicModel,
}


def create_model(provider: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, **kwargs):
    """Build a model for HypothesisOS.

    provider: openai | anthropic | gemini | groq | openrouter | ollama |
              heuristic. Aliases: claude, google, local, offline.
              None = auto-detect from env (OPENAI_API_KEY, ANTHROPIC_API_KEY,
              GEMINI_API_KEY/GOOGLE_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY,
              OLLAMA_HOST). No keys found -> offline HeuristicModel.
    OpenAI-compatible proxies: create_model("openai", base_url="http://...").
    """
    if provider is not None:
        cls = _PROVIDERS.get(provider.strip().lower())
        if cls is None:
            raise ValueError(f"Unknown provider {provider!r}. "
                             f"Choose from {sorted(_PROVIDERS)}")
        if cls is HeuristicModel:
            return HeuristicModel()
        kw = dict(kwargs)
        if model is not None:
            kw["model"] = model
        if api_key is not None and cls is not OllamaModel:
            kw["api_key"] = api_key
        return cls(**kw)
    if model is not None:  # model id given, provider inferred elsewhere -> compat
        return OpenAICompatibleModel(
            api_key or os.getenv("OPENAI_API_KEY", ""), model,
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    for env_var, cls in (("OPENAI_API_KEY", OpenAIModel),
                         ("ANTHROPIC_API_KEY", AnthropicModel),
                         ("GEMINI_API_KEY", GeminiModel),
                         ("GOOGLE_API_KEY", GeminiModel),
                         ("GROQ_API_KEY", GroqModel),
                         ("OPENROUTER_API_KEY", OpenRouterModel)):
        if os.getenv(env_var):
            return cls()
    if os.getenv("OLLAMA_HOST"):
        return OllamaModel()
    return HeuristicModel()  # offline default: reproducible, no network
