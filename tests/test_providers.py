"""Provider contract tests: mocked HTTP, no network, no keys."""
import io
import json
import os
import urllib.error
from unittest import mock

import pytest

from hypothesisos import HypothesisOS, create_model
from hypothesisos.models import HeuristicModel
from hypothesisos.providers import (
    AnthropicModel,
    GeminiModel,
    GroqModel,
    OllamaModel,
    OpenAICompatibleModel,
    OpenAIModel,
    OpenRouterModel,
)


class FakeHTTP:
    """Records the request, replays a canned JSON body."""

    def __init__(self, body):
        self.body = body
        self.requests = []

    def __call__(self, req, timeout=None, context=None):
        self.requests.append(req)
        payload = io.BytesIO(json.dumps(self.body).encode())
        resp = mock.MagicMock()
        resp.read.return_value = payload.read()
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        return resp

    @property
    def last(self):
        req = self.requests[-1]
        return {
            "url": req.full_url,
            "body": json.loads(req.data.decode()),
            "get": req.get_header,
        }


def _run(model, body):
    fake = FakeHTTP(body)
    with mock.patch("urllib.request.urlopen", fake):
        text = model.complete("sys", "hello")
    return text, fake.last


def test_openai_request_shape_and_parse():
    m = OpenAIModel(api_key="sk-test", model="gpt-4o-mini")
    text, req = _run(m, {"choices": [{"message": {"content": "hi"}}]})
    assert text == "hi"
    assert req["url"] == "https://api.openai.com/v1/chat/completions"
    assert req["get"]("Authorization") == "Bearer sk-test"
    assert req["body"]["model"] == "gpt-4o-mini"
    assert req["body"]["messages"][0] == {"role": "system", "content": "sys"}


def test_groq_defaults():
    m = GroqModel(api_key="gsk-test")
    _, req = _run(m, {"choices": [{"message": {"content": "x"}}]})
    assert req["url"].startswith("https://api.groq.com/openai/v1")
    assert req["body"]["model"] == "llama-3.3-70b-versatile"


def test_openrouter_backcompat_import():
    from hypothesisos.tools import OpenRouterModel as Reexported
    assert Reexported is OpenRouterModel
    m = OpenRouterModel(api_key="sk-or-test", model="m")
    _, req = _run(m, {"choices": [{"message": {"content": "x"}}]})
    assert req["url"] == "https://openrouter.ai/api/v1/chat/completions"


def test_anthropic_request_shape_and_parse():
    m = AnthropicModel(api_key="sk-ant-test", model="claude-x")
    text, req = _run(m, {"content": [{"type": "text", "text": "yo"},
                                     {"type": "tool_use", "id": "1"}]})
    assert text == "yo"
    assert req["url"] == "https://api.anthropic.com/v1/messages"
    assert req["get"]("X-api-key") == "sk-ant-test"
    assert req["get"]("Anthropic-version") == "2023-06-01"
    assert req["body"]["system"] == "sys"
    assert req["body"]["max_tokens"] == 800


def test_gemini_request_shape_and_parse():
    m = GeminiModel(api_key="AIza-test", model="gemini-2.5-flash")
    text, req = _run(m, {"candidates": [{"content": {"parts": [{"text": "a"},
                                                               {"text": "b"}]}}]})
    assert text == "ab"
    assert req["url"].endswith("/v1beta/models/gemini-2.5-flash:generateContent")
    assert req["get"]("X-goog-api-key") == "AIza-test"
    assert req["body"]["contents"][0]["parts"] == [{"text": "hello"}]


def test_ollama_no_key_and_options():
    m = OllamaModel(model="llama3.1", host="http://box:11434")
    text, req = _run(m, {"message": {"content": "local hi"}})
    assert text == "local hi"
    assert req["url"] == "http://box:11434/api/chat"
    assert req["get"]("Authorization") is None
    assert req["body"]["stream"] is False
    assert req["body"]["options"]["num_predict"] == 800


def test_bad_response_shape_raises():
    m = OpenAIModel(api_key="sk-test")
    with mock.patch("urllib.request.urlopen",
                    FakeHTTP({"unexpected": True})):
        with pytest.raises(RuntimeError):
            m.complete("sys", "hi")


def test_retry_on_429_then_success():
    err = urllib.error.HTTPError("http://x", 429, "slow down", {}, io.BytesIO(b"{}"))
    calls = []

    def fake_urlopen(req, timeout=None, context=None):
        calls.append(req)
        if len(calls) == 1:
            raise err
        return FakeHTTP({"choices": [{"message": {"content": "ok"}}]})(req)

    m = OpenAIModel(api_key="sk-test")
    with mock.patch("urllib.request.urlopen", fake_urlopen), \
         mock.patch("time.sleep", lambda s: None):
        assert m.complete("sys", "hi") == "ok"
    assert len(calls) == 2


def test_missing_key_raises():
    with pytest.raises(ValueError):
        OpenAIModel(api_key="")
    with pytest.raises(ValueError):
        AnthropicModel(api_key="replace-me")
    with pytest.raises(ValueError):
        GeminiModel(api_key=None)


def test_create_model_explicit_and_unknown():
    assert isinstance(create_model("heuristic"), HeuristicModel)
    assert isinstance(create_model("offline"), HeuristicModel)
    assert isinstance(create_model("openai", api_key="sk-x")._model, str)
    with pytest.raises(ValueError):
        create_model("skynet")


def test_create_model_autodetect(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
                "OLLAMA_HOST"):
        monkeypatch.delenv(var, raising=False)
    assert isinstance(create_model(), HeuristicModel)
    monkeypatch.setenv("OLLAMA_HOST", "http://box:11434")
    assert isinstance(create_model(), OllamaModel)
    monkeypatch.delenv("OLLAMA_HOST")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    assert isinstance(create_model(), AnthropicModel)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")  # openai wins priority
    assert isinstance(create_model(), OpenAIModel)


def test_from_env_builds_investigator(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
                "OLLAMA_HOST"):
        monkeypatch.delenv(var, raising=False)
    inv = HypothesisOS.from_env(tools=[])
    assert isinstance(inv.model, HeuristicModel)
    r = inv.investigate("Revenue is down. Why?", context="dashboard -27%")
    assert r.hypotheses and r.observations
