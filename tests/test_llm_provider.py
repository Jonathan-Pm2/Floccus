"""LLM provider switch: Ollama stays the default, Anthropic is opt-in via LLM_PROVIDER."""

from __future__ import annotations

import pytest

from biofloc_review.dedup import llm_judge


class _Resp:
    def __init__(self, data: dict):
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._data


MESSAGES = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "USER"},
]


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Keep the real .env (which may set LLM_PROVIDER) out of these tests."""
    monkeypatch.setattr(llm_judge._load_env_once, "_loaded", True, raising=False)


@pytest.fixture
def captured(monkeypatch):
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        if "anthropic" in url:
            return _Resp({"content": [{"type": "text", "text": "VERDICT: DUPLICATE"}]})
        return _Resp({"message": {"content": "VERDICT: NOT_DUPLICATE"}})

    monkeypatch.setattr(llm_judge.requests, "post", fake_post)
    return calls


def test_default_provider_is_ollama_unchanged(monkeypatch, captured):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OLLAMA_URL", "http://ollama.test")
    out = llm_judge._ollama_chat(MESSAGES, model="qwen3:14b")
    assert out == "VERDICT: NOT_DUPLICATE"
    assert captured[0]["url"] == "http://ollama.test/api/chat"
    assert captured[0]["json"] == {
        "model": "qwen3:14b",
        "messages": MESSAGES,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 400},
    }


def test_anthropic_provider_same_prompt_temperature_and_token_cap(monkeypatch, captured):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    out = llm_judge._ollama_chat(MESSAGES, model="claude-test")
    assert out == "VERDICT: DUPLICATE"
    call = captured[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["json"] == {
        "model": "claude-test",
        "system": "SYS",
        "messages": [{"role": "user", "content": "USER"}],
        "temperature": 0.0,
        "max_tokens": 400,
    }
    assert call["headers"]["x-api-key"] == "test-key"
    assert call["headers"]["anthropic-version"] == "2023-06-01"


def test_anthropic_can_omit_temperature_for_models_that_reject_it(monkeypatch, captured):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_TEMPERATURE", "none")
    llm_judge._ollama_chat(MESSAGES, model="claude-test")
    assert "temperature" not in captured[0]["json"]
    assert captured[0]["json"]["max_tokens"] == 400


def test_anthropic_default_model_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-x")
    monkeypatch.setenv("OLLAMA_LLM_MODEL", "qwen3:14b")
    assert llm_judge.default_llm_model() == "claude-x"


def test_ollama_default_model_unchanged(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OLLAMA_LLM_MODEL", "qwen3:14b")
    assert llm_judge.default_llm_model() == "qwen3:14b"


def test_anthropic_without_key_fails_loudly(monkeypatch, captured):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm_judge._ollama_chat.__wrapped__(MESSAGES, model="claude-test")
    assert captured == []
