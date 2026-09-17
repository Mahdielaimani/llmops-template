from __future__ import annotations

import pytest
from pydantic import SecretStr

from llmops_core.config import Environment, LLMProviderKind, Settings, get_settings


def test_defaults_are_safe_for_local_mock() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.app.env is Environment.LOCAL
    assert s.llm.provider is LLMProviderKind.MOCK
    assert s.llm.api_key is None
    assert s.server.port == 8080


def test_nested_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMOPS_SERVER__PORT", "9001")
    monkeypatch.setenv("LLMOPS_LLM__PROVIDER", "vllm")
    monkeypatch.setenv("LLMOPS_LLM__BASE_URL", "http://vllm:8000/v1")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.server.port == 9001
    assert s.llm.provider is LLMProviderKind.VLLM
    assert s.llm.base_url == "http://vllm:8000/v1"


def test_invalid_provider_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMOPS_LLM__PROVIDER", "gpt-magic")
    with pytest.raises(ValueError, match="provider"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_secrets_never_leak_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMOPS_LLM__API_KEY", "sk-super-secret")
    monkeypatch.setenv("LLMOPS_POSTGRES__PASSWORD", "pg-secret")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    dumped = repr(s) + s.model_dump_json()
    assert "sk-super-secret" not in dumped
    assert "pg-secret" not in dumped
    assert isinstance(s.llm.api_key, SecretStr)
    assert s.llm.api_key.get_secret_value() == "sk-super-secret"
    # DSN is the one deliberate place the password is materialised.
    assert "pg-secret" in s.postgres.dsn


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMOPS_SERVER__PORT", "9100")
    a = get_settings()
    monkeypatch.setenv("LLMOPS_SERVER__PORT", "9200")
    b = get_settings()
    assert a is b and a.server.port == 9100
    get_settings.cache_clear()
    assert get_settings().server.port == 9200
