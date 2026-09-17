from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from llmops_core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clean_settings_cache() -> Iterator[None]:
    """Each test sees a fresh Settings() so monkeypatched env vars take effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def test_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("LLMOPS_APP__ENV", "test")
    monkeypatch.setenv("LLMOPS_APP__LOG_FORMAT", "console")
    monkeypatch.setenv("LLMOPS_LLM__PROVIDER", "mock")
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def client(test_settings: Settings) -> Iterator[TestClient]:
    from llm_application.main import create_app

    with TestClient(create_app(test_settings)) as c:
        yield c
