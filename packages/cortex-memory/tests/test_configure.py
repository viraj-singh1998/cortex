import json
from pathlib import Path

import pytest

import cortex.memory._client as client_module
from cortex.memory._client import _resolved_provider, read_config, write_config


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Redirect CONFIG_PATH to a temp dir and clear LLM env vars for every test."""
    config_file = tmp_path / "config.json"
    monkeypatch.setattr(client_module, "CONFIG_PATH", config_file)
    for var in ("CORTEX_LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield config_file


class TestReadWriteConfig:
    def test_read_missing_returns_empty(self):
        assert read_config() == {}

    def test_write_then_read_roundtrips(self, isolated_config):
        write_config({"provider": "openai"})
        assert read_config() == {"provider": "openai"}

    def test_write_creates_parent_dirs(self, tmp_path, monkeypatch):
        deep = tmp_path / "a" / "b" / "config.json"
        monkeypatch.setattr(client_module, "CONFIG_PATH", deep)
        write_config({"provider": "anthropic"})
        assert deep.exists()

    def test_corrupt_config_returns_empty(self, isolated_config):
        isolated_config.write_text("not json")
        assert read_config() == {}


class TestResolvedProvider:
    def test_env_var_wins_over_config(self, isolated_config, monkeypatch):
        write_config({"provider": "openai"})
        monkeypatch.setenv("CORTEX_LLM_PROVIDER", "anthropic")
        assert _resolved_provider() == "anthropic"

    def test_env_var_is_lowercased(self, monkeypatch):
        monkeypatch.setenv("CORTEX_LLM_PROVIDER", "OPENAI")
        assert _resolved_provider() == "openai"

    def test_config_file_used_when_no_env_var(self, isolated_config):
        write_config({"provider": "openrouter"})
        assert _resolved_provider() == "openrouter"

    def test_falls_back_to_first_api_key_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert _resolved_provider() == "openai"

    def test_falls_back_to_anthropic_if_no_openai(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert _resolved_provider() == "anthropic"

    def test_falls_back_to_openrouter_last(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        assert _resolved_provider() == "openrouter"

    def test_openai_beats_anthropic_in_fallback_order(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert _resolved_provider() == "openai"

    def test_returns_none_when_nothing_set(self):
        assert _resolved_provider() is None
