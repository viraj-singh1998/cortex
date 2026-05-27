from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_PATH = Path.home() / ".cortex" / "config.json"

_PROVIDER_ORDER = ["openai", "anthropic", "openrouter"]
_ENV_KEYS = {
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def read_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            return {}
    return {}


def write_config(data: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, indent=2))


def _resolved_provider() -> str | None:
    # CORTEX_LLM_PROVIDER env var overrides everything except explicit constructor args
    if p := os.environ.get("CORTEX_LLM_PROVIDER"):
        return p.lower()
    cfg = read_config()
    if p := cfg.get("provider"):
        return str(p)
    # Fall back to first API key found in env
    for provider in _PROVIDER_ORDER:
        if os.environ.get(_ENV_KEYS[provider]):
            return provider
    return None


def auto_client() -> tuple[Any, Any]:
    """Return (anthropic_client, openai_client) based on resolved provider.

    Resolution order:
      1. CORTEX_LLM_PROVIDER env var
      2. ~/.cortex/config.json provider preference (set by cortex-memory-configure)
      3. First API key found in env: openai → anthropic → openrouter
      4. (None, None) — heuristic-only mode, no LLM enrichment
    """
    provider = _resolved_provider()

    if provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("pip install cortex-memory[anthropic]")
        return Anthropic(), None

    if provider in ("openai", "openrouter"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install cortex-memory[openai]")
        if provider == "openrouter":
            return None, OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            )
        return None, OpenAI()

    return None, None
