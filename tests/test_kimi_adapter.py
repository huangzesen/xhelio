"""Tests for KimiAdapter."""
from agent.llm.kimi.adapter import DEFAULTS, create_kimi_adapter


def test_kimi_defaults():
    assert DEFAULTS["api_compat"] == "openai"
    assert DEFAULTS["base_url"] == "https://api.moonshot.ai/v1"
    assert DEFAULTS["api_key_env"] == "KIMI_API_KEY"
    assert DEFAULTS["web_search_provider"] == "kimi"
    assert DEFAULTS["vision_provider"] == "kimi"


def test_kimi_has_anthropic_base_url():
    assert "base_url_anthropic" in DEFAULTS
