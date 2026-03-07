"""Tests for DeepSeekAdapter."""
from agent.llm.deepseek.adapter import DeepSeekAdapter, DEFAULTS


def test_deepseek_defaults():
    assert DEFAULTS["api_compat"] == "openai"
    assert DEFAULTS["base_url"] == "https://api.deepseek.com"
    assert DEFAULTS["api_key_env"] == "DEEPSEEK_API_KEY"
    assert DEFAULTS["web_search_provider"] is None
    assert DEFAULTS["vision_provider"] is None


def test_deepseek_capability_flags():
    assert DeepSeekAdapter.supports_web_search is False
    assert DeepSeekAdapter.supports_vision is False


def test_deepseek_inherits_openai():
    from agent.llm.openai.adapter import OpenAIAdapter
    assert issubclass(DeepSeekAdapter, OpenAIAdapter)
