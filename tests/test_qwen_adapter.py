"""Tests for QwenAdapter."""
from agent.llm.qwen.adapter import QwenAdapter, DEFAULTS


def test_qwen_defaults():
    assert DEFAULTS["api_compat"] == "openai"
    assert DEFAULTS["base_url"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert DEFAULTS["api_key_env"] == "QWEN_API_KEY"
    assert DEFAULTS["web_search_provider"] == "qwen"
    assert DEFAULTS["vision_provider"] == "qwen"


def test_qwen_capability_flags():
    assert QwenAdapter.supports_web_search is True
    assert QwenAdapter.supports_vision is True


def test_qwen_inherits_openai():
    from agent.llm.openai.adapter import OpenAIAdapter
    assert issubclass(QwenAdapter, OpenAIAdapter)
