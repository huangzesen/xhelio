"""Tests for Custom adapter."""
import pytest
from unittest.mock import patch, MagicMock
from agent.llm.custom.adapter import DEFAULTS, create_custom_adapter


def test_custom_defaults():
    assert DEFAULTS["api_compat"] == "openai"
    assert DEFAULTS["base_url"] is None
    assert DEFAULTS["api_key_env"] == "CUSTOM_API_KEY"
    assert DEFAULTS["web_search_provider"] is None
    assert DEFAULTS["vision_provider"] is None


def test_custom_requires_base_url():
    with pytest.raises(ValueError, match="base_url"):
        create_custom_adapter(api_key="key")


@patch("agent.llm.openai.adapter.openai")
def test_custom_openai_compat(mock_openai_mod):
    mock_openai_mod.OpenAI.return_value = MagicMock()
    adapter = create_custom_adapter(
        api_key="key", base_url="http://localhost:8000/v1"
    )
    from agent.llm.openai.adapter import OpenAIAdapter
    assert isinstance(adapter, OpenAIAdapter)
    assert adapter.supports_web_search is False
    assert adapter.supports_vision is False


@patch("agent.llm.openai.adapter.openai")
def test_custom_with_capabilities(mock_openai_mod):
    mock_openai_mod.OpenAI.return_value = MagicMock()
    adapter = create_custom_adapter(
        api_key="key",
        base_url="http://localhost:8000/v1",
        supports_web_search=True,
        supports_vision=True,
    )
    assert adapter.supports_web_search is True
    assert adapter.supports_vision is True
