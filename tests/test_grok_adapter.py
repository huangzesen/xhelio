"""Tests for GrokAdapter — thin subclass of OpenAIAdapter."""
import pytest
from unittest.mock import patch, MagicMock

from agent.llm.grok.adapter import GrokAdapter, DEFAULTS


def test_grok_defaults():
    """DEFAULTS has required fields."""
    assert DEFAULTS["api_compat"] == "openai"
    assert DEFAULTS["base_url"] == "https://api.x.ai/v1"
    assert DEFAULTS["api_key_env"] == "GROK_API_KEY"
    assert DEFAULTS["web_search_provider"] == "grok"
    assert DEFAULTS["vision_provider"] == "grok"


def test_grok_adapter_inherits_openai():
    """GrokAdapter is a subclass of OpenAIAdapter."""
    from agent.llm.openai.adapter import OpenAIAdapter
    assert issubclass(GrokAdapter, OpenAIAdapter)


def test_grok_capability_flags():
    """GrokAdapter declares web search and vision support."""
    assert GrokAdapter.supports_web_search is True
    assert GrokAdapter.supports_vision is True


@patch("agent.llm.openai.adapter.openai")
def test_grok_adapter_uses_correct_base_url(mock_openai_mod):
    """GrokAdapter passes the Grok base URL to the OpenAI client."""
    mock_openai_mod.OpenAI.return_value = MagicMock()
    adapter = GrokAdapter(api_key="test-key")
    mock_openai_mod.OpenAI.assert_called_once()
    call_kwargs = mock_openai_mod.OpenAI.call_args
    assert call_kwargs.kwargs.get("base_url") == "https://api.x.ai/v1"
