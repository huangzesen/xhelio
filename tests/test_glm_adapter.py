"""Tests for GLMAdapter."""
from agent.llm.glm.adapter import GLMAdapter, DEFAULTS


def test_glm_defaults():
    assert DEFAULTS["api_compat"] == "openai"
    assert DEFAULTS["base_url"] == "https://api.z.ai"
    assert DEFAULTS["api_key_env"] == "GLM_API_KEY"
    assert DEFAULTS["web_search_provider"] == "glm"
    assert DEFAULTS["vision_provider"] == "glm"


def test_glm_capability_flags():
    assert GLMAdapter.supports_web_search is True
    assert GLMAdapter.supports_vision is True


def test_glm_inherits_openai():
    from agent.llm.openai.adapter import OpenAIAdapter
    assert issubclass(GLMAdapter, OpenAIAdapter)
