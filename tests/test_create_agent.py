"""Tests for the create_agent factory function and core.py utilities."""


def test_sanitize_for_json():
    """_sanitize_for_json replaces NaN/Inf with None."""
    import math
    from agent.core import _sanitize_for_json

    assert _sanitize_for_json(float("nan")) is None
    assert _sanitize_for_json(float("inf")) is None
    assert _sanitize_for_json(42.0) == 42.0
    assert _sanitize_for_json({"a": float("nan"), "b": 1}) == {"a": None, "b": 1}
    assert _sanitize_for_json([float("inf"), 2]) == [None, 2]


def test_extract_turns_gemini_style():
    """_extract_turns handles Gemini-style history entries."""
    from agent.core import _extract_turns

    history = [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Hi there"}]},
    ]
    turns = _extract_turns(history, max_text=1000)
    assert len(turns) == 2
    assert turns[0].startswith("User: Hello")
    assert turns[1].startswith("Agent: Hi there")


def test_extract_turns_openai_style():
    """_extract_turns handles OpenAI-style history entries."""
    from agent.core import _extract_turns

    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    turns = _extract_turns(history, max_text=1000)
    assert len(turns) == 2


def test_orchestrator_reexport():
    """OrchestratorAgent is re-exported from core for backward compatibility."""
    from agent.core import OrchestratorAgent
    from agent.orchestrator_agent import OrchestratorAgent as DirectImport

    assert OrchestratorAgent is DirectImport
