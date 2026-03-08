"""Tests for TokenTracker."""
from unittest.mock import MagicMock
from agent.token_tracker import TokenTracker


def _make_response(input_t=100, output_t=50, thinking_t=10, cached_t=20):
    """Create a mock LLMResponse with usage data."""
    resp = MagicMock()
    resp.usage.input_tokens = input_t
    resp.usage.output_tokens = output_t
    resp.usage.thinking_tokens = thinking_t
    resp.usage.cached_tokens = cached_t
    return resp


def test_initial_state():
    t = TokenTracker("test")
    usage = t.get_usage()
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["thinking_tokens"] == 0
    assert usage["cached_tokens"] == 0
    assert usage["api_calls"] == 0
    assert t.latest_input_tokens == 0
    assert t.api_calls == 0


def test_track_accumulates(monkeypatch):
    # Mock track_llm_usage to just update the state dict
    def fake_track(response, token_state, agent_name, last_tool_context="", system_tokens=0, tools_tokens=0):
        token_state["input"] += response.usage.input_tokens
        token_state["output"] += response.usage.output_tokens
        token_state["thinking"] += response.usage.thinking_tokens
        token_state["cached"] += response.usage.cached_tokens
        token_state["api_calls"] += 1

    monkeypatch.setattr("agent.token_tracker.track_llm_usage", fake_track)

    t = TokenTracker("orchestrator")
    resp = _make_response(100, 50, 10, 20)
    t.track(resp)

    usage = t.get_usage()
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    assert usage["thinking_tokens"] == 10
    assert usage["cached_tokens"] == 20
    assert usage["api_calls"] == 1
    assert t.latest_input_tokens == 100


def test_track_multiple(monkeypatch):
    def fake_track(response, token_state, **kwargs):
        token_state["input"] += response.usage.input_tokens
        token_state["output"] += response.usage.output_tokens
        token_state["thinking"] += response.usage.thinking_tokens
        token_state["cached"] += response.usage.cached_tokens
        token_state["api_calls"] += 1

    monkeypatch.setattr("agent.token_tracker.track_llm_usage", fake_track)

    t = TokenTracker("orch")
    t.track(_make_response(100, 50, 10, 20))
    t.track(_make_response(200, 100, 20, 40))

    usage = t.get_usage()
    assert usage["input_tokens"] == 300
    assert usage["output_tokens"] == 150
    assert usage["thinking_tokens"] == 30
    assert usage["cached_tokens"] == 60
    assert usage["api_calls"] == 2
    assert t.latest_input_tokens == 200  # latest, not cumulative


def test_restore():
    t = TokenTracker("test")
    t.restore({"input_tokens": 500, "output_tokens": 200, "thinking_tokens": 30, "cached_tokens": 10, "api_calls": 3})
    usage = t.get_usage()
    assert usage["input_tokens"] == 500
    assert usage["output_tokens"] == 200
    assert usage["thinking_tokens"] == 30
    assert usage["cached_tokens"] == 10
    assert usage["api_calls"] == 3


def test_restore_missing_keys():
    t = TokenTracker("test")
    t.restore({})  # All should default to 0
    usage = t.get_usage()
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["thinking_tokens"] == 0
    assert usage["cached_tokens"] == 0
    assert usage["api_calls"] == 0


def test_restore_partial_keys():
    t = TokenTracker("test")
    t.restore({"input_tokens": 100, "api_calls": 2})
    usage = t.get_usage()
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 0
    assert usage["api_calls"] == 2


def test_agent_name_stored():
    t = TokenTracker("MyAgent")
    assert t.agent_name == "MyAgent"
