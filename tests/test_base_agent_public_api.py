"""Tests for BaseAgent public API methods used by session persistence."""
import threading
from unittest.mock import MagicMock, patch

from agent.base_agent import BaseAgent
from agent.session_context import SessionContext


def _make_mock_session_ctx():
    """Minimal SessionContext for BaseAgent construction."""
    ctx = MagicMock(spec=SessionContext)
    ctx.event_bus = MagicMock()
    ctx.cancel_event = threading.Event()
    return ctx


def _make_agent(**kwargs):
    """Create a minimal BaseAgent for testing."""
    ctx = _make_mock_session_ctx()
    service = MagicMock()
    service.provider = "mock"
    return BaseAgent(
        agent_id="test-agent",
        service=service,
        tool_schemas=[],
        system_prompt="test",
        session_ctx=ctx,
        event_bus=ctx.event_bus,
        cancel_event=ctx.cancel_event,
        **kwargs,
    )


class TestGetChatState:
    def test_returns_empty_when_no_chat(self):
        agent = _make_agent()
        state = agent.get_chat_state()
        assert state == {}

    def test_returns_serialized_interface(self):
        agent = _make_agent()
        agent._chat = MagicMock()
        agent._chat.interface.to_dict.return_value = [{"role": "user", "content": "hi"}]
        state = agent.get_chat_state()
        assert state["messages"] == [{"role": "user", "content": "hi"}]


class TestRestoreChat:
    def test_resume_from_saved_state(self):
        agent = _make_agent()
        agent.service.resume_session.return_value = MagicMock()
        agent.restore_chat({"session_id": "abc", "messages": [{"role": "user"}]})
        agent.service.resume_session.assert_called_once()
        assert agent._chat is not None

    def test_creates_fresh_on_empty_state(self):
        agent = _make_agent()
        agent.service.create_session.return_value = MagicMock()
        agent.restore_chat({})
        agent.service.create_session.assert_called_once()

    def test_creates_fresh_on_resume_failure(self):
        agent = _make_agent()
        agent.service.resume_session.side_effect = RuntimeError("fail")
        agent.service.create_session.return_value = MagicMock()
        agent.restore_chat({"session_id": "abc", "messages": [{"role": "user"}]})
        agent.service.create_session.assert_called_once()


class TestRestoreTokenState:
    def test_restores_all_counters(self):
        agent = _make_agent()
        agent.restore_token_state({
            "input_tokens": 100,
            "output_tokens": 200,
            "thinking_tokens": 50,
            "cached_tokens": 30,
            "api_calls": 5,
        })
        usage = agent.get_token_usage()
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 200
        assert usage["thinking_tokens"] == 50
        assert usage["cached_tokens"] == 30
        assert usage["api_calls"] == 5

    def test_ignores_missing_keys(self):
        agent = _make_agent()
        agent.restore_token_state({})
        usage = agent.get_token_usage()
        assert usage["input_tokens"] == 0
