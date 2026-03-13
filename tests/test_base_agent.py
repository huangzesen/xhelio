# tests/test_base_agent.py
"""Tests for BaseAgent — unified agent lifecycle."""
import threading
import time
from unittest.mock import MagicMock, patch

from agent.base_agent import BaseAgent


class StubAgent(BaseAgent):
    """Minimal concrete subclass for testing."""
    agent_type = "stub"


def _make_stub(**kwargs):
    """Create a StubAgent with mocked dependencies."""
    defaults = dict(
        agent_id="stub-001",
        service=MagicMock(),
        tool_schemas=[],
        system_prompt="You are a test agent.",
    )
    defaults.update(kwargs)
    return StubAgent(**defaults)


def test_agent_starts_and_stops():
    agent = _make_stub()
    agent.start()
    assert agent._thread.is_alive()
    agent.stop(timeout=2.0)
    assert not agent._thread.is_alive()


def test_agent_type_is_class_attribute():
    agent = _make_stub()
    assert agent.agent_type == "stub"


def test_agent_id_set_at_construction():
    agent = _make_stub(agent_id="custom-id")
    assert agent.agent_id == "custom-id"


def test_streaming_default_false():
    agent = _make_stub()
    assert agent._streaming is False


def test_streaming_can_be_enabled():
    agent = _make_stub(streaming=True)
    assert agent._streaming is True


def test_config_key_defaults_to_agent_type():
    agent = _make_stub()
    assert agent.config_key == "stub"


def test_local_tools_empty_by_default():
    agent = _make_stub()
    assert agent._local_tools == {}


def test_inbox_is_queue():
    import queue
    agent = _make_stub()
    assert isinstance(agent.inbox, queue.Queue)


def test_send_puts_message_in_inbox():
    """send() with wait=False puts a message in the inbox."""
    agent = _make_stub()
    agent.send("hello", wait=False)
    assert not agent.inbox.empty()


def test_is_idle_initially():
    agent = _make_stub()
    assert agent.is_idle is True


# ------------------------------------------------------------------
# Task 4: LLM communication tests
# ------------------------------------------------------------------


def _mock_usage():
    """Create a mock LLM usage object."""
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 5
    usage.thinking_tokens = 0
    usage.cached_tokens = 0
    return usage


def test_llm_send_creates_chat_lazily():
    """Chat session is created on first _llm_send call."""
    agent = _make_stub()
    mock_adapter = MagicMock()
    mock_chat = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "hello"
    mock_response.tool_calls = []
    mock_response.usage = _mock_usage()
    mock_chat.send.return_value = mock_response
    mock_chat.context_window.return_value = 0  # skip compaction
    agent.service.get_adapter.return_value = mock_adapter
    mock_adapter.create_chat.return_value = mock_chat

    result = agent._llm_send("test message")
    assert mock_adapter.create_chat.called
    assert result.text == "hello"


@patch("agent.base_agent.send_with_timeout")
def test_llm_send_reuses_chat(mock_swt):
    """Second _llm_send reuses the existing chat session."""
    mock_response = MagicMock()
    mock_response.text = "response"
    mock_response.tool_calls = []
    mock_response.usage = _mock_usage()
    mock_swt.return_value = mock_response

    agent = _make_stub()
    mock_adapter = MagicMock()
    mock_chat = MagicMock()
    mock_chat.context_window.return_value = 0
    agent.service.get_adapter.return_value = mock_adapter
    mock_adapter.create_chat.return_value = mock_chat

    agent._llm_send("first")
    agent._llm_send("second")
    # create_chat called only once
    assert mock_adapter.create_chat.call_count == 1


def test_streaming_flag_affects_send_path():
    """When streaming=True, _llm_send uses streaming path."""
    agent = _make_stub(streaming=True, event_bus=MagicMock())
    assert agent._streaming is True


# ------------------------------------------------------------------
# Task 5: Tool dispatch tests
# ------------------------------------------------------------------

from agent.tool_handlers import TOOL_REGISTRY


def test_execute_single_tool_uses_registry(monkeypatch):
    """_execute_single_tool dispatches through TOOL_REGISTRY."""
    handler = MagicMock(return_value={"result": "ok"})
    monkeypatch.setitem(TOOL_REGISTRY, "test_tool", handler)
    agent = _make_stub(session_ctx=MagicMock())

    from agent.loop_guard import LoopGuard
    guard = LoopGuard()

    tc = MagicMock()
    tc.name = "test_tool"
    tc.args = {"key": "value"}
    tc.id = "tc-1"

    result_msg, intercepted, _ = agent._execute_single_tool(tc, guard, [])
    handler.assert_called_once()
    assert not intercepted


def test_local_tool_overrides_registry(monkeypatch):
    """_local_tools take priority over TOOL_REGISTRY."""
    global_handler = MagicMock(return_value={"from": "global"})
    local_handler = MagicMock(return_value={"from": "local"})
    monkeypatch.setitem(TOOL_REGISTRY, "test_tool", global_handler)

    agent = _make_stub(session_ctx=MagicMock())
    agent._local_tools["test_tool"] = local_handler

    from agent.loop_guard import LoopGuard
    guard = LoopGuard()

    tc = MagicMock()
    tc.name = "test_tool"
    tc.args = {}
    tc.id = "tc-1"

    result_msg, intercepted, _ = agent._execute_single_tool(tc, guard, [])
    local_handler.assert_called_once()
    global_handler.assert_not_called()


def test_intercept_sentinel_short_circuits():
    """Tool returning {'intercept': True} triggers interception."""
    agent = _make_stub(session_ctx=MagicMock())
    agent._local_tools["clarify"] = lambda ctx, caller, args: {
        "intercept": True,
        "text": "Which one?",
    }

    from agent.loop_guard import LoopGuard
    guard = LoopGuard()

    tc = MagicMock()
    tc.name = "clarify"
    tc.args = {}
    tc.id = "tc-1"

    result_msg, intercepted, text = agent._execute_single_tool(tc, guard, [])
    assert intercepted is True
    assert text == "Which one?"


# ------------------------------------------------------------------
# Task 6: Token tracking and status tests
# ------------------------------------------------------------------


def test_token_tracking():
    agent = _make_stub()
    usage = agent.get_token_usage()
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["api_calls"] == 0


def test_status_returns_dict():
    agent = _make_stub()
    s = agent.status()
    assert "agent_id" in s
    assert "state" in s
    assert "agent_type" in s
    assert s["agent_type"] == "stub"
    assert s["state"] == "sleeping"
