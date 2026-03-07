"""Tests for ChatSession.rollback_last_turn() — cancellation history cleanup."""

import pytest


class TestAnthropicRollback:
    """Anthropic adapter: messages use role='assistant' with tool_use content blocks,
    and role='user' with tool_result content blocks."""

    def test_rollback_strips_last_assistant_turn(self):
        """After an assistant message with tool_use, rollback removes it."""
        from agent.llm.anthropic.adapter import AnthropicAdapter

        adapter = AnthropicAdapter(api_key="fake")
        chat = adapter.create_chat(model="claude-sonnet-4-20250514", system_prompt="test")

        # Manually inject history: user → assistant (with tool_use)
        chat._messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me fetch that."},
                {"type": "tool_use", "id": "tc_001", "name": "fetch_data",
                 "input": {"dataset": "ACE_MAG"}},
            ]},
        ]

        chat.rollback_last_turn()

        assert len(chat._messages) == 1
        assert chat._messages[0]["role"] == "user"

    def test_rollback_strips_orphaned_tool_results(self):
        """If tool_results were committed after the assistant turn,
        rollback removes both the assistant turn AND the orphaned tool_results."""
        from agent.llm.anthropic.adapter import AnthropicAdapter

        adapter = AnthropicAdapter(api_key="fake")
        chat = adapter.create_chat(model="claude-sonnet-4-20250514", system_prompt="test")

        chat._messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tc_001", "name": "fetch_data",
                 "input": {"dataset": "ACE_MAG"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tc_001",
                 "content": '{"status": "ok"}'},
            ]},
        ]

        chat.rollback_last_turn()

        assert len(chat._messages) == 1
        assert chat._messages[0]["role"] == "user"
        assert chat._messages[0]["content"] == "hello"

    def test_rollback_noop_when_no_assistant_turn(self):
        """If history has no assistant turn, rollback does nothing."""
        from agent.llm.anthropic.adapter import AnthropicAdapter

        adapter = AnthropicAdapter(api_key="fake")
        chat = adapter.create_chat(model="claude-sonnet-4-20250514", system_prompt="test")

        chat._messages = [
            {"role": "user", "content": "hello"},
        ]

        chat.rollback_last_turn()

        assert len(chat._messages) == 1

    def test_rollback_preserves_earlier_turns(self):
        """Only the LAST assistant turn is removed, not earlier ones."""
        from agent.llm.anthropic.adapter import AnthropicAdapter

        adapter = AnthropicAdapter(api_key="fake")
        chat = adapter.create_chat(model="claude-sonnet-4-20250514", system_prompt="test")

        chat._messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": [{"type": "text", "text": "first reply"}]},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tc_002", "name": "plot",
                 "input": {"type": "line"}},
            ]},
        ]

        chat.rollback_last_turn()

        assert len(chat._messages) == 3
        assert chat._messages[-1]["role"] == "user"
        assert chat._messages[-1]["content"] == "second"


class TestGeminiRollback:
    """Gemini adapter: uses role='model' (not 'assistant') and _client_history."""

    def test_rollback_strips_last_model_turn(self):
        from agent.llm.gemini.adapter import GeminiAdapter

        adapter = GeminiAdapter(api_key="fake")
        chat = adapter.create_chat(model="gemini-2.5-flash", system_prompt="test")

        chat._client_history = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "model", "content": [
                {"type": "function_call", "id": "fc_001", "name": "fetch_data",
                 "arguments": {"dataset": "ACE_MAG"}},
            ]},
        ]

        chat.rollback_last_turn()

        assert len(chat._client_history) == 1
        assert chat._client_history[0]["role"] == "user"

    def test_rollback_strips_orphaned_tool_results(self):
        from agent.llm.gemini.adapter import GeminiAdapter

        adapter = GeminiAdapter(api_key="fake")
        chat = adapter.create_chat(model="gemini-2.5-flash", system_prompt="test")

        chat._client_history = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "model", "content": [
                {"type": "function_call", "id": "fc_001", "name": "fetch_data",
                 "arguments": {"dataset": "ACE_MAG"}},
            ]},
            {"role": "user", "content": [
                {"type": "function_result", "id": "fc_001", "name": "fetch_data",
                 "result": '{"status": "ok"}'},
            ]},
        ]

        chat.rollback_last_turn()

        assert len(chat._client_history) == 1


class TestOpenAIRollback:
    """OpenAI adapter: tool results are separate messages with role='tool'."""

    def test_rollback_strips_assistant_and_tool_messages(self):
        from agent.llm.openai.adapter import OpenAIAdapter

        adapter = OpenAIAdapter(api_key="fake", base_url="http://fake")
        chat = adapter.create_chat(model="gpt-4o", system_prompt="test")

        chat._messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc_001", "type": "function",
                 "function": {"name": "fetch_data", "arguments": '{"dataset": "ACE_MAG"}'}},
            ]},
            {"role": "tool", "tool_call_id": "tc_001", "content": '{"status": "ok"}'},
        ]

        chat.rollback_last_turn()

        assert len(chat._messages) == 1
        assert chat._messages[0]["role"] == "user"

    def test_rollback_strips_multiple_tool_results(self):
        from agent.llm.openai.adapter import OpenAIAdapter

        adapter = OpenAIAdapter(api_key="fake", base_url="http://fake")
        chat = adapter.create_chat(model="gpt-4o", system_prompt="test")

        chat._messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc_001", "type": "function",
                 "function": {"name": "fetch_data", "arguments": "{}"}},
                {"id": "tc_002", "type": "function",
                 "function": {"name": "plot", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc_001", "content": "ok"},
            {"role": "tool", "tool_call_id": "tc_002", "content": "ok"},
        ]

        chat.rollback_last_turn()

        assert len(chat._messages) == 1


class TestSendCancelAware:
    """agent.send(wait=True) should return early when cancel_event is set."""

    def test_send_returns_cancelled_when_event_set(self):
        """If cancel_event fires while waiting for reply, return cancelled result."""
        import threading
        import time
        from unittest.mock import MagicMock
        from agent.sub_agent import SubAgent

        cancel = threading.Event()
        adapter = MagicMock()
        adapter.create_chat.return_value = MagicMock()

        agent = SubAgent(
            agent_id="TestAgent",
            adapter=adapter,
            model_name="test-model",
            tool_executor=lambda *a, **kw: {},
            system_prompt="test",
            cancel_event=cancel,
        )
        # Do NOT start the agent thread — we want the message to sit
        # in the inbox unprocessed so reply_event never fires.
        # This simulates a long-running LLM call that hasn't replied yet.

        # Set cancel after a short delay
        def set_cancel():
            time.sleep(0.3)
            cancel.set()
        threading.Thread(target=set_cancel, daemon=True).start()

        # send(wait=True) should return within ~2s, not wait 300s
        start = time.monotonic()
        result = agent.send("test", sender="orchestrator", wait=True, timeout=10)
        elapsed = time.monotonic() - start

        assert elapsed < 3.0, f"send() took {elapsed:.1f}s, should have returned on cancel"
        assert result["failed"] is True
        assert "cancel" in result["text"].lower() or "interrupt" in result["text"].lower()


class TestSubAgentCancelRollback:
    """Sub-agent should rollback history on cancel, not commit tool results."""

    def test_cancel_before_tools_rolls_back_history(self):
        """When cancel fires before tool execution, the last assistant turn
        (containing tool_use blocks) should be stripped from history."""
        import threading
        from unittest.mock import MagicMock, patch
        from agent.envoy_agent import EnvoyAgent

        cancel = threading.Event()
        cancel.set()  # Pre-set cancel

        chat = MagicMock()
        chat.rollback_last_turn = MagicMock()
        chat.commit_tool_results = MagicMock()

        adapter = MagicMock()

        with patch("agent.envoy_agent.build_envoy_prompt", return_value="test prompt"):
            agent = EnvoyAgent(
                mission_id="TEST",
                adapter=adapter,
                model_name="test",
                tool_executor=lambda *a, **kw: {},
                cancel_event=cancel,
            )
        agent._chat = chat

        # Simulate: response has tool_calls, cancel is already set
        response = MagicMock()
        response.text = ""
        tc = MagicMock()
        tc.name = "fetch_data"
        tc.args = {}
        tc.id = "tc_001"
        response.tool_calls = [tc]

        result = agent._process_response(response)

        chat.rollback_last_turn.assert_called_once()
        chat.commit_tool_results.assert_not_called()
        assert result["failed"] is True
