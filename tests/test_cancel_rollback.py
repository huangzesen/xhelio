"""Tests for ChatSession.rollback_last_turn() — cancellation history cleanup."""

import pytest


class TestInterfaceStripUnansweredToolCalls:
    """ChatInterface strips unanswered tool_use from the trailing assistant
    entry when add_user_message is called — provider-agnostic fix."""

    def test_user_message_strips_trailing_tool_use(self):
        """add_user_message auto-strips assistant entry with unanswered tool_use."""
        from agent.llm.interface import ChatInterface, TextBlock, ToolCallBlock

        iface = ChatInterface()
        iface.add_user_message("hello")
        iface.add_assistant_message([
            TextBlock(text="Let me fetch that."),
            ToolCallBlock(id="tc_001", name="fetch_data", args={"dataset": "ACE_MAG"}),
        ])

        # Now a user message arrives (e.g. after cancel)
        iface.add_user_message("make a white background one")

        roles = [e.role for e in iface.entries]
        # assistant with tool_use should be gone: user, user
        assert roles == ["user", "user"]
        assert isinstance(iface.entries[-1].content[0], TextBlock)
        assert iface.entries[-1].content[0].text == "make a white background one"

    def test_user_message_keeps_text_only_assistant(self):
        """Assistant entries without tool_use are NOT stripped."""
        from agent.llm.interface import ChatInterface, TextBlock

        iface = ChatInterface()
        iface.add_user_message("hello")
        iface.add_assistant_message([TextBlock(text="Hi there!")])

        iface.add_user_message("follow up")

        roles = [e.role for e in iface.entries]
        assert roles == ["user", "assistant", "user"]

    def test_strip_preserves_earlier_turns(self):
        """Only the trailing assistant entry is stripped, not earlier ones."""
        from agent.llm.interface import ChatInterface, TextBlock, ToolCallBlock

        iface = ChatInterface()
        iface.add_user_message("first")
        iface.add_assistant_message([TextBlock(text="first reply")])
        iface.add_user_message("second")
        iface.add_assistant_message([
            ToolCallBlock(id="tc_002", name="plot", args={"type": "line"}),
        ])

        iface.add_user_message("third")

        roles = [e.role for e in iface.entries]
        assert roles == ["user", "assistant", "user", "user"]
        assert iface.entries[-1].content[0].text == "third"

    def test_noop_when_no_trailing_assistant(self):
        """No crash when history ends with a user entry."""
        from agent.llm.interface import ChatInterface

        iface = ChatInterface()
        iface.add_user_message("hello")
        iface.add_user_message("another")

        roles = [e.role for e in iface.entries]
        assert roles == ["user", "user"]

    def test_noop_on_empty_interface(self):
        """No crash on empty interface."""
        from agent.llm.interface import ChatInterface

        iface = ChatInterface()
        iface.add_user_message("hello")

        assert len(iface.entries) == 1


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
        svc = MagicMock()
        svc.get_adapter.return_value = adapter
        svc.provider = "gemini"
        svc.make_tool_result.side_effect = adapter.make_tool_result_message

        agent = SubAgent(
            agent_id="TestAgent",
            service=svc,
            agent_type="test",
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
        svc = MagicMock()
        svc.get_adapter.return_value = adapter
        svc.provider = "gemini"
        svc.make_tool_result.side_effect = adapter.make_tool_result_message

        with patch("agent.envoy_agent.build_envoy_prompt", return_value="test prompt"):
            agent = EnvoyAgent(
                mission_id="TEST",
                service=svc,
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
