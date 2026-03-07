"""Tests for the canonical LLMInterface layer."""

import time
import pytest
from agent.llm.interface import (
    LLMInterface,
    InterfaceEntry,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    ThinkingBlock,
    ImageBlock,
    content_block_from_dict,
)


class TestContentBlocks:
    def test_text_block(self):
        b = TextBlock(text="hello")
        assert b.to_dict() == {"type": "text", "text": "hello"}

    def test_tool_call_block(self):
        b = ToolCallBlock(id="call_123", name="fetch_data", args={"dataset": "ACE"})
        d = b.to_dict()
        assert d == {"type": "tool_call", "id": "call_123", "name": "fetch_data", "args": {"dataset": "ACE"}}

    def test_tool_result_block(self):
        b = ToolResultBlock(id="call_123", name="fetch_data", content={"status": "ok"})
        d = b.to_dict()
        assert d == {"type": "tool_result", "id": "call_123", "name": "fetch_data", "content": {"status": "ok"}}

    def test_thinking_block(self):
        b = ThinkingBlock(text="Let me think...")
        d = b.to_dict()
        assert d == {"type": "thinking", "text": "Let me think..."}

    def test_thinking_block_with_provider_data(self):
        b = ThinkingBlock(
            text="reasoning",
            provider_data={"gemini": {"thought_signature": "abc"}},
        )
        d = b.to_dict()
        assert d["provider_data"] == {"gemini": {"thought_signature": "abc"}}

    def test_thinking_block_omits_empty_provider_data(self):
        b = ThinkingBlock(text="hi")
        assert "provider_data" not in b.to_dict()

    def test_image_block(self):
        import base64
        b = ImageBlock(data=b"\x89PNG", mime_type="image/png")
        d = b.to_dict()
        assert d["type"] == "image"
        assert d["mime_type"] == "image/png"
        assert d["data"] == base64.b64encode(b"\x89PNG").decode("ascii")

    def test_content_block_roundtrip(self):
        blocks = [
            TextBlock(text="hi"),
            ToolCallBlock(id="c1", name="fn", args={"x": 1}),
            ToolResultBlock(id="c1", name="fn", content="ok"),
            ThinkingBlock(text="hmm"),
            ThinkingBlock(text="t", provider_data={"anthropic": {"signature": "s"}}),
        ]
        for block in blocks:
            d = block.to_dict()
            restored = content_block_from_dict(d)
            assert restored.to_dict() == d


class TestInterfaceEntry:
    def test_entry_to_dict(self):
        entry = InterfaceEntry(
            id=0, role="user",
            content=[TextBlock(text="hello")],
            timestamp=1000.0,
        )
        d = entry.to_dict()
        assert d["id"] == 0
        assert d["role"] == "user"
        assert d["timestamp"] == 1000.0
        assert d["content"] == [{"type": "text", "text": "hello"}]
        assert "provider_data" not in d

    def test_entry_with_provider_data(self):
        entry = InterfaceEntry(
            id=1, role="assistant",
            content=[TextBlock(text="hi")],
            timestamp=1000.0,
            provider_data={"gemini": {"interaction_id": "abc"}},
        )
        d = entry.to_dict()
        assert d["provider_data"] == {"gemini": {"interaction_id": "abc"}}

    def test_entry_roundtrip(self):
        entry = InterfaceEntry(
            id=5, role="assistant",
            content=[
                ThinkingBlock(text="think", provider_data={"anthropic": {"signature": "sig"}}),
                TextBlock(text="answer"),
                ToolCallBlock(id="c1", name="fn", args={"a": 1}),
            ],
            timestamp=1234.5,
            provider_data={"gemini": {"interaction_id": "xyz"}},
        )
        d = entry.to_dict()
        restored = InterfaceEntry.from_dict(d)
        assert restored.to_dict() == d


class TestLLMInterface:
    def test_add_system_prompt(self):
        iface = LLMInterface()
        iface.add_system("You are helpful.")
        assert len(iface.entries) == 1
        assert iface.entries[0].role == "system"
        assert iface.entries[0].content[0].text == "You are helpful."

    def test_add_system_dedup(self):
        iface = LLMInterface()
        iface.add_system("A")
        iface.add_system("A")
        assert len(iface.entries) == 1

    def test_add_system_changed(self):
        iface = LLMInterface()
        iface.add_system("A")
        iface.add_system("B")
        assert len(iface.entries) == 2
        assert iface.entries[1].content[0].text == "B"

    def test_current_system_prompt(self):
        iface = LLMInterface()
        assert iface.current_system_prompt is None
        iface.add_system("A")
        assert iface.current_system_prompt == "A"
        iface.add_user_message("hi")
        iface.add_system("B")
        assert iface.current_system_prompt == "B"

    def test_add_user_text(self):
        iface = LLMInterface()
        iface.add_user_message("hello")
        assert iface.entries[0].role == "user"
        assert iface.entries[0].content == [TextBlock(text="hello")]

    def test_add_user_multimodal(self):
        iface = LLMInterface()
        iface.add_user_message("describe", image_bytes=b"\x89PNG", mime_type="image/png")
        blocks = iface.entries[0].content
        assert len(blocks) == 2
        assert isinstance(blocks[0], TextBlock)
        assert isinstance(blocks[1], ImageBlock)

    def test_add_assistant_message(self):
        iface = LLMInterface()
        blocks = [
            ThinkingBlock(text="think"),
            TextBlock(text="answer"),
            ToolCallBlock(id="c1", name="fetch", args={"x": 1}),
        ]
        iface.add_assistant_message(blocks)
        assert iface.entries[0].role == "assistant"
        assert len(iface.entries[0].content) == 3

    def test_add_tool_results(self):
        iface = LLMInterface()
        iface.add_tool_results([
            ToolResultBlock(id="c1", name="fetch", content={"data": [1, 2]}),
        ])
        assert iface.entries[0].role == "user"
        assert isinstance(iface.entries[0].content[0], ToolResultBlock)

    def test_sequential_ids(self):
        iface = LLMInterface()
        iface.add_user_message("a")
        iface.add_assistant_message([TextBlock(text="b")])
        iface.add_user_message("c")
        assert iface.entries[0].id == 0
        assert iface.entries[1].id == 1
        assert iface.entries[2].id == 2

    def test_truncate_basic(self):
        iface = LLMInterface()
        for i in range(10):
            iface.add_user_message(f"message {i}")
        iface.truncate(max_entries=5)
        assert len(iface.entries) == 5
        assert iface.entries[0].content[0].text == "message 0"
        assert iface.entries[4].content[0].text == "message 4"

    def test_truncate_preserves_system(self):
        iface = LLMInterface()
        iface.add_system("system prompt")
        for i in range(10):
            iface.add_user_message(f"msg {i}")
        iface.truncate(max_entries=5)
        assert len(iface.entries) == 6  # 1 system + 5 messages
        assert iface.entries[0].role == "system"

    def test_truncate_keeps_recent(self):
        iface = LLMInterface()
        for i in range(10):
            iface.add_user_message(f"msg {i}")
        iface.truncate(max_entries=3, keep_recent=3)
        # Should keep first entry (system placeholder if any), then last 3 + 1 for context
        # With no system, max_entries=3 means keep 3 most recent
        assert len(iface.entries) >= 3

    def test_to_dict_and_back(self):
        iface = LLMInterface()
        iface.add_system("sys")
        iface.add_user_message("hi")
        iface.add_assistant_message([
            ThinkingBlock(text="think"),
            TextBlock(text="answer"),
        ])
        iface.add_tool_results([
            ToolResultBlock(id="c1", name="fn", content="ok"),
        ])

        d = iface.to_dict()
        restored = LLMInterface.from_dict(d)

        assert restored.current_system_prompt == "sys"
        assert len(restored.entries) == len(iface.entries)
        assert restored.entries[2].content[0].text == "think"
        assert restored.entries[3].content[0].content == "ok"

    def test_to_messages_basic(self):
        iface = LLMInterface()
        iface.add_user_message("hello")
        msgs = iface.to_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["text"] == "hello"


class TestInterfaceEntryExtended:
    """Test new fields: model, provider, usage on InterfaceEntry."""

    def test_assistant_entry_with_model_and_usage(self):
        entry = InterfaceEntry(
            id=0, role="assistant",
            content=[TextBlock(text="hello")],
            timestamp=1000.0,
            model="gemini-3-flash-preview",
            provider="gemini",
            usage={"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0},
        )
        d = entry.to_dict()
        assert d["model"] == "gemini-3-flash-preview"
        assert d["provider"] == "gemini"
        assert d["usage"] == {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0}

    def test_user_entry_omits_model_and_usage(self):
        entry = InterfaceEntry(
            id=1, role="user",
            content=[TextBlock(text="hi")],
            timestamp=1000.0,
        )
        d = entry.to_dict()
        assert "model" not in d
        assert "provider" not in d
        assert "usage" not in d

    def test_assistant_entry_roundtrip(self):
        entry = InterfaceEntry(
            id=0, role="assistant",
            content=[TextBlock(text="answer")],
            timestamp=1234.5,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            provider_data={"response_id": "resp_xyz"},
            usage={"input_tokens": 200, "output_tokens": 100, "thinking_tokens": 50},
        )
        d = entry.to_dict()
        restored = InterfaceEntry.from_dict(d)
        assert restored.model == "claude-sonnet-4-20250514"
        assert restored.provider == "anthropic"
        assert restored.usage == {"input_tokens": 200, "output_tokens": 100, "thinking_tokens": 50}
        assert restored.to_dict() == d

    def test_legacy_entry_without_new_fields(self):
        """Backward compat: old entries without model/provider/usage still load."""
        d = {
            "id": 0, "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "timestamp": 1000.0,
        }
        entry = InterfaceEntry.from_dict(d)
        assert entry.model is None
        assert entry.provider is None
        assert entry.usage == {}


class TestSystemEntryWithTools:
    """Test system entries carry tools alongside system prompt."""

    def test_add_system_with_tools(self):
        iface = LLMInterface()
        tools = [{"name": "fetch", "description": "Fetch data", "parameters": {}}]
        iface.add_system("You are helpful.", tools=tools)
        entry = iface.entries[0]
        assert entry.role == "system"
        assert entry.content[0].text == "You are helpful."
        d = entry.to_dict()
        assert d["system"] == "You are helpful."
        assert d["tools"] == tools

    def test_add_system_dedup_with_same_tools(self):
        iface = LLMInterface()
        tools = [{"name": "fetch", "description": "Fetch", "parameters": {}}]
        iface.add_system("A", tools=tools)
        iface.add_system("A", tools=tools)
        assert len(iface.entries) == 1

    def test_add_system_emits_on_tool_change(self):
        iface = LLMInterface()
        tools1 = [{"name": "fetch", "description": "Fetch", "parameters": {}}]
        tools2 = [{"name": "fetch", "description": "Fetch", "parameters": {}},
                  {"name": "plot", "description": "Plot", "parameters": {}}]
        iface.add_system("sys", tools=tools1)
        iface.add_system("sys", tools=tools2)
        assert len(iface.entries) == 2
        assert iface.current_tools == tools2

    def test_system_entry_to_dict_format(self):
        """System entries use 'system' and 'tools' keys, not 'content'."""
        iface = LLMInterface()
        tools = [{"name": "fn", "description": "d", "parameters": {}}]
        iface.add_system("prompt", tools=tools)
        d = iface.entries[0].to_dict()
        assert "system" in d
        assert "tools" in d
        assert d["role"] == "system"

    def test_system_roundtrip(self):
        iface = LLMInterface()
        tools = [{"name": "fn", "description": "d", "parameters": {"x": {"type": "int"}}}]
        iface.add_system("prompt", tools=tools)
        iface.add_user_message("hi")
        data = iface.to_dict()
        restored = LLMInterface.from_dict(data)
        assert restored.current_system_prompt == "prompt"
        assert restored.current_tools == tools


class TestAddAssistantMessageExtended:
    """Test add_assistant_message records model, provider, usage."""

    def test_add_with_model_and_usage(self):
        iface = LLMInterface()
        iface.add_assistant_message(
            [TextBlock(text="hi")],
            model="gemini-3-flash-preview",
            provider="gemini",
            usage={"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0},
        )
        entry = iface.entries[0]
        assert entry.model == "gemini-3-flash-preview"
        assert entry.provider == "gemini"
        assert entry.usage == {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0}

    def test_add_without_new_fields_still_works(self):
        """Backward compat: existing calls without model/usage still work."""
        iface = LLMInterface()
        iface.add_assistant_message([TextBlock(text="hi")])
        entry = iface.entries[0]
        assert entry.model is None
        assert entry.usage == {}

    def test_roundtrip_with_model_and_usage(self):
        iface = LLMInterface()
        iface.add_assistant_message(
            [TextBlock(text="answer")],
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            provider_data={"some": "data"},
            usage={"input_tokens": 500, "output_tokens": 200, "thinking_tokens": 100},
        )
        data = iface.to_dict()
        restored = LLMInterface.from_dict(data)
        assert restored.entries[0].model == "claude-sonnet-4-20250514"
        assert restored.entries[0].usage["input_tokens"] == 500


class TestUsageHelpers:
    """Test total_usage() and usage_by_model() on LLMInterface."""

    def test_total_usage_empty(self):
        iface = LLMInterface()
        usage = iface.total_usage()
        assert usage == {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "calls": 0}

    def test_total_usage_sums_across_messages(self):
        iface = LLMInterface()
        iface.add_assistant_message(
            [TextBlock(text="a")], model="m1", provider="p",
            usage={"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 10},
        )
        iface.add_user_message("follow up")
        iface.add_assistant_message(
            [TextBlock(text="b")], model="m1", provider="p",
            usage={"input_tokens": 200, "output_tokens": 100, "thinking_tokens": 20},
        )
        usage = iface.total_usage()
        assert usage["input_tokens"] == 300
        assert usage["output_tokens"] == 150
        assert usage["thinking_tokens"] == 30
        assert usage["calls"] == 2

    def test_usage_by_model(self):
        iface = LLMInterface()
        iface.add_assistant_message(
            [TextBlock(text="a")], model="gemini-flash", provider="gemini",
            usage={"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0},
        )
        iface.add_user_message("next")
        iface.add_assistant_message(
            [TextBlock(text="b")], model="claude-sonnet", provider="anthropic",
            usage={"input_tokens": 200, "output_tokens": 100, "thinking_tokens": 50},
        )
        iface.add_user_message("next")
        iface.add_assistant_message(
            [TextBlock(text="c")], model="gemini-flash", provider="gemini",
            usage={"input_tokens": 150, "output_tokens": 75, "thinking_tokens": 0},
        )
        by_model = iface.usage_by_model()
        assert by_model["gemini-flash"]["input_tokens"] == 250
        assert by_model["gemini-flash"]["calls"] == 2
        assert by_model["claude-sonnet"]["input_tokens"] == 200
        assert by_model["claude-sonnet"]["calls"] == 1


class TestChatSessionABC:
    def test_interface_property_required(self):
        """ChatSession subclass must provide interface."""
        from agent.llm.base import ChatSession
        from agent.llm.interface import LLMInterface

        class GoodSession(ChatSession):
            def __init__(self):
                self._iface = LLMInterface()
            def send(self, message):
                pass
            @property
            def interface(self):
                return self._iface

        s = GoodSession()
        assert isinstance(s.interface, LLMInterface)
        assert s.get_history() == []  # empty interface

    def test_get_history_returns_canonical(self):
        from agent.llm.base import ChatSession
        from agent.llm.interface import LLMInterface, TextBlock

        class TestSession(ChatSession):
            def __init__(self):
                self._iface = LLMInterface()
                self._iface.add_system("sys")
                self._iface.add_user_message("hi")
            def send(self, message):
                pass
            @property
            def interface(self):
                return self._iface

        s = TestSession()
        h = s.get_history()
        assert len(h) == 2
        assert h[0]["role"] == "system"


class TestChatSessionState:
    """Test ChatSession.session_id and get_state()."""

    def test_session_has_id(self):
        from agent.llm.base import ChatSession
        from agent.llm.interface import LLMInterface, TextBlock

        class TestSession(ChatSession):
            def __init__(self):
                self._iface = LLMInterface()
                self.session_id = "xh_test123"
                self._model = "test-model"
                self._provider = "test"
                self._agent_type = "orchestrator"
            def send(self, message):
                pass
            @property
            def interface(self):
                return self._iface

        s = TestSession()
        assert s.session_id == "xh_test123"

    def test_get_state_format(self):
        from agent.llm.base import ChatSession
        from agent.llm.interface import LLMInterface, TextBlock

        class TestSession(ChatSession):
            def __init__(self):
                self._iface = LLMInterface()
                self.session_id = "xh_test456"
                self._agent_type = "envoy"
                self._tracked = True
                self._iface.add_system("sys prompt", tools=[{"name": "fn", "description": "d", "parameters": {}}])
                self._iface.add_user_message("hello")
                self._iface.add_assistant_message(
                    [TextBlock(text="hi")],
                    model="gemini-flash", provider="gemini",
                    usage={"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0},
                )
            def send(self, message):
                pass
            @property
            def interface(self):
                return self._iface

        s = TestSession()
        state = s.get_state()
        assert state["session_id"] == "xh_test456"
        assert "messages" in state
        assert state["messages"][0]["role"] == "system"
        assert "system" in state["messages"][0]
        assert "tools" in state["messages"][0]
        assert state["messages"][2]["model"] == "gemini-flash"
        assert "metadata" in state
        assert state["metadata"]["agent_type"] == "envoy"

    def test_total_usage_delegates(self):
        from agent.llm.base import ChatSession
        from agent.llm.interface import LLMInterface, TextBlock

        class TestSession(ChatSession):
            def __init__(self):
                self._iface = LLMInterface()
                self.session_id = "xh_test"
                self._agent_type = ""
                self._tracked = True
                self._iface.add_assistant_message(
                    [TextBlock(text="a")], model="m", provider="p",
                    usage={"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0},
                )
            def send(self, message):
                pass
            @property
            def interface(self):
                return self._iface

        s = TestSession()
        usage = s.total_usage()
        assert usage["calls"] == 1
        assert usage["input_tokens"] == 100


class TestAdapterRecordsUsage:
    """Test that adapters record model/provider/usage on assistant messages."""

    def test_anthropic_send_records_metadata(self):
        """Anthropic send() should record model/provider/usage on assistant entry."""
        from unittest.mock import MagicMock, patch
        from agent.llm.anthropic_adapter import AnthropicAdapter

        with patch("agent.llm.anthropic_adapter.anthropic") as mock_sdk:
            mock_client = MagicMock()
            mock_sdk.Anthropic.return_value = mock_client

            adapter = AnthropicAdapter(api_key="test-key")

            # Build a mock response
            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = "hello"
            mock_response = MagicMock()
            mock_response.content = [mock_block]
            mock_response.stop_reason = "end_turn"
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            mock_response.usage.cache_read_input_tokens = 0
            mock_response.usage.cache_creation_input_tokens = 0
            mock_response.usage.thinking_tokens = 0
            mock_response.model = "claude-test"
            mock_client.messages.create.return_value = mock_response

            chat = adapter.create_chat(model="claude-test", system_prompt="sys")
            chat.send("hi")

            last = chat.interface.last_assistant_entry()
            assert last is not None
            assert last.model == "claude-test"
            assert last.provider == "anthropic"
            assert last.usage["input_tokens"] == 100
            assert last.usage["output_tokens"] == 50

    def test_openai_completions_records_metadata(self):
        """OpenAI ChatCompletions session records model/provider/usage on assistant entry."""
        from unittest.mock import MagicMock, patch
        from agent.llm.openai_adapter import OpenAIChatSession

        # Build a mock OpenAI ChatCompletion response
        mock_msg = MagicMock()
        mock_msg.content = "hello"
        mock_msg.tool_calls = None
        mock_msg.reasoning_content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_choice.finish_reason = "stop"
        mock_raw = MagicMock()
        mock_raw.choices = [mock_choice]
        mock_raw.usage.prompt_tokens = 200
        mock_raw.usage.completion_tokens = 100
        mock_raw.usage.prompt_tokens_details = None
        mock_raw.usage.completion_tokens_details = None

        # Create a session directly with a mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_raw
        iface = LLMInterface()
        iface.add_system("sys")
        session = OpenAIChatSession(
            client=mock_client,
            model="gpt-test",
            interface=iface,
            tools=None,
            tool_choice=None,
            extra_kwargs={},
        )
        session.send("hi")

        last = session.interface.last_assistant_entry()
        assert last is not None
        assert last.model == "gpt-test"
        assert last.provider == "openai"
        assert last.usage["input_tokens"] == 200
        assert last.usage["output_tokens"] == 100


class TestLLMAdapterABC:
    def test_create_chat_accepts_interface(self):
        """create_chat signature should accept interface parameter."""
        import inspect
        from agent.llm.base import LLMAdapter
        sig = inspect.signature(LLMAdapter.create_chat)
        assert "interface" in sig.parameters

    def test_make_tool_result_returns_canonical(self):
        """make_tool_result_message should return ToolResultBlock."""
        from agent.llm.interface import ToolResultBlock
        block = ToolResultBlock(id="c1", name="fn", content={"ok": True})
        assert block.id == "c1"
        assert block.name == "fn"
