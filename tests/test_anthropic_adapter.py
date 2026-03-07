"""Tests for the Anthropic LLM adapter (agent/llm/anthropic_adapter.py).

These tests mock the ``anthropic`` SDK so no API key is needed.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from agent.llm.base import (
    ChatSession,
    FunctionSchema,
    LLMAdapter,
    LLMResponse,
    ToolCall,
    UsageMetadata,
)
from agent.llm.anthropic_adapter import (
    AnthropicAdapter,
    AnthropicChatSession,
    _build_tools,
    _parse_response,
    _ensure_alternation,
    _response_to_messages,
    _filter_invalid_tool_results,
)


# ---------------------------------------------------------------------------
# Helper builder tests
# ---------------------------------------------------------------------------


class TestBuildTools:
    def test_none_returns_none(self):
        assert _build_tools(None) is None

    def test_empty_returns_none(self):
        assert _build_tools([]) is None

    def test_converts_schemas(self):
        schemas = [
            FunctionSchema(
                name="fetch_data",
                description="Fetch data",
                parameters={"type": "object", "properties": {"id": {"type": "string"}}},
            ),
        ]
        result = _build_tools(schemas)
        assert len(result) == 1
        assert result[0]["name"] == "fetch_data"
        assert result[0]["description"] == "Fetch data"
        # Anthropic uses input_schema, not parameters
        assert "input_schema" in result[0]
        assert "parameters" not in result[0]


class TestEnsureAlternation:
    def test_empty_list(self):
        assert _ensure_alternation([]) == []

    def test_already_alternating(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]
        result = _ensure_alternation(msgs)
        assert len(result) == 3

    def test_merges_consecutive_user(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]
        result = _ensure_alternation(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        # Content should be merged into list form
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2

    def test_merges_list_content(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "{}"}]},
        ]
        result = _ensure_alternation(msgs)
        assert len(result) == 1
        assert len(result[0]["content"]) == 2

    def test_merges_mixed_content_types(self):
        msgs = [
            {"role": "user", "content": "Text message"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "{}"}]},
        ]
        result = _ensure_alternation(msgs)
        assert len(result) == 1
        assert len(result[0]["content"]) == 2

    def test_preserves_different_roles(self):
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A2"},
        ]
        result = _ensure_alternation(msgs)
        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"
        assert result[3]["role"] == "assistant"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# _filter_invalid_tool_results tests (new cleaner design)
# ---------------------------------------------------------------------------


class TestFilterInvalidToolResults:
    def test_invalid_tool_result_replaced_with_text(self):
        """tool_result with non-existent tool_use_id is replaced with explanatory text."""
        from agent.llm.anthropic_adapter import _filter_invalid_tool_results

        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_valid123", "name": "fetch_data", "input": {}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_invalid456",  # Does not exist!
                        "content": '{"status": "success", "data": [1,2,3]}'
                    }
                ]
            }
        ]

        result = _filter_invalid_tool_results(messages)

        # The invalid tool_result should be replaced with text
        user_msg = result[1]
        assert user_msg["content"][0]["type"] == "text"
        assert "toolu_invalid456" in user_msg["content"][0]["text"]
        assert "does not match" in user_msg["content"][0]["text"].lower()

    def test_valid_tool_result_kept_as_is(self):
        """Valid tool_results are kept unchanged."""
        from agent.llm.anthropic_adapter import _filter_invalid_tool_results

        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_valid123", "name": "fetch_data", "input": {}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_valid123",  # Valid!
                        "content": '{"status": "success"}'
                    }
                ]
            }
        ]

        result = _filter_invalid_tool_results(messages)
        # Valid tool_result should be kept
        assert result[1]["content"][0]["type"] == "tool_result"
        assert result[1]["content"][0]["tool_use_id"] == "toolu_valid123"

    def test_mixed_valid_and_invalid(self):
        """Mixed valid and invalid tool_results - valid kept, invalid replaced."""
        from agent.llm.anthropic_adapter import _filter_invalid_tool_results

        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_aaa", "name": "fetch_ace", "input": {}},
                    {"type": "tool_use", "id": "toolu_bbb", "name": "fetch_psp", "input": {}},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_aaa", "content": "ace data"},
                    {"type": "tool_result", "tool_use_id": "toolu_ccc", "content": "invalid!"},  # Invalid
                    {"type": "tool_result", "tool_use_id": "toolu_bbb", "content": "psp data"},
                ]
            }
        ]

        result = _filter_invalid_tool_results(messages)
        user_content = result[1]["content"]

        # First and third should be kept
        assert user_content[0]["type"] == "tool_result"
        assert user_content[0]["tool_use_id"] == "toolu_aaa"
        assert user_content[2]["type"] == "tool_result"
        assert user_content[2]["tool_use_id"] == "toolu_bbb"

        # Second should be replaced with text
        assert user_content[1]["type"] == "text"
        assert "toolu_ccc" in user_content[1]["text"]

    def test_no_tool_use_in_history(self):
        """When no tool_use in history, all tool_results are invalid."""
        from agent.llm.anthropic_adapter import _filter_invalid_tool_results

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_orphan", "content": "data"}
                ]
            }
        ]

        result = _filter_invalid_tool_results(messages)
        # Should be replaced with text
        assert result[2]["content"][0]["type"] == "text"
        assert "toolu_orphan" in result[2]["content"][0]["text"]

    def test_preserves_non_tool_result_blocks(self):
        """Non-tool_result blocks in user messages are preserved."""
        from agent.llm.anthropic_adapter import _filter_invalid_tool_results

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Some context"},
                    {"type": "tool_result", "tool_use_id": "toolu_valid", "content": "result"}
                ]
            }
        ]

        result = _filter_invalid_tool_results(messages)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Some context"


# ---------------------------------------------------------------------------
# _parse_response tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    def _make_response(
        self,
        content_blocks=None,
        input_tokens=10,
        output_tokens=5,
    ):
        """Build a mock Anthropic Messages response."""
        if content_blocks is None:
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "Hello"
            content_blocks = [text_block]

        usage = MagicMock(spec=[])
        usage.input_tokens = input_tokens
        usage.output_tokens = output_tokens
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 0

        raw = MagicMock()
        raw.content = content_blocks
        raw.usage = usage
        return raw

    def test_text_only(self):
        result = _parse_response(self._make_response())
        assert result.text == "Hello"
        assert result.tool_calls == []
        assert result.thoughts == []
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_tool_use(self):
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_abc123"
        tool_block.name = "fetch_data"
        tool_block.input = {"dataset": "AC_H0_MFI"}

        raw = self._make_response(content_blocks=[tool_block])
        result = _parse_response(raw)
        assert result.text == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "fetch_data"
        assert result.tool_calls[0].id == "toolu_abc123"
        assert result.tool_calls[0].args == {"dataset": "AC_H0_MFI"}

    def test_thinking_block(self):
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Let me analyze this..."

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Here's the answer."

        raw = self._make_response(content_blocks=[thinking_block, text_block])
        result = _parse_response(raw)
        assert result.text == "Here's the answer."
        assert result.thoughts == ["Let me analyze this..."]

    def test_mixed_content(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I'll search."

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_xyz"
        tool_block.name = "search"
        tool_block.input = {"q": "ACE"}

        raw = self._make_response(content_blocks=[text_block, tool_block])
        result = _parse_response(raw)
        assert result.text == "I'll search."
        assert len(result.tool_calls) == 1

    def test_raw_preserved(self):
        raw = self._make_response()
        result = _parse_response(raw)
        assert result.raw is raw


# ---------------------------------------------------------------------------
# _response_to_messages tests
# ---------------------------------------------------------------------------


class TestResponseToMessages:
    def test_text_response(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello"

        raw = MagicMock()
        raw.content = [text_block]

        msgs = _response_to_messages(raw)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"][0]["type"] == "text"
        assert msgs[0]["content"][0]["text"] == "Hello"

    def test_tool_use_response(self):
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_abc"
        tool_block.name = "search"
        tool_block.input = {"q": "test"}

        raw = MagicMock()
        raw.content = [tool_block]

        msgs = _response_to_messages(raw)
        assert msgs[0]["content"][0]["type"] == "tool_use"
        assert msgs[0]["content"][0]["id"] == "toolu_abc"

    def test_thinking_response(self):
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Thinking..."
        thinking_block.signature = "sig123"

        raw = MagicMock()
        raw.content = [thinking_block]

        msgs = _response_to_messages(raw)
        assert msgs[0]["content"][0]["type"] == "thinking"
        assert msgs[0]["content"][0]["thinking"] == "Thinking..."
        assert msgs[0]["content"][0]["signature"] == "sig123"


# ---------------------------------------------------------------------------
# AnthropicAdapter tests (mocked SDK)
# ---------------------------------------------------------------------------


class TestAnthropicAdapterMocked:
    @pytest.fixture
    def adapter(self):
        with patch("agent.llm.anthropic_adapter.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_anthropic.RateLimitError = type("RateLimitError", (Exception,), {})

            a = AnthropicAdapter(api_key="test-key")
            a._mock_client = mock_client
            a._mock_anthropic = mock_anthropic
            return a

    def test_create_chat_basic(self, adapter):
        session = adapter.create_chat(
            model="claude-sonnet-4-5-20250929",
            system_prompt="You are helpful.",
        )
        assert isinstance(session, AnthropicChatSession)
        assert session._system == "You are helpful."
        # No messages yet (no history)
        assert session._messages == []

    def test_create_chat_with_tools(self, adapter):
        tools = [
            FunctionSchema(name="test", description="A test tool", parameters={"type": "object"}),
        ]
        session = adapter.create_chat(
            model="claude-sonnet-4-5-20250929",
            system_prompt="System",
            tools=tools,
        )
        assert session._tools is not None
        assert len(session._tools) == 1
        assert session._tools[0]["name"] == "test"
        assert "input_schema" in session._tools[0]

    def test_create_chat_force_tool_call(self, adapter):
        tools = [FunctionSchema(name="t", description="T", parameters={})]
        session = adapter.create_chat(
            model="claude-sonnet-4-5-20250929",
            system_prompt="System",
            tools=tools,
            force_tool_call=True,
        )
        assert session._tool_choice == {"type": "any"}

    def test_create_chat_json_schema(self, adapter):
        schema = {
            "type": "object",
            "title": "my_output",
            "properties": {"name": {"type": "string"}},
        }
        session = adapter.create_chat(
            model="claude-sonnet-4-5-20250929",
            system_prompt="System",
            json_schema=schema,
        )
        # Should add a tool for structured output and force it
        assert any(t["name"] == "my_output" for t in session._tools)
        assert session._tool_choice == {"type": "tool", "name": "my_output"}

    def test_create_chat_thinking_high(self, adapter):
        session = adapter.create_chat(
            model="claude-sonnet-4-5-20250929",
            system_prompt="System",
            thinking="high",
        )
        assert session._extra_kwargs["thinking"]["type"] == "enabled"
        assert session._extra_kwargs["thinking"]["budget_tokens"] == 16384

    def test_create_chat_thinking_low(self, adapter):
        session = adapter.create_chat(
            model="claude-sonnet-4-5-20250929",
            system_prompt="System",
            thinking="low",
        )
        assert session._extra_kwargs["thinking"]["budget_tokens"] == 2048

    def test_create_chat_with_history(self, adapter):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ]
        session = adapter.create_chat(
            model="claude-sonnet-4-5-20250929",
            system_prompt="System",
            history=history,
        )
        assert len(session._messages) == 2

    def test_make_tool_result_message(self, adapter):
        result = adapter.make_tool_result_message(
            "fetch_data",
            {"status": "success", "rows": 100},
            tool_call_id="toolu_abc123",
        )
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "toolu_abc123"
        content = json.loads(result["content"])
        assert content["status"] == "success"

    def test_make_tool_result_message_no_id(self, adapter):
        result = adapter.make_tool_result_message(
            "fetch_data", {"status": "success"}
        )
        assert result["tool_use_id"].startswith("toolu_")

    def test_is_quota_error_true(self, adapter):
        import anthropic as real_anthropic
        try:
            exc = real_anthropic.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            )
        except TypeError:
            pytest.skip("Cannot construct RateLimitError in this SDK version")
        assert adapter.is_quota_error(exc) is True

    def test_is_quota_error_false(self, adapter):
        assert adapter.is_quota_error(ValueError("something")) is False

    def test_generate(self, adapter):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Generated text"

        usage = MagicMock(spec=[])
        usage.input_tokens = 20
        usage.output_tokens = 10
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 0

        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.usage = usage

        adapter._mock_client.messages.create.return_value = mock_response

        result = adapter.generate(
            model="claude-sonnet-4-5-20250929",
            contents="What is 2+2?",
            system_prompt="You are a math tutor.",
            temperature=0.5,
        )
        assert isinstance(result, LLMResponse)
        assert result.text == "Generated text"
        assert result.usage.input_tokens == 20

        call_kwargs = adapter._mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are a math tutor."
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["messages"][0]["role"] == "user"

    def test_generate_json_schema(self, adapter):
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_x"
        tool_block.name = "my_schema"
        tool_block.input = {"name": "test"}

        usage = MagicMock(spec=[])
        usage.input_tokens = 10
        usage.output_tokens = 5
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 0

        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_response.usage = usage

        adapter._mock_client.messages.create.return_value = mock_response

        result = adapter.generate(
            model="claude-sonnet-4-5-20250929",
            contents="Generate a name",
            json_schema={"type": "object", "title": "my_schema", "properties": {"name": {"type": "string"}}},
        )
        # JSON schema via tool-based structured output
        call_kwargs = adapter._mock_client.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"]["type"] == "tool"

    def test_client_property(self, adapter):
        assert adapter.client is adapter._mock_client


# ---------------------------------------------------------------------------
# AnthropicChatSession tests
# ---------------------------------------------------------------------------


class TestAnthropicChatSession:
    def _make_mock_response(self, content="Hi", tool_calls=None):
        """Build a mock Anthropic Messages response."""
        blocks = []
        if content:
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = content
            blocks.append(text_block)
        if tool_calls:
            for tc in tool_calls:
                tool_block = MagicMock()
                tool_block.type = "tool_use"
                tool_block.id = tc["id"]
                tool_block.name = tc["name"]
                tool_block.input = tc["input"]
                blocks.append(tool_block)

        usage = MagicMock(spec=[])
        usage.input_tokens = 5
        usage.output_tokens = 3
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 0

        raw = MagicMock()
        raw.content = blocks
        raw.usage = usage
        return raw

    def test_send_user_message(self):
        mock_client = MagicMock()
        mock_response = self._make_mock_response(content="Hello!")
        mock_client.messages.create.return_value = mock_response

        session = AnthropicChatSession(
            client=mock_client,
            model="claude-sonnet-4-5-20250929",
            system_prompt="You are helpful.",
            messages=[],
            tools=None,
            tool_choice=None,
            extra_kwargs={},
        )

        result = session.send("Hi there")
        assert isinstance(result, LLMResponse)
        assert result.text == "Hello!"

        # user message + assistant response
        assert len(session._messages) == 2
        assert session._messages[0]["role"] == "user"
        assert session._messages[1]["role"] == "assistant"

    def test_send_tool_results(self):
        mock_client = MagicMock()
        mock_response = self._make_mock_response(content="Got the results.")
        mock_client.messages.create.return_value = mock_response

        session = AnthropicChatSession(
            client=mock_client,
            model="claude-sonnet-4-5-20250929",
            system_prompt="System",
            messages=[
                {"role": "user", "content": "Search for ACE"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_abc", "name": "search", "input": {"q": "ACE"}},
                ]},
            ],
            tools=[{"name": "search", "description": "Search", "input_schema": {}}],
            tool_choice=None,
            extra_kwargs={},
        )

        tool_results = [
            {"type": "tool_result", "tool_use_id": "toolu_abc", "content": '{"status":"success"}'}
        ]
        result = session.send(tool_results)
        assert result.text == "Got the results."

        # Tool results wrapped in a user message
        assert session._messages[2]["role"] == "user"
        assert session._messages[2]["content"][0]["type"] == "tool_result"

    def test_get_history(self):
        session = AnthropicChatSession(
            client=MagicMock(),
            model="claude-sonnet-4-5-20250929",
            system_prompt="System",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            ],
            tools=None,
            tool_choice=None,
            extra_kwargs={},
        )
        history = session.get_history()
        assert len(history) == 2
        assert history is not session._messages

    def test_send_invalid_type_raises(self):
        session = AnthropicChatSession(
            client=MagicMock(),
            model="claude-sonnet-4-5-20250929",
            system_prompt="System",
            messages=[],
            tools=None,
            tool_choice=None,
            extra_kwargs={},
        )
        with pytest.raises(TypeError, match="Unsupported message type"):
            session.send(12345)

    def test_system_prompt_passed_to_api(self):
        mock_client = MagicMock()
        mock_response = self._make_mock_response()
        mock_client.messages.create.return_value = mock_response

        session = AnthropicChatSession(
            client=mock_client,
            model="claude-sonnet-4-5-20250929",
            system_prompt="Be helpful.",
            messages=[],
            tools=None,
            tool_choice=None,
            extra_kwargs={},
        )
        session.send("test")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "Be helpful."

    def test_tools_passed_to_api(self):
        mock_client = MagicMock()
        mock_response = self._make_mock_response()
        mock_client.messages.create.return_value = mock_response

        tools = [{"name": "t", "description": "t", "input_schema": {}}]
        session = AnthropicChatSession(
            client=mock_client,
            model="claude-sonnet-4-5-20250929",
            system_prompt="Sys",
            messages=[],
            tools=tools,
            tool_choice={"type": "any"},
            extra_kwargs={},
        )
        session.send("test")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == {"type": "any"}

    def test_response_with_tool_calls_in_history(self):
        mock_client = MagicMock()
        mock_response = self._make_mock_response(
            content=None,
            tool_calls=[{"id": "toolu_xyz", "name": "fetch_data", "input": {"id": "test"}}],
        )
        mock_client.messages.create.return_value = mock_response

        session = AnthropicChatSession(
            client=mock_client,
            model="claude-sonnet-4-5-20250929",
            system_prompt="Sys",
            messages=[],
            tools=[{"name": "fetch_data", "description": "F", "input_schema": {}}],
            tool_choice=None,
            extra_kwargs={},
        )
        result = session.send("Do something")
        assert len(result.tool_calls) == 1

        # Assistant message with tool_use should be in history
        assistant_msg = session._messages[-1]
        assert assistant_msg["role"] == "assistant"
        assert any(b["type"] == "tool_use" for b in assistant_msg["content"])
