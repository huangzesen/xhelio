"""Tests for the OpenAI LLM adapter (agent/llm/openai_adapter.py).

These tests mock the ``openai`` SDK so no API key is needed.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from agent.llm.base import (
    ChatSession,
    FunctionSchema,
    LLMAdapter,
    LLMResponse,
    ToolCall,
    UsageMetadata,
)
from agent.llm.openai_adapter import (
    OpenAIAdapter,
    OpenAIChatSession,
    _build_tools,
    _parse_response,
    _parse_tool_calls,
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
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "fetch_data"
        assert result[0]["function"]["description"] == "Fetch data"
        assert "properties" in result[0]["function"]["parameters"]


class TestParseToolCalls:
    def test_none_returns_empty(self):
        assert _parse_tool_calls(None) == []

    def test_empty_returns_empty(self):
        assert _parse_tool_calls([]) == []

    def test_parses_tool_calls(self):
        tc = MagicMock()
        tc.id = "call_abc123"
        tc.function.name = "fetch_data"
        tc.function.arguments = '{"dataset": "AC_H0_MFI"}'

        result = _parse_tool_calls([tc])
        assert len(result) == 1
        assert result[0].name == "fetch_data"
        assert result[0].args == {"dataset": "AC_H0_MFI"}
        assert result[0].id == "call_abc123"

    def test_handles_invalid_json(self):
        tc = MagicMock()
        tc.id = "call_xyz"
        tc.function.name = "some_tool"
        tc.function.arguments = "not valid json"

        result = _parse_tool_calls([tc])
        assert len(result) == 1
        assert result[0].args == {}

    def test_handles_empty_arguments(self):
        tc = MagicMock()
        tc.id = "call_empty"
        tc.function.name = "list_data"
        tc.function.arguments = ""

        result = _parse_tool_calls([tc])
        assert len(result) == 1
        assert result[0].args == {}


# ---------------------------------------------------------------------------
# _parse_response tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    def _make_response(
        self,
        content="Hello",
        tool_calls=None,
        reasoning_content=None,
        prompt_tokens=10,
        completion_tokens=5,
        reasoning_tokens=0,
        finish_reason="stop",
    ):
        """Build a mock OpenAI ChatCompletion response."""
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls
        msg.reasoning_content = reasoning_content

        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = finish_reason

        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        if reasoning_tokens:
            details = MagicMock()
            details.reasoning_tokens = reasoning_tokens
            usage.completion_tokens_details = details
        else:
            usage.completion_tokens_details = None

        raw = MagicMock()
        raw.choices = [choice]
        raw.usage = usage
        return raw

    def test_text_only(self):
        raw = self._make_response(content="Hello world")
        result = _parse_response(raw)
        assert isinstance(result, LLMResponse)
        assert result.text == "Hello world"
        assert result.tool_calls == []
        assert result.thoughts == []
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_tool_calls(self):
        tc = MagicMock()
        tc.id = "call_abc"
        tc.function.name = "search"
        tc.function.arguments = '{"query": "ACE"}'

        raw = self._make_response(content=None, tool_calls=[tc])
        result = _parse_response(raw)
        assert result.text == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].id == "call_abc"

    def test_reasoning_content(self):
        raw = self._make_response(
            content="Answer",
            reasoning_content="Let me think about this...",
            reasoning_tokens=100,
        )
        result = _parse_response(raw)
        assert result.text == "Answer"
        assert result.thoughts == ["Let me think about this..."]
        assert result.usage.thinking_tokens == 100

    def test_empty_choices(self):
        raw = MagicMock()
        raw.choices = []
        raw.usage = None
        result = _parse_response(raw)
        assert result.text == ""
        assert result.tool_calls == []

    def test_no_usage(self):
        raw = self._make_response()
        raw.usage = None
        result = _parse_response(raw)
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0

    def test_raw_preserved(self):
        raw = self._make_response()
        result = _parse_response(raw)
        assert result.raw is raw


# ---------------------------------------------------------------------------
# OpenAIAdapter tests (mocked SDK)
# ---------------------------------------------------------------------------


class TestOpenAIAdapterMocked:
    @pytest.fixture
    def adapter(self):
        with patch("agent.llm.openai_adapter.openai") as mock_openai:
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})

            a = OpenAIAdapter(api_key="test-key", base_url="https://api.test.com/v1")
            a._mock_client = mock_client
            a._mock_openai = mock_openai
            return a

    def test_constructor_sets_base_url(self):
        with patch("agent.llm.openai_adapter.openai") as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock()
            OpenAIAdapter(api_key="key", base_url="https://custom.api/v1")
            mock_openai.OpenAI.assert_called_once_with(
                api_key="key",
                base_url="https://custom.api/v1",
                timeout=300.0,
            )

    def test_constructor_no_base_url(self):
        with patch("agent.llm.openai_adapter.openai") as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock()
            OpenAIAdapter(api_key="key")
            mock_openai.OpenAI.assert_called_once_with(
                api_key="key",
                timeout=300.0,
            )

    def test_create_chat_basic(self, adapter):
        session = adapter.create_chat(
            model="gpt-4.1",
            system_prompt="You are helpful.",
        )
        assert isinstance(session, OpenAIChatSession)
        # System message should be first
        assert session._messages[0]["role"] == "system"
        assert session._messages[0]["content"] == "You are helpful."

    def test_create_chat_with_tools(self, adapter):
        tools = [
            FunctionSchema(name="test", description="A test tool", parameters={"type": "object"}),
        ]
        session = adapter.create_chat(
            model="gpt-4.1",
            system_prompt="System",
            tools=tools,
        )
        assert session._tools is not None
        assert len(session._tools) == 1
        assert session._tools[0]["function"]["name"] == "test"

    def test_create_chat_force_tool_call(self, adapter):
        tools = [
            FunctionSchema(name="test", description="Test", parameters={}),
        ]
        session = adapter.create_chat(
            model="gpt-4.1",
            system_prompt="System",
            tools=tools,
            force_tool_call=True,
        )
        assert session._tool_choice == "required"

    def test_create_chat_json_schema(self, adapter):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        session = adapter.create_chat(
            model="gpt-4.1",
            system_prompt="System",
            json_schema=schema,
        )
        assert "response_format" in session._extra_kwargs
        assert session._extra_kwargs["response_format"]["type"] == "json_schema"

    def test_create_chat_thinking(self, adapter):
        session = adapter.create_chat(
            model="o3",
            system_prompt="System",
            thinking="high",
        )
        assert session._extra_kwargs.get("reasoning_effort") == "high"

    def test_create_chat_with_history(self, adapter):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        session = adapter.create_chat(
            model="gpt-4.1",
            system_prompt="System",
            history=history,
        )
        # system + 2 history messages
        assert len(session._messages) == 3
        assert session._messages[1]["role"] == "user"
        assert session._messages[2]["role"] == "assistant"

    def test_make_tool_result_message(self, adapter):
        result = adapter.make_tool_result_message(
            "fetch_data",
            {"status": "success", "rows": 100},
            tool_call_id="call_abc123",
        )
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_abc123"
        content = json.loads(result["content"])
        assert content["status"] == "success"

    def test_make_tool_result_message_no_id(self, adapter):
        """When no tool_call_id is provided, generates a placeholder."""
        result = adapter.make_tool_result_message(
            "fetch_data", {"status": "success"}
        )
        assert result["tool_call_id"].startswith("call_")

    def test_is_quota_error_true(self, adapter):
        # Use the real openai.RateLimitError so isinstance() works
        import openai as real_openai
        try:
            # RateLimitError needs specific constructor args
            exc = real_openai.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            )
        except TypeError:
            # Older SDK versions may have different constructor
            pytest.skip("Cannot construct RateLimitError in this SDK version")
        assert adapter.is_quota_error(exc) is True

    def test_is_quota_error_false(self, adapter):
        assert adapter.is_quota_error(ValueError("something")) is False

    def test_generate(self, adapter):
        # Build mock response
        msg = MagicMock()
        msg.content = "Generated text"
        msg.tool_calls = None
        msg.reasoning_content = None

        choice = MagicMock()
        choice.message = msg

        usage = MagicMock()
        usage.prompt_tokens = 20
        usage.completion_tokens = 10
        usage.completion_tokens_details = None

        mock_response = MagicMock()
        mock_response.choices = [choice]
        mock_response.usage = usage

        adapter._mock_client.chat.completions.create.return_value = mock_response

        result = adapter.generate(
            model="gpt-4.1",
            contents="What is 2+2?",
            system_prompt="You are a math tutor.",
            temperature=0.5,
        )
        assert isinstance(result, LLMResponse)
        assert result.text == "Generated text"
        assert result.usage.input_tokens == 20

        # Check the call was made with correct messages
        call_args = adapter._mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is 2+2?"
        assert call_args.kwargs["temperature"] == 0.5

    def test_client_property(self, adapter):
        assert adapter.client is adapter._mock_client


# ---------------------------------------------------------------------------
# OpenAIChatSession tests
# ---------------------------------------------------------------------------


class TestOpenAIChatSession:
    def _make_mock_response(self, content="Hi", tool_calls=None):
        """Build a mock ChatCompletion for the session."""
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls
        msg.reasoning_content = None

        choice = MagicMock()
        choice.message = msg

        usage = MagicMock()
        usage.prompt_tokens = 5
        usage.completion_tokens = 3
        usage.completion_tokens_details = None

        raw = MagicMock()
        raw.choices = [choice]
        raw.usage = usage
        return raw

    def test_send_user_message(self):
        mock_client = MagicMock()
        mock_response = self._make_mock_response(content="Hello!")
        mock_client.chat.completions.create.return_value = mock_response

        session = OpenAIChatSession(
            client=mock_client,
            model="gpt-4.1",
            messages=[{"role": "system", "content": "You are helpful."}],
            tools=None,
            tool_choice=None,
            extra_kwargs={},
        )

        result = session.send("Hi there")
        assert isinstance(result, LLMResponse)
        assert result.text == "Hello!"

        # Check messages were accumulated
        assert len(session._messages) == 3  # system + user + assistant
        assert session._messages[1]["role"] == "user"
        assert session._messages[1]["content"] == "Hi there"
        assert session._messages[2]["role"] == "assistant"

    def test_send_tool_results(self):
        mock_client = MagicMock()
        mock_response = self._make_mock_response(content="Got it.")
        mock_client.chat.completions.create.return_value = mock_response

        session = OpenAIChatSession(
            client=mock_client,
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Search for ACE"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q":"ACE"}'},
                        }
                    ],
                },
            ],
            tools=[{"type": "function", "function": {"name": "search", "description": "Search", "parameters": {}}}],
            tool_choice=None,
            extra_kwargs={},
        )

        tool_results = [
            {"role": "tool", "tool_call_id": "call_abc", "content": '{"status":"success"}'}
        ]
        result = session.send(tool_results)
        assert result.text == "Got it."

        # Tool result should be in messages
        assert session._messages[3]["role"] == "tool"
        assert session._messages[3]["tool_call_id"] == "call_abc"

    def test_get_history(self):
        session = OpenAIChatSession(
            client=MagicMock(),
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Hello"},
            ],
            tools=None,
            tool_choice=None,
            extra_kwargs={},
        )
        history = session.get_history()
        assert len(history) == 2
        # Should be a copy
        assert history is not session._messages

    def test_send_invalid_type_raises(self):
        session = OpenAIChatSession(
            client=MagicMock(),
            model="gpt-4.1",
            messages=[],
            tools=None,
            tool_choice=None,
            extra_kwargs={},
        )
        with pytest.raises(TypeError, match="Unsupported message type"):
            session.send(12345)

    def test_tools_passed_to_api(self):
        mock_client = MagicMock()
        mock_response = self._make_mock_response()
        mock_client.chat.completions.create.return_value = mock_response

        tools = [{"type": "function", "function": {"name": "t", "description": "t", "parameters": {}}}]
        session = OpenAIChatSession(
            client=mock_client,
            model="gpt-4.1",
            messages=[{"role": "system", "content": "Sys"}],
            tools=tools,
            tool_choice="required",
            extra_kwargs={},
        )
        session.send("test")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "required"

    def test_response_with_tool_calls_in_history(self):
        """When the model responds with tool calls, they should be in history."""
        mock_client = MagicMock()

        tc = MagicMock()
        tc.id = "call_xyz"
        tc.function.name = "fetch_data"
        tc.function.arguments = '{"id": "test"}'

        mock_response = self._make_mock_response(content=None, tool_calls=[tc])
        mock_client.chat.completions.create.return_value = mock_response

        session = OpenAIChatSession(
            client=mock_client,
            model="gpt-4.1",
            messages=[{"role": "system", "content": "Sys"}],
            tools=[{"type": "function", "function": {"name": "fetch_data", "description": "F", "parameters": {}}}],
            tool_choice=None,
            extra_kwargs={},
        )
        result = session.send("Do something")
        assert len(result.tool_calls) == 1

        # Assistant message with tool_calls should be in history
        assistant_msg = session._messages[-1]
        assert assistant_msg["role"] == "assistant"
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"][0]["id"] == "call_xyz"


# ---------------------------------------------------------------------------
# ToolCall.id field tests
# ---------------------------------------------------------------------------


class TestToolCallId:
    def test_default_none(self):
        tc = ToolCall(name="test", args={})
        assert tc.id is None

    def test_explicit_id(self):
        tc = ToolCall(name="test", args={}, id="call_abc")
        assert tc.id == "call_abc"

    def test_backward_compat(self):
        """Existing code creating ToolCall(name=..., args=...) still works."""
        tc = ToolCall(name="fetch", args={"x": 1})
        assert tc.name == "fetch"
        assert tc.args == {"x": 1}
        assert tc.id is None
