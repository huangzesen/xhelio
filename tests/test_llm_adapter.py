"""Tests for the LLM adapter layer (agent/llm/).

These tests verify the adapter types and GeminiAdapter without needing an API key,
using mocks for the google-genai SDK.
"""

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


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_basic(self):
        tc = ToolCall(name="fetch_data", args={"dataset": "AC_H0_MFI"})
        assert tc.name == "fetch_data"
        assert tc.args == {"dataset": "AC_H0_MFI"}

    def test_empty_args(self):
        tc = ToolCall(name="list_data", args={})
        assert tc.args == {}


class TestUsageMetadata:
    def test_defaults(self):
        u = UsageMetadata()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.thinking_tokens == 0

    def test_values(self):
        u = UsageMetadata(input_tokens=100, output_tokens=50, thinking_tokens=25)
        assert u.input_tokens == 100
        assert u.output_tokens == 50
        assert u.thinking_tokens == 25


class TestLLMResponse:
    def test_defaults(self):
        r = LLMResponse()
        assert r.text == ""
        assert r.tool_calls == []
        assert r.usage.input_tokens == 0
        assert r.thoughts == []
        assert r.raw is None

    def test_with_data(self):
        tc = ToolCall(name="search", args={"q": "ACE"})
        r = LLMResponse(
            text="Found ACE data.",
            tool_calls=[tc],
            usage=UsageMetadata(input_tokens=10, output_tokens=5),
            thoughts=["I should search for ACE."],
            raw="raw_object",
        )
        assert r.text == "Found ACE data."
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "search"
        assert r.usage.input_tokens == 10
        assert r.thoughts == ["I should search for ACE."]
        assert r.raw == "raw_object"

    def test_independent_defaults(self):
        """Ensure default mutable fields are independent across instances."""
        r1 = LLMResponse()
        r2 = LLMResponse()
        r1.tool_calls.append(ToolCall(name="a", args={}))
        assert r2.tool_calls == []


class TestFunctionSchema:
    def test_basic(self):
        fs = FunctionSchema(
            name="fetch_data",
            description="Fetch spacecraft data",
            parameters={"type": "object", "properties": {"id": {"type": "string"}}},
        )
        assert fs.name == "fetch_data"
        assert "properties" in fs.parameters


# ---------------------------------------------------------------------------
# GeminiAdapter tests (mocked SDK)
# ---------------------------------------------------------------------------


class TestGeminiAdapterMocked:
    """Test GeminiAdapter with the google-genai SDK fully mocked."""

    @pytest.fixture
    def adapter(self):
        with patch("agent.llm.gemini_adapter.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            from agent.llm.gemini_adapter import GeminiAdapter
            a = GeminiAdapter(api_key="test-key")
            a._mock_client = mock_client
            return a

    def test_make_tool_result_message_chat_api(self, adapter):
        """make_tool_result_message (Chat API) should delegate to types.Part.from_function_response."""
        adapter._use_interactions = False
        with patch("agent.llm.gemini_adapter.types") as mock_types:
            mock_types.Part.from_function_response.return_value = "mock_part"
            result = adapter.make_tool_result_message("fetch_data", {"status": "success"})
            mock_types.Part.from_function_response.assert_called_once_with(
                name="fetch_data",
                response={"result": {"status": "success"}},
            )
            assert result == "mock_part"

    def test_make_tool_result_message_interactions_api(self, adapter):
        """make_tool_result_message (Interactions API) should return a FunctionResultContentParam dict."""
        adapter._use_interactions = True
        result = adapter.make_tool_result_message(
            "fetch_data", {"status": "success"}, tool_call_id="call_123"
        )
        assert isinstance(result, dict)
        assert result["type"] == "function_result"
        assert result["call_id"] == "call_123"
        assert result["name"] == "fetch_data"
        import json
        assert json.loads(result["result"]) == {"status": "success"}

    def test_is_quota_error_true(self, adapter):
        """Should detect 429 quota errors via RESOURCE_EXHAUSTED string."""
        from google.genai import errors as real_errors
        # Use a real ClientError if available, otherwise test the string heuristic
        try:
            exc = real_errors.ClientError("RESOURCE_EXHAUSTED: quota exceeded")
            exc.code = 429
        except Exception:
            # Fallback: create an exception that contains the marker string
            exc = Exception("RESOURCE_EXHAUSTED: quota exceeded")
        # The adapter checks isinstance first, then string fallback
        # With a real ClientError, code=429 should match
        # With a generic Exception, is_quota_error returns False (correct)
        if isinstance(exc, real_errors.ClientError):
            assert adapter.is_quota_error(exc) is True
        else:
            assert adapter.is_quota_error(exc) is False

    def test_is_quota_error_false(self, adapter):
        """Non-quota exceptions should return False."""
        assert adapter.is_quota_error(ValueError("something")) is False

    def test_create_chat_chat_api(self, adapter):
        """create_chat with use_interactions_api=False should create a GeminiChatSession."""
        mock_chat = MagicMock()
        adapter._mock_client.chats.create.return_value = mock_chat

        with patch("agent.llm.gemini_adapter.types") as mock_types, \
             patch("config.get", return_value=False):
            mock_types.GenerateContentConfig.return_value = "config"
            mock_types.Tool.return_value = "tool"
            mock_types.ThinkingConfig.return_value = "thinking"
            mock_types.FunctionDeclaration.return_value = "fd"
            mock_types.FunctionCallingConfig.return_value = "fcc"
            mock_types.ToolConfig.return_value = "tc"

            session = adapter.create_chat(
                model="gemini-test",
                system_prompt="You are helpful.",
                tools=[FunctionSchema(name="test", description="test", parameters={})],
            )

        from agent.llm.gemini_adapter import GeminiChatSession
        assert isinstance(session, GeminiChatSession)

    def test_create_chat_interactions_api(self, adapter):
        """create_chat with use_interactions_api=True should create an InteractionsChatSession."""
        with patch("config.get", return_value=True):
            session = adapter.create_chat(
                model="gemini-test",
                system_prompt="You are helpful.",
                tools=[FunctionSchema(name="test", description="test", parameters={})],
            )

        from agent.llm.gemini_adapter import InteractionsChatSession
        assert isinstance(session, InteractionsChatSession)
        assert session.interaction_id is None  # No previous interaction
        assert adapter._use_interactions is True

    def test_generate(self, adapter):
        """generate should call client.models.generate_content and parse."""
        # Build a mock response
        mock_part = MagicMock()
        mock_part.thought = False
        mock_part.function_call = None
        mock_part.text = "Hello!"

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=10,
            candidates_token_count=5,
            thoughts_token_count=0,
        )

        adapter._mock_client.models.generate_content.return_value = mock_response

        with patch("agent.llm.gemini_adapter.types") as mock_types:
            mock_types.GenerateContentConfig.return_value = "config"
            result = adapter.generate(
                model="gemini-test",
                contents="Hi",
                temperature=0.5,
            )

        assert isinstance(result, LLMResponse)
        assert result.text == "Hello!"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.raw is mock_response


# ---------------------------------------------------------------------------
# GeminiChatSession parse response tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Test the _parse_response helper directly."""

    def test_text_only(self):
        from agent.llm.gemini_adapter import _parse_response

        part = MagicMock()
        part.thought = False
        part.text = "Hello"
        part.function_call = None

        content = MagicMock()
        content.parts = [part]

        candidate = MagicMock()
        candidate.content = content

        raw = MagicMock()
        raw.candidates = [candidate]
        raw.usage_metadata = None

        resp = _parse_response(raw)
        assert resp.text == "Hello"
        assert resp.tool_calls == []
        assert resp.thoughts == []

    def test_tool_calls(self):
        from agent.llm.gemini_adapter import _parse_response

        fc = MagicMock()
        fc.name = "fetch_data"
        fc.args = {"dataset": "AC_H0_MFI"}

        part = MagicMock()
        part.thought = False
        part.text = None
        part.function_call = fc

        content = MagicMock()
        content.parts = [part]

        candidate = MagicMock()
        candidate.content = content

        raw = MagicMock()
        raw.candidates = [candidate]
        raw.usage_metadata = MagicMock(
            prompt_token_count=50,
            candidates_token_count=20,
            thoughts_token_count=10,
        )

        resp = _parse_response(raw)
        assert resp.text == ""
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "fetch_data"
        assert resp.tool_calls[0].args == {"dataset": "AC_H0_MFI"}
        assert resp.usage.input_tokens == 50
        assert resp.usage.output_tokens == 20
        assert resp.usage.thinking_tokens == 10

    def test_thinking_parts(self):
        from agent.llm.gemini_adapter import _parse_response

        thought_part = MagicMock()
        thought_part.thought = True
        thought_part.text = "Let me think..."
        thought_part.function_call = None

        text_part = MagicMock()
        text_part.thought = False
        text_part.text = "Result"
        text_part.function_call = None

        content = MagicMock()
        content.parts = [thought_part, text_part]

        candidate = MagicMock()
        candidate.content = content

        raw = MagicMock()
        raw.candidates = [candidate]
        raw.usage_metadata = None

        resp = _parse_response(raw)
        assert resp.text == "Result"
        assert resp.thoughts == ["Let me think..."]

    def test_empty_response(self):
        from agent.llm.gemini_adapter import _parse_response

        raw = MagicMock()
        raw.candidates = []
        raw.usage_metadata = None

        resp = _parse_response(raw)
        assert resp.text == ""
        assert resp.tool_calls == []


# ---------------------------------------------------------------------------
# Interactions API response parsing tests
# ---------------------------------------------------------------------------


class TestParseInteractionResponse:
    """Test the _parse_interaction_response helper directly."""

    def test_text_output(self):
        from agent.llm.gemini_adapter import _parse_interaction_response

        text_block = MagicMock(type="text", text="Hello from interactions!")
        interaction = MagicMock()
        interaction.outputs = [text_block]
        interaction.usage = MagicMock(
            total_input_tokens=100,
            total_output_tokens=50,
            total_thought_tokens=0,
            total_cached_tokens=20,
        )

        resp = _parse_interaction_response(interaction)
        assert resp.text == "Hello from interactions!"
        assert resp.tool_calls == []
        assert resp.usage.input_tokens == 100
        assert resp.usage.output_tokens == 50
        assert resp.usage.cached_tokens == 20

    def test_function_call_output(self):
        from agent.llm.gemini_adapter import _parse_interaction_response

        fc = MagicMock()
        fc.type = "function_call"
        fc.id = "call_abc123"
        fc.configure_mock(name="fetch_data")  # MagicMock reserves 'name' kwarg
        fc.arguments = {"dataset": "AC_H0_MFI"}
        interaction = MagicMock()
        interaction.outputs = [fc]
        interaction.usage = None

        resp = _parse_interaction_response(interaction)
        assert resp.text == ""
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "fetch_data"
        assert resp.tool_calls[0].args == {"dataset": "AC_H0_MFI"}
        assert resp.tool_calls[0].id == "call_abc123"

    def test_thought_output(self):
        from agent.llm.gemini_adapter import _parse_interaction_response

        summary_text = MagicMock(type="text", text="Let me think about this...")
        thought = MagicMock(type="thought", summary=[summary_text])
        interaction = MagicMock()
        interaction.outputs = [thought]
        interaction.usage = None

        resp = _parse_interaction_response(interaction)
        assert resp.text == ""
        assert resp.tool_calls == []
        assert len(resp.thoughts) == 1
        assert resp.thoughts[0] == "Let me think about this..."

    def test_mixed_outputs(self):
        from agent.llm.gemini_adapter import _parse_interaction_response

        thought_text = MagicMock(type="text", text="Thinking...")
        thought = MagicMock(type="thought", summary=[thought_text])
        fc = MagicMock()
        fc.type = "function_call"
        fc.id = "call_1"
        fc.configure_mock(name="fetch_data")
        fc.arguments = {"a": "b"}
        text = MagicMock(type="text", text="Here's what I found")
        interaction = MagicMock()
        interaction.outputs = [thought, fc, text]
        interaction.usage = MagicMock(
            total_input_tokens=200, total_output_tokens=100,
            total_thought_tokens=30, total_cached_tokens=0,
        )

        resp = _parse_interaction_response(interaction)
        assert len(resp.thoughts) == 1
        assert len(resp.tool_calls) == 1
        assert resp.text == "Here's what I found"

    def test_empty_outputs(self):
        from agent.llm.gemini_adapter import _parse_interaction_response

        interaction = MagicMock()
        interaction.outputs = None
        interaction.usage = None

        resp = _parse_interaction_response(interaction)
        assert resp.text == ""
        assert resp.tool_calls == []
        assert resp.thoughts == []


class TestConvertHistoryToTurns:
    """Test _convert_history_to_turns helper."""

    def test_text_parts(self):
        from agent.llm.gemini_adapter import _convert_history_to_turns

        history = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there!"}]},
        ]
        turns = _convert_history_to_turns(history)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == [{"type": "text", "text": "Hello"}]
        assert turns[1]["role"] == "model"
        assert turns[1]["content"] == [{"type": "text", "text": "Hi there!"}]

    def test_function_call_and_response(self):
        from agent.llm.gemini_adapter import _convert_history_to_turns

        history = [
            {
                "role": "model",
                "parts": [{
                    "function_call": {"name": "fetch_data", "args": {"dataset": "ACE"}}
                }],
            },
            {
                "role": "user",
                "parts": [{
                    "function_response": {"name": "fetch_data", "response": {"status": "ok"}}
                }],
            },
        ]
        turns = _convert_history_to_turns(history)
        assert len(turns) == 2
        assert turns[0]["content"][0]["type"] == "function_call"
        assert turns[0]["content"][0]["name"] == "fetch_data"
        assert turns[1]["content"][0]["type"] == "function_result"
        assert turns[1]["content"][0]["name"] == "fetch_data"

    def test_empty_history(self):
        from agent.llm.gemini_adapter import _convert_history_to_turns

        assert _convert_history_to_turns([]) == []
