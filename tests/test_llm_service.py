"""Tests for LLMService facade."""

from unittest.mock import MagicMock, patch
import pytest
from agent.llm.service import LLMService
from agent.llm.base import LLMResponse, FunctionSchema, UsageMetadata
from agent.llm.interface import TextBlock, ToolResultBlock


class TestLLMServiceInit:
    def test_creates_gemini_adapter(self):
        with patch("agent.llm.gemini_adapter.GeminiAdapter") as mock_cls:
            mock_cls.return_value = MagicMock()
            svc = LLMService(provider="gemini", model="gemini-flash")
            mock_cls.assert_called_once()
            assert svc._model == "gemini-flash"

    def test_creates_anthropic_adapter(self):
        with patch("agent.llm.anthropic_adapter.AnthropicAdapter") as mock_cls:
            mock_cls.return_value = MagicMock()
            svc = LLMService(provider="anthropic", model="claude-sonnet", api_key="sk-test")
            mock_cls.assert_called_once()

    def test_creates_openai_adapter(self):
        with patch("agent.llm.openai_adapter.OpenAIAdapter") as mock_cls:
            mock_cls.return_value = MagicMock()
            svc = LLMService(provider="openai", model="gpt-4o", api_key="sk-test")
            mock_cls.assert_called_once()

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMService(provider="unknown", model="m")

    def test_adapter_property(self):
        with patch("agent.llm.gemini_adapter.GeminiAdapter") as mock_cls:
            mock_adapter = MagicMock()
            mock_cls.return_value = mock_adapter
            svc = LLMService(provider="gemini", model="m")
            assert svc.adapter is mock_adapter


class TestLLMServiceCreateSession:
    def test_creates_session_with_id(self):
        with patch("agent.llm.gemini_adapter.GeminiAdapter") as mock_cls:
            mock_adapter = MagicMock()
            mock_session = MagicMock()
            mock_adapter.create_chat.return_value = mock_session
            mock_cls.return_value = mock_adapter
            svc = LLMService(provider="gemini", model="m")
            chat = svc.create_session(system_prompt="sys", agent_type="orchestrator")
            assert mock_session.session_id != ""
            assert mock_session.session_id.startswith("xh_")
            assert svc.get_session(mock_session.session_id) is mock_session

    def test_untracked_session_no_registry(self):
        with patch("agent.llm.gemini_adapter.GeminiAdapter") as mock_cls:
            mock_adapter = MagicMock()
            mock_session = MagicMock()
            mock_adapter.create_chat.return_value = mock_session
            mock_cls.return_value = mock_adapter
            svc = LLMService(provider="gemini", model="m")
            chat = svc.create_session(system_prompt="sys", tracked=False)
            assert svc.get_session(mock_session.session_id) is None


class TestLLMServiceGenerate:
    def test_generate_tracked(self):
        with patch("agent.llm.gemini_adapter.GeminiAdapter") as mock_cls:
            mock_adapter = MagicMock()
            mock_adapter.generate.return_value = LLMResponse(
                text="answer", usage=UsageMetadata(input_tokens=10, output_tokens=5),
            )
            mock_cls.return_value = mock_adapter
            svc = LLMService(provider="gemini", model="m")
            resp = svc.generate("hello")
            assert resp.text == "answer"
            mock_adapter.generate.assert_called_once()

    def test_generate_untracked(self):
        with patch("agent.llm.gemini_adapter.GeminiAdapter") as mock_cls:
            mock_adapter = MagicMock()
            mock_adapter.generate.return_value = LLMResponse(text="x")
            mock_cls.return_value = mock_adapter
            svc = LLMService(provider="gemini", model="m")
            resp = svc.generate("hello", tracked=False)
            assert resp.text == "x"


class TestLLMServiceMakeToolResult:
    def test_returns_tool_result_block(self):
        with patch("agent.llm.gemini_adapter.GeminiAdapter") as mock_cls:
            mock_adapter = MagicMock()
            mock_adapter.make_tool_result_message.return_value = ToolResultBlock(
                id="c1", name="fn", content={"ok": True},
            )
            mock_cls.return_value = mock_adapter
            svc = LLMService(provider="gemini", model="m")
            result = svc.make_tool_result("fn", {"ok": True}, tool_call_id="c1")
            assert isinstance(result, ToolResultBlock)
            assert result.name == "fn"
