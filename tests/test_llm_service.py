"""Tests for LLMService facade."""

from unittest.mock import MagicMock, patch
import pytest
from agent.llm.service import LLMService
from agent.llm.base import LLMResponse, FunctionSchema, UsageMetadata
from agent.llm.interface import TextBlock, ToolResultBlock


class TestLLMServiceInit:
    def test_creates_gemini_adapter(self):
        with patch("agent.llm.gemini.adapter.GeminiAdapter") as mock_cls:
            mock_cls.return_value = MagicMock()
            svc = LLMService(provider="gemini", model="gemini-flash")
            mock_cls.assert_called_once()
            assert svc._model == "gemini-flash"

    def test_creates_anthropic_adapter(self):
        with patch("agent.llm.anthropic.adapter.AnthropicAdapter") as mock_cls:
            mock_cls.return_value = MagicMock()
            svc = LLMService(provider="anthropic", model="claude-sonnet", api_key="sk-test")
            mock_cls.assert_called_once()

    def test_creates_openai_adapter(self):
        with patch("agent.llm.openai.adapter.OpenAIAdapter") as mock_cls:
            mock_cls.return_value = MagicMock()
            svc = LLMService(provider="openai", model="gpt-4o", api_key="sk-test")
            mock_cls.assert_called_once()

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMService(provider="unknown", model="m")

    def test_get_adapter_returns_primary(self):
        with patch("agent.llm.gemini.adapter.GeminiAdapter") as mock_cls:
            mock_adapter = MagicMock()
            mock_cls.return_value = mock_adapter
            svc = LLMService(provider="gemini", model="m")
            assert svc.get_adapter("gemini") is mock_adapter


class TestLLMServiceCreateSession:
    def test_creates_session_with_id(self):
        with patch("agent.llm.gemini.adapter.GeminiAdapter") as mock_cls:
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
        with patch("agent.llm.gemini.adapter.GeminiAdapter") as mock_cls:
            mock_adapter = MagicMock()
            mock_session = MagicMock()
            mock_adapter.create_chat.return_value = mock_session
            mock_cls.return_value = mock_adapter
            svc = LLMService(provider="gemini", model="m")
            chat = svc.create_session(system_prompt="sys", tracked=False)
            assert svc.get_session(mock_session.session_id) is None


class TestLLMServiceGenerate:
    def test_generate_tracked(self):
        with patch("agent.llm.gemini.adapter.GeminiAdapter") as mock_cls:
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
        with patch("agent.llm.gemini.adapter.GeminiAdapter") as mock_cls:
            mock_adapter = MagicMock()
            mock_adapter.generate.return_value = LLMResponse(text="x")
            mock_cls.return_value = mock_adapter
            svc = LLMService(provider="gemini", model="m")
            resp = svc.generate("hello", tracked=False)
            assert resp.text == "x"


class TestLLMServiceMakeToolResult:
    def test_returns_tool_result_block(self):
        with patch("agent.llm.gemini.adapter.GeminiAdapter") as mock_cls:
            mock_adapter = MagicMock()
            mock_adapter.make_tool_result_message.return_value = ToolResultBlock(
                id="c1", name="fn", content={"ok": True},
            )
            mock_cls.return_value = mock_adapter
            svc = LLMService(provider="gemini", model="m")
            result = svc.make_tool_result("fn", {"ok": True}, tool_call_id="c1")
            assert isinstance(result, ToolResultBlock)
            assert result.name == "fn"


class TestGetAdapter:
    def test_primary_adapter_cached(self):
        """Primary adapter is returned for primary provider."""
        with patch.object(LLMService, '_create_adapter') as mock_create:
            mock_adapter = MagicMock()
            mock_create.return_value = mock_adapter
            svc = LLMService(provider="gemini", model="test-model", api_key="fake")
            assert svc.get_adapter("gemini") is mock_adapter
            assert svc.get_adapter("gemini") is mock_adapter
            mock_create.assert_called_once()

    def test_secondary_adapter_created_on_demand(self):
        with patch.object(LLMService, '_create_adapter') as mock_create:
            primary = MagicMock()
            secondary = MagicMock()
            mock_create.side_effect = [primary, secondary]
            svc = LLMService(provider="gemini", model="test-model", api_key="fake")

            with patch('agent.llm.service.get_api_key', return_value="fake-key"):
                result = svc.get_adapter("anthropic")
            assert result is secondary
            assert svc.get_adapter("anthropic") is secondary

    def test_missing_api_key_raises(self):
        with patch.object(LLMService, '_create_adapter') as mock_create:
            mock_create.return_value = MagicMock()
            svc = LLMService(provider="gemini", model="test-model", api_key="fake")

            with patch('agent.llm.service.get_api_key', return_value=None):
                with pytest.raises(RuntimeError, match="API key"):
                    svc.get_adapter("anthropic")

    def test_no_adapter_property(self):
        """The old escape hatch 'adapter' property must not exist."""
        with patch.object(LLMService, '_create_adapter') as mock_create:
            mock_create.return_value = MagicMock()
            svc = LLMService(provider="gemini", model="test-model", api_key="fake")
            assert not hasattr(svc, 'adapter')
            assert not hasattr(svc, 'primary_adapter')
            # Use get_adapter() instead
            assert svc.get_adapter("gemini") is mock_create.return_value

    def test_adapter_cache_keyed_by_provider_and_base_url(self):
        """Same provider with different base_urls gets separate adapters."""
        with patch.object(LLMService, '_create_adapter') as mock_create:
            primary = MagicMock()
            custom_url = MagicMock()
            mock_create.side_effect = [primary, custom_url]
            svc = LLMService(provider="openai", model="gpt-4o", api_key="fake")

            # Primary adapter (base_url=None)
            assert svc.get_adapter("openai") is primary

            # Same provider, different base_url — new adapter
            with patch('agent.llm.service.get_api_key', return_value="fake-key"):
                result = svc.get_adapter("openai", base_url="https://vllm.local/v1")
            assert result is custom_url
            assert result is not primary

            # Both are cached independently
            assert svc.get_adapter("openai") is primary
            assert svc.get_adapter("openai", base_url="https://vllm.local/v1") is custom_url

    def test_adapter_cache_tuple_keys(self):
        """Internal cache uses (provider, base_url) tuple keys."""
        with patch.object(LLMService, '_create_adapter') as mock_create:
            mock_adapter = MagicMock()
            mock_create.return_value = mock_adapter
            svc = LLMService(provider="gemini", model="m", api_key="fake")
            # Cache should use tuple key
            assert ("gemini", None) in svc._adapters
            assert svc._adapters[("gemini", None)] is mock_adapter

    def test_base_url_stored_on_service(self):
        """LLMService stores base_url for primary adapter lookup."""
        with patch.object(LLMService, '_create_adapter') as mock_create:
            mock_create.return_value = MagicMock()
            svc = LLMService(provider="openai", model="m", api_key="fake",
                             base_url="https://custom.api/v1")
            assert svc._base_url == "https://custom.api/v1"
            assert ("openai", "https://custom.api/v1") in svc._adapters
