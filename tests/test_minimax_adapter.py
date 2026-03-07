"""Tests for the MiniMax LLM adapter (agent/llm/minimax_adapter.py) and MCP client.

These tests mock external dependencies (MCP subprocess, Anthropic SDK) so no
API key is needed.
"""

import logging
import pytest
from unittest.mock import MagicMock, patch

from agent.llm.base import LLMResponse
from agent.llm import MiniMaxAdapter
from agent.minimax_mcp_client import MiniMaxMCPClient, get_minimax_mcp_client
import agent.minimax_mcp_client as _mcp_mod


@pytest.fixture(autouse=True)
def _reset_mcp_singleton():
    """Reset the MiniMax MCP singleton between tests."""
    old = _mcp_mod._client
    yield
    _mcp_mod._client = old


class TestMiniMaxAdapter:
    def test_constructor_sets_minimax_base_url(self):
        with patch("agent.llm.anthropic_adapter.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client

            adapter = MiniMaxAdapter(api_key="test-key")

            mock_anthropic.Anthropic.assert_called_once()
            call_kwargs = mock_anthropic.Anthropic.call_args.kwargs
            assert "api.minimaxi.com/anthropic" in call_kwargs["base_url"]

    def test_make_multimodal_message_text_only(self):
        adapter = MiniMaxAdapter.__new__(MiniMaxAdapter)
        result = adapter.make_multimodal_message(
            "analyze this", b"\x89PNG...", "image/png"
        )
        assert result["role"] == "user"
        contents = result["content"]
        assert any(c["type"] == "text" for c in contents)
        assert not any(c.get("type") == "image" for c in contents)

    def test_make_multimodal_message_logs_warning(self, caplog):
        adapter = MiniMaxAdapter.__new__(MiniMaxAdapter)
        with caplog.at_level(logging.WARNING):
            adapter.make_multimodal_message("test", b"data")
        assert "image" in caplog.text.lower() or "minimax" in caplog.text.lower()

    def test_web_search_disabled_by_config(self):
        adapter = MiniMaxAdapter.__new__(MiniMaxAdapter)
        with patch("config._provider_get", return_value=False):
            result = adapter.web_search("test query", "model")
            assert result.text == ""

    def test_web_search_via_mcp(self):
        adapter = MiniMaxAdapter.__new__(MiniMaxAdapter)
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {
            "status": "success",
            "text": "Search results here",
        }
        with (
            patch("config._provider_get", return_value=True),
            patch(
                "agent.minimax_mcp_client.get_minimax_mcp_client",
                return_value=mock_client,
            ),
        ):
            result = adapter.web_search("solar flares", "MiniMax-M2.5")
            assert "Search results" in result.text
            mock_client.call_tool.assert_called_once_with(
                "web_search", {"query": "solar flares"}
            )

    def test_web_search_mcp_failure(self):
        adapter = MiniMaxAdapter.__new__(MiniMaxAdapter)
        with (
            patch("config._provider_get", return_value=True),
            patch(
                "agent.minimax_mcp_client.get_minimax_mcp_client",
                side_effect=RuntimeError("no uvx"),
            ),
        ):
            result = adapter.web_search("test", "model")
            assert result.text == ""

    def test_web_search_mcp_error_response(self):
        adapter = MiniMaxAdapter.__new__(MiniMaxAdapter)
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {
            "status": "error",
            "message": "API key invalid",
        }
        with (
            patch("config._provider_get", return_value=True),
            patch(
                "agent.minimax_mcp_client.get_minimax_mcp_client",
                return_value=mock_client,
            ),
        ):
            result = adapter.web_search("test", "model")
            assert result.text == ""

    def test_web_search_handles_answer_field(self):
        adapter = MiniMaxAdapter.__new__(MiniMaxAdapter)
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {
            "status": "success",
            "answer": "Direct answer from MCP",
        }
        with (
            patch("config._provider_get", return_value=True),
            patch(
                "agent.minimax_mcp_client.get_minimax_mcp_client",
                return_value=mock_client,
            ),
        ):
            result = adapter.web_search("query", "model")
            assert "Direct answer" in result.text

    def test_is_quota_error_inherited(self):
        adapter = MiniMaxAdapter.__new__(MiniMaxAdapter)
        assert hasattr(adapter, "is_quota_error")


class TestMiniMaxMCPClient:
    def test_singleton_returns_same_client(self):
        _mcp_mod._client = None
        with patch.object(MiniMaxMCPClient, "_start"):
            # After _start is mocked (no real loop), is_connected() returns False.
            # So get_minimax_mcp_client() will create a new client each time.
            # To test singleton behaviour, mock is_connected to return True.
            client1 = get_minimax_mcp_client()
            with patch.object(client1, "is_connected", return_value=True):
                client2 = get_minimax_mcp_client()
            assert client1 is client2

    def test_singleton_different_after_close(self):
        _mcp_mod._client = None
        with patch.object(MiniMaxMCPClient, "_start"):
            client1 = get_minimax_mcp_client()
            client1.close()
            client2 = get_minimax_mcp_client()
            assert client1 is not client2

    def test_missing_uvx_raises(self):
        with patch("shutil.which", return_value=None):
            client = MiniMaxMCPClient()
            with pytest.raises(RuntimeError, match="uvx"):
                client._start()

    def test_missing_api_key_raises(self):
        with (
            patch("shutil.which", return_value="/usr/bin/uvx"),
            patch(
                "os.getenv",
                side_effect=lambda k: None if k == "MINIMAX_API_KEY" else "dummy",
            ),
        ):
            client = MiniMaxMCPClient()
            with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
                client._start()

    def test_call_tool_not_connected_raises(self):
        client = MiniMaxMCPClient()
        with pytest.raises(RuntimeError, match="not connected"):
            client.call_tool("web_search", {"query": "test"})

    def test_call_tool_after_close_raises(self):
        _mcp_mod._client = None
        with patch.object(MiniMaxMCPClient, "_start"):
            client = get_minimax_mcp_client()
            client.close()
            with pytest.raises(RuntimeError, match="closed"):
                client.call_tool("web_search", {"query": "test"})

    def test_is_connected_false_before_start(self):
        client = MiniMaxMCPClient()
        assert client.is_connected() is False

    def test_close_idempotent(self):
        _mcp_mod._client = None
        with patch.object(MiniMaxMCPClient, "_start"):
            client = get_minimax_mcp_client()
            client.close()
            client.close()
