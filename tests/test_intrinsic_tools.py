"""Tests for intrinsic agent tools (vision, web_search)."""

from unittest.mock import MagicMock, patch
import pytest

from agent.llm.base import LLMResponse


class TestLLMServiceGenerateVision:
    """LLMService.generate_vision routes to vision_provider adapter."""

    def test_generate_vision_routes_to_provider(self):
        mock_adapter = MagicMock()
        mock_adapter.generate_vision.return_value = LLMResponse(text="I see a plot.")
        from agent.llm.service import LLMService
        svc = MagicMock(spec=LLMService)
        svc._config = {"vision_provider": "gemini"}
        svc.get_adapter = MagicMock(return_value=mock_adapter)
        svc._get_provider_defaults = MagicMock(return_value={"model": "gemini-2.5-flash"})
        result = LLMService.generate_vision(svc, "What is this?", b"png_bytes")
        mock_adapter.generate_vision.assert_called_once_with(
            "What is this?", b"png_bytes", model="gemini-2.5-flash", mime_type="image/png"
        )
        assert result.text == "I see a plot."

    def test_generate_vision_no_provider_returns_empty(self):
        from agent.llm.service import LLMService
        svc = MagicMock(spec=LLMService)
        svc._config = {}
        result = LLMService.generate_vision(svc, "What is this?", b"png_bytes")
        assert result.text == ""

    def test_generate_vision_bad_provider_returns_empty(self):
        from agent.llm.service import LLMService
        svc = MagicMock(spec=LLMService)
        svc._config = {"vision_provider": "nonexistent"}
        svc.get_adapter = MagicMock(side_effect=RuntimeError("No API key"))
        result = LLMService.generate_vision(svc, "What is this?", b"png_bytes")
        assert result.text == ""

    def test_generate_vision_custom_mime_type(self):
        mock_adapter = MagicMock()
        mock_adapter.generate_vision.return_value = LLMResponse(text="JPEG image.")
        from agent.llm.service import LLMService
        svc = MagicMock(spec=LLMService)
        svc._config = {"vision_provider": "openai"}
        svc.get_adapter = MagicMock(return_value=mock_adapter)
        svc._get_provider_defaults = MagicMock(return_value={"model": "gpt-4o"})
        result = LLMService.generate_vision(
            svc, "Describe this", b"jpeg_bytes", mime_type="image/jpeg"
        )
        mock_adapter.generate_vision.assert_called_once_with(
            "Describe this", b"jpeg_bytes", model="gpt-4o", mime_type="image/jpeg"
        )
        assert result.text == "JPEG image."

    def test_generate_vision_no_defaults_uses_empty_model(self):
        mock_adapter = MagicMock()
        mock_adapter.generate_vision.return_value = LLMResponse(text="ok")
        from agent.llm.service import LLMService
        svc = MagicMock(spec=LLMService)
        svc._config = {"vision_provider": "gemini"}
        svc.get_adapter = MagicMock(return_value=mock_adapter)
        svc._get_provider_defaults = MagicMock(return_value=None)
        result = LLMService.generate_vision(svc, "Q?", b"data")
        mock_adapter.generate_vision.assert_called_once_with(
            "Q?", b"data", model="", mime_type="image/png"
        )


class TestLLMAdapterGenerateVisionDefault:
    """Base LLMAdapter.generate_vision returns empty response."""

    def test_default_returns_empty(self):
        from agent.llm.base import LLMAdapter
        # LLMAdapter is abstract, so we create a minimal concrete subclass
        class StubAdapter(LLMAdapter):
            def create_session(self, *a, **kw): ...
            def create_chat(self, *a, **kw): ...
            def generate(self, *a, **kw): ...
            def make_tool_result_message(self, *a, **kw): ...
            def make_multimodal_message(self, *a, **kw): ...
            def is_quota_error(self, exc): return False

        adapter = StubAdapter.__new__(StubAdapter)
        result = adapter.generate_vision("question", b"bytes")
        assert result.text == ""


# ---------------------------------------------------------------------------
# BaseAgent intrinsic tool registration
# ---------------------------------------------------------------------------


class TestBaseAgentIntrinsicTools:
    """BaseAgent registers vision and web_search as intrinsic local tools."""

    def _make_agent(self, **kwargs):
        """Create a minimal BaseAgent for testing."""
        from agent.base_agent import BaseAgent

        svc = MagicMock()
        svc._config = {}
        svc._provider = "gemini"
        svc._base_url = None
        with patch("agent.base_agent._config") as mock_cfg:
            mock_cfg.resolve_agent_model.return_value = ("gemini", "gemini-2.5-flash", None)
            agent = BaseAgent(
                agent_id="test:001",
                service=svc,
                system_prompt="Test agent.",
                **kwargs,
            )
        return agent

    def test_intrinsic_tools_registered(self):
        agent = self._make_agent()
        assert "vision" in agent._local_tools
        assert "web_search" in agent._local_tools

    def test_intrinsic_schemas_added(self):
        agent = self._make_agent()
        names = {s.name for s in agent._tool_schemas}
        assert "vision" in names
        assert "web_search" in names

    def test_vision_handler_missing_path(self):
        agent = self._make_agent()
        handler = agent._local_tools["vision"]
        result = handler(MagicMock(), {"question": "What do you see?"}, None)
        assert result["status"] == "error"
        assert "image_path" in result["message"]

    def test_vision_handler_file_not_found(self):
        agent = self._make_agent()
        handler = agent._local_tools["vision"]
        result = handler(MagicMock(), {"image_path": "/nonexistent/plot.png", "question": "?"}, None)
        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_vision_handler_reads_file(self, tmp_path):
        agent = self._make_agent()
        agent.service.generate_vision.return_value = LLMResponse(text="A line plot.")
        handler = agent._local_tools["vision"]
        img = tmp_path / "figure.png"
        img.write_bytes(b"fake_png_data")
        result = handler(MagicMock(), {"image_path": str(img), "question": "Describe this."}, None)
        assert result["status"] == "ok"
        assert result["analysis"] == "A line plot."
        agent.service.generate_vision.assert_called_once_with(
            "Describe this.", b"fake_png_data", mime_type="image/png"
        )

    def test_vision_handler_jpeg_mime(self, tmp_path):
        agent = self._make_agent()
        agent.service.generate_vision.return_value = LLMResponse(text="Photo.")
        handler = agent._local_tools["vision"]
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake_jpeg")
        result = handler(MagicMock(), {"image_path": str(img), "question": "What?"}, None)
        assert result["status"] == "ok"
        agent.service.generate_vision.assert_called_once_with(
            "What?", b"fake_jpeg", mime_type="image/jpeg"
        )

    def test_web_search_handler(self):
        agent = self._make_agent()
        agent.service.web_search.return_value = LLMResponse(text="Solar storm on Jan 1.")
        handler = agent._local_tools["web_search"]
        result = handler(MagicMock(), {"query": "solar storms 2024"}, None)
        assert result["status"] == "ok"
        assert result["results"] == "Solar storm on Jan 1."

    def test_web_search_handler_missing_query(self):
        agent = self._make_agent()
        handler = agent._local_tools["web_search"]
        result = handler(MagicMock(), {}, None)
        assert result["status"] == "error"

    def test_web_search_handler_no_results(self):
        agent = self._make_agent()
        agent.service.web_search.return_value = LLMResponse(text="")
        handler = agent._local_tools["web_search"]
        result = handler(MagicMock(), {"query": "nothing"}, None)
        assert result["status"] == "error"


class TestEurekaAgentIntrinsicTools:
    """EurekaAgent inherits intrinsic tools and has its own local tools."""

    def test_eureka_has_vision_and_web_search(self):
        """EurekaAgent inherits vision and web_search from BaseAgent."""
        from agent.eureka_agent import EurekaAgent
        svc = MagicMock()
        svc._config = {}
        svc._provider = "gemini"
        svc._base_url = None
        with patch("agent.base_agent._config") as mock_cfg:
            mock_cfg.resolve_agent_model.return_value = ("gemini", "gemini-2.5-flash", None)
            agent = EurekaAgent(service=svc)
        assert "vision" in agent._local_tools
        assert "web_search" in agent._local_tools
        assert "submit_finding" in agent._local_tools
        assert "submit_suggestion" in agent._local_tools
        assert "get_session_figure" not in agent._local_tools


class TestIntrinsicToolOptOut:
    """Agents can opt out of intrinsic tools."""

    def test_memory_agent_no_intrinsic_tools(self):
        """MemoryAgent has no vision or web_search tools."""
        from agent.memory_agent import MemoryAgent
        svc = MagicMock()
        svc._config = {}
        svc._provider = "gemini"
        svc._base_url = None
        with patch("agent.base_agent._config") as mock_cfg:
            mock_cfg.resolve_agent_model.return_value = ("gemini", "gemini-2.5-flash", None)
            agent = MemoryAgent(service=svc)
        assert "vision" not in agent._local_tools
        assert "web_search" not in agent._local_tools
        schema_names = {s.name for s in agent._tool_schemas}
        assert "vision" not in schema_names
        assert "web_search" not in schema_names
