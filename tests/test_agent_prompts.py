"""Test that all agents use prompt_builder for system prompts, not stubs."""

from unittest.mock import MagicMock

from agent.llm import LLMService


def _make_service() -> LLMService:
    """Create a minimal mock LLMService."""
    svc = MagicMock(spec=LLMService)
    svc.create_session.return_value = MagicMock()
    svc.get_adapter.return_value = MagicMock()
    return svc


class TestVizAgentPrompt:
    """VizAgent should get its prompt from build_viz_system_prompt."""

    def test_plotly_uses_prompt_builder(self):
        from agent.viz_agent import VizAgent
        agent = VizAgent(backend="plotly", service=MagicMock(), session_ctx=MagicMock())
        assert len(agent.system_prompt) > 100, \
            f"Viz plotly got stub prompt ({len(agent.system_prompt)} chars)"

    def test_matplotlib_alias_uses_prompt_builder(self):
        from agent.viz_agent import VizAgent
        agent = VizAgent(backend="matplotlib", service=MagicMock(), session_ctx=MagicMock())
        assert agent.backend == "mpl"
        assert agent.config_key == "viz_mpl"
        assert len(agent.system_prompt) > 100

    def test_mpl_uses_prompt_builder(self):
        from agent.viz_agent import VizAgent
        agent = VizAgent(backend="mpl", service=MagicMock(), session_ctx=MagicMock())
        assert len(agent.system_prompt) > 100

    def test_jsx_uses_prompt_builder(self):
        from agent.viz_agent import VizAgent
        agent = VizAgent(backend="jsx", service=MagicMock(), session_ctx=MagicMock())
        assert len(agent.system_prompt) > 100


class TestDataOpsAgentPrompt:
    """DataOpsAgent should get its prompt from build_data_ops_system_prompt."""

    def test_uses_prompt_builder(self):
        from agent.data_ops_agent import DataOpsAgent
        agent = DataOpsAgent(service=MagicMock(), session_ctx=MagicMock())
        assert len(agent.system_prompt) > 100, \
            f"DataOps got stub prompt ({len(agent.system_prompt)} chars)"


class TestDataIOAgentPrompt:
    """DataIOAgent should get its prompt from build_data_io_system_prompt."""

    def test_uses_prompt_builder(self):
        from agent.data_io_agent import DataIOAgent
        agent = DataIOAgent(service=MagicMock(), session_ctx=MagicMock())
        assert len(agent.system_prompt) > 100, \
            f"DataIO got stub prompt ({len(agent.system_prompt)} chars)"


class TestEurekaAgentPrompt:
    """EurekaAgent should get its prompt from build_eureka_system_prompt."""

    def test_uses_prompt_builder(self):
        from agent.eureka_agent import EurekaAgent
        agent = EurekaAgent(service=MagicMock(), session_ctx=MagicMock())
        assert len(agent.system_prompt) > 100, \
            f"Eureka got stub prompt ({len(agent.system_prompt)} chars)"
