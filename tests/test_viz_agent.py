from unittest.mock import MagicMock
from agent.viz_agent import VizAgent


def test_agent_type():
    assert VizAgent.agent_type == "viz"


def test_backend_metadata():
    agent = VizAgent(backend="plotly", service=MagicMock(), session_ctx=MagicMock())
    assert agent.backend == "plotly"


def test_config_key_includes_backend():
    agent = VizAgent(backend="plotly", service=MagicMock(), session_ctx=MagicMock())
    assert agent.config_key == "viz_plotly"


def test_config_key_mpl():
    agent = VizAgent(backend="mpl", service=MagicMock(), session_ctx=MagicMock())
    assert agent.config_key == "viz_mpl"
