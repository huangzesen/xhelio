"""Tests for thin BaseAgent subclasses (v2 agents)."""

from unittest.mock import MagicMock

from agent.data_ops_agent import DataOpsAgent
from agent.data_io_agent import DataIOAgent
from agent.memory_agent import MemoryAgent
from agent.eureka_agent import EurekaAgent


def _mock_deps():
    return dict(service=MagicMock(), session_ctx=MagicMock())


def test_data_ops_agent_type():
    assert DataOpsAgent.agent_type == "data_ops"


def test_data_io_agent_type():
    assert DataIOAgent.agent_type == "data_io"


def test_memory_agent_type():
    assert MemoryAgent.agent_type == "memory"


def test_eureka_agent_type():
    assert EurekaAgent.agent_type == "eureka"


def test_eureka_local_tools():
    agent = EurekaAgent(**_mock_deps())
    assert "submit_finding" in agent._local_tools
    assert "submit_suggestion" in agent._local_tools
    # vision is now an intrinsic tool on BaseAgent, not a local EurekaAgent tool
    assert "vision" in agent._local_tools


def test_memory_local_tools():
    agent = MemoryAgent(**_mock_deps())
    assert "manage_memory" in agent._local_tools
    # get_event_details was replaced by xhelio__events (server tool, not local)
    assert "get_event_details" not in agent._local_tools
