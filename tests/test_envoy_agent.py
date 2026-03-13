"""Tests for EnvoyAgent — BaseAgent subclass with kind/instance."""
from unittest.mock import MagicMock
from agent.envoy_agent import EnvoyAgent


def test_agent_type():
    assert EnvoyAgent.agent_type == "envoy"


def test_kind_and_instance():
    agent = EnvoyAgent(
        kind="cdaweb",
        instance_id="ace",
        service=MagicMock(),
        session_ctx=MagicMock(),
        tool_schemas=[],
        system_prompt="test",
    )
    assert agent.kind == "cdaweb"
    assert agent.instance_id == "ace"


def test_config_key_includes_kind():
    agent = EnvoyAgent(
        kind="cdaweb",
        instance_id="ace",
        service=MagicMock(),
        session_ctx=MagicMock(),
        tool_schemas=[],
        system_prompt="test",
    )
    assert agent.config_key == "envoy_cdaweb"


def test_agent_id_contains_kind_and_instance():
    agent = EnvoyAgent(
        kind="cdaweb",
        instance_id="ace",
        service=MagicMock(),
        session_ctx=MagicMock(),
        tool_schemas=[],
        system_prompt="test",
    )
    assert "envoy:cdaweb:ace" in agent.agent_id
