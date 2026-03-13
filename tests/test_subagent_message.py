"""Tests for subagent sending messages to orchestrator."""
from agent.base_agent import BaseAgent


def test_subagent_has_send_to_orchestrator():
    """BaseAgent should have method to send to orchestrator inbox."""
    assert hasattr(BaseAgent, 'send_to_orchestrator')
