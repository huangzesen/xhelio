"""Tests for subagent sending messages to orchestrator."""
from agent.sub_agent import SubAgent


def test_subagent_has_send_to_orchestrator():
    """SubAgent should have method to send to orchestrator inbox."""
    assert hasattr(SubAgent, 'send_to_orchestrator')
