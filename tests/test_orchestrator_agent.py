# tests/test_orchestrator_agent.py
"""Tests for OrchestratorAgent — BaseAgent subclass."""
import queue
from unittest.mock import MagicMock

from agent.orchestrator_agent import OrchestratorAgent
from agent.base_agent import _make_message


def _make_orch(**kwargs):
    """Create OrchestratorAgent with mocked dependencies."""
    defaults = dict(
        session_ctx=MagicMock(),
        service=MagicMock(),
        system_prompt="Test orchestrator prompt.",
        tool_schemas=[],
    )
    defaults.update(kwargs)
    return OrchestratorAgent(**defaults)


def test_agent_type():
    assert OrchestratorAgent.agent_type == "orchestrator"


def test_streaming_enabled_by_default():
    agent = _make_orch()
    assert agent._streaming is True


def test_config_key():
    agent = _make_orch()
    assert agent.config_key == "orchestrator"


def test_parallel_safe_tools_are_delegation_tools():
    """Orchestrator's parallel safe tools are delegation tools."""
    safe = OrchestratorAgent._PARALLEL_SAFE_TOOLS
    assert "delegate_to_envoy" in safe
    assert "delegate_to_viz" in safe
    assert "delegate_to_data_ops" in safe


def test_guard_limits_use_orchestrator_limits():
    """Orchestrator uses orchestrator-specific loop guard limits."""
    agent = _make_orch()
    max_calls, dup_free, dup_hard = agent._get_guard_limits()
    # These should be integers > 0
    assert isinstance(max_calls, int) and max_calls > 0
    assert isinstance(dup_free, int) and dup_free > 0
    assert isinstance(dup_hard, int) and dup_hard > 0


def test_cancel_holdback():
    """Cancellation sets _was_cancelled flag."""
    agent = _make_orch()
    cancel_msg = _make_message("cancel", "user", "stop")
    agent._handle_cancel(cancel_msg)
    assert agent._was_cancelled is True


def test_hold_result():
    """hold_result() buffers results for later injection."""
    agent = _make_orch()
    agent.hold_result({"agent": "viz", "summary": "plot done"})
    assert len(agent._held_results) == 1
    assert agent._held_results[0]["agent"] == "viz"


def test_pre_request_drains_held_results():
    """_pre_request injects held results into content."""
    agent = _make_orch()
    agent.hold_result({"agent": "viz", "summary": "plot done"})
    msg = _make_message("request", "user", "What happened?")
    content = agent._pre_request(msg)
    assert "Background tasks completed" in content
    assert "viz" in content
    assert len(agent._held_results) == 0  # drained


def test_pre_request_injects_cancel_context():
    """_pre_request adds cancel context when previous op was cancelled."""
    agent = _make_orch()
    agent._was_cancelled = True
    msg = _make_message("request", "user", "Do something else")
    content = agent._pre_request(msg)
    assert "cancelled" in content.lower()
    assert agent._was_cancelled is False  # cleared after injection


def test_cycle_tracking():
    """Orchestrator tracks cycle and turn numbers."""
    agent = _make_orch()
    assert agent._cycle_number == 0
    assert agent._turn_number == 0
