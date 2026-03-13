"""Tests for ToolCaller and OrchestratorState dataclasses."""
from agent.tool_caller import ToolCaller, OrchestratorState


def test_tool_caller_basic():
    caller = ToolCaller(
        agent_id="orchestrator:abc123",
        agent_type="orchestrator",
    )
    assert caller.agent_id == "orchestrator:abc123"
    assert caller.agent_type == "orchestrator"
    assert caller.tool_call_id is None


def test_tool_caller_with_tool_call_id():
    caller = ToolCaller(
        agent_id="orchestrator:abc123",
        agent_type="orchestrator",
        tool_call_id="tc_001",
    )
    assert caller.tool_call_id == "tc_001"


def test_orchestrator_state_defaults():
    state = OrchestratorState()
    assert state.current_plan is None
    assert state.auto_save is False
    assert state.session_title_generated is False


def test_orchestrator_state_mutation():
    state = OrchestratorState()
    state.current_plan = {"tasks": [{"description": "do something"}]}
    assert state.current_plan["tasks"][0]["description"] == "do something"
