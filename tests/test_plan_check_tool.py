"""Tests for plan_check tool."""
from agent.core import OrchestratorAgent


def test_orchestrator_has_plan_check_tool():
    """Orchestrator should have plan_check tool."""
    # Check tool is registered
    from agent.tools import get_function_schemas
    schemas = get_function_schemas()
    tool_names = [s.name for s in schemas]
    assert "plan_check" in tool_names
