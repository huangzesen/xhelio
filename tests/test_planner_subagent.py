"""Tests for PlannerAgent as a SubAgent subclass."""

import threading
import pytest
from unittest.mock import MagicMock

from agent.sub_agent import SubAgent
from agent.planner import PlannerAgent


def test_planner_is_subagent():
    """PlannerAgent must be a SubAgent subclass."""
    assert issubclass(PlannerAgent, SubAgent)


def test_planner_init():
    """PlannerAgent can be instantiated with standard SubAgent args."""
    adapter = MagicMock()
    agent = PlannerAgent(
        adapter=adapter,
        model_name="test-model",
        tool_executor=lambda name, args, tc_id=None: {"status": "ok"},
        cancel_event=threading.Event(),
    )
    assert agent.agent_id == "PlannerAgent"
    assert agent.model_name == "test-model"


def test_planner_has_produce_plan_tool():
    """PlannerAgent tool schemas must include produce_plan."""
    adapter = MagicMock()
    agent = PlannerAgent(
        adapter=adapter,
        model_name="test-model",
        tool_executor=lambda name, args, tc_id=None: {"status": "ok"},
        cancel_event=threading.Event(),
    )
    schemas = agent._tool_schemas
    names = [s.name for s in schemas]
    assert "produce_plan" in names
