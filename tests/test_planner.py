"""
Tests for agent.planner — complexity detection, PlannerAgent, and plan formatting.

Run with: python -m pytest tests/test_planner.py
"""

import threading

import pytest
from unittest.mock import MagicMock

from agent.planner import (
    format_plan_for_display,
    PlannerAgent,
)
from agent.agent_registry import PLANNER_TOOLS
from agent.tasks import Task, TaskPlan, TaskStatus, PlanStatus, create_task, create_plan
from datetime import datetime


def _make_mock_service():
    """Create a mock LLMService for testing."""
    svc = MagicMock()
    svc.get_adapter.return_value = MagicMock()
    svc.provider = "gemini"
    svc.make_tool_result.side_effect = lambda name, result, **kw: {
        "tool_name": name, "result": result,
    }
    return svc


class TestPlannerAgentInterface:
    """Test PlannerAgent class structure and interface (SubAgent-based)."""

    def _make_agent(self, **kwargs):
        defaults = dict(
            service=_make_mock_service(),
            cancel_event=threading.Event(),
        )
        defaults.update(kwargs)
        return PlannerAgent(**defaults)

    def test_has_required_methods(self):
        """PlannerAgent should have the expected public methods."""
        assert hasattr(PlannerAgent, "send")
        assert hasattr(PlannerAgent, "get_token_usage")

    def test_is_subagent(self):
        """PlannerAgent should be a SubAgent subclass."""
        from agent.sub_agent import SubAgent
        assert issubclass(PlannerAgent, SubAgent)

    def test_init_sets_agent_id(self):
        """PlannerAgent should set its agent_id."""
        agent = self._make_agent()
        assert agent.agent_id == "PlannerAgent"

    def test_init_has_produce_plan_in_schemas(self):
        """PlannerAgent always includes produce_plan in its tool schemas."""
        agent = self._make_agent()
        schema_names = {s.name for s in agent._tool_schemas}
        assert "produce_plan" in schema_names

    def test_get_token_usage_initial(self):
        """Token usage starts at zero."""
        agent = self._make_agent()
        usage = agent.get_token_usage()
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert usage["thinking_tokens"] == 0


class TestTaskRoundField:
    """Test the round field on Task dataclass."""

    def test_default_round_is_zero(self):
        task = create_task("Test", "instruction")
        assert task.round == 0

    def test_round_in_to_dict(self):
        task = create_task("Test", "instruction")
        task.round = 2
        d = task.to_dict()
        assert d["round"] == 2

    def test_round_from_dict(self):
        d = {
            "id": "test-id",
            "description": "Test",
            "instruction": "Do something",
            "status": "pending",
            "round": 3,
        }
        task = Task.from_dict(d)
        assert task.round == 3

    def test_round_from_dict_missing(self):
        """Missing round field should default to 0."""
        d = {
            "id": "test-id",
            "description": "Test",
            "instruction": "Do something",
            "status": "pending",
        }
        task = Task.from_dict(d)
        assert task.round == 0


class TestTaskPlanAddTasks:
    """Test the add_tasks() method on TaskPlan."""

    def test_add_tasks_appends(self):
        plan = create_plan("Test", [])
        assert len(plan.tasks) == 0

        tasks = [create_task("A", "a"), create_task("B", "b")]
        plan.add_tasks(tasks)
        assert len(plan.tasks) == 2

    def test_add_tasks_incremental(self):
        t1 = create_task("A", "a")
        plan = create_plan("Test", [t1])
        assert len(plan.tasks) == 1

        t2 = create_task("B", "b")
        t3 = create_task("C", "c")
        plan.add_tasks([t2, t3])
        assert len(plan.tasks) == 3
        assert plan.tasks[0].description == "A"
        assert plan.tasks[1].description == "B"
        assert plan.tasks[2].description == "C"


class TestFormatPlanForDisplay:
    """Test the plan formatting function."""

    def test_format_pending_plan(self):
        tasks = [
            create_task("Fetch PSP data", "Use fetch_data..."),
            create_task("Compute magnitude", "Use compute_magnitude..."),
            create_task("Plot result", "Use plot_computed_data..."),
        ]
        plan = create_plan("Test request", tasks)

        output = format_plan_for_display(plan)
        assert "Plan: 3 steps" in output
        assert "Fetch PSP data" in output
        assert "Compute magnitude" in output
        assert "Plot result" in output
        assert "0/3 completed" in output
        # All tasks pending, should show ASCII 'o' (Windows-compatible)
        assert "[o]" in output

    def test_format_in_progress_plan(self):
        tasks = [
            Task(
                id="1",
                description="Step 1",
                instruction="I1",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="2",
                description="Step 2",
                instruction="I2",
                status=TaskStatus.IN_PROGRESS,
            ),
            Task(
                id="3",
                description="Step 3",
                instruction="I3",
                status=TaskStatus.PENDING,
            ),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Test",
            tasks=tasks,
            created_at=datetime.now(),
            status=PlanStatus.EXECUTING,
        )

        output = format_plan_for_display(plan)
        assert "[+]" in output  # Completed
        assert "[*]" in output  # In progress
        assert "[o]" in output  # Pending
        assert "1/3 completed" in output

    def test_format_failed_task(self):
        tasks = [
            Task(
                id="1",
                description="Fetch data",
                instruction="I1",
                status=TaskStatus.FAILED,
                error="Network timeout",
            ),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Test",
            tasks=tasks,
            created_at=datetime.now(),
        )

        output = format_plan_for_display(plan)
        assert "[x]" in output  # Failed
        assert "Network timeout" in output
        assert "1 failed" in output

    def test_format_completed_plan(self):
        tasks = [
            Task(
                id="1",
                description="Step 1",
                instruction="I1",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="2",
                description="Step 2",
                instruction="I2",
                status=TaskStatus.COMPLETED,
            ),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Test",
            tasks=tasks,
            created_at=datetime.now(),
            status=PlanStatus.COMPLETED,
        )

        output = format_plan_for_display(plan)
        assert "2/2 completed" in output
        assert "failed" not in output

    def test_format_skipped_task(self):
        tasks = [
            Task(
                id="1",
                description="Step 1",
                instruction="I1",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="2",
                description="Step 2",
                instruction="I2",
                status=TaskStatus.SKIPPED,
            ),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Test",
            tasks=tasks,
            created_at=datetime.now(),
        )

        output = format_plan_for_display(plan)
        assert "[-]" in output  # Skipped

    def test_format_mission_tagged_task(self):
        tasks = [
            Task(
                id="1",
                description="Fetch PSP data",
                instruction="I1",
                mission="PSP",
                status=TaskStatus.PENDING,
            ),
            Task(
                id="2",
                description="Fetch ACE data",
                instruction="I2",
                mission="ACE",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="3",
                description="Compare",
                instruction="I3",
                status=TaskStatus.PENDING,
            ),  # No mission
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Compare PSP and ACE",
            tasks=tasks,
            created_at=datetime.now(),
        )

        output = format_plan_for_display(plan)
        assert "[PSP]" in output
        assert "[ACE]" in output
        # Task 3 has no mission tag
        lines = output.split("\n")
        compare_line = [l for l in lines if "Compare" in l][0]
        assert "[PSP]" not in compare_line
        assert "[ACE]" not in compare_line

    def test_format_with_rounds(self):
        """Tasks with non-zero rounds should be grouped by round."""
        tasks = [
            Task(
                id="1",
                description="Fetch ACE",
                instruction="I1",
                mission="ACE",
                status=TaskStatus.COMPLETED,
                round=1,
            ),
            Task(
                id="2",
                description="Fetch Wind",
                instruction="I2",
                mission="WIND",
                status=TaskStatus.COMPLETED,
                round=1,
            ),
            Task(
                id="3",
                description="Compute magnitude",
                instruction="I3",
                mission="__data_ops__",
                status=TaskStatus.COMPLETED,
                round=2,
            ),
            Task(
                id="4",
                description="Plot comparison",
                instruction="I4",
                mission="__visualization__",
                status=TaskStatus.PENDING,
                round=3,
            ),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Compare ACE and Wind mag",
            tasks=tasks,
            created_at=datetime.now(),
        )

        output = format_plan_for_display(plan)
        assert "Round 1:" in output
        assert "Round 2:" in output
        assert "Round 3:" in output
        assert "Plan: 4 steps" in output
        assert "Fetch ACE" in output
        assert "Plot comparison" in output

    def test_format_without_rounds(self):
        """Tasks with round=0 should display without round headers."""
        tasks = [
            Task(
                id="1",
                description="Step A",
                instruction="I1",
                status=TaskStatus.PENDING,
            ),
            Task(
                id="2",
                description="Step B",
                instruction="I2",
                status=TaskStatus.PENDING,
            ),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Test",
            tasks=tasks,
            created_at=datetime.now(),
        )

        output = format_plan_for_display(plan)
        assert "Round" not in output
        assert "Step A" in output
        assert "Step B" in output


class TestPlannerAgentWithTools:
    """Test PlannerAgent tool integration (no API calls needed)."""

    def _dummy_executor(self, tool_name, tool_args, tc_id=None):
        return {"status": "success", "message": "mock"}

    def _make_agent(self, **kwargs):
        defaults = dict(
            service=_make_mock_service(),
            tool_executor=self._dummy_executor,
            cancel_event=threading.Event(),
        )
        defaults.update(kwargs)
        return PlannerAgent(**defaults)

    def test_init_with_tool_executor_has_declarations(self):
        """When tool_executor is provided, function declarations should be built."""
        agent = self._make_agent()
        assert agent.tool_executor is not None
        assert len(agent._tool_schemas) > 0

    def test_init_without_tool_executor_has_produce_plan(self):
        """When tool_executor is None, schemas still include PLANNER_TOOLS + produce_plan."""
        agent = self._make_agent(tool_executor=None)
        schema_names = {s.name for s in agent._tool_schemas}
        assert "produce_plan" in schema_names
        # PLANNER_TOOLS are always included (research + discovery)
        assert len(agent._tool_schemas) > 1

    def test_tool_names_include_research_tools(self):
        """Function declarations should include research, discovery, and produce_plan tools."""
        agent = self._make_agent()
        tool_names = {fd.name for fd in agent._tool_schemas}
        # Research tools
        assert "list_missions" in tool_names
        assert "web_search" in tool_names
        assert "list_fetched_data" in tool_names
        # Discovery tools (planner uses these to validate plans)
        assert "search_datasets" in tool_names
        assert "list_parameters" in tool_names
        assert "browse_datasets" in tool_names
        assert "get_dataset_docs" in tool_names
        assert "search_full_catalog" in tool_names
        # Plan submission tool
        assert "produce_plan" in tool_names

    def test_tool_names_exclude_routing_and_visualization(self):
        """Function declarations should NOT include routing or visualization tools."""
        agent = self._make_agent()
        tool_names = {fd.name for fd in agent._tool_schemas}
        # Routing tools should NOT be present
        assert "delegate_to_envoy" not in tool_names
        assert "delegate_to_viz" not in tool_names
        assert "delegate_to_data_ops" not in tool_names
        assert "delegate_to_planner" not in tool_names
        # Visualization tools should NOT be present
        assert "plot_data" not in tool_names
        assert "style_plot" not in tool_names
        assert "manage_plot" not in tool_names
        # Fetch / compute tools should NOT be present
        assert "fetch_data" not in tool_names
        assert "run_code" not in tool_names

    def test_planner_tools_constant(self):
        """PLANNER_TOOLS should include research and discovery tools."""
        assert "list_missions" in PLANNER_TOOLS
        assert "web_search" in PLANNER_TOOLS
        assert "list_fetched_data" in PLANNER_TOOLS
        # Discovery tools (planner uses these to validate plans)
        assert "search_datasets" in PLANNER_TOOLS
        assert "list_parameters" in PLANNER_TOOLS
        assert "browse_datasets" in PLANNER_TOOLS
        assert "get_dataset_docs" in PLANNER_TOOLS
        assert "search_full_catalog" in PLANNER_TOOLS

