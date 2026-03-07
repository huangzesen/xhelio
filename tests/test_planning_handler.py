"""Tests for the request_planning tool handler returning structured plans."""

import pytest
from unittest.mock import MagicMock, patch


class TestHandleRequestPlanningReturnsStructuredPlan:
    """handle_request_planning should return a structured plan, not execute it."""

    def test_returns_plan_on_success(self):
        """Successful planning returns the plan dict in the tool result."""
        from agent.tool_handlers.planning import handle_request_planning

        mock_plan = {
            "reasoning": "Need to fetch PSP data and plot",
            "tasks": [
                {
                    "description": "Fetch PSP mag data",
                    "instruction": "Fetch magnetic field for 2024-06-01 to 2024-07-15",
                    "mission": "PSP",
                    "candidate_datasets": ["PSP_FLD_L2_MAG_RTN_4_SA_PER_CYC"],
                },
                {
                    "description": "Plot data",
                    "instruction": "Plot the magnetic field data",
                    "mission": "__visualization__",
                },
            ],
            "summary": "Fetch PSP mag and plot",
            "time_range_validated": True,
        }

        mock_orch = MagicMock()
        mock_planner = MagicMock()
        mock_planner.start_planning.return_value = mock_plan
        mock_orch._get_or_create_planner_agent.return_value = mock_planner
        mock_orch._event_bus = MagicMock()
        mock_orch._cancel_event = MagicMock()
        mock_orch._cancel_event.is_set.return_value = False

        result = handle_request_planning(
            mock_orch,
            {
                "request": "show PSP mag data",
                "reasoning": "data fetch + plot",
                "time_start": "2024-06-01",
                "time_end": "2024-07-15",
            },
        )

        assert result["status"] == "success"
        assert result["planning_used"] is True
        assert "plan" in result
        assert len(result["plan"]["tasks"]) == 2
        assert result["plan"]["tasks"][0]["mission"] == "PSP"

    def test_returns_error_on_planning_failure(self):
        """When planner returns None, tool result should indicate failure."""
        from agent.tool_handlers.planning import handle_request_planning

        mock_orch = MagicMock()
        mock_planner = MagicMock()
        mock_planner.start_planning.return_value = None
        mock_orch._get_or_create_planner_agent.return_value = mock_planner
        mock_orch._event_bus = MagicMock()
        mock_orch._cancel_event = MagicMock()
        mock_orch._cancel_event.is_set.return_value = False

        result = handle_request_planning(
            mock_orch,
            {
                "request": "show data",
                "reasoning": "test",
                "time_start": "2024-01-01",
                "time_end": "2024-01-15",
            },
        )

        assert result["status"] == "error"
        assert "planning_used" in result

    def test_does_not_call_handle_planning_request(self):
        """The handler must NOT call _handle_planning_request (the old loop)."""
        from agent.tool_handlers.planning import handle_request_planning

        mock_orch = MagicMock()
        mock_planner = MagicMock()
        mock_planner.start_planning.return_value = {
            "reasoning": "x",
            "tasks": [],
            "summary": "y",
            "time_range_validated": True,
        }
        mock_orch._get_or_create_planner_agent.return_value = mock_planner
        mock_orch._event_bus = MagicMock()
        mock_orch._cancel_event = MagicMock()
        mock_orch._cancel_event.is_set.return_value = False

        handle_request_planning(
            mock_orch,
            {
                "request": "test",
                "reasoning": "test",
                "time_start": "2024-01-01",
                "time_end": "2024-01-15",
            },
        )

        mock_orch._handle_planning_request.assert_not_called()

    def test_injects_time_range_into_planning_message(self):
        """When time_start and time_end are provided, they should be passed to the planner."""
        from agent.tool_handlers.planning import handle_request_planning

        mock_orch = MagicMock()
        mock_planner = MagicMock()
        mock_planner.start_planning.return_value = {
            "reasoning": "x",
            "tasks": [],
            "summary": "y",
            "time_range_validated": True,
        }
        mock_orch._get_or_create_planner_agent.return_value = mock_planner
        mock_orch._event_bus = MagicMock()
        mock_orch._cancel_event = MagicMock()
        mock_orch._cancel_event.is_set.return_value = False

        handle_request_planning(
            mock_orch,
            {
                "request": "show PSP data",
                "reasoning": "fetch + plot",
                "time_start": "2024-06-01",
                "time_end": "2024-07-15",
            },
        )

        call_args = mock_planner.start_planning.call_args[0][0]
        assert "2024-06-01" in call_args
        assert "2024-07-15" in call_args

    def test_resets_planner_after_use(self):
        """Planner should be reset after each use so the next call gets a fresh session."""
        from agent.tool_handlers.planning import handle_request_planning

        mock_orch = MagicMock()
        mock_planner = MagicMock()
        mock_planner.start_planning.return_value = {
            "reasoning": "x",
            "tasks": [],
            "summary": "y",
            "time_range_validated": True,
        }
        mock_orch._get_or_create_planner_agent.return_value = mock_planner
        mock_orch._event_bus = MagicMock()
        mock_orch._cancel_event = MagicMock()
        mock_orch._cancel_event.is_set.return_value = False

        handle_request_planning(
            mock_orch,
            {
                "request": "test",
                "reasoning": "test",
                "time_start": "2024-01-01",
                "time_end": "2024-01-15",
            },
        )

        mock_planner.reset.assert_called_once()

    def test_sets_current_plan_on_success(self):
        """Orchestrator._current_plan should be set to the plan dict after successful planning."""
        from agent.tool_handlers.planning import handle_request_planning

        mock_plan = {
            "reasoning": "Need PSP data",
            "tasks": [
                {"description": "Fetch", "instruction": "Fetch PSP", "mission": "PSP"}
            ],
            "summary": "Fetch PSP",
            "time_range_validated": True,
        }

        mock_orch = MagicMock()
        mock_orch._current_plan = None  # Initialize to None
        mock_planner = MagicMock()
        mock_planner.start_planning.return_value = mock_plan
        mock_orch._get_or_create_planner_agent.return_value = mock_planner
        mock_orch._event_bus = MagicMock()
        mock_orch._cancel_event = MagicMock()
        mock_orch._cancel_event.is_set.return_value = False

        result = handle_request_planning(
            mock_orch,
            {
                "request": "show PSP data",
                "reasoning": "fetch + plot",
                "time_start": "2024-06-01",
                "time_end": "2024-07-15",
            },
        )

        assert result["status"] == "success"
        assert mock_orch._current_plan == mock_plan
