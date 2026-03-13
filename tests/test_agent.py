"""
Tests for the agent core logic.

Run with: python -m pytest tests/test_agent.py

Note: Most tests mock external dependencies to avoid network calls and JVM startup.
"""

import math
import pytest
from unittest.mock import patch, MagicMock
from agent.time_utils import parse_time_range, TimeRange, TimeRangeError
from agent.core import _sanitize_for_json
from datetime import datetime, timedelta, timezone


class TestParseTimeRange:
    def test_last_week(self):
        result = parse_time_range("last week")
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        assert result.end == today
        assert result.start == today - timedelta(days=7)

    def test_last_month(self):
        result = parse_time_range("last month")
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        assert result.end == today
        assert result.start == today - timedelta(days=30)

    def test_last_n_days(self):
        result = parse_time_range("last 3 days")
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        assert result.end == today
        assert result.start == today - timedelta(days=3)

    def test_month_year(self):
        result = parse_time_range("January 2024")
        assert result.to_time_range_string() == "2024-01-01 to 2024-02-01"

    def test_december_year(self):
        result = parse_time_range("December 2024")
        assert result.to_time_range_string() == "2024-12-01 to 2025-01-01"

    def test_single_date(self):
        result = parse_time_range("2024-06-15")
        assert result.to_time_range_string() == "2024-06-15 to 2024-06-16"

    def test_already_formatted(self):
        result = parse_time_range("2024-01-01 to 2024-01-31")
        assert result.to_time_range_string() == "2024-01-01 to 2024-01-31"

    def test_abbreviated_month(self):
        result = parse_time_range("Jan 2024")
        assert result.to_time_range_string() == "2024-01-01 to 2024-02-01"

    def test_datetime_range(self):
        result = parse_time_range("2024-01-15T06:00 to 2024-01-15T18:00")
        assert result.start == datetime(2024, 1, 15, 6, 0)
        assert result.end == datetime(2024, 1, 15, 18, 0)
        assert "T" in result.to_time_range_string()

    def test_datetime_range_with_seconds(self):
        result = parse_time_range("2024-01-15T06:00:30 to 2024-01-15T18:30:45")
        assert result.start == datetime(2024, 1, 15, 6, 0, 30)
        assert result.end == datetime(2024, 1, 15, 18, 30, 45)

    def test_unparseable_raises_error(self):
        with pytest.raises(TimeRangeError, match="Could not parse"):
            parse_time_range("gobbledygook")

    def test_bad_date_format_raises_error(self):
        with pytest.raises(TimeRangeError, match="Could not parse"):
            parse_time_range("15/01/2024")

    def test_start_after_end_raises_error(self):
        with pytest.raises(ValueError, match="must be before"):
            parse_time_range("2024-01-20 to 2024-01-15")

    def test_day_precision_omits_time(self):
        result = parse_time_range("2024-01-15 to 2024-01-20")
        s = result.to_time_range_string()
        assert "T" not in s
        assert s == "2024-01-15 to 2024-01-20"

    def test_sub_day_includes_time(self):
        result = parse_time_range("2024-01-15T06:00 to 2024-01-15T18:00")
        s = result.to_time_range_string()
        assert "T" in s
        assert s == "2024-01-15T06:00:00 to 2024-01-15T18:00:00"


class TestAgentToolExecution:
    """Tests for agent tool execution logic (mocked)."""

    @pytest.fixture
    def mock_genai(self):
        """Mock google.generativeai module."""
        with patch("agent.core.genai") as mock:
            yield mock

    @pytest.fixture
    def mock_renderer(self):
        """Mock renderer commands."""
        with patch("agent.core.get_commands") as mock:
            mock_commands = MagicMock()
            mock.return_value = mock_commands
            yield mock_commands

    def _make_agent(self):
        """Create a minimal OrchestratorAgent with _execute_tool prerequisites."""
        import threading
        from agent.core import OrchestratorAgent

        agent = OrchestratorAgent.__new__(OrchestratorAgent)
        agent.verbose = False
        agent._renderer = None
        agent._tls = threading.local()
        agent._tls.current_agent_type = "orchestrator"
        agent._event_bus = MagicMock()
        return agent

    def test_ask_clarification_tool(self):
        """Test that ask_clarification returns question data."""
        agent = self._make_agent()
        result = agent._execute_tool("ask_clarification", {
            "question": "Which parameter?",
            "options": ["Magnitude", "Vector"],
            "context": "Multiple parameters available",
        })

        assert result["status"] == "clarification_needed"
        assert result["question"] == "Which parameter?"
        assert result["options"] == ["Magnitude", "Vector"]


class TestSanitizeForJson:
    """Tests for _sanitize_for_json NaN/Inf cleaning."""

    def test_nan_replaced_with_none(self):
        assert _sanitize_for_json(float("nan")) is None

    def test_inf_replaced_with_none(self):
        assert _sanitize_for_json(float("inf")) is None

    def test_neg_inf_replaced_with_none(self):
        assert _sanitize_for_json(float("-inf")) is None

    def test_normal_float_unchanged(self):
        assert _sanitize_for_json(3.14) == 3.14

    def test_zero_unchanged(self):
        assert _sanitize_for_json(0.0) == 0.0

    def test_int_unchanged(self):
        assert _sanitize_for_json(42) == 42

    def test_string_unchanged(self):
        assert _sanitize_for_json("hello") == "hello"

    def test_none_unchanged(self):
        assert _sanitize_for_json(None) is None

    def test_nested_dict(self):
        result = _sanitize_for_json({
            "status": "success",
            "statistics": {
                "col1": {"mean": 1.5, "std": float("nan"), "min": float("-inf")},
            },
        })
        assert result["statistics"]["col1"]["mean"] == 1.5
        assert result["statistics"]["col1"]["std"] is None
        assert result["statistics"]["col1"]["min"] is None

    def test_list_with_nan(self):
        result = _sanitize_for_json([1.0, float("nan"), 3.0, float("inf")])
        assert result == [1.0, None, 3.0, None]

    def test_tuple_with_nan(self):
        result = _sanitize_for_json((float("nan"), 2.0))
        assert result == [None, 2.0]

    def test_deeply_nested(self):
        result = _sanitize_for_json({"a": [{"b": float("nan")}]})
        assert result == {"a": [{"b": None}]}

    def test_empty_structures(self):
        assert _sanitize_for_json({}) == {}
        assert _sanitize_for_json([]) == []

    def test_describe_data_scenario(self):
        """Simulate the exact describe_data output for a 1-row DataFrame."""
        stats = {
            "col1": {
                "min": 5.0,
                "max": 5.0,
                "mean": 5.0,
                "std": float("nan"),  # std of 1 value is NaN
                "25%": 5.0,
                "50%": 5.0,
                "75%": 5.0,
            }
        }
        result = _sanitize_for_json({"status": "success", "statistics": stats})
        assert result["statistics"]["col1"]["std"] is None
        assert result["statistics"]["col1"]["mean"] == 5.0


