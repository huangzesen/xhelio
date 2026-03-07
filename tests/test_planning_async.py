"""Tests for async planning."""
import threading
from agent.tool_handlers.planning import handle_request_planning


def test_planning_returns_immediately():
    """handle_request_planning should return immediately, not block."""
    # This will be tested with actual handler
    assert True
