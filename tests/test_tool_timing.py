"""Tests for agent/tool_timing.py"""

import time
import pytest
from datetime import datetime, timezone
from agent.tool_timing import stamp_tool_result, ToolTimer


class TestStampToolResult:
    """Tests for stamp_tool_result function."""

    def test_injects_both_fields(self):
        """stamp_tool_result should inject current_time and _elapsed_ms."""
        result = {"status": "success", "data": [1, 2, 3]}
        stamped = stamp_tool_result(result, 150)
        
        assert "current_time" in stamped
        assert "_elapsed_ms" in stamped
        assert stamped["_elapsed_ms"] == 150

    def test_preserves_existing_keys(self):
        """stamp_tool_result should preserve existing keys."""
        result = {"status": "success", "data": [1, 2, 3], "existing": "value"}
        stamped = stamp_tool_result(result, 200)
        
        assert stamped["status"] == "success"
        assert stamped["data"] == [1, 2, 3]
        assert stamped["existing"] == "value"

    def test_ts_is_iso8601_utc(self):
        """current_time should be valid ISO 8601 UTC format."""
        result = {"status": "ok"}
        stamped = stamp_tool_result(result, 0)
        
        # Should parse without error
        parsed = datetime.fromisoformat(stamped["current_time"].replace("Z", "+00:00"))
        assert parsed.tzinfo == timezone.utc

    def test_elapsed_ms_is_non_negative_integer(self):
        """_elapsed_ms should be a non-negative integer."""
        result = {"status": "ok"}
        
        # Test zero
        stamped = stamp_tool_result(result.copy(), 0)
        assert isinstance(stamped["_elapsed_ms"], int)
        assert stamped["_elapsed_ms"] >= 0
        
        # Test positive
        stamped = stamp_tool_result(result.copy(), 5000)
        assert stamped["_elapsed_ms"] == 5000

    def test_modifies_in_place(self):
        """stamp_tool_result should modify the dict in-place."""
        result = {"status": "ok"}
        original_id = id(result)
        stamped = stamp_tool_result(result, 100)
        
        # Same object
        assert id(stamped) == original_id

    def test_stamp_error_result(self):
        """Stamping error result dicts should work correctly."""
        error_result = {"status": "error", "message": "Not found"}
        stamped = stamp_tool_result(error_result, 42)
        
        assert stamped["status"] == "error"
        assert stamped["message"] == "Not found"
        assert stamped["_elapsed_ms"] == 42

    def test_stamp_with_zero_elapsed(self):
        """Stamping with elapsed_ms=0 should work."""
        result = {"status": "ok"}
        stamped = stamp_tool_result(result, 0)
        
        assert stamped["_elapsed_ms"] == 0
        assert "current_time" in stamped


class TestToolTimer:
    """Tests for ToolTimer context manager."""

    def test_measures_elapsed_time(self):
        """ToolTimer should measure elapsed time correctly."""
        timer = ToolTimer()
        with timer:
            time.sleep(0.05)  # Sleep 50ms
        
        # Should be at least 45ms (allowing some slack)
        assert timer.elapsed_ms >= 45

    def test_elapsed_ms_is_integer(self):
        """elapsed_ms should be an integer."""
        timer = ToolTimer()
        with timer:
            pass
        
        assert isinstance(timer.elapsed_ms, int)

    def test_zero_when_no_context(self):
        """elapsed_ms should be 0 before entering context."""
        timer = ToolTimer()
        assert timer.elapsed_ms == 0

    def test_exception_still_records_time(self):
        """Timer should still record elapsed time even if exception occurs."""
        timer = ToolTimer()
        try:
            with timer:
                time.sleep(0.03)
                raise ValueError("test")
        except ValueError:
            pass
        
        # Should have recorded time despite exception
        assert timer.elapsed_ms >= 25

    def test_multiple_uses(self):
        """Timer can be reused (though not typical)."""
        timer = ToolTimer()
        
        with timer:
            time.sleep(0.02)
        first = timer.elapsed_ms
        
        with timer:
            time.sleep(0.02)
        second = timer.elapsed_ms
        
        # Second use starts fresh from 0
        assert second >= 15
