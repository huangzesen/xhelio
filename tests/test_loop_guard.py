"""Tests for agent/loop_guard.py — LoopGuard (max-calls + dup tracking)."""

import pytest
from agent.loop_guard import LoopGuard, DupVerdict


# =====================================================================
# Existing tests — total call limit
# =====================================================================

class TestLoopGuardCheckLimit:
    """Tests for check_limit() — the only protection layer."""

    def test_under_limit_returns_none(self):
        guard = LoopGuard(max_total_calls=10)
        assert guard.check_limit(3) is None

    def test_at_exact_limit_returns_none(self):
        guard = LoopGuard(max_total_calls=5)
        assert guard.check_limit(5) is None

    def test_over_limit_returns_reason(self):
        guard = LoopGuard(max_total_calls=5)
        reason = guard.check_limit(6)
        assert reason is not None
        assert "total call limit" in reason

    def test_accumulated_calls_hit_limit(self):
        guard = LoopGuard(max_total_calls=5)
        assert guard.check_limit(3) is None
        guard.record_calls(3)
        assert guard.total_calls == 3

        # 2 more — at limit
        assert guard.check_limit(2) is None
        guard.record_calls(2)
        assert guard.total_calls == 5

        # 1 more — over limit
        reason = guard.check_limit(1)
        assert reason is not None
        assert "total call limit" in reason

    def test_zero_calls_always_ok(self):
        guard = LoopGuard(max_total_calls=0)
        assert guard.check_limit(0) is None

    def test_negative_calls_always_ok(self):
        guard = LoopGuard(max_total_calls=0)
        assert guard.check_limit(-1) is None


class TestLoopGuardRecordCalls:
    """Tests for record_calls() — simple counter."""

    def test_increments_total(self):
        guard = LoopGuard(max_total_calls=100)
        guard.record_calls(3)
        assert guard.total_calls == 3
        guard.record_calls(2)
        assert guard.total_calls == 5

    def test_zero_record_no_change(self):
        guard = LoopGuard(max_total_calls=100)
        guard.record_calls(0)
        assert guard.total_calls == 0


class TestLoopGuardInit:
    """Tests for initialization and kwargs."""

    def test_default_max(self):
        guard = LoopGuard()
        assert guard.max_total_calls == 10
        assert guard.total_calls == 0

    def test_custom_max(self):
        guard = LoopGuard(max_total_calls=42)
        assert guard.max_total_calls == 42

    def test_ignores_unknown_kwargs(self):
        """Backwards compat: old callers may pass max_iterations etc."""
        guard = LoopGuard(max_total_calls=10, max_iterations=5)
        assert guard.max_total_calls == 10

    def test_default_dup_params(self):
        guard = LoopGuard()
        assert guard._dup_free_passes == 2
        assert guard._dup_hard_block == 8

    def test_custom_dup_params(self):
        guard = LoopGuard(dup_free_passes=5, dup_hard_block=15)
        assert guard._dup_free_passes == 5
        assert guard._dup_hard_block == 15

    def test_backward_compat_without_dup_params(self):
        """Old callers that only pass max_total_calls still work."""
        guard = LoopGuard(max_total_calls=20)
        assert guard.max_total_calls == 20
        assert guard._dup_free_passes == 2
        assert guard._dup_hard_block == 8


class TestLoopGuardIntegration:
    """End-to-end tests simulating real agent behavior."""

    def test_normal_viz_flow(self):
        """3 rounds of tool calls within limit should all pass."""
        guard = LoopGuard(max_total_calls=10)

        for n_calls in [1, 1, 1]:
            assert guard.check_limit(n_calls) is None
            guard.record_calls(n_calls)

        assert guard.total_calls == 3

    def test_repeated_calls_allowed(self):
        """Same tool called repeatedly is fine — no dedup, just counting."""
        guard = LoopGuard(max_total_calls=10)

        for _ in range(5):
            assert guard.check_limit(1) is None
            guard.record_calls(1)

        assert guard.total_calls == 5

    def test_total_limit_as_safety_ceiling(self):
        """Total call limit eventually stops even repeated unique calls."""
        guard = LoopGuard(max_total_calls=5)

        for _ in range(5):
            assert guard.check_limit(1) is None
            guard.record_calls(1)

        # 6th call — over limit
        reason = guard.check_limit(1)
        assert reason is not None
        assert "total call limit" in reason

    def test_large_batch_hits_limit(self):
        """A single large batch exceeding the limit is caught."""
        guard = LoopGuard(max_total_calls=10)

        reason = guard.check_limit(11)
        assert reason is not None
        assert "total call limit" in reason


# =====================================================================
# Duplicate call tracking tests
# =====================================================================

class TestDedupKey:
    """Tests for LoopGuard._dedup_key() — key generation."""

    def test_basic_key(self):
        key = LoopGuard._dedup_key("fetch_data", {"dataset": "AC_H0_MFI"})
        assert key == ("fetch_data", '{"dataset": "AC_H0_MFI"}')

    def test_strips_commentary(self):
        """commentary key should be stripped before hashing."""
        key_with = LoopGuard._dedup_key("fetch_data", {"dataset": "AC_H0_MFI", "commentary": "getting data"})
        key_without = LoopGuard._dedup_key("fetch_data", {"dataset": "AC_H0_MFI"})
        assert key_with == key_without

    def test_strips_sync(self):
        """_sync key should be stripped before hashing."""
        key_with = LoopGuard._dedup_key("events", {"_sync": True})
        key_without = LoopGuard._dedup_key("events", {})
        assert key_with == key_without

    def test_none_args_same_as_empty(self):
        key_none = LoopGuard._dedup_key("events", None)
        key_empty = LoopGuard._dedup_key("events", {})
        assert key_none == key_empty

    def test_different_args_different_keys(self):
        key1 = LoopGuard._dedup_key("fetch_data", {"dataset": "AC_H0_MFI"})
        key2 = LoopGuard._dedup_key("fetch_data", {"dataset": "AC_H1_SWE"})
        assert key1 != key2

    def test_different_tools_different_keys(self):
        key1 = LoopGuard._dedup_key("fetch_data", {"dataset": "AC_H0_MFI"})
        key2 = LoopGuard._dedup_key("search_data", {"dataset": "AC_H0_MFI"})
        assert key1 != key2

    def test_sorted_keys(self):
        """Arg order should not matter."""
        key1 = LoopGuard._dedup_key("tool", {"a": 1, "b": 2})
        key2 = LoopGuard._dedup_key("tool", {"b": 2, "a": 1})
        assert key1 == key2


class TestRecordToolCall:
    """Tests for record_tool_call() — duplicate detection with escalation."""

    def test_first_calls_no_warning(self):
        """First N calls (dup_free_passes) produce no warning."""
        guard = LoopGuard(dup_free_passes=2, dup_hard_block=8)

        v1 = guard.record_tool_call("events", {})
        assert v1.count == 1
        assert v1.blocked is False
        assert v1.warning is None

        v2 = guard.record_tool_call("events", {})
        assert v2.count == 2
        assert v2.blocked is False
        assert v2.warning is None

    def test_mild_warning_after_free_passes(self):
        """Calls N+1 and N+2 produce a mild warning."""
        guard = LoopGuard(dup_free_passes=2, dup_hard_block=8)

        # Burn free passes
        for _ in range(2):
            guard.record_tool_call("events", {})

        v3 = guard.record_tool_call("events", {})
        assert v3.count == 3
        assert v3.blocked is False
        assert v3.warning is not None
        assert "identical arguments" in v3.warning
        assert "STOP POLLING" not in v3.warning

        v4 = guard.record_tool_call("events", {})
        assert v4.count == 4
        assert v4.blocked is False
        assert v4.warning is not None
        assert "STOP POLLING" not in v4.warning

    def test_strong_warning_escalation(self):
        """Calls N+3+ produce a strong STOP POLLING warning."""
        guard = LoopGuard(dup_free_passes=2, dup_hard_block=8)

        # Burn free passes + mild warnings
        for _ in range(4):
            guard.record_tool_call("events", {})

        # Call #5 — should be strong
        v5 = guard.record_tool_call("events", {})
        assert v5.count == 5
        assert v5.blocked is False
        assert "STOP POLLING" in v5.warning

    def test_hard_block_at_threshold(self):
        """At dup_hard_block threshold, execution is blocked."""
        guard = LoopGuard(dup_free_passes=2, dup_hard_block=5)

        for i in range(4):
            v = guard.record_tool_call("events", {})
            assert v.blocked is False, f"Call {i+1} should not be blocked"

        # Call #5 — at hard block threshold
        v5 = guard.record_tool_call("events", {})
        assert v5.count == 5
        assert v5.blocked is True
        assert "BLOCKED" in v5.warning

    def test_blocked_calls_stay_blocked(self):
        """Calls beyond the hard block threshold remain blocked."""
        guard = LoopGuard(dup_free_passes=2, dup_hard_block=5)

        for _ in range(4):
            guard.record_tool_call("events", {})

        for i in range(5, 10):
            v = guard.record_tool_call("events", {})
            assert v.blocked is True, f"Call #{i} should be blocked"
            assert v.count == i

    def test_different_args_tracked_separately(self):
        """Different args produce independent counts — no false positives."""
        guard = LoopGuard(dup_free_passes=2, dup_hard_block=5)

        # Call events with {} 4 times (just under hard block)
        for _ in range(4):
            guard.record_tool_call("events", {})

        # Call events with different args — should start fresh
        v = guard.record_tool_call("events", {"filter": "data"})
        assert v.count == 1
        assert v.blocked is False
        assert v.warning is None

    def test_different_tools_tracked_separately(self):
        """Different tool names produce independent counts."""
        guard = LoopGuard(dup_free_passes=2, dup_hard_block=5)

        for _ in range(4):
            guard.record_tool_call("events", {})

        v = guard.record_tool_call("fetch_data", {})
        assert v.count == 1
        assert v.blocked is False
        assert v.warning is None


class TestDupCounts:
    """Tests for the dup_counts property."""

    def test_empty_initially(self):
        guard = LoopGuard()
        assert guard.dup_counts == {}

    def test_tracks_counts(self):
        guard = LoopGuard()
        guard.record_tool_call("events", {})
        guard.record_tool_call("events", {})
        guard.record_tool_call("fetch_data", {"id": 1})

        counts = guard.dup_counts
        assert len(counts) == 2
        ce_key = LoopGuard._dedup_key("events", {})
        fd_key = LoopGuard._dedup_key("fetch_data", {"id": 1})
        assert counts[ce_key] == 2
        assert counts[fd_key] == 1

    def test_returns_copy(self):
        """dup_counts returns a copy, not the internal dict."""
        guard = LoopGuard()
        guard.record_tool_call("tool_a", {})
        counts = guard.dup_counts
        counts[("fake", "{}")] = 999
        assert ("fake", "{}") not in guard.dup_counts


class TestDupVerdictDataclass:
    """Tests for DupVerdict immutability and fields."""

    def test_fields(self):
        v = DupVerdict(count=3, blocked=False, warning="test")
        assert v.count == 3
        assert v.blocked is False
        assert v.warning == "test"

    def test_frozen(self):
        v = DupVerdict(count=1, blocked=False, warning=None)
        with pytest.raises(AttributeError):
            v.count = 5


class TestPollingSimulation:
    """Integration test simulating the 43x events polling pattern."""

    def test_events_polling_detected_and_blocked(self):
        """Simulates LLM calling events 43 times with empty args.

        With default settings (free_passes=2, hard_block=8):
          - Calls 1-2: no warning
          - Calls 3-4: mild warning
          - Calls 5-7: strong STOP POLLING warning
          - Calls 8+: blocked entirely
        """
        guard = LoopGuard(dup_free_passes=2, dup_hard_block=8)

        blocked_count = 0
        warned_count = 0
        clean_count = 0

        for i in range(43):
            v = guard.record_tool_call("events", {})
            if v.blocked:
                blocked_count += 1
            elif v.warning:
                warned_count += 1
            else:
                clean_count += 1

        # First 2 clean, 5 warned (3-7), 36 blocked (8-43)
        assert clean_count == 2
        assert warned_count == 5
        assert blocked_count == 36

        # Verify the tool was tracked
        key = LoopGuard._dedup_key("events", {})
        assert guard.dup_counts[key] == 43

    def test_diverse_tools_no_false_positives(self):
        """Normal session with diverse tool calls never triggers dup warnings."""
        guard = LoopGuard(dup_free_passes=2, dup_hard_block=8)

        # Simulate a realistic session: different tools, different args
        tools = [
            ("search_datasets", {"query": "ACE magnetic field"}),
            ("fetch_data", {"dataset": "AC_H0_MFI", "start": "2024-01-01", "end": "2024-01-02"}),
            ("describe_data", {"label": "ac_h0_mfi"}),
            ("render_plotly_json", {"figure_json": "{}"}),
            ("manage_plot", {"action": "export", "format": "png"}),
            ("search_datasets", {"query": "Wind solar wind"}),
            ("fetch_data", {"dataset": "WI_H1_SWE", "start": "2024-01-01", "end": "2024-01-02"}),
        ]

        for name, args in tools:
            v = guard.record_tool_call(name, args)
            assert v.blocked is False, f"{name} should not be blocked"
            assert v.warning is None, f"{name} should have no warning"

    def test_legitimate_retry_with_same_args(self):
        """A tool retried once (e.g. after transient error) should not warn."""
        guard = LoopGuard(dup_free_passes=2, dup_hard_block=8)

        # First call
        v1 = guard.record_tool_call("fetch_data", {"dataset": "AC_H0_MFI"})
        assert v1.warning is None

        # Retry — still within free passes
        v2 = guard.record_tool_call("fetch_data", {"dataset": "AC_H0_MFI"})
        assert v2.warning is None
        assert v2.blocked is False
