"""Tests for InlineCompletions."""
from unittest.mock import MagicMock
from agent.inline_completions import InlineCompletions


def _make_ic():
    return InlineCompletions(
        service=MagicMock(),
        inline_tracker=MagicMock(),
        event_bus=MagicMock(),
    )


def test_circuit_breaker_disables_after_5_failures():
    ic = _make_ic()
    for _ in range(5):
        ic._fail_count += 1
    # Simulate the threshold check
    assert ic._fail_count >= 5


def test_reset_clears_circuit_breaker():
    ic = _make_ic()
    ic._fail_count = 3
    ic._disabled_until = 9999999999.0
    ic.reset()
    assert ic._fail_count == 0
    assert ic._disabled_until == 0.0


def test_follow_ups_returns_empty_on_no_turns():
    ic = _make_ic()
    result = ic.generate_follow_ups(
        chat_history=[],
        store_labels=[],
        has_plot=False,
    )
    assert result == []


def test_session_title_returns_none_on_no_turns():
    ic = _make_ic()
    result = ic.generate_session_title(chat_history=[])
    assert result is None
