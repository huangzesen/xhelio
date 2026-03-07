"""Tests for the retry/rollback logic in llm_utils.

Two mechanisms:
1. Retry every 10s for up to 8 attempts
2. After 3 consecutive failures (~30s), on_reset creates a new chat
   with the last assistant turn dropped
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from agent.event_bus import EventBus, set_event_bus
from agent.llm_utils import (
    _LLM_MAX_RETRIES,
    _SESSION_RESET_THRESHOLD,
    _is_precondition_error,
    send_with_timeout,
)
from agent.llm import LLMResponse, ChatSession


# ---------------------------------------------------------------------------
# Fake 500 error
# ---------------------------------------------------------------------------

class FakeInternalServerError(Exception):
    pass


_orig_is_retryable = None


def _setup_retryable_patch():
    from agent.llm_utils import _is_retryable_api_error
    global _orig_is_retryable
    _orig_is_retryable = _is_retryable_api_error

    def _patched(exc):
        if isinstance(exc, FakeInternalServerError):
            return True
        return _orig_is_retryable(exc)
    return _patched


# ---------------------------------------------------------------------------
# Mock ChatSession
# ---------------------------------------------------------------------------

class MockChatSession(ChatSession):
    def __init__(self, *, fail_count=999):
        self.call_count = 0
        self.fail_count = fail_count

    def send(self, message) -> LLMResponse:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise FakeInternalServerError(f"error call #{self.call_count}")
        return LLMResponse(text=f"Success on call #{self.call_count}")

    def get_history(self) -> list[dict]:
        return [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "tool call response"},
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _setup_event_bus():
    bus = EventBus()
    set_event_bus(bus)
    yield
    set_event_bus(None)


@pytest.fixture
def pool():
    with ThreadPoolExecutor(max_workers=2) as p:
        yield p


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_send(chat, pool, *, on_reset=None):
    fast_delays = [0.01] * 8
    patched = _setup_retryable_patch()
    with (
        patch("agent.llm_utils._is_retryable_api_error", patched),
        patch("agent.llm_utils._API_ERROR_RETRY_DELAYS", fast_delays),
        patch("agent.llm_utils._LLM_WARN_INTERVAL", 0.5),
    ):
        return send_with_timeout(
            chat=chat,
            message="test message",
            timeout_pool=pool,
            cancel_event=None,
            retry_timeout=60.0,
            agent_name="TestAgent",
            logger=MagicMock(),
            on_reset=on_reset,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRetryBasics:
    """Verify basic retry behavior."""

    def test_all_retries_exhausted_raises(self, pool):
        chat = MockChatSession(fail_count=999)
        with pytest.raises(FakeInternalServerError):
            _run_send(chat, pool)
        assert chat.call_count == 1 + _LLM_MAX_RETRIES

    def test_recovery_on_first_retry(self, pool):
        chat = MockChatSession(fail_count=1)
        result = _run_send(chat, pool)
        assert result.text == "Success on call #2"
        assert chat.call_count == 2

    def test_recovery_just_before_exhaustion(self, pool):
        chat = MockChatSession(fail_count=_LLM_MAX_RETRIES)
        result = _run_send(chat, pool)
        assert result.text == f"Success on call #{_LLM_MAX_RETRIES + 1}"
        assert chat.call_count == _LLM_MAX_RETRIES + 1

    def test_consecutive_error_counter_resets_on_success(self, pool):
        chat = MockChatSession(fail_count=1)
        result = _run_send(chat, pool)
        assert result.text == "Success on call #2"


class TestRollbackReset:
    """Verify on_reset (rollback) is called at the threshold."""

    def test_on_reset_called_at_threshold(self, pool):
        """on_reset should be called after _SESSION_RESET_THRESHOLD (2) failures."""
        chat = MockChatSession(fail_count=999)
        reset_calls = []

        def mock_on_reset(old_chat, old_message):
            reset_calls.append(len(reset_calls) + 1)
            return old_chat, "retry message"

        with pytest.raises(FakeInternalServerError):
            _run_send(chat, pool, on_reset=mock_on_reset)

        # With 5 attempts (1+4) and threshold=2:
        # attempts 0,1 → rollback (2 errors, reset_calls=1), consecutive=0
        # attempts 2,3 → rollback (2 errors, reset_calls=2), consecutive=0
        # attempt 4 → fails, consecutive=1, last attempt → no retry branch
        assert len(reset_calls) == 2

    def test_on_reset_recovery_after_rollback(self, pool):
        """After rollback, the new chat should be used for subsequent retries."""
        original_chat = MockChatSession(fail_count=999)
        new_chat = MockChatSession(fail_count=0)  # succeeds immediately

        def mock_on_reset(old_chat, old_message):
            return new_chat, "retry after rollback"

        result = _run_send(original_chat, pool, on_reset=mock_on_reset)

        assert result.text == "Success on call #1"
        # Original failed 2 times (threshold), then rollback gave us new_chat
        assert original_chat.call_count == _SESSION_RESET_THRESHOLD
        assert new_chat.call_count == 1

    def test_no_on_reset_means_no_rollback(self, pool):
        """Without on_reset, just retry without rollback."""
        chat = MockChatSession(fail_count=999)

        with pytest.raises(FakeInternalServerError):
            _run_send(chat, pool, on_reset=None)

        # All attempts should be on the same chat
        assert chat.call_count == 1 + _LLM_MAX_RETRIES

    def test_on_reset_changes_message(self, pool):
        """on_reset can change the message sent on retry."""
        messages_seen = []

        class TrackingChat(ChatSession):
            def __init__(self):
                self.call_count = 0
            def send(self, message) -> LLMResponse:
                self.call_count += 1
                messages_seen.append(message)
                if self.call_count <= 2:
                    raise FakeInternalServerError("fail")
                return LLMResponse(text="ok")
            def get_history(self):
                return []

        chat = TrackingChat()

        def mock_on_reset(old_chat, old_message):
            return old_chat, "retrying after rollback"

        result = _run_send(chat, pool, on_reset=mock_on_reset)

        assert result.text == "ok"
        # First 2 calls use original "test message", after rollback uses new message
        assert messages_seen[:2] == ["test message"] * 2
        assert messages_seen[2] == "retrying after rollback"

    def test_on_reset_failure_is_swallowed(self, pool):
        """If on_reset raises, retries continue without rollback."""
        chat = MockChatSession(fail_count=999)

        def bad_on_reset(old_chat, old_message):
            raise RuntimeError("rollback failed")

        with pytest.raises(FakeInternalServerError):
            _run_send(chat, pool, on_reset=bad_on_reset)

        # Should still have attempted all retries
        assert chat.call_count == 1 + _LLM_MAX_RETRIES


class TestCancellation:
    def test_cancel_during_retry_loop(self, pool):
        chat = MockChatSession(fail_count=999)
        cancel = threading.Event()

        def _cancel_later():
            time.sleep(0.05)
            cancel.set()

        t = threading.Thread(target=_cancel_later, daemon=True)
        t.start()

        fast_delays = [0.01] * 8
        patched = _setup_retryable_patch()
        with (
            patch("agent.llm_utils._is_retryable_api_error", patched),
            patch("agent.llm_utils._API_ERROR_RETRY_DELAYS", fast_delays),
            patch("agent.llm_utils._LLM_WARN_INTERVAL", 0.01),
        ):
            with pytest.raises(Exception):
                send_with_timeout(
                    chat=chat, message="test",
                    timeout_pool=pool, cancel_event=cancel,
                    retry_timeout=60.0, agent_name="TestAgent",
                    logger=MagicMock(),
                )

        assert chat.call_count <= 1 + _LLM_MAX_RETRIES
        t.join(timeout=1)


class TestTiming:
    def test_total_wall_time_bounded(self, pool):
        chat = MockChatSession(fail_count=999)
        t0 = time.monotonic()
        with pytest.raises(FakeInternalServerError):
            _run_send(chat, pool)
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0


# ---------------------------------------------------------------------------
# Precondition error detection & recovery
# ---------------------------------------------------------------------------

class FakePreconditionError(Exception):
    """Fake error for testing precondition detection."""
    pass


def _run_send_precondition(chat, pool, *, on_reset=None):
    """Helper that patches _is_precondition_error to recognize FakePreconditionError."""
    from agent.llm_utils import _is_precondition_error as _orig_precond

    def _patched_precond(exc):
        if isinstance(exc, FakePreconditionError):
            return True
        return _orig_precond(exc)

    with (
        patch("agent.llm_utils._is_precondition_error", _patched_precond),
        patch("agent.llm_utils._LLM_WARN_INTERVAL", 0.5),
    ):
        return send_with_timeout(
            chat=chat,
            message="test message",
            timeout_pool=pool,
            cancel_event=None,
            retry_timeout=60.0,
            agent_name="TestAgent",
            logger=MagicMock(),
            on_reset=on_reset,
        )


class PreconditionMockChat(ChatSession):
    """Chat that raises FakePreconditionError the first N times, then succeeds."""

    def __init__(self, *, fail_count=1):
        self.call_count = 0
        self.fail_count = fail_count

    def send(self, message) -> LLMResponse:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise FakePreconditionError(f"precondition error call #{self.call_count}")
        return LLMResponse(text=f"Success on call #{self.call_count}")

    def get_history(self) -> list[dict]:
        return [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "tool call response"},
        ]


class TestPreconditionDetection:
    """Unit tests for _is_precondition_error() detection."""

    def test_detects_precondition_clienterror(self):
        """ClientError with FAILED_PRECONDITION status is detected."""
        from google.genai import errors as genai_errors

        exc = genai_errors.ClientError(
            400,
            {"error": {"status": "FAILED_PRECONDITION", "message": "Precondition check failed"}},
        )
        assert _is_precondition_error(exc) is True

    def test_rejects_non_precondition_clienterror(self):
        """ClientError with different status is not detected."""
        from google.genai import errors as genai_errors

        exc = genai_errors.ClientError(
            400,
            {"error": {"status": "INVALID_ARGUMENT", "message": "Bad request"}},
        )
        assert _is_precondition_error(exc) is False

    def test_rejects_non_genai_exception(self):
        """Non-genai exception with 'precondition' in message is not detected."""
        exc = ValueError("Precondition check failed")
        assert _is_precondition_error(exc) is False

    def test_rejects_quota_error(self):
        """429 RESOURCE_EXHAUSTED is not detected as precondition error."""
        from google.genai import errors as genai_errors

        exc = genai_errors.ClientError(
            429,
            {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded"}},
        )
        assert _is_precondition_error(exc) is False


class TestPreconditionError:
    """Integration tests for precondition error recovery in send_with_timeout."""

    def test_precondition_triggers_reset(self, pool):
        """Precondition error triggers on_reset and recovery succeeds."""
        original_chat = PreconditionMockChat(fail_count=1)
        new_chat = PreconditionMockChat(fail_count=0)  # succeeds immediately

        reset_calls = []

        def mock_on_reset(old_chat, old_message):
            reset_calls.append(1)
            return new_chat, "retry after precondition reset"

        result = _run_send_precondition(original_chat, pool, on_reset=mock_on_reset)

        assert result.text == "Success on call #1"
        assert len(reset_calls) == 1
        assert original_chat.call_count == 1  # failed once
        assert new_chat.call_count == 1  # succeeded on first try

    def test_precondition_without_on_reset_raises(self, pool):
        """Without on_reset, precondition error is re-raised."""
        chat = PreconditionMockChat(fail_count=999)

        with pytest.raises(FakePreconditionError):
            _run_send_precondition(chat, pool, on_reset=None)

        # Without on_reset, the precondition error falls through to raise
        assert chat.call_count == 1

    def test_precondition_reset_failure_reraises_original(self, pool):
        """If on_reset raises, original precondition error is surfaced."""
        chat = PreconditionMockChat(fail_count=1)

        def bad_on_reset(old_chat, old_message):
            raise RuntimeError("reset failed")

        with pytest.raises(FakePreconditionError):
            _run_send_precondition(chat, pool, on_reset=bad_on_reset)

        assert chat.call_count == 1
