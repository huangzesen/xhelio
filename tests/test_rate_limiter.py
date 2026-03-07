"""Tests for the shared rate limiter."""
import time
import threading
import pytest
from agent.llm.rate_limiter import RateLimiter


def test_rate_limiter_allows_immediate_call_when_disabled():
    """When interval is 0, calls should not be blocked."""
    limiter = RateLimiter(min_interval=0.0)
    start = time.monotonic()
    limiter.wait()
    limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed < 0.01, "Should be nearly instant with 0 interval"


def test_rate_limiter_blocks_when_interval_set():
    """When interval is set, calls should be throttled."""
    limiter = RateLimiter(min_interval=0.1)
    start = time.monotonic()
    limiter.wait()
    limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.1, "Should wait at least 0.1s between calls"


def test_rate_limiter_is_thread_safe():
    """Rate limiter should work correctly with multiple threads."""
    limiter = RateLimiter(min_interval=0.01)
    call_times = []

    def make_call():
        limiter.wait()
        call_times.append(time.monotonic())

    threads = [threading.Thread(target=make_call) for _ in range(3)]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All calls should complete
    assert len(call_times) == 3
    # Each call should be at least 0.01s apart from the previous
    for i in range(1, len(call_times)):
        diff = call_times[i] - call_times[i-1]
        assert diff >= 0.009, f"Calls too close: {diff}"


def test_llm_adapter_has_rate_limiter():
    """LLMAdapter should support rate limiting."""
    from agent.llm.base import LLMAdapter

    # Check that LLMAdapter has rate limiter support
    assert hasattr(LLMAdapter, '_rate_limiter') or hasattr(LLMAdapter, '_setup_rate_limiter')


def test_minimax_adapter_uses_base_rate_limiter():
    """MiniMaxAdapter should use base class rate limiter, not custom."""
    from agent.llm.minimax.adapter import MiniMaxAdapter
    from agent.llm.base import LLMAdapter

    # MiniMaxAdapter should inherit rate limiter from base
    assert issubclass(MiniMaxAdapter, LLMAdapter)
    # Rate limiter should come from base, not be custom implementation
    # Check that _RateLimiter class doesn't exist in minimax_adapter module
    import agent.llm.minimax.adapter as mm_module
    assert not hasattr(mm_module, '_RateLimiter'), "Should not have custom _RateLimiter"
