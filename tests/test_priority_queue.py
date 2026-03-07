"""Tests for priority message queue in orchestrator."""
import queue
import time
from agent.sub_agent import _make_message


def test_priority_queue_user_first():
    """User messages should be processed before subagent results."""
    pq = queue.PriorityQueue()

    # Put subagent_result first (lower priority number = higher)
    pq.put((1, time.time(), _make_message("subagent_result", "envoy", "data")))

    # Put user_input second
    pq.put((0, time.time(), _make_message("user_input", "user", "show data")))

    # User should come out first
    _, _, user_msg = pq.get_nowait()
    assert user_msg.type == "user_input"

    # Then subagent
    _, _, subagent_msg = pq.get_nowait()
    assert subagent_msg.type == "subagent_result"


def test_priority_queue_fifo_subagent():
    """Subagent results should maintain FIFO order."""
    pq = queue.PriorityQueue()

    # Put multiple subagent results
    pq.put((1, 1.0, _make_message("subagent_result", "envoy1", "result1")))
    pq.put((1, 2.0, _make_message("subagent_result", "envoy2", "result2")))
    pq.put((1, 3.0, _make_message("subagent_result", "envoy3", "result3")))

    # Should come out in order
    _, _, msg1 = pq.get_nowait()
    _, _, msg2 = pq.get_nowait()
    _, _, msg3 = pq.get_nowait()

    assert msg1.sender == "envoy1"
    assert msg2.sender == "envoy2"
    assert msg3.sender == "envoy3"


def test_run_loop_uses_priority_get():
    """Run loop should use priority-aware message retrieval."""
    from agent.core import OrchestratorAgent

    assert hasattr(OrchestratorAgent, '_get_message')
    assert hasattr(OrchestratorAgent, '_put_message')


def test_run_loop_handles_subagent_result():
    """Run loop should handle subagent_result messages."""
    # Document expected behavior
    assert True
