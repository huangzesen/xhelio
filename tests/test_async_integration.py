"""Integration test for async message flow."""
import queue
import time
from agent.base_agent import _make_message


def test_full_priority_flow():
    """Test priority queue with mixed messages."""
    pq = queue.PriorityQueue()

    # Simulate arrival order:
    # 1. subagent result from envoy
    pq.put((1, time.time(), _make_message("subagent_result", "envoy", "data ready")))

    # 2. user message
    pq.put((0, time.time(), _make_message("user_input", "user", "show me data")))

    # 3. another subagent result
    pq.put((1, time.time(), _make_message("subagent_result", "viz", "plot ready")))

    # User should come first
    _, _, msg1 = pq.get_nowait()
    assert msg1.type == "user_input"

    # Then subagents in order
    _, _, msg2 = pq.get_nowait()
    assert msg2.sender == "envoy"

    _, _, msg3 = pq.get_nowait()
    assert msg3.sender == "viz"
