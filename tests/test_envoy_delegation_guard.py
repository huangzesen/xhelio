"""
Tests for agent-state-based mission delegation guard.

Validates that only one delegate_to_envoy call per mission_id can run
at a time — if an agent is busy (state != SLEEPING or inbox non-empty),
concurrent same-mission calls are rejected immediately, while
different-mission calls proceed in parallel.

Run with: python -m pytest tests/test_envoy_delegation_guard.py -v
"""

import queue
import threading
from unittest.mock import MagicMock

import pytest

from agent.sub_agent import AgentState
from agent.event_bus import EventBus


# ---------------------------------------------------------------------------
# Helpers — lightweight stub that replicates the agent-state delegation
# guard from OrchestratorAgent without the heavyweight __init__.
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Minimal agent-like object with state and inbox for testing."""

    def __init__(self):
        self.inbox: queue.Queue = queue.Queue()
        self._state = AgentState.SLEEPING

    @property
    def state(self) -> AgentState:
        return self._state


class _DelegationGuardStub:
    """Minimal stub replicating the agent-state delegation guard.

    Mirrors the delegate_to_envoy handler in OrchestratorAgent._execute_tool
    without requiring LLM adapters, tool schemas, or session setup.
    """

    def __init__(self):
        self._agents: dict[str, _FakeAgent] = {}
        self._event_bus = EventBus(session_id="test")

        # Configurable: controls how long _delegate_to_agent blocks
        self._delegate_delay = 0.0
        self._delegate_result = {"status": "success", "summary": "done"}

    def _get_or_create_envoy_agent(self, mission_id: str) -> _FakeAgent:
        """Get or create a fake agent for the given mission."""
        agent_id = f"EnvoyAgent[{mission_id}]"
        if agent_id not in self._agents:
            self._agents[agent_id] = _FakeAgent()
        return self._agents[agent_id]

    def handle_delegate_to_envoy(self, mission_id: str, request: str) -> dict:
        """Replicate the delegate_to_envoy handler from OrchestratorAgent._execute_tool."""
        agent = self._get_or_create_envoy_agent(mission_id)
        # Reject if agent is already processing a delegation
        if agent.state != AgentState.SLEEPING or agent.inbox.qsize() > 0:
            return {
                "status": "error",
                "message": (
                    f"A delegation to mission '{mission_id}' is already in progress. "
                    f"Only one delegation per mission is allowed at a time. "
                    f"Combine all {mission_id} requests into a single delegate_to_envoy call. "
                    f"The mission agent can parallelize multiple fetch_data calls internally by calling multiple tools in one response."
                ),
            }
        # Simulate delegation (in real code this is _delegate_to_agent)
        result = dict(self._delegate_result)
        result["mission"] = mission_id
        return result


class TestMissionDelegationGuard:
    """Tests for agent-state-based delegation guard enforcement."""

    def test_single_delegation_succeeds(self):
        """A single delegate_to_envoy call to a sleeping agent succeeds."""
        stub = _DelegationGuardStub()
        result = stub.handle_delegate_to_envoy("ACE", "fetch magnetic field data")
        assert result["status"] == "success"
        assert result["mission"] == "ACE"

    def test_busy_agent_rejected(self):
        """An agent in ACTIVE state rejects a delegation."""
        stub = _DelegationGuardStub()
        # Create agent and set it to ACTIVE (simulating in-progress work)
        agent = stub._get_or_create_envoy_agent("ACE")
        agent._state = AgentState.ACTIVE

        result = stub.handle_delegate_to_envoy("ACE", "fetch magnetic field data")
        assert result["status"] == "error"
        assert "already in progress" in result["message"]
        assert "ACE" in result["message"]

    def test_active_agent_rejected_different_mission(self):
        """An agent in ACTIVE state rejects a delegation."""
        stub = _DelegationGuardStub()
        agent = stub._get_or_create_envoy_agent("PSP")
        agent._state = AgentState.ACTIVE

        result = stub.handle_delegate_to_envoy("PSP", "fetch data")
        assert result["status"] == "error"
        assert "already in progress" in result["message"]

    def test_different_missions_concurrent_allowed(self):
        """Two calls to different missions (both sleeping) both succeed."""
        stub = _DelegationGuardStub()

        result_ace = stub.handle_delegate_to_envoy("ACE", "request 1")
        result_psp = stub.handle_delegate_to_envoy("PSP", "request 2")

        assert result_ace["status"] == "success"
        assert result_ace["mission"] == "ACE"
        assert result_psp["status"] == "success"
        assert result_psp["mission"] == "PSP"

    def test_sequential_same_mission_succeeds(self):
        """After first delegation completes (agent returns to SLEEPING), second succeeds."""
        stub = _DelegationGuardStub()

        result1 = stub.handle_delegate_to_envoy("ACE", "first request")
        assert result1["status"] == "success"

        # Actor remains SLEEPING after stub delegation (no real work done)
        result2 = stub.handle_delegate_to_envoy("ACE", "second request")
        assert result2["status"] == "success"

    def test_error_message_includes_mission_and_parallel_hint(self):
        """The rejection error message should include the mission_id and mention parallel tools."""
        stub = _DelegationGuardStub()
        agent = stub._get_or_create_envoy_agent("WIND")
        agent._state = AgentState.ACTIVE

        result = stub.handle_delegate_to_envoy("WIND", "request")
        assert result["status"] == "error"
        assert "WIND" in result["message"]
        assert "parallelize" in result["message"]
        assert "already in progress" in result["message"]

    def test_queued_message_rejected(self):
        """Actor is SLEEPING but inbox has queued message — still rejected.

        This covers the race window between inbox.put() and the agent
        transitioning to ACTIVE.
        """
        stub = _DelegationGuardStub()
        agent = stub._get_or_create_envoy_agent("ACE")
        # Actor is SLEEPING but has a queued message
        assert agent.state == AgentState.SLEEPING
        agent.inbox.put({"type": "request", "content": "previous request"})

        result = stub.handle_delegate_to_envoy("ACE", "new request")
        assert result["status"] == "error"
        assert "already in progress" in result["message"]

    def test_busy_one_mission_sleeping_other_allowed(self):
        """One mission busy, another mission sleeping — the sleeping one accepts."""
        stub = _DelegationGuardStub()
        # Make ACE busy
        agent_ace = stub._get_or_create_envoy_agent("ACE")
        agent_ace._state = AgentState.ACTIVE

        # PSP should still accept (its agent is sleeping)
        result = stub.handle_delegate_to_envoy("PSP", "fetch data")
        assert result["status"] == "success"
        assert result["mission"] == "PSP"

        # ACE should be rejected
        result_ace = stub.handle_delegate_to_envoy("ACE", "fetch data")
        assert result_ace["status"] == "error"
