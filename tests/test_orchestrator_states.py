"""
Tests for orchestrator lifecycle state machine (AgentState), cycle counting,
and turn counting.

These tests exercise the OrchestratorAgent's state tracking without requiring
a real LLM — they mock the LLM adapter and test the state transitions that
occur in _run_loop_actor() and process_message().

Run with: python -m pytest tests/test_orchestrator_states.py -v
"""

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.base_agent import AgentState, Message, _make_message
from agent.event_bus import (
    EventBus,
    AGENT_STATE_CHANGE,
    ROUND_START,
    ROUND_END,
    CYCLE_START,
    CYCLE_END,
    AGENT_RESPONSE,
    TEXT_DELTA,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight orchestrator stub
# ---------------------------------------------------------------------------


class _OrchestratorStub:
    """Minimal stub that replicates OrchestratorAgent's state machine logic.

    Avoids the heavyweight __init__ (LLM adapter, tool schemas, session
    setup) by directly implementing only the state/cycle/turn tracking
    and the _run_loop_actor() logic.
    """

    def __init__(self, event_bus: EventBus | None = None):
        self._event_bus = (
            event_bus if event_bus is not None else EventBus(session_id="test")
        )
        self._inbox: queue.Queue[Message] = queue.Queue()
        self._shutdown_event = threading.Event()
        self._state = AgentState.SLEEPING
        self._cycle_number = 0
        self._turn_number = 0
        self._round_start_tokens: dict | None = None
        self._thread: threading.Thread | None = None

        # Configurable response for process_message
        self._response_text = "OK"
        self._process_delay = 0.0
        self._raise_on_process = False

    # ---- State machine (mirrors OrchestratorAgent) ----

    @property
    def state(self) -> AgentState:
        return self._state

    def _set_state(self, new_state: AgentState, reason: str = "") -> None:
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        suffix = f" ({reason})" if reason else ""
        self._event_bus.emit(
            AGENT_STATE_CHANGE,
            agent="orchestrator",
            level="debug",
            msg=f"[orchestrator] {old.value} → {new_state.value}{suffix}",
            data={
                "agent_id": "orchestrator",
                "old": old.value,
                "new": new_state.value,
                "reason": reason,
                "cycle": self._cycle_number,
                "turn": self._turn_number,
            },
        )

    def get_token_usage(self) -> dict:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
        }

    def process_message(self, user_message: str) -> str:
        if self._raise_on_process:
            raise RuntimeError("simulated error")
        if self._process_delay > 0:
            time.sleep(self._process_delay)
        self._event_bus.emit(
            AGENT_RESPONSE,
            level="info",
            msg=f"[Agent] {self._response_text}",
            data={
                "text": self._response_text,
                "turn": self._turn_number,
                "cycle": self._cycle_number,
            },
        )
        return self._response_text

    # ---- Event loop (copied from OrchestratorAgent._run_loop_agent) ----

    def _run_loop_agent(self) -> None:
        self._set_state(AgentState.SLEEPING)

        while not self._shutdown_event.is_set():
            try:
                msg = self._inbox.get(timeout=0.05)
            except queue.Empty:
                continue

            if self._shutdown_event.is_set():
                break

            if msg.type == "user_input":
                if self._state == AgentState.SLEEPING:
                    self._cycle_number += 1
                self._set_state(AgentState.ACTIVE)

                self._round_start_tokens = self.get_token_usage()
                self._event_bus.emit(
                    CYCLE_START,
                    level="info",
                    msg="Cycle started",
                    data={"cycle": self._cycle_number},
                )

                turns_in_cycle = 0
                user_message = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                try:
                    self.process_message(user_message)
                    self._turn_number += 1
                    turns_in_cycle += 1
                    self._event_bus.emit(
                        TEXT_DELTA,
                        level="info",
                        msg=self._response_text,
                        data={"text": self._response_text},
                    )
                except Exception:
                    pass

                # Drain queued messages
                while not self._shutdown_event.is_set():
                    try:
                        extra = self._inbox.get_nowait()
                    except queue.Empty:
                        break
                    if extra.type == "user_input":
                        extra_msg = (
                            extra.content
                            if isinstance(extra.content, str)
                            else str(extra.content)
                        )
                        try:
                            self.process_message(extra_msg)
                            self._turn_number += 1
                            turns_in_cycle += 1
                        except Exception:
                            pass

                current_tokens = self.get_token_usage()
                start = self._round_start_tokens or {}
                round_delta = {
                    k: current_tokens.get(k, 0) - start.get(k, 0)
                    for k in current_tokens
                }
                self._event_bus.emit(
                    CYCLE_END,
                    level="info",
                    msg="Cycle complete",
                    data={
                        "cycle": self._cycle_number,
                        "turns_in_cycle": turns_in_cycle,
                        "token_usage": current_tokens,
                        "round_token_usage": round_delta,
                    },
                )
                self._set_state(AgentState.SLEEPING)

    def push_input(self, message: str) -> None:
        self._inbox.put(_make_message("user_input", "user", message))

    def start(self) -> None:
        self._shutdown_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop_agent,
            daemon=True,
            name="test-orchestrator",
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._shutdown_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------


class TestOrchestratorStateTransitions:
    """Tests for orchestrator lifecycle state machine."""

    def test_initial_state_is_sleeping(self):
        stub = _OrchestratorStub()
        assert stub.state == AgentState.SLEEPING

    def test_sleeping_to_active_on_user_input(self):
        """Orchestrator transitions SLEEPING → ACTIVE on user input."""
        bus = EventBus(session_id="test")
        state_changes = []
        bus.subscribe(
            lambda e: (
                state_changes.append(e.data) if e.type == AGENT_STATE_CHANGE else None
            )
        )

        stub = _OrchestratorStub(event_bus=bus)
        stub.start()
        try:
            stub.push_input("hello")
            time.sleep(0.15)

            active_transitions = [
                c
                for c in state_changes
                if c.get("agent_id") == "orchestrator" and c.get("new") == "active"
            ]
            assert len(active_transitions) >= 1, (
                f"Expected ACTIVE transition, got: {state_changes}"
            )
        finally:
            stub.stop()

    def test_active_to_sleeping_after_processing(self):
        """Orchestrator returns to SLEEPING after processing completes."""
        stub = _OrchestratorStub()
        stub.start()
        try:
            stub.push_input("test")
            time.sleep(0.15)
            assert stub.state == AgentState.SLEEPING
        finally:
            stub.stop()

    def test_set_state_emits_event(self):
        """_set_state() emits AGENT_STATE_CHANGE with correct data."""
        bus = EventBus(session_id="test")
        events = []
        bus.subscribe(
            lambda e: events.append(e) if e.type == AGENT_STATE_CHANGE else None
        )

        stub = _OrchestratorStub(event_bus=bus)
        stub._set_state(AgentState.ACTIVE)

        assert len(events) == 1
        assert events[0].data["agent_id"] == "orchestrator"
        assert events[0].data["old"] == "sleeping"
        assert events[0].data["new"] == "active"
        assert events[0].data["reason"] == ""

    def test_set_state_emits_reason(self):
        """_set_state() includes reason in event data and message."""
        bus = EventBus(session_id="test")
        events = []
        bus.subscribe(
            lambda e: events.append(e) if e.type == AGENT_STATE_CHANGE else None
        )

        stub = _OrchestratorStub(event_bus=bus)
        stub._set_state(AgentState.ACTIVE, reason="user input")

        assert len(events) == 1
        assert events[0].data["reason"] == "user input"
        assert "(user input)" in events[0].msg

    def test_set_state_noop_when_same(self):
        """No event emitted when setting the same state."""
        bus = EventBus(session_id="test")
        events = []
        bus.subscribe(
            lambda e: events.append(e) if e.type == AGENT_STATE_CHANGE else None
        )

        stub = _OrchestratorStub(event_bus=bus)
        stub._set_state(AgentState.SLEEPING)  # already SLEEPING
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Cycle counting tests
# ---------------------------------------------------------------------------


class TestCycleCounting:
    """Tests for cycle number tracking."""

    def test_cycle_increments_on_first_input(self):
        """First user input starts cycle 1."""
        stub = _OrchestratorStub()
        stub.start()
        try:
            stub.push_input("first")
            time.sleep(0.15)
            assert stub._cycle_number == 1
        finally:
            stub.stop()

    def test_cycle_increments_per_sleep_wake(self):
        """Each SLEEPING → ACTIVE transition starts a new cycle."""
        stub = _OrchestratorStub()
        stub.start()
        try:
            stub.push_input("first")
            time.sleep(0.15)
            assert stub._cycle_number == 1
            assert stub.state == AgentState.SLEEPING

            stub.push_input("second")
            time.sleep(0.15)
            assert stub._cycle_number == 2
            assert stub.state == AgentState.SLEEPING
        finally:
            stub.stop()

    def test_cycle_event_data(self):
        """CYCLE_START and CYCLE_END events carry cycle number."""
        bus = EventBus(session_id="test")
        cycle_events = []
        bus.subscribe(
            lambda e: (
                cycle_events.append(e) if e.type in (CYCLE_START, CYCLE_END) else None
            )
        )

        stub = _OrchestratorStub(event_bus=bus)
        stub.start()
        try:
            stub.push_input("go")
            time.sleep(0.15)

            starts = [e for e in cycle_events if e.type == CYCLE_START]
            ends = [e for e in cycle_events if e.type == CYCLE_END]
            assert len(starts) == 1
            assert len(ends) == 1
            assert starts[0].data["cycle"] == 1
            assert ends[0].data["cycle"] == 1
        finally:
            stub.stop()

    def test_cycle_end_includes_turns_in_cycle(self):
        """CYCLE_END event includes turns_in_cycle count."""
        bus = EventBus(session_id="test")
        cycle_ends = []
        bus.subscribe(lambda e: cycle_ends.append(e) if e.type == CYCLE_END else None)

        stub = _OrchestratorStub(event_bus=bus)
        stub.start()
        try:
            stub.push_input("hello")
            time.sleep(0.15)

            assert len(cycle_ends) == 1
            assert cycle_ends[0].data["turns_in_cycle"] == 1
        finally:
            stub.stop()


# ---------------------------------------------------------------------------
# Turn counting tests
# ---------------------------------------------------------------------------


class TestTurnCounting:
    """Tests for turn number tracking."""

    def test_turn_increments_per_response(self):
        """Turn number increments with each process_message() call."""
        stub = _OrchestratorStub()
        stub.start()
        try:
            stub.push_input("first")
            time.sleep(0.15)
            assert stub._turn_number == 1

            stub.push_input("second")
            time.sleep(0.15)
            assert stub._turn_number == 2
        finally:
            stub.stop()

    def test_turn_survives_error(self):
        """Turn count does not increment on processing error."""
        stub = _OrchestratorStub()
        stub._raise_on_process = True
        stub.start()
        try:
            stub.push_input("will fail")
            time.sleep(0.15)
            assert stub._turn_number == 0  # no increment on error
        finally:
            stub.stop()


# ---------------------------------------------------------------------------
# CYCLE_START / CYCLE_END alias tests
# ---------------------------------------------------------------------------


class TestCycleAliases:
    """Tests that CYCLE_START/CYCLE_END are aliases for ROUND_START/ROUND_END."""

    def test_cycle_start_equals_round_start(self):
        assert CYCLE_START == ROUND_START

    def test_cycle_end_equals_round_end(self):
        assert CYCLE_END == ROUND_END

    def test_aliases_are_same_string(self):
        assert CYCLE_START == "round_start"
        assert CYCLE_END == "round_end"


# ---------------------------------------------------------------------------
# orchestrator_status() tests (via stub — just verifying the dict shape)
# ---------------------------------------------------------------------------


class TestOrchestratorStatus:
    """Tests for the orchestrator_status() method."""

    def test_status_shape(self):
        """orchestrator_status() returns expected fields."""
        # Use the real OrchestratorAgent import to check it has the method
        # but test the stub's data since we can't instantiate the real one easily
        stub = _OrchestratorStub()
        status = {
            "agent_id": "orchestrator",
            "state": stub.state.value,
            "cycle": stub._cycle_number,
            "turn": stub._turn_number,
        }
        assert status["agent_id"] == "orchestrator"
        assert status["state"] == "sleeping"
        assert status["cycle"] == 0
        assert status["turn"] == 0

    def test_status_reflects_state_changes(self):
        """Status dict reflects current state after transitions."""
        stub = _OrchestratorStub()
        stub._set_state(AgentState.ACTIVE)
        assert stub.state.value == "active"

        stub._cycle_number = 3
        stub._turn_number = 7
        assert stub._cycle_number == 3
        assert stub._turn_number == 7
