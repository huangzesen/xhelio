"""Control Center — thread-safe registry of all in-flight work units.

Replaces ``async_delegation.py`` with a superset that provides:
- Per-unit cancel events (selective cancellation)
- Compact status line for automatic context injection
- Cancel by unit ID, agent type, or all
- Shared ``threading.Condition`` with ``InputQueue`` so the orchestrator
  wakes on either new user input or work completion

The orchestrator LLM interacts with the Control Center via two tools:
``list_active_work`` and ``cancel_work``.
"""

from __future__ import annotations

import enum
import threading
import time
import typing
import uuid
from dataclasses import dataclass, field
from typing import Any


class WorkStatus(enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkUnit:
    """Tracks a single unit of in-flight work (delegation, planner, etc.)."""

    id: str
    kind: str  # "delegation", "planner"
    agent_type: str  # "mission", "data_ops", "data_extraction", "viz", "planner"
    agent_name: str  # "MissionAgent[ACE]", "DataOpsAgent", etc.
    task_summary: str  # Human-readable summary
    status: WorkStatus
    cancel_event: threading.Event  # Per-unit cancellation signal
    thread: threading.Thread | None = None
    started_at: float = 0.0
    completed_at: float | None = None
    tool_call_id: str | None = None  # LLM's tool_call_id for result mapping
    result: dict | None = None
    operation_log: list[dict] = field(default_factory=list)
    error: str | None = None
    _collected: bool = field(default=False, repr=False)


def _make_unit_id() -> str:
    return f"wu_{uuid.uuid4().hex[:8]}"


class ControlCenter:
    """Thread-safe registry of all in-flight work units.

    Shares a ``threading.Condition`` with ``InputQueue`` so the orchestrator's
    ``_wait_for_event()`` wakes on either new user input or work completion.

    Args:
        cond: Shared condition variable. If None, creates its own.
    """

    def __init__(self, cond: threading.Condition | None = None) -> None:
        if cond is None:
            self._lock = threading.RLock()
            self._cond = threading.Condition(self._lock)
        else:
            self._cond = cond
            self._lock = cond._lock  # type: ignore[attr-defined]
        self._units: dict[str, WorkUnit] = {}
        # Completion callbacks keyed by agent_type — fire-and-forget when a
        # unit of that type completes or fails.
        self._completion_callbacks: dict[str, list[typing.Callable[[WorkUnit], None]]] = {}

    # ── Registration (called from orchestrator thread) ────────────

    def register(
        self,
        kind: str,
        agent_type: str,
        agent_name: str,
        task_summary: str,
        tool_call_id: str | None = None,
    ) -> WorkUnit:
        """Create and register a new RUNNING work unit.

        Returns the unit with a fresh per-unit cancel_event.
        """
        unit = WorkUnit(
            id=_make_unit_id(),
            kind=kind,
            agent_type=agent_type,
            agent_name=agent_name,
            task_summary=task_summary,
            status=WorkStatus.RUNNING,
            cancel_event=threading.Event(),
            started_at=time.monotonic(),
            tool_call_id=tool_call_id,
        )
        with self._cond:
            self._units[unit.id] = unit
        return unit

    # ── Completion (called from worker threads) ───────────────────

    def mark_completed(
        self,
        unit_id: str,
        result: dict,
        operation_log: list[dict] | None = None,
    ) -> None:
        """Mark a work unit as completed. Wakes the orchestrator."""
        callbacks: list[typing.Callable] = []
        with self._cond:
            unit = self._units.get(unit_id)
            if unit is None or unit.status != WorkStatus.RUNNING:
                return
            unit.status = WorkStatus.COMPLETED
            unit.result = result
            unit.operation_log = operation_log or []
            unit.completed_at = time.monotonic()
            # Pop one-shot callbacks for this agent_type
            callbacks = self._completion_callbacks.pop(unit.agent_type, [])
            self._cond.notify_all()
        # Fire callbacks outside the lock
        for cb in callbacks:
            try:
                cb(unit)
            except Exception:
                pass  # fire-and-forget

    def mark_failed(self, unit_id: str, error: str) -> None:
        """Mark a work unit as failed. Wakes the orchestrator."""
        callbacks: list[typing.Callable] = []
        with self._cond:
            unit = self._units.get(unit_id)
            if unit is None or unit.status != WorkStatus.RUNNING:
                return
            unit.status = WorkStatus.FAILED
            unit.error = error
            unit.result = {"status": "error", "message": error}
            unit.completed_at = time.monotonic()
            callbacks = self._completion_callbacks.pop(unit.agent_type, [])
            self._cond.notify_all()
        for cb in callbacks:
            try:
                cb(unit)
            except Exception:
                pass

    # ── Query (called from orchestrator thread) ───────────────────

    def list_active(self) -> list[dict]:
        """Return summary dicts of all RUNNING units. Used by LLM tool."""
        now = time.monotonic()
        with self._cond:
            return [
                {
                    "id": u.id,
                    "kind": u.kind,
                    "agent_type": u.agent_type,
                    "agent_name": u.agent_name,
                    "task_summary": u.task_summary,
                    "elapsed_s": round(now - u.started_at, 1),
                }
                for u in self._units.values()
                if u.status == WorkStatus.RUNNING
            ]

    def status_line(self) -> str:
        """Compact one-liner for automatic context injection.

        Returns empty string if nothing is running.
        Example: ``[Active: MissionAgent[ACE] (3.2s), MissionAgent[PSP] (1.8s)]``
        """
        active = self.list_active()
        if not active:
            return ""
        parts = [f"{a['agent_name']} ({a['elapsed_s']}s)" for a in active]
        return f"[Active work: {', '.join(parts)}]"

    def has_pending(self) -> bool:
        """Return True if any work unit is still RUNNING."""
        with self._cond:
            return any(
                u.status == WorkStatus.RUNNING for u in self._units.values()
            )

    def get_pending_types(self) -> set[str]:
        """Return the set of ``agent_type`` strings for all RUNNING work units."""
        with self._cond:
            return {
                u.agent_type
                for u in self._units.values()
                if u.status == WorkStatus.RUNNING
            }

    def get(self, unit_id: str) -> WorkUnit | None:
        """Get a specific unit by ID."""
        with self._cond:
            return self._units.get(unit_id)

    def register_completion_callback(
        self,
        agent_type: str,
        callback: typing.Callable[[WorkUnit], None],
    ) -> None:
        """Register a fire-and-forget callback for when a unit of *agent_type* completes.

        The callback is invoked on the worker thread (inside ``mark_completed``
        or ``mark_failed``) with the ``WorkUnit`` as its sole argument.
        Callbacks are one-shot: consumed when fired.
        """
        with self._cond:
            self._completion_callbacks.setdefault(agent_type, []).append(callback)

    # ── Control (called from orchestrator thread or tool handler) ─

    def cancel(self, unit_id: str) -> bool:
        """Cancel a specific work unit. Returns False if not found or not running."""
        with self._cond:
            unit = self._units.get(unit_id)
            if unit is None or unit.status != WorkStatus.RUNNING:
                return False
            unit.cancel_event.set()
            unit.status = WorkStatus.CANCELLED
            unit.completed_at = time.monotonic()
            self._cond.notify_all()
            return True

    def cancel_by_type(self, agent_type: str) -> int:
        """Cancel all RUNNING units of a given agent type. Returns count cancelled."""
        count = 0
        with self._cond:
            for unit in self._units.values():
                if unit.status == WorkStatus.RUNNING and unit.agent_type == agent_type:
                    unit.cancel_event.set()
                    unit.status = WorkStatus.CANCELLED
                    unit.completed_at = time.monotonic()
                    count += 1
            if count:
                self._cond.notify_all()
        return count

    def cancel_all(self) -> int:
        """Cancel all RUNNING work units. Returns count cancelled."""
        count = 0
        with self._cond:
            for unit in self._units.values():
                if unit.status == WorkStatus.RUNNING:
                    unit.cancel_event.set()
                    unit.status = WorkStatus.CANCELLED
                    unit.completed_at = time.monotonic()
                    count += 1
            if count:
                self._cond.notify_all()
        return count

    # ── Wait (called from orchestrator event loop) ────────────────

    def wait_for_any(
        self,
        cancel_event: threading.Event | None = None,
        timeout: float = 300.0,
        wake_on_input: typing.Callable[[], bool] | None = None,
    ) -> list[WorkUnit]:
        """Block until at least one unit completes/fails/is cancelled,
        or until user input arrives (if *wake_on_input* is provided).

        Args:
            cancel_event: If set, return immediately with [].
            timeout: Maximum seconds to wait.
            wake_on_input: Callable returning True when the orchestrator
                should wake to process user input (e.g.
                ``input_queue.has_pending``).  When it fires, returns
                whatever is ready (may be []).

        Returns newly-finished units (collected exactly once).
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                ready = [
                    u for u in self._units.values()
                    if u.status in (WorkStatus.COMPLETED, WorkStatus.FAILED, WorkStatus.CANCELLED)
                    and not u._collected
                ]
                if ready:
                    for u in ready:
                        u._collected = True
                    return ready

                if cancel_event and cancel_event.is_set():
                    return []

                # Wake early when user input is pending
                if wake_on_input and wake_on_input():
                    return []

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []

                self._cond.wait(timeout=min(0.5, remaining))

    def drain_completed(self) -> list[WorkUnit]:
        """Non-blocking: return all completed/failed/cancelled units not yet collected."""
        with self._cond:
            ready = [
                u for u in self._units.values()
                if u.status in (WorkStatus.COMPLETED, WorkStatus.FAILED, WorkStatus.CANCELLED)
                and not u._collected
            ]
            for u in ready:
                u._collected = True
            return ready

    def has_newly_finished(self) -> bool:
        """Check if any units have finished but haven't been collected yet."""
        with self._cond:
            return any(
                u.status in (WorkStatus.COMPLETED, WorkStatus.FAILED, WorkStatus.CANCELLED)
                and not u._collected
                for u in self._units.values()
            )

    # ── Cleanup ───────────────────────────────────────────────────

    def clear(self) -> None:
        """Cancel all running units and remove all units. Called at session reset."""
        with self._cond:
            for unit in self._units.values():
                if unit.status == WorkStatus.RUNNING:
                    unit.cancel_event.set()
            self._units.clear()
            self._completion_callbacks.clear()
            self._cond.notify_all()
